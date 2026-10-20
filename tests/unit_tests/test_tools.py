import json

import httpx
import pytest
import respx
from eth_account import Account
from langchain_core.tools import ToolException
from x402.http.utils import decode_payment_signature_header, encode_payment_required_header
from x402.schemas import PaymentRequired, PaymentRequirements

from langchain_agent_press_wire import (
    AgentPressWireStatusTool,
    AgentPressWireTool,
    AgentPressWireWireTool,
)
from langchain_agent_press_wire._payment import PAY_TO, USDC_BASE

BASE = "https://agentpresswire.dev"
KEY = Account.create().key.hex()
ARGS = {"product_url": "https://example.com", "announcement": "Example 2.0 ships"}
WIRE_ARGS = {**ARGS, "city": "Los Angeles", "country": "USA", "contact_email": "a@example.com"}


def terms(amount="1000000", pay_to=PAY_TO, asset=USDC_BASE, network="eip155:8453"):
    pr = PaymentRequired(
        accepts=[
            PaymentRequirements(
                scheme="exact",
                network=network,
                asset=asset,
                amount=amount,
                pay_to=pay_to,
                max_timeout_seconds=300,
                extra={"name": "USD Coin", "version": "2"},
            )
        ]
    )
    return httpx.Response(
        402, json={}, headers={"PAYMENT-REQUIRED": encode_payment_required_header(pr)}
    )


def paid_only(success: httpx.Response):
    """First call returns 402 terms, the paid retry returns ``success``."""
    calls = []

    def handler(request: httpx.Request):
        calls.append(request)
        if "PAYMENT-SIGNATURE" in request.headers:
            return success
        return terms()

    return handler, calls


def run(tool, args):
    # Invoke through the public API; handle_tool_error=False surfaces ToolException.
    tool.handle_tool_error = False
    return json.loads(tool.invoke(args))


@respx.mock
def test_quote_only_does_not_sign_or_pay():
    route = respx.post(f"{BASE}/v1/press-release").mock(return_value=terms())
    out = run(AgentPressWireTool(private_key=KEY), {**ARGS, "quote_only": True})
    assert out["mode"] == "quote" and out["paid"] is False
    assert out["quote"]["price_usd"] == "1.00" and out["quote"]["within_cap"] is True
    assert route.call_count == 1
    assert "PAYMENT-SIGNATURE" not in route.calls[0].request.headers


@respx.mock
def test_quote_works_without_key():
    respx.post(f"{BASE}/v1/wire-release").mock(return_value=terms("99000000"))
    out = run(AgentPressWireWireTool(private_key=None, quote_only=True), WIRE_ARGS)
    assert out["quote"]["price_usd"] == "99.00" and out["quote"]["within_cap"] is False


@respx.mock
def test_pays_one_dollar_and_returns_release():
    release = {
        "id": "r_1",
        "headline": "Example Ships 2.0",
        "newsroom_url": f"{BASE}/r/r_1",
        "release_markdown": "text",
    }
    handler, calls = paid_only(httpx.Response(200, json=release))
    respx.post(f"{BASE}/v1/press-release").mock(side_effect=handler)
    out = run(AgentPressWireTool(private_key=KEY), ARGS)
    assert out["mode"] == "paid" and out["paid"] is True and out["price_usd"] == "1.00"
    assert out["result"]["newsroom_url"].endswith("/r/r_1")
    assert len(calls) == 2
    sent = json.loads(calls[1].content)
    assert sent == ARGS
    payload = decode_payment_signature_header(calls[1].headers["PAYMENT-SIGNATURE"])
    auth = payload.payload["authorization"]
    assert auth["to"].lower() == PAY_TO.lower() and auth["value"] == "1000000"
    assert auth["from"].lower() == Account.from_key(KEY).address.lower()


@respx.mock
async def test_async_pays():
    handler, calls = paid_only(httpx.Response(200, json={"id": "r_2"}))
    respx.post(f"{BASE}/v1/press-release").mock(side_effect=handler)
    tool = AgentPressWireTool(private_key=KEY, handle_tool_error=False)
    out = json.loads(await tool.ainvoke(ARGS))
    assert out["paid"] is True and len(calls) == 2


@respx.mock
def test_default_cap_refuses_wire_tier():
    route = respx.post(f"{BASE}/v1/wire-release").mock(return_value=terms("99000000"))
    with pytest.raises(ToolException, match="above max_payment_usd"):
        run(AgentPressWireWireTool(private_key=KEY), WIRE_ARGS)
    assert route.call_count == 1  # never retried with a payment


@respx.mock
def test_raised_cap_pays_premium_tier():
    handler, calls = paid_only(httpx.Response(200, json={"order_id": "w_ab", "status": "queued"}))

    def premium(request):
        if "PAYMENT-SIGNATURE" in request.headers:
            return handler(request)
        calls.append(request)
        return terms("299000000")

    respx.post(f"{BASE}/v1/wire-release/premium").mock(side_effect=premium)
    out = run(
        AgentPressWireWireTool(private_key=KEY, max_payment_usd=299),
        {**WIRE_ARGS, "tier": "premium"},
    )
    assert out["paid"] is True and out["result"]["order_id"] == "w_ab"
    assert json.loads(calls[-1].content)["contact_email"] == "a@example.com"


@pytest.mark.parametrize(
    "bad",
    [
        {"pay_to": "0x0000000000000000000000000000000000000001"},
        {"asset": "0x0000000000000000000000000000000000000002"},
        {"network": "eip155:1"},
        {"amount": "500000"},  # differs from the listed price even though under the cap
    ],
)
@respx.mock
def test_refuses_terms_that_do_not_match_pinned(bad):
    route = respx.post(f"{BASE}/v1/press-release").mock(return_value=terms(**bad))
    with pytest.raises(ToolException, match="Refusing to pay"):
        run(AgentPressWireTool(private_key=KEY, max_payment_usd=5), ARGS)
    assert route.call_count == 1


@respx.mock
def test_422_after_payment_is_reported_not_charged():
    details = {"error": "rejected", "details": ["headline must name the company"]}
    handler, _ = paid_only(httpx.Response(422, json=details))
    respx.post(f"{BASE}/v1/press-release").mock(side_effect=handler)
    with pytest.raises(ToolException, match="422, not charged") as e:
        run(AgentPressWireTool(private_key=KEY), ARGS)
    assert "headline must name the company" in str(e.value)


@respx.mock
def test_422_before_payment_is_reported_not_charged():
    respx.post(f"{BASE}/v1/press-release").mock(
        return_value=httpx.Response(422, json={"error": "Tell us what you are announcing"})
    )
    with pytest.raises(ToolException, match="not charged"):
        run(AgentPressWireTool(private_key=KEY), ARGS)


@respx.mock
def test_409_duplicate_is_not_charged_again():
    handler, _ = paid_only(httpx.Response(409, json={"id": "r_1", "status_url": "x"}))
    respx.post(f"{BASE}/v1/press-release").mock(side_effect=handler)
    out = run(AgentPressWireTool(private_key=KEY), ARGS)
    assert out["mode"] == "duplicate" and out["charged_again"] is False


@respx.mock
def test_missing_key_refuses_to_pay():
    respx.post(f"{BASE}/v1/press-release").mock(return_value=terms())
    with pytest.raises(ToolException, match="X402_PRIVATE_KEY"):
        run(AgentPressWireTool(private_key=None), ARGS)


@respx.mock
def test_tool_error_returned_to_agent_by_default():
    respx.post(f"{BASE}/v1/wire-release").mock(return_value=terms("99000000"))
    msg = AgentPressWireWireTool(private_key=KEY).invoke(WIRE_ARGS)
    assert "Refusing to pay" in msg


def test_input_validation():
    tool = AgentPressWireTool(private_key=KEY)
    with pytest.raises(Exception):
        tool.invoke({**ARGS, "announcement": "x" * 501})


@respx.mock
def test_status_tool():
    respx.get(f"{BASE}/v1/wire-release/w_ab").mock(
        return_value=httpx.Response(200, json={"order_id": "w_ab", "status": "distributed"})
    )
    out = json.loads(AgentPressWireStatusTool().invoke({"order_id": "w_ab"}))
    assert out["status"] == "distributed"
