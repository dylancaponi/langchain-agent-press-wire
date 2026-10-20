"""x402 payment handling for Agent Press Wire.

Pure helpers: read the 402 terms, check them against the pinned Agent Press Wire
terms and the caller's cap, and sign an exact EIP-3009 USDC authorization with the
official ``x402`` SDK.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from x402 import x402Client, x402ClientSync
from x402.http.utils import (
    decode_payment_required_header,
    decode_payment_response_header,
    encode_payment_signature_header,
)
from x402.schemas import PaymentPayload, PaymentRequired, PaymentRequirements

NETWORK = "eip155:8453"  # Base mainnet
USDC_BASE = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
USDC_DECIMALS = 6
# Agent Press Wire's receiving wallet. Pinned so a changed base_url or a
# compromised server cannot redirect funds.
PAY_TO = "0xF7887C1fAf0DbAedc80C78F9D396D594B69eba95"
# Listed price per endpoint, in atomic USDC. Any other amount is refused.
EXPECTED_ATOMIC: dict[str, int] = {
    "/v1/press-release": 1_000_000,
    "/v1/wire-release": 99_000_000,
    "/v1/wire-release/premium": 299_000_000,
}
MAX_TIMEOUT_SECONDS = 600

PAYMENT_REQUIRED_HEADER = "PAYMENT-REQUIRED"
PAYMENT_SIGNATURE_HEADER = "PAYMENT-SIGNATURE"
PAYMENT_RESPONSE_HEADER = "PAYMENT-RESPONSE"


class PaymentRefused(Exception):
    """Raised when the client refuses to pay. Nothing was signed or sent."""


def usd_to_atomic(usd: float | Decimal | str) -> int:
    return int((Decimal(str(usd)) * 10**USDC_DECIMALS).to_integral_value())


def atomic_to_usd(atomic: int | str) -> str:
    return f"{Decimal(int(atomic)) / 10**USDC_DECIMALS:.2f}"


def parse_payment_required(headers: Any) -> PaymentRequired:
    """Decode the x402 v2 PAYMENT-REQUIRED header of a 402 response."""
    raw = headers.get(PAYMENT_REQUIRED_HEADER)
    if not raw:
        raise PaymentRefused("402 response has no PAYMENT-REQUIRED header")
    try:
        decoded = decode_payment_required_header(raw)
    except Exception as e:  # noqa: BLE001
        raise PaymentRefused(f"could not parse x402 payment terms: {e}") from e
    if not isinstance(decoded, PaymentRequired):
        raise PaymentRefused("server sent x402 v1 terms; only v2 is supported")
    return decoded


def check_pinned_terms(path: str, req: PaymentRequirements) -> str | None:
    """Return None when the terms match Agent Press Wire's listed terms, else a reason."""
    expected = EXPECTED_ATOMIC.get(path)
    if expected is None:
        return f"no pinned price for endpoint {path}"
    if req.scheme != "exact":
        return f"unexpected x402 scheme {req.scheme!r} (expected 'exact')"
    if req.network != NETWORK:
        return f"unexpected network {req.network!r} (expected {NETWORK})"
    if str(req.asset).lower() != USDC_BASE:
        return f"unexpected asset {req.asset!r} (expected USDC on Base)"
    method = (req.extra or {}).get("assetTransferMethod")
    if method is not None and method != "eip3009":
        return f"unexpected assetTransferMethod {method!r} (only eip3009 is allowed)"
    if str(req.pay_to).lower() != PAY_TO.lower():
        return f"payTo {req.pay_to} is not the Agent Press Wire wallet {PAY_TO}"
    if not str(req.amount).isdigit() or int(req.amount) != expected:
        return (
            f"amount {req.amount} does not match the listed price "
            f"{expected} atomic USDC (${atomic_to_usd(expected)}) for {path}"
        )
    t = req.max_timeout_seconds
    if not isinstance(t, int) or t <= 0 or t > MAX_TIMEOUT_SECONDS:
        return f"maxTimeoutSeconds {t!r} is outside 1..{MAX_TIMEOUT_SECONDS}"
    return None


def select_requirement(path: str, terms: PaymentRequired) -> PaymentRequirements:
    """Pick the accepted option that matches the pinned terms, or refuse."""
    reasons = []
    for req in terms.accepts:
        reason = check_pinned_terms(path, req)
        if reason is None:
            return req
        reasons.append(reason)
    raise PaymentRefused(
        "the server's x402 terms do not match Agent Press Wire's listed terms: "
        + ("; ".join(reasons) or "no payment options offered")
    )


def build_quote(path: str, req: PaymentRequirements, max_payment_usd: float) -> dict:
    return {
        "endpoint": path,
        "price_usd": atomic_to_usd(req.amount),
        "amount_atomic": str(req.amount),
        "asset": "USDC",
        "network": req.network,
        "pay_to": req.pay_to,
        "max_payment_usd": f"{Decimal(str(max_payment_usd)):.2f}",
        "within_cap": int(req.amount) <= usd_to_atomic(max_payment_usd),
    }


def check_cap(req: PaymentRequirements, max_payment_usd: float) -> None:
    cap = usd_to_atomic(max_payment_usd)
    if int(req.amount) > cap:
        raise PaymentRefused(
            f"price ${atomic_to_usd(req.amount)} is above max_payment_usd "
            f"${atomic_to_usd(cap)}. Confirm the price with the user and raise "
            "max_payment_usd for this call if they approve."
        )


def _register(client: Any, private_key: str, cap_atomic: int) -> Any:
    from eth_account import Account
    from x402 import max_amount
    from x402.mechanisms.evm import EthAccountSigner
    from x402.mechanisms.evm.exact import register_exact_evm_client

    signer = EthAccountSigner(Account.from_key(private_key))
    # Second guard: the SDK's own spend control and requirement filter, set to
    # the same cap. (The SDK defaults to $1 per payment; this replaces it.)
    client.set_spend_controls({"max_amount_per_payment": f"${atomic_to_usd(cap_atomic)}"})
    return register_exact_evm_client(
        client, signer, networks=NETWORK, policies=[max_amount(cap_atomic)]
    )


def _payment_headers(payload: PaymentPayload, req: PaymentRequirements) -> dict:
    auth = (payload.payload or {}).get("authorization") or {}
    if str(auth.get("to", "")).lower() != PAY_TO.lower() or str(auth.get("value")) != str(
        req.amount
    ):
        raise PaymentRefused("signed authorization does not match the pinned payTo and price")
    return {PAYMENT_SIGNATURE_HEADER: encode_payment_signature_header(payload)}


def sign_payment(
    terms: PaymentRequired,
    req: PaymentRequirements,
    private_key: str,
    max_payment_usd: float,
) -> dict:
    """Sign a payment for exactly ``req``. Returns headers for the paid retry."""
    client = _register(x402ClientSync(), private_key, usd_to_atomic(max_payment_usd))
    only = terms.model_copy(update={"accepts": [req]})
    try:
        payload = client.create_payment_payload(only)
    except Exception as e:  # noqa: BLE001
        raise PaymentRefused(f"could not create payment: {e}") from e
    return _payment_headers(payload, req)


async def asign_payment(
    terms: PaymentRequired,
    req: PaymentRequirements,
    private_key: str,
    max_payment_usd: float,
) -> dict:
    client = _register(x402Client(), private_key, usd_to_atomic(max_payment_usd))
    only = terms.model_copy(update={"accepts": [req]})
    try:
        payload = await client.create_payment_payload(only)
    except Exception as e:  # noqa: BLE001
        raise PaymentRefused(f"could not create payment: {e}") from e
    return _payment_headers(payload, req)


def read_settlement(headers: Any) -> dict | None:
    raw = headers.get(PAYMENT_RESPONSE_HEADER) or headers.get("X-PAYMENT-RESPONSE")
    if not raw:
        return None
    try:
        return decode_payment_response_header(raw).model_dump(by_alias=True, exclude_none=True)
    except Exception:  # noqa: BLE001
        return None
