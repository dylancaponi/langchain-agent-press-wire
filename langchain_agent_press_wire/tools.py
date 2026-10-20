"""LangChain tools for Agent Press Wire (https://agentpresswire.dev)."""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from langchain_agent_press_wire._payment import (
    PaymentRefused,
    asign_payment,
    build_quote,
    check_cap,
    parse_payment_required,
    read_settlement,
    select_requirement,
    sign_payment,
)

DEFAULT_BASE_URL = "https://agentpresswire.dev"
USER_AGENT = "langchain-agent-press-wire"

_QUOTE_ONLY_FIELD = Field(
    default=False,
    description=(
        "If true, do not pay. Return the x402 price quote only, so the price can "
        "be shown to the user first."
    ),
)


class PressReleaseInput(BaseModel):
    """Input for the $1 newsroom release."""

    product_url: str = Field(
        description="Public URL of the product, launch page, or repo being announced."
    )
    announcement: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "What is being announced (new version, feature, launch, funding, "
            "milestone). Strongly recommended: without it the release only covers "
            "news shown on the page, and the API returns 422 (not charged) if "
            "there is none."
        ),
    )
    company: str | None = Field(
        default=None, max_length=100, description="Company or project name."
    )
    contact: str | None = Field(
        default=None,
        max_length=200,
        description="Media contact line (name, email or URL) printed at the end.",
    )
    headline: str | None = Field(
        default=None,
        max_length=200,
        description=(
            "Bring your own copy (send together with body): finished headline, 3 to "
            "20 words, naming the announcing company. Leave headline and body empty "
            "to have the release drafted from the page."
        ),
    )
    body: str | None = Field(
        default=None,
        max_length=12000,
        description=(
            "Bring your own copy (send together with headline): plain English text, "
            "paragraphs separated by blank lines, 150 to 900 words (380 to 900 on "
            "wire tiers), at most 5 links and only to the product_url site."
        ),
    )
    quote_only: bool = _QUOTE_ONLY_FIELD


class WireReleaseInput(PressReleaseInput):
    """Input for newswire distribution ($99 standard, $299 premium)."""

    city: str = Field(max_length=80, description="Dateline city, e.g. 'San Francisco'.")
    country: str = Field(max_length=80, description="Dateline country, e.g. 'USA'.")
    contact_email: str = Field(
        description="Media contact email printed on the release and used by the wire."
    )
    tier: Literal["standard", "premium"] = Field(
        default="standard",
        description=(
            "'standard' ($99): 500+ news sites plus 100+ TV and radio network sites. "
            "'premium' ($299): adds AP News, USA Today, Barchart and StreetInsider."
        ),
    )


class WireStatusInput(BaseModel):
    """Input for the free wire order status lookup."""

    order_id: str = Field(
        pattern=r"^w_[0-9a-f]+$", description="order_id returned by a wire release."
    )


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def _body_of(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:2000]


class _AgentPressWireBase(BaseTool):
    """Shared config and HTTP handling for the Agent Press Wire tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    private_key: SecretStr | None = Field(
        default_factory=secret_from_env("X402_PRIVATE_KEY", default=None)
    )
    """Hex private key of the Base wallet that pays (USDC). Read from
    ``X402_PRIVATE_KEY`` when not passed. Not needed for quotes."""

    max_payment_usd: float = 1.0
    """Refuse any 402 asking more than this, in USD. Default $1. Wire tiers need
    99 or 299; raise it only after the user approves the price."""

    quote_only: bool = False
    """If true, never pay: every call returns the 402 price quote."""

    base_url: str = DEFAULT_BASE_URL
    timeout: float = 120.0
    handle_tool_error: bool | str | Any = True

    def _headers(self) -> dict:
        return {"user-agent": USER_AGENT, "accept": "application/json"}

    def _key(self) -> str:
        if self.private_key is None or not self.private_key.get_secret_value():
            raise ToolException(
                "No wallet key: set X402_PRIVATE_KEY or pass private_key to pay. "
                "Use quote_only to see the price without paying."
            )
        return self.private_key.get_secret_value()

    # ---------- response handling ----------

    @staticmethod
    def _unpaid_result(path: str, response: httpx.Response) -> str:
        """Handle a non-402 answer to the unpaid request."""
        status = response.status_code
        body = _body_of(response)
        if 200 <= status < 300:
            return _json({"mode": "result", "paid": False, "result": body})
        if status == 409:
            return _json(
                {"mode": "duplicate", "paid": False, "charged_again": False, "result": body}
            )
        raise ToolException(
            f"Agent Press Wire rejected the request to {path} (HTTP {status}, not "
            f"charged): {_json(body)}"
        )

    @staticmethod
    def _paid_result(
        path: str, response: httpx.Response, quote: dict, settlement: dict | None
    ) -> str:
        status = response.status_code
        body = _body_of(response)
        if 200 <= status < 300:
            return _json(
                {
                    "mode": "paid",
                    "paid": True,
                    "price_usd": quote["price_usd"],
                    "settlement": settlement,
                    "result": body,
                }
            )
        if status == 409:
            return _json(
                {
                    "mode": "duplicate",
                    "paid": False,
                    "charged_again": False,
                    "note": "Same request from this wallet within 24 h; existing order returned.",
                    "result": body,
                }
            )
        if status == 422:
            raise ToolException(
                f"Agent Press Wire could not accept this release (HTTP 422, not "
                f"charged). Fix the listed issues and retry: {_json(body)}"
            )
        if status == 402:
            raise ToolException(
                f"Payment was rejected (HTTP 402, not charged), for example "
                f"insufficient USDC balance on Base: {_json(body)}"
            )
        raise ToolException(
            f"Agent Press Wire returned HTTP {status} after payment was sent. The "
            f"API settles only on success, so this should not be charged; check the "
            f"wallet before retrying. Response: {_json(body)}"
        )

    def _prepare(self, path: str, terms_headers: Any, quote_only: bool):
        try:
            terms = parse_payment_required(terms_headers)
            req = select_requirement(path, terms)
        except PaymentRefused as e:
            raise ToolException(f"Refusing to pay: {e}") from e
        quote = build_quote(path, req, self.max_payment_usd)
        if quote_only or self.quote_only:
            return (
                terms,
                req,
                quote,
                _json(
                    {
                        "mode": "quote",
                        "paid": False,
                        "quote": quote,
                        "next_step": (
                            "Show the user this price. To pay, call again with "
                            "quote_only=false; max_payment_usd must be at least the price."
                        ),
                    }
                ),
            )
        try:
            check_cap(req, self.max_payment_usd)
        except PaymentRefused as e:
            raise ToolException(f"Refusing to pay: {e} Quote: {_json(quote)}") from e
        return terms, req, quote, None

    # ---------- sync / async request flows ----------

    def _post(self, path: str, payload: dict, quote_only: bool) -> str:
        url = self.base_url.rstrip("/") + path
        with httpx.Client(timeout=self.timeout, headers=self._headers()) as http:
            first = http.post(url, json=payload)
            if first.status_code != 402:
                return self._unpaid_result(path, first)
            terms, req, quote, early = self._prepare(path, first.headers, quote_only)
            if early is not None:
                return early
            try:
                pay_headers = sign_payment(terms, req, self._key(), self.max_payment_usd)
            except PaymentRefused as e:
                raise ToolException(f"Refusing to pay: {e}") from e
            try:
                paid = http.post(url, json=payload, headers=pay_headers)
            except httpx.HTTPError as e:
                raise ToolException(_lost_answer(e)) from e
            return self._paid_result(path, paid, quote, read_settlement(paid.headers))

    async def _apost(self, path: str, payload: dict, quote_only: bool) -> str:
        url = self.base_url.rstrip("/") + path
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers()) as http:
            first = await http.post(url, json=payload)
            if first.status_code != 402:
                return self._unpaid_result(path, first)
            terms, req, quote, early = self._prepare(path, first.headers, quote_only)
            if early is not None:
                return early
            try:
                pay_headers = await asign_payment(terms, req, self._key(), self.max_payment_usd)
            except PaymentRefused as e:
                raise ToolException(f"Refusing to pay: {e}") from e
            try:
                paid = await http.post(url, json=payload, headers=pay_headers)
            except httpx.HTTPError as e:
                raise ToolException(_lost_answer(e)) from e
            return self._paid_result(path, paid, quote, read_settlement(paid.headers))


def _lost_answer(e: Exception) -> str:
    return (
        f"Network error after sending a signed payment: {e}. Settlement status is "
        "unknown. Repeating the same request from the same wallet within 24 h is "
        "safe: the API returns the existing order (HTTP 409) instead of charging again."
    )


def _release_payload(fields: dict, extra: tuple[str, ...] = ()) -> dict:
    keys = ("product_url", "announcement", "company", "contact", "headline", "body") + extra
    return {k: fields[k] for k in keys if fields.get(k) not in (None, "")}


class AgentPressWireTool(_AgentPressWireBase):
    """Write and publish a press release on the Agent Press Wire newsroom ($1 USDC).

    Pays per call over x402 in USDC on Base. Refuses any price above
    ``max_payment_usd`` (default $1). Only charged if the release is published.

    Setup:
        .. code-block:: bash

            pip install -U langchain-agent-press-wire
            export X402_PRIVATE_KEY=0x...  # Base wallet holding USDC

    Instantiate:
        .. code-block:: python

            from langchain_agent_press_wire import AgentPressWireTool

            tool = AgentPressWireTool()  # max_payment_usd=1.0

    Invoke:
        .. code-block:: python

            tool.invoke(
                {
                    "product_url": "https://example.com",
                    "announcement": "Example 2.0 adds offline mode",
                }
            )
    """

    name: str = "agent_press_wire_release"
    description: str = (
        "Write and publish a press release for a product launch, new version, "
        "feature, funding round or milestone. Publishes it on a public, "
        "search-indexed newsroom page that links to the product. Costs $1.00 USDC, "
        "charged only on success. Returns the headline, newsroom_url and the full "
        "release text. Use quote_only=true to get the price without paying."
    )
    args_schema: type[BaseModel] = PressReleaseInput

    def _run(
        self,
        product_url: str,
        quote_only: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        payload = _release_payload({"product_url": product_url, **kwargs})
        return self._post("/v1/press-release", payload, quote_only)

    async def _arun(
        self,
        product_url: str,
        quote_only: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        payload = _release_payload({"product_url": product_url, **kwargs})
        return await self._apost("/v1/press-release", payload, quote_only)


class AgentPressWireWireTool(_AgentPressWireBase):
    """Distribute a press release over a newswire ($99 standard, $299 premium).

    The default cap of $1 refuses both tiers. Set ``max_payment_usd`` to 99 or
    299 only after the user approves the price. Returns an ``order_id``; check
    progress with ``AgentPressWireStatusTool``.

    Instantiate:
        .. code-block:: python

            from langchain_agent_press_wire import AgentPressWireWireTool

            tool = AgentPressWireWireTool(max_payment_usd=99)
    """

    name: str = "agent_press_wire_wire_release"
    description: str = (
        "Distribute a product-launch press release over a newswire: 'standard' "
        "tier ($99 USDC) reaches 500+ news sites plus 100+ TV and radio network "
        "sites; 'premium' ($299 USDC) adds AP News, USA Today, Barchart and "
        "StreetInsider. Charged only on success. Always confirm the price with the "
        "user first (call with quote_only=true). Returns an order_id to check with "
        "agent_press_wire_wire_status. Financial or market-moving news is refused."
    )
    args_schema: type[BaseModel] = WireReleaseInput

    def _payload(self, kwargs: dict) -> tuple[str, dict]:
        tier = kwargs.pop("tier", "standard") or "standard"
        path = "/v1/wire-release/premium" if tier == "premium" else "/v1/wire-release"
        return path, _release_payload(kwargs, ("city", "country", "contact_email"))

    def _run(
        self,
        quote_only: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        path, payload = self._payload(kwargs)
        return self._post(path, payload, quote_only)

    async def _arun(
        self,
        quote_only: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> str:
        path, payload = self._payload(kwargs)
        return await self._apost(path, payload, quote_only)


class AgentPressWireStatusTool(BaseTool):
    """Check a wire order's status and live links (free, no wallet needed)."""

    name: str = "agent_press_wire_wire_status"
    description: str = (
        "Check the status of an Agent Press Wire newswire order by order_id. Free. "
        "Returns status, status_reason, headline, newsroom_url and live links once "
        "distributed."
    )
    args_schema: type[BaseModel] = WireStatusInput
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0
    handle_tool_error: bool | str | Any = True

    def _url(self, order_id: str) -> str:
        return f"{self.base_url.rstrip('/')}/v1/wire-release/{order_id}"

    @staticmethod
    def _result(response: httpx.Response) -> str:
        body = _body_of(response)
        if response.status_code == 200:
            return _json(body)
        raise ToolException(f"Status lookup failed (HTTP {response.status_code}): {_json(body)}")

    def _run(self, order_id: str, run_manager: CallbackManagerForToolRun | None = None) -> str:
        with httpx.Client(timeout=self.timeout, headers={"user-agent": USER_AGENT}) as http:
            return self._result(http.get(self._url(order_id)))

    async def _arun(
        self, order_id: str, run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> str:
        async with httpx.AsyncClient(
            timeout=self.timeout, headers={"user-agent": USER_AGENT}
        ) as http:
            return self._result(await http.get(self._url(order_id)))
