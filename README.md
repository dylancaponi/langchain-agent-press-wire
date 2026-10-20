# langchain-agent-press-wire

LangChain tools for [Agent Press Wire](https://agentpresswire.dev). An agent can write and publish a press release for a product launch, or send it over a newswire. Each call is paid in USDC on Base over [x402](https://x402.org). No account or API key.

| Tool | Endpoint | Price |
| --- | --- | --- |
| `AgentPressWireTool` | `POST /v1/press-release` (release on a public newsroom page) | $1 |
| `AgentPressWireWireTool` tier `standard` | `POST /v1/wire-release` (500+ news sites, 100+ TV and radio network sites) | $99 |
| `AgentPressWireWireTool` tier `premium` | `POST /v1/wire-release/premium` (adds AP News, USA Today, Barchart, StreetInsider) | $299 |
| `AgentPressWireStatusTool` | `GET /v1/wire-release/{id}` (wire order status and links) | free |

You are charged only if the job succeeds. Validation failures return HTTP 422 with the reasons and nothing is charged.

## Install

```bash
pip install -U langchain-agent-press-wire
export X402_PRIVATE_KEY=0x...   # private key of a Base wallet holding USDC
```

Use a dedicated wallet with only the USDC you plan to spend.

## Use

```python
from langchain_agent_press_wire import AgentPressWireTool

tool = AgentPressWireTool()  # pays at most $1 per call

tool.invoke({
    "product_url": "https://example.com",
    "announcement": "Example 2.0 adds offline mode",
})
# {"mode": "paid", "paid": true, "price_usd": "1.00", "result": {"id": ..., "headline": ...,
#  "newsroom_url": ..., "release_markdown": ...}, "settlement": {...}}
```

With an agent:

```python
from langchain.agents import create_agent
from langchain_agent_press_wire import AgentPressWireStatusTool, AgentPressWireTool

agent = create_agent("openai:gpt-5", tools=[AgentPressWireTool(), AgentPressWireStatusTool()])
```

## Spending limits

- `max_payment_usd` (default `1.0`): the tool refuses any 402 asking more than this. Nothing is signed. The wire tiers cost $99 and $299, so `AgentPressWireWireTool()` with the default cap only quotes. Raise the cap for a call after the user approves the price:

  ```python
  AgentPressWireWireTool(max_payment_usd=99)
  ```

- The 402 terms must also match Agent Press Wire's listed terms: scheme `exact`, Base mainnet, USDC, the Agent Press Wire wallet `0xF7887C1fAf0DbAedc80C78F9D396D594B69eba95`, and the listed price for that endpoint. Anything else is refused, even under the cap.
- Quote without paying: pass `quote_only=True` in the tool input, or construct the tool with `quote_only=True` to never pay. No wallet key is needed for quotes.

  ```python
  AgentPressWireWireTool(quote_only=True).invoke({..., "tier": "premium"})
  # {"mode": "quote", "paid": false, "quote": {"price_usd": "299.00", "within_cap": false, ...}}
  ```

## Inputs

All release tools take `product_url` (required), `announcement`, `company`, `contact`, and optional `headline` + `body` to publish your own copy as written. The wire tool also requires `city`, `country`, `contact_email` and takes `tier` (`standard` or `premium`). Field limits match the API's [OpenAPI spec](https://agentpresswire.dev/openapi.json).

## Errors

Errors are returned to the agent as tool messages (`handle_tool_error=True`). Set `handle_tool_error=False` to get a `ToolException` instead.

- 422: the release was not accepted (thin page, no news, refused topic, copy failed checks). Not charged. The message lists what to fix.
- 409: the same request from the same wallet within 24 hours. The existing order is returned and nothing is charged again.
- Network error after payment was sent: settlement status is unknown. Retrying the same request is safe because of the 409 rule.

Wire tiers refuse financial or market-moving news (earnings, acquisitions, token prices and similar). Prohibited topics are listed in the [API docs](https://agentpresswire.dev/llms.txt) and [terms](https://agentpresswire.dev/terms).

## Development

```bash
uv venv && uv pip install -e . pytest pytest-asyncio pytest-socket respx langchain-tests
pytest tests/unit_tests          # mocked HTTP, no network
pytest tests/integration_tests   # live API, quote mode only, nothing is paid
```

## License

MIT
