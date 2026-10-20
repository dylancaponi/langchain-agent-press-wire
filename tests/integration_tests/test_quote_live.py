"""Live checks against agentpresswire.dev. Quote mode only: nothing is paid."""

import json

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_agent_press_wire import AgentPressWireTool, AgentPressWireWireTool


class TestReleaseQuoteLive(ToolsIntegrationTests):
    @property
    def tool_constructor(self):
        return AgentPressWireTool

    @property
    def tool_constructor_params(self):
        return {"quote_only": True, "private_key": None}

    @property
    def tool_invoke_params_example(self):
        return {"product_url": "https://example.com", "announcement": "Example 2.0 ships"}


def test_live_prices():
    r = json.loads(
        AgentPressWireTool(quote_only=True, private_key=None).invoke(
            {"product_url": "https://example.com", "announcement": "test"}
        )
    )
    assert r["quote"]["price_usd"] == "1.00"
    w = json.loads(
        AgentPressWireWireTool(quote_only=True, private_key=None).invoke(
            {
                "product_url": "https://example.com",
                "announcement": "test",
                "city": "LA",
                "country": "USA",
                "contact_email": "a@example.com",
                "tier": "premium",
            }
        )
    )
    assert w["quote"]["price_usd"] == "299.00"
