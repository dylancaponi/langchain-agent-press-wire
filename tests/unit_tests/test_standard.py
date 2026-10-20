from langchain_tests.unit_tests import ToolsUnitTests

from langchain_agent_press_wire import (
    AgentPressWireStatusTool,
    AgentPressWireTool,
    AgentPressWireWireTool,
)

EXAMPLE = {"product_url": "https://example.com", "announcement": "Example 2.0 ships"}
KEY = "0x" + "11" * 32


class TestReleaseToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return AgentPressWireTool

    @property
    def tool_constructor_params(self):
        return {"private_key": KEY}

    @property
    def tool_invoke_params_example(self):
        return EXAMPLE

    @property
    def init_from_env_params(self):
        return {"X402_PRIVATE_KEY": KEY}, {}, {"private_key": KEY}


class TestWireToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return AgentPressWireWireTool

    @property
    def tool_constructor_params(self):
        return {"private_key": KEY, "max_payment_usd": 99}

    @property
    def tool_invoke_params_example(self):
        return {
            **EXAMPLE,
            "city": "Los Angeles",
            "country": "USA",
            "contact_email": "press@example.com",
        }


class TestStatusToolStandard(ToolsUnitTests):
    @property
    def tool_constructor(self):
        return AgentPressWireStatusTool

    @property
    def tool_constructor_params(self):
        return {}

    @property
    def tool_invoke_params_example(self):
        return {"order_id": "w_ab12"}
