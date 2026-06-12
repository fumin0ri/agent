from slack_agent.mcp_client import MCPTool, is_write_tool


def _tool(name: str, description: str = "") -> MCPTool:
    return MCPTool(name=name, description=description, input_schema={"type": "object"})


def test_mcp_tool_converts_to_llm_tool() -> None:
    tool = _tool("search_messages", "Search Slack messages")

    assert tool.as_llm_tool()["function"] == {
        "name": "search_messages",
        "description": "Search Slack messages",
        "parameters": {"type": "object"},
    }


def test_read_and_write_tools_are_classified_conservatively() -> None:
    assert not is_write_tool(_tool("search_messages"))
    assert not is_write_tool(_tool("read_channel"))
    assert is_write_tool(_tool("send_message"))
    assert is_write_tool(_tool("mystery_action"))
