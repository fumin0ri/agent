from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

import slack_agent.agent as agent_module
from slack_agent.agent import create_agent, run_agent
from slack_agent.mcp_client import MCPTool


class FakeAgent:
    def invoke(self, state):
        assert state["allow_writes"] is False
        return {"messages": [*state["messages"], AIMessage(content="完了しました。")]}


def test_run_agent_returns_final_ai_message() -> None:
    assert run_agent(FakeAgent(), "最新情報を調べて", allow_writes=False) == "完了しました。"


class FakeChatOllama:
    def __init__(self, **kwargs):
        pass

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage(content=f"結果: {messages[-1].content}")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": self.tool_name,
                    "args": {"query": "project"},
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )


class FakeMCPClient:
    def __init__(self):
        self.calls = []

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return "MCP result"


def _settings():
    return SimpleNamespace(
        ollama_model="fake",
        ollama_base_url="http://localhost",
        max_tool_steps=3,
    )


def test_agent_executes_read_tool(monkeypatch) -> None:
    FakeChatOllama.tool_name = "search_messages"
    monkeypatch.setattr(agent_module, "ChatOllama", FakeChatOllama)
    client = FakeMCPClient()
    tool = MCPTool("search_messages", "Search Slack messages", {"type": "object"})
    agent = create_agent(_settings(), mcp_client=client, tools=[tool])

    answer = run_agent(agent, "projectを検索して", allow_writes=False)

    assert client.calls == [("search_messages", {"query": "project"})]
    assert "MCP result" in answer


def test_agent_blocks_write_tool_without_permission(monkeypatch) -> None:
    FakeChatOllama.tool_name = "send_message"
    monkeypatch.setattr(agent_module, "ChatOllama", FakeChatOllama)
    client = FakeMCPClient()
    tool = MCPTool("send_message", "Send a Slack message", {"type": "object"})
    agent = create_agent(_settings(), mcp_client=client, tools=[tool])

    answer = run_agent(agent, "generalに投稿して", allow_writes=False)

    assert client.calls == []
    assert "was not executed" in answer
