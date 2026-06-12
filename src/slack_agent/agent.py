from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import Settings
from .mcp_client import MCPTool, SlackMCPClient, is_write_tool

SYSTEM_PROMPT = """
あなたはSlackワークスペースを支援する秘書エージェントです。
利用可能なSlack MCPツールを使い、利用者の指示を理解して必要な処理を行ってください。

方針:
- 最初に必要な情報を検索・取得し、その結果を踏まえて次の処理を判断する。
- チャンネル名やユーザー名しか指定されていない場合は、検索ツールでIDを確認してから使う。
- ツール結果にない事実を推測しない。
- 指示が曖昧で、安全に実行できない場合は確認事項を日本語で尋ねる。
- 実行後は、使用したツールと結果を日本語で簡潔に報告する。
""".strip()


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    allow_writes: bool
    tool_steps: int


def create_agent(
    settings: Settings,
    mcp_client: SlackMCPClient | None = None,
    tools: list[MCPTool] | None = None,
):
    client = mcp_client or SlackMCPClient(settings)
    available_tools = tools if tools is not None else client.list_tools()
    tools_by_name = {tool.name: tool for tool in available_tools}
    model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    ).bind_tools([tool.as_llm_tool() for tool in available_tools])

    def reason(state: AgentState) -> AgentState:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        return {"messages": [model.invoke(messages)]}

    def execute_tools(state: AgentState) -> AgentState:
        response = state["messages"][-1]
        if not isinstance(response, AIMessage):
            return {}
        results: list[ToolMessage] = []
        for call in response.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                content = f"Unknown MCP tool: {call['name']}"
            elif is_write_tool(tool) and not state.get("allow_writes", False):
                content = (
                    f"Write operation '{tool.name}' was not executed. "
                    "Ask the user to rerun with --allow-writes after confirming the action."
                )
            else:
                try:
                    content = client.call_tool(tool.name, call["args"])
                except Exception as exc:
                    content = f"MCP tool '{tool.name}' failed: {exc}"
            results.append(ToolMessage(content=content, tool_call_id=call["id"]))
        return {
            "messages": results,
            "tool_steps": state.get("tool_steps", 0) + 1,
        }

    def route_after_reason(state: AgentState) -> str:
        response = state["messages"][-1]
        if not isinstance(response, AIMessage) or not response.tool_calls:
            return "end"
        if state.get("tool_steps", 0) >= settings.max_tool_steps:
            return "end"
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason)
    graph.add_node("tools", execute_tools)
    graph.add_edge(START, "reason")
    graph.add_conditional_edges("reason", route_after_reason, {"tools": "tools", "end": END})
    graph.add_edge("tools", "reason")
    return graph.compile()


def run_agent(agent, instruction: str, allow_writes: bool) -> str:
    result = agent.invoke(
        {
            "messages": [HumanMessage(content=instruction)],
            "allow_writes": allow_writes,
            "tool_steps": 0,
        }
    )
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    return "エージェントは最終回答を生成できませんでした。"
