from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from .config import Settings


@dataclass(frozen=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]

    def as_llm_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class SlackMCPClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        if not self.settings.slack_mcp_token:
            raise ValueError(
                "SLACK_MCP_TOKEN is required. Complete Slack OAuth and set the resulting user token."
            )
        headers = {"Authorization": f"Bearer {self.settings.slack_mcp_token}"}
        async with httpx.AsyncClient(headers=headers, timeout=60) as http_client:
            async with streamable_http_client(
                self.settings.slack_mcp_url,
                http_client=http_client,
            ) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session

    async def list_tools_async(self) -> list[MCPTool]:
        async with self._session() as session:
            result = await session.list_tools()
        return [
            MCPTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
            )
            for tool in result.tools
        ]

    def list_tools(self) -> list[MCPTool]:
        return asyncio.run(self.list_tools_async())

    async def call_tool_async(self, name: str, arguments: dict[str, Any]) -> str:
        async with self._session() as session:
            result = await session.call_tool(name, arguments=arguments)
        if result.isError:
            raise RuntimeError(f"Slack MCP tool '{name}' failed: {result.content}")
        return result.model_dump_json(exclude_none=True)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return asyncio.run(self.call_tool_async(name, arguments))


READ_ONLY_HINTS = ("search", "read", "get", "fetch", "list", "find", "lookup")
WRITE_HINTS = ("send", "post", "create", "update", "add", "react", "delete", "remove", "write")


def is_write_tool(tool: MCPTool) -> bool:
    text = f"{tool.name} {tool.description}".lower()
    if any(hint in text for hint in WRITE_HINTS):
        return True
    return not any(hint in text for hint in READ_ONLY_HINTS)
