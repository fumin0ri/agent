from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_model: str
    slack_mcp_url: str
    slack_mcp_token: str
    allow_writes_by_default: bool
    max_tool_steps: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            slack_mcp_url=os.getenv("SLACK_MCP_URL", "https://mcp.slack.com/mcp"),
            slack_mcp_token=os.getenv("SLACK_MCP_TOKEN", ""),
            allow_writes_by_default=os.getenv("SLACK_ALLOW_WRITES_BY_DEFAULT", "false").lower() == "true",
            max_tool_steps=int(os.getenv("MAX_TOOL_STEPS", "8")),
        )
