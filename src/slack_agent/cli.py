from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from .agent import create_agent, run_agent
from .config import Settings
from .mcp_client import SlackMCPClient, is_write_tool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LangGraph agent for the official Slack MCP server")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Connect to Slack MCP and summarize available capabilities")
    subparsers.add_parser("tools", help="List tools discovered from Slack MCP")
    ask = subparsers.add_parser("ask", help="Understand an instruction and execute Slack MCP tools")
    ask.add_argument("instruction", help="Natural-language instruction")
    ask.add_argument(
        "--allow-writes",
        action="store_true",
        help="Allow message sending, channel creation, reactions, canvas updates, and other writes",
    )
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    args = _parser().parse_args()
    settings = Settings.from_env()
    client = SlackMCPClient(settings)
    try:
        tools = client.list_tools()
        if args.command in {"doctor", "tools"}:
            print(f"Connected to {settings.slack_mcp_url}. Discovered {len(tools)} tools.\n")
            for tool in tools:
                mode = "WRITE" if is_write_tool(tool) else "READ"
                print(f"[{mode}] {tool.name}: {tool.description}")
            return

        agent = create_agent(settings, mcp_client=client, tools=tools)
        answer = run_agent(
            agent,
            args.instruction,
            args.allow_writes or settings.allow_writes_by_default,
        )
        print(answer)
    except Exception as exc:
        raise SystemExit(f"Slack agent failed: {exc}") from None


if __name__ == "__main__":
    main()
