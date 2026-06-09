from __future__ import annotations

import argparse

from dotenv import load_dotenv

from .agent import create_secretary
from .config import Settings
from .index import build_index


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LangGraph paper secretary")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("index", help="Build or update the paper search index")
    ask = subparsers.add_parser("ask", help="Ask the secretary to find papers")
    ask.add_argument("request", help="Natural-language request")
    return parser


def main() -> None:
    load_dotenv()
    args = _parser().parse_args()
    settings = Settings.from_env()
    if args.command == "index":
        indexed, skipped, failed = build_index(settings.paper_library, settings.paper_index)
        print(f"Indexed: {indexed}, unchanged: {skipped}, failed: {failed}")
        return

    graph = create_secretary(settings)
    result = graph.invoke({"request": args.request})
    print(result["answer"])


if __name__ == "__main__":
    main()
