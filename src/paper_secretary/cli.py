from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv()
    args = _parser().parse_args()
    settings = Settings.from_env()
    if args.command == "index":
        embeddings = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )
        indexed, skipped, failed = build_index(
            settings.paper_library,
            settings.paper_index,
            embeddings,
            settings.ollama_embedding_model,
            settings.chunk_size,
            settings.chunk_overlap,
            settings.embedding_batch_size,
        )
        print(f"Indexed: {indexed}, unchanged: {skipped}, failed: {failed}")
        return

    graph = create_secretary(settings)
    result = graph.invoke({"request": args.request})
    print(result["answer"])


if __name__ == "__main__":
    main()
