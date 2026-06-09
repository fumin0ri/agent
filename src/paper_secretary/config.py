from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    ollama_model: str
    paper_library: Path
    paper_index: Path
    search_result_limit: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            paper_library=Path(os.getenv("PAPER_LIBRARY", "./papers")).expanduser().resolve(),
            paper_index=Path(os.getenv("PAPER_INDEX", "./work/papers.sqlite3")).expanduser().resolve(),
            search_result_limit=int(os.getenv("SEARCH_RESULT_LIMIT", "8")),
        )
