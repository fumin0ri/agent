from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class SearchResult:
    path: str
    title: str
    snippet: str
    score: float


def _connect(index_path: Path) -> sqlite3.Connection:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS papers USING fts5(
            path UNINDEXED,
            fingerprint UNINDEXED,
            title,
            content,
            tokenize='unicode61'
        )
        """
    )
    return connection


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    value = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(value.encode()).hexdigest()


def _read_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="replace")


def build_index(library: Path, index_path: Path) -> tuple[int, int, int]:
    if not library.is_dir():
        raise FileNotFoundError(f"Paper library does not exist: {library}")

    files = sorted(
        path for path in library.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    indexed = skipped = failed = 0
    with _connect(index_path) as connection:
        known = {row["path"]: row["fingerprint"] for row in connection.execute("SELECT path, fingerprint FROM papers")}
        current_paths = {str(path.resolve()) for path in files}
        for stale_path in set(known) - current_paths:
            connection.execute("DELETE FROM papers WHERE path = ?", (stale_path,))

        for path in files:
            resolved = str(path.resolve())
            fingerprint = _fingerprint(path)
            if known.get(resolved) == fingerprint:
                skipped += 1
                continue
            try:
                content = _read_document(path)
                connection.execute("DELETE FROM papers WHERE path = ?", (resolved,))
                connection.execute(
                    "INSERT INTO papers(path, fingerprint, title, content) VALUES (?, ?, ?, ?)",
                    (resolved, fingerprint, path.stem, content),
                )
                indexed += 1
            except Exception as exc:
                print(f"[warning] Could not index {path}: {exc}")
                failed += 1
    return indexed, skipped, failed


def _fts_query(query: str) -> str:
    terms = re.findall(r"[\w-]+", query, flags=re.UNICODE)
    return " OR ".join(f'"{term}"' for term in terms[:20])


def search(index_path: Path, query: str, limit: int = 8) -> list[SearchResult]:
    if not index_path.exists():
        raise FileNotFoundError(f"Index does not exist: {index_path}. Run the index command first.")
    fts_query = _fts_query(query)
    if not fts_query:
        return []
    with _connect(index_path) as connection:
        rows = connection.execute(
            """
            SELECT path, title,
                   snippet(papers, 3, '[', ']', ' ... ', 32) AS snippet,
                   bm25(papers, 4.0, 1.0) AS score
            FROM papers
            WHERE papers MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
    return [SearchResult(**dict(row)) for row in rows]


def format_results(results: Iterable[SearchResult]) -> str:
    blocks = []
    for number, result in enumerate(results, 1):
        blocks.append(
            f"[{number}] {result.title}\n"
            f"Path: {result.path}\n"
            f"Excerpt: {result.snippet}"
        )
    return "\n\n".join(blocks)
