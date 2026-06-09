from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


class Embeddings(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class DocumentChunk:
    page: int | None
    index: int
    content: str


@dataclass(frozen=True)
class SearchResult:
    path: str
    title: str
    snippet: str
    score: float
    page: int | None
    chunk_index: int


def _connect(index_path: Path) -> sqlite3.Connection:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS vector_documents(
            path TEXT PRIMARY KEY,
            fingerprint TEXT NOT NULL,
            title TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS vector_chunks(
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL,
            page INTEGER,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL,
            FOREIGN KEY(path) REFERENCES vector_documents(path) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS vector_chunks_path ON vector_chunks(path);
        CREATE TABLE IF NOT EXISTS vector_meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    value = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(value.encode()).hexdigest()


def _read_pages(path: Path) -> list[tuple[int | None, str]]:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, 1)]
    return [(None, path.read_text(encoding="utf-8", errors="replace"))]


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _document_chunks(path: Path, chunk_size: int, overlap: int) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    chunk_index = 0
    for page, content in _read_pages(path):
        for text in _split_text(content, chunk_size, overlap):
            chunks.append(DocumentChunk(page=page, index=chunk_index, content=text))
            chunk_index += 1
    return chunks


def _batches(items: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])


def build_index(
    library: Path,
    index_path: Path,
    embeddings: Embeddings,
    embedding_model: str,
    chunk_size: int = 1800,
    overlap: int = 250,
    batch_size: int = 16,
) -> tuple[int, int, int]:
    if not library.is_dir():
        raise FileNotFoundError(f"Paper library does not exist: {library}")
    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    probe_vector = embeddings.embed_query("paper vector index health check")
    if not probe_vector:
        raise ValueError("Embedding API returned an empty health-check vector.")

    files = sorted(
        path for path in library.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    indexed = skipped = failed = 0
    with _connect(index_path) as connection:
        stored_model = connection.execute(
            "SELECT value FROM vector_meta WHERE key = 'embedding_model'"
        ).fetchone()
        if stored_model and stored_model["value"] != embedding_model:
            connection.execute("DELETE FROM vector_chunks")
            connection.execute("DELETE FROM vector_documents")
        connection.execute(
            "INSERT OR REPLACE INTO vector_meta(key, value) VALUES ('embedding_model', ?)",
            (embedding_model,),
        )
        connection.execute(
            """
            DELETE FROM vector_documents
            WHERE NOT EXISTS (
                SELECT 1 FROM vector_chunks WHERE vector_chunks.path = vector_documents.path
            )
            """
        )

        known = {
            row["path"]: row["fingerprint"]
            for row in connection.execute("SELECT path, fingerprint FROM vector_documents")
        }
        current_paths = {str(path.resolve()) for path in files}
        for stale_path in set(known) - current_paths:
            connection.execute("DELETE FROM vector_documents WHERE path = ?", (stale_path,))

        for path in files:
            resolved = str(path.resolve())
            fingerprint = _fingerprint(path)
            if known.get(resolved) == fingerprint:
                skipped += 1
                continue
            try:
                chunks = _document_chunks(path, chunk_size, overlap)
                if not chunks:
                    raise ValueError("No text could be extracted from this document.")
                vectors: list[list[float]] = []
                for batch in _batches([chunk.content for chunk in chunks], batch_size):
                    vectors.extend(embeddings.embed_documents(batch))
                if len(vectors) != len(chunks):
                    raise ValueError("Embedding API returned an unexpected number of vectors.")
                if any(len(vector) != len(probe_vector) for vector in vectors):
                    raise ValueError("Embedding API returned vectors with inconsistent dimensions.")

                connection.execute("DELETE FROM vector_documents WHERE path = ?", (resolved,))
                connection.execute(
                    "INSERT INTO vector_documents(path, fingerprint, title) VALUES (?, ?, ?)",
                    (resolved, fingerprint, path.stem),
                )
                connection.executemany(
                    """
                    INSERT INTO vector_chunks(path, page, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (resolved, chunk.page, chunk.index, chunk.content, json.dumps(vector))
                        for chunk, vector in zip(chunks, vectors, strict=True)
                    ],
                )
                indexed += 1
            except Exception as exc:
                print(f"[warning] Could not index {path}: {exc}")
                failed += 1
    return indexed, skipped, failed


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return -1.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return -1.0
    return dot / (left_norm * right_norm)


def search(
    index_path: Path,
    query: str,
    embeddings: Embeddings,
    embedding_model: str,
    limit: int = 8,
    max_chunks_per_paper: int = 2,
) -> list[SearchResult]:
    if not index_path.exists():
        raise FileNotFoundError(f"Index does not exist: {index_path}. Run the index command first.")
    with _connect(index_path) as connection:
        stored_model = connection.execute(
            "SELECT value FROM vector_meta WHERE key = 'embedding_model'"
        ).fetchone()
        if not stored_model or stored_model["value"] != embedding_model:
            raise ValueError(
                "The vector index was built with a different embedding model. Run the index command again."
            )
        rows = connection.execute(
            """
            SELECT d.path, d.title, c.page, c.chunk_index, c.content, c.embedding
            FROM vector_chunks c
            JOIN vector_documents d ON d.path = c.path
            """
        ).fetchall()

    query_vector = embeddings.embed_query(query)
    results = [
        SearchResult(
            path=row["path"],
            title=row["title"],
            snippet=row["content"],
            score=_cosine_similarity(query_vector, json.loads(row["embedding"])),
            page=row["page"],
            chunk_index=row["chunk_index"],
        )
        for row in rows
    ]
    ranked = sorted(results, key=lambda result: result.score, reverse=True)
    selected: list[SearchResult] = []
    counts: dict[str, int] = {}
    for result in ranked:
        if counts.get(result.path, 0) >= max_chunks_per_paper:
            continue
        selected.append(result)
        counts[result.path] = counts.get(result.path, 0) + 1
        if len(selected) == limit:
            break
    return selected


def format_results(results: Iterable[SearchResult]) -> str:
    blocks = []
    for number, result in enumerate(results, 1):
        location = f"Page: {result.page}" if result.page is not None else f"Chunk: {result.chunk_index}"
        blocks.append(
            f"[{number}] {result.title}\n"
            f"Path: {result.path}\n"
            f"{location}\n"
            f"Similarity: {result.score:.4f}\n"
            f"Relevant passage: {result.snippet}"
        )
    return "\n\n".join(blocks)
