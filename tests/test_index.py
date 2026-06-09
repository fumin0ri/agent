from pathlib import Path

from paper_secretary.index import build_index, search


class FakeEmbeddings:
    terms = ("transformer", "attention", "distillation")

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(term)) for term in self.terms]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def test_build_and_vector_search_index(tmp_path: Path) -> None:
    library = tmp_path / "papers"
    library.mkdir()
    attention = library / "attention.md"
    attention.write_text(
        "# Attention Is All You Need\nTransformer architecture with multi-head attention.",
        encoding="utf-8",
    )
    (library / "distillation.md").write_text(
        "# Distillation\nKnowledge distillation compresses a teacher model.",
        encoding="utf-8",
    )
    index_path = tmp_path / "papers.sqlite3"
    embeddings = FakeEmbeddings()

    assert build_index(library, index_path, embeddings, "fake") == (2, 0, 0)
    results = search(index_path, "Transformer attention", embeddings, "fake")

    assert len(results) == 2
    assert results[0].title == "attention"
    assert results[0].path == str(attention.resolve())
    assert results[0].score > results[1].score
    assert build_index(library, index_path, embeddings, "fake") == (0, 2, 0)


def test_changing_embedding_model_rebuilds_index(tmp_path: Path) -> None:
    library = tmp_path / "papers"
    library.mkdir()
    (library / "paper.md").write_text("Transformer attention.", encoding="utf-8")
    index_path = tmp_path / "papers.sqlite3"
    embeddings = FakeEmbeddings()

    build_index(library, index_path, embeddings, "first-model")

    assert build_index(library, index_path, embeddings, "second-model") == (1, 0, 0)


def test_search_rejects_different_embedding_model(tmp_path: Path) -> None:
    library = tmp_path / "papers"
    library.mkdir()
    (library / "paper.md").write_text("Transformer attention.", encoding="utf-8")
    index_path = tmp_path / "papers.sqlite3"
    embeddings = FakeEmbeddings()
    build_index(library, index_path, embeddings, "first-model")

    try:
        search(index_path, "attention", embeddings, "second-model")
    except ValueError as exc:
        assert "different embedding model" in str(exc)
    else:
        raise AssertionError("Expected search to reject a mismatched embedding model.")


def test_search_diversifies_papers(tmp_path: Path) -> None:
    library = tmp_path / "papers"
    library.mkdir()
    (library / "many.md").write_text("attention " * 20, encoding="utf-8")
    (library / "other.md").write_text("transformer attention", encoding="utf-8")
    index_path = tmp_path / "papers.sqlite3"
    embeddings = FakeEmbeddings()
    build_index(library, index_path, embeddings, "fake", chunk_size=20, overlap=5)

    results = search(index_path, "attention", embeddings, "fake", limit=3, max_chunks_per_paper=1)

    assert len(results) == 2
    assert len({result.path for result in results}) == 2
