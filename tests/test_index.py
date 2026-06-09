from pathlib import Path

from paper_secretary.index import build_index, search


def test_build_and_search_index(tmp_path: Path) -> None:
    library = tmp_path / "papers"
    library.mkdir()
    paper = library / "attention.md"
    paper.write_text(
        "# Attention Is All You Need\nTransformer architecture with multi-head attention.",
        encoding="utf-8",
    )
    index_path = tmp_path / "papers.sqlite3"

    assert build_index(library, index_path) == (1, 0, 0)
    results = search(index_path, "Transformer attention")

    assert len(results) == 1
    assert results[0].title == "attention"
    assert results[0].path == str(paper.resolve())
    assert build_index(library, index_path) == (0, 1, 0)
