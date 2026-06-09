from __future__ import annotations

from typing import TypedDict

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .index import SearchResult, format_results, search


class SecretaryState(TypedDict, total=False):
    request: str
    results: list[SearchResult]
    answer: str


def _content(response: object) -> str:
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def create_secretary(settings: Settings):
    model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
    embeddings = OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )

    def search_library(state: SecretaryState) -> SecretaryState:
        results = search(
            settings.paper_index,
            state["request"],
            embeddings,
            settings.ollama_embedding_model,
            settings.search_result_limit,
            settings.max_chunks_per_paper,
        )
        return {"results": results}

    def answer_request(state: SecretaryState) -> SecretaryState:
        if not state["results"]:
            return {
                "answer": "条件に合う論文をベクトル索引から見つけられませんでした。"
            }
        context = format_results(state["results"])
        prompt = f"""
あなたは研究者の秘書です。利用者の依頼に対し、関連する論文本文だけを根拠に日本語で回答してください。
適合度が高い候補を先に示し、各候補の理由、ファイルパス、ページ番号を含めてください。
論文の主張と利用者の条件がどう対応するかを具体的に説明してください。
各事実には根拠となる検索結果番号とページ番号を付けてください。
関連箇所に明記されていない手法や処理を、一般知識や推測で補わないでください。
根拠が不足する点は「検索された関連箇所からは確認できない」と明記してください。

依頼:
{state["request"]}

検索結果:
{context}
""".strip()
        return {"answer": _content(model.invoke(prompt))}

    graph = StateGraph(SecretaryState)
    graph.add_node("search_library", search_library)
    graph.add_node("answer_request", answer_request)
    graph.add_edge(START, "search_library")
    graph.add_edge("search_library", "answer_request")
    graph.add_edge("answer_request", END)
    return graph.compile()
