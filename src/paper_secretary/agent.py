from __future__ import annotations

import json
import re
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from .config import Settings
from .index import SearchResult, format_results, search


class SecretaryState(TypedDict, total=False):
    request: str
    search_query: str
    results: list[SearchResult]
    answer: str


def _content(response: object) -> str:
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def _extract_json(text: str) -> dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def create_secretary(settings: Settings):
    model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )

    def plan_search(state: SecretaryState) -> SecretaryState:
        prompt = f"""
あなたは論文検索の司書です。利用者の依頼から、ローカル全文検索に使う短い検索語を抽出してください。
著者名、手法名、分野、重要な専門用語を優先してください。
JSONだけを返してください。形式: {{"query": "検索語"}}

依頼: {state["request"]}
""".strip()
        parsed = _extract_json(_content(model.invoke(prompt)))
        query = str(parsed.get("query") or state["request"]).strip()
        return {"search_query": query}

    def search_library(state: SecretaryState) -> SecretaryState:
        results = search(settings.paper_index, state["search_query"], settings.search_result_limit)
        return {"results": results}

    def answer_request(state: SecretaryState) -> SecretaryState:
        if not state["results"]:
            return {
                "answer": (
                    "条件に合う論文を索引から見つけられませんでした。"
                    f"\n使用した検索語: {state['search_query']}"
                )
            }
        context = format_results(state["results"])
        prompt = f"""
あなたは研究者の秘書です。利用者の依頼に対し、検索結果だけを根拠に日本語で簡潔に回答してください。
適合度が高い候補を先に示し、各候補の理由とファイルパスを含めてください。
検索結果にない内容を推測で補わないでください。

依頼:
{state["request"]}

検索結果:
{context}
""".strip()
        return {"answer": _content(model.invoke(prompt))}

    graph = StateGraph(SecretaryState)
    graph.add_node("plan_search", plan_search)
    graph.add_node("search_library", search_library)
    graph.add_node("answer_request", answer_request)
    graph.add_edge(START, "plan_search")
    graph.add_edge("plan_search", "search_library")
    graph.add_edge("search_library", "answer_request")
    graph.add_edge("answer_request", END)
    return graph.compile()
