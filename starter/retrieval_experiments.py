"""Day 7 retrieval experiment runner.

Runs three retrieval strategies against the real Day 6 corpus without touching
Day 6 source files:
  - dense: real vector retrieval via the Day 6 Chroma index
  - lexical: TF-IDF on the real Day 6 chunk texts
  - hybrid: weighted combination of dense + lexical scores

This uses the actual Day 6 benchmark JSON and the persistent Chroma collection it
built when the Day 6 project was run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baseline import TfidfIndex, hit, pack_context  # noqa: E402
from day6_real_connector import Day6Retriever, normalize_day6_filters  # noqa: E402

DAY6_ROOT = Path(r"D:\Python\VinAI\Lab6_L3A\K5-L3-Day06-ChuMaiHong-2A202603377-DataIngestionEngineering")
DAY6_BENCHMARK = DAY6_ROOT / "starter" / "benchmark" / "questions.json"
DAY6_CHROMA_DIR = DAY6_ROOT / "starter" / "results" / "chroma"


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object with 'questions' list in {path}")
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"Missing 'questions' array in {path}")
    return questions


def fetch_day6_chunks() -> list[dict[str, Any]]:
    """Read the real Day 6 Chroma collection and normalize it to Day 7 chunk schema."""
    from src.store import ChromaVectorStore  # type: ignore

    store = ChromaVectorStore(persist_directory=DAY6_CHROMA_DIR, dimensions=384)
    strategy = "B"
    collection = store._collection(strategy)
    result = collection.get(include=["documents", "metadatas"])
    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    chunks: list[dict[str, Any]] = []
    for chunk_id, text, metadata in zip(ids, documents, metadatas):
        meta = metadata or {}
        chunks.append({
            "id": str(chunk_id),
            "text": str(text),
            "source_id": str(meta.get("document_id") or "owasp-llm-top-10-2025"),
            "source": str(meta.get("source") or "https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/"),
            "page": int(meta.get("page") or 0),
            "metadata": {k: v for k, v in meta.items() if k not in {"document_id", "source", "page"}},
        })
    return chunks


def dense_search(question: str, *, top_k: int = 3, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    retriever = Day6Retriever(strategy="B")
    results = retriever.search(question, filters=filters, top_k=top_k)
    for item in results:
        item["strategy"] = "dense"
    return results


def _filter_chunks(chunks: list[dict[str, Any]], filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not filters:
        return chunks
    return [
        chunk for chunk in chunks
        if all((chunk.get("metadata") or {}).get(key) == value for key, value in filters.items())
    ]


def lexical_search(
    question: str,
    *,
    chunks: list[dict[str, Any]],
    top_k: int = 3,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    index = TfidfIndex(_filter_chunks(chunks, filters))
    results = index.search(question, top_k=top_k)
    for item in results:
        item["strategy"] = "lexical"
    return results


def hybrid_search(
    question: str,
    *,
    chunks: list[dict[str, Any]],
    top_k: int = 3,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    dense = dense_search(question, top_k=top_k, filters=filters)
    lexical = lexical_search(question, chunks=chunks, top_k=top_k, filters=filters)
    dense_by_id = {item["id"]: float(item.get("score", 0.0)) for item in dense}
    lexical_by_id = {item["id"]: float(item.get("score", 0.0)) for item in lexical}
    all_ids = sorted(set(dense_by_id) | set(lexical_by_id))

    merged: dict[str, dict[str, Any]] = {}
    for item in dense:
        merged[item["id"]] = {**item, "strategy": "hybrid"}
    for item in lexical:
        if item["id"] not in merged:
            merged[item["id"]] = {**item, "strategy": "hybrid"}

    if not all_ids:
        return []

    dense_max = max((abs(v) for v in dense_by_id.values()), default=1.0)
    lexical_max = max((abs(v) for v in lexical_by_id.values()), default=1.0)

    combined: list[dict[str, Any]] = []
    for chunk_id in all_ids:
        dense_score = dense_by_id.get(chunk_id, 0.0)
        lexical_score = lexical_by_id.get(chunk_id, 0.0)
        dense_normalized = 0.0 if dense_max <= 0 else dense_score / dense_max
        lexical_normalized = 0.0 if lexical_max <= 0 else lexical_score / lexical_max
        merged_score = 0.6 * dense_normalized + 0.4 * lexical_normalized
        base = merged.get(chunk_id, {})
        base.update({
            "id": chunk_id,
            "score": merged_score,
            "strategy": "hybrid",
            "source": base.get("source") or "unknown",
            "page": base.get("page") or 0,
            "metadata": base.get("metadata") or {},
            "text": base.get("text") or "",
        })
        combined.append(base)

    return sorted(combined, key=lambda x: (-float(x.get("score", 0.0)), str(x.get("id", ""))))[:top_k]


def evaluate_strategy(
    strategy: str,
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    budget: int = 800,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    if strategy == "dense":
        def fn(question: dict[str, Any]):
            raw_filter = question.get("filter") if isinstance(question.get("filter"), dict) else None
            filters = normalize_day6_filters(raw_filter)
            return dense_search(question.get("question") or "", top_k=3, filters=filters)
    elif strategy == "lexical":
        def fn(question: dict[str, Any]):
            raw_filter = question.get("filter") if isinstance(question.get("filter"), dict) else None
            filters = normalize_day6_filters(raw_filter)
            return lexical_search(question.get("question") or "", chunks=chunks, top_k=3, filters=filters)
    elif strategy == "hybrid":
        def fn(question: dict[str, Any]):
            raw_filter = question.get("filter") if isinstance(question.get("filter"), dict) else None
            filters = normalize_day6_filters(raw_filter)
            return hybrid_search(question.get("question") or "", chunks=chunks, top_k=3, filters=filters)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    for question in questions:
        start = time.perf_counter()
        qtext = question.get("question") or ""
        results = fn(question)
        elapsed_ms = (time.perf_counter() - start) * 1000
        gold = question.get("gold") or [{"source_id": question.get("gold_source_id"), "page": question.get("gold_page"), "evidence": question.get("evidence_quote")}]
        scored_results = []
        for result in results:
            scored_results.append({**result, "hit": hit(result, gold)})
        context, context_ids = pack_context(scored_results, budget=budget)
        traces.append({
            "question_id": question.get("id"),
            "split": question.get("split"),
            "question": qtext,
            "strategy": strategy,
            "hit1": bool(scored_results and scored_results[0].get("hit", False)),
            "hit3": any(item.get("hit") for item in scored_results),
            "context_hit": any(item.get("hit") and item.get("id") in context_ids for item in scored_results),
            "context_ids": context_ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "results": scored_results,
            "context": context,
        })

    summary = {
        "strategy": strategy,
        "split": questions[0].get("split") if questions else "unknown",
        "n": len(questions),
        "hit1": sum(1 for t in traces if t["hit1"]),
        "hit3": sum(1 for t in traces if t["hit3"]),
        "context_hit": sum(1 for t in traces if t["context_hit"]),
        "mean_context_chars": sum(t["context_chars"] for t in traces) / len(traces) if traces else 0,
        "mean_elapsed_ms": sum(t["elapsed_ms"] for t in traces) / len(traces) if traces else 0,
    }
    return traces, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 7 dense/lexical/hybrid retrieval experiments on the real Day 6 corpus.")
    parser.add_argument("--benchmark", type=Path, default=DAY6_BENCHMARK)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "retrieval")
    parser.add_argument("--budget", type=int, default=800, help="Context budget in characters")
    args = parser.parse_args()

    if args.budget < 1:
        parser.error("--budget must be >= 1")

    questions = [
        q for q in load_questions(args.benchmark)
        if q.get("split") == args.split and q.get("answerable") is not False
    ]
    if not questions:
        raise ValueError(f"No {args.split} questions found in {DAY6_BENCHMARK}")

    chunks = fetch_day6_chunks()
    if not chunks:
        raise RuntimeError(f"No chunks found in Day 6 Chroma collection under {DAY6_CHROMA_DIR}")

    args.output.mkdir(parents=True, exist_ok=True)
    overall: dict[str, Any] = {}
    for strategy in ("dense", "lexical", "hybrid"):
        traces, summary = evaluate_strategy(strategy, questions, chunks, budget=args.budget)
        (args.output / strategy).mkdir(parents=True, exist_ok=True)
        (args.output / strategy / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output / strategy / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        overall[strategy] = summary

    (args.output / "overall_summary.json").write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
