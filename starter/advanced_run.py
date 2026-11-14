"""Run the advanced multimodal-document agent and retain page-resolved traces."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STARTER_DIR = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = STARTER_DIR / "advanced_benchmark.json"
DEFAULT_DATA_DIR = STARTER_DIR / "data" / "advanced" / "extracted"
PRICING_USD_PER_MILLION = {
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}
TEST_CATEGORY_MINIMUMS = {
    "table": 6,
    "image_diagram": 5,
    "chart": 5,
    "multi_hop": 4,
    "text": 4,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_benchmark(benchmark: dict[str, Any]) -> None:
    questions = benchmark.get("questions")
    if not isinstance(questions, list):
        raise ValueError("Benchmark must contain a questions array")
    ids: set[str] = set()
    normalized_questions: dict[str, str] = {}
    by_split: dict[str, list[dict[str, Any]]] = {"dev": [], "test": []}
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Every benchmark question must be an object")
        query_id = question.get("id")
        split = question.get("split")
        category = question.get("category")
        text = question.get("question")
        answerable = question.get("answerable")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in ids:
            raise ValueError(f"Invalid or duplicate benchmark question ID: {query_id!r}")
        ids.add(query_id)
        if split not in by_split or category not in (*TEST_CATEGORY_MINIMUMS, "no_evidence"):
            raise ValueError(f"{query_id}: invalid split or category")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{query_id}: question must be a non-empty string")
        normalized = " ".join(text.casefold().split())
        previous_id = normalized_questions.get(normalized)
        if previous_id is not None:
            raise ValueError(f"Duplicate question text in {previous_id} and {query_id}")
        normalized_questions[normalized] = query_id
        if not isinstance(answerable, bool):
            raise ValueError(f"{query_id}: answerable must be a boolean")
        if answerable:
            if category == "no_evidence" or question.get("expected_behavior") != "answer":
                raise ValueError(f"{query_id}: answerable case has inconsistent labels")
            if not str(question.get("gold_answer") or "").strip() or not question.get("evidence"):
                raise ValueError(f"{query_id}: answerable case needs a gold answer and evidence")
        else:
            if category != "no_evidence" or question.get("expected_behavior") != "abstain":
                raise ValueError(f"{query_id}: no-evidence case has inconsistent labels")
            if question.get("gold_answer") is not None or not str(question.get("absence_check") or "").strip():
                raise ValueError(f"{query_id}: no-evidence case needs an absence check and no gold answer")
        by_split[split].append(question)

    dev = by_split["dev"]
    test = by_split["test"]
    if sum(bool(item["answerable"]) for item in dev) < 15 or sum(
        not item["answerable"] for item in dev
    ) < 3:
        raise ValueError("Dev benchmark needs at least 15 answerable and 3 no-evidence cases")
    if len(test) < 30 or sum(bool(item["answerable"]) for item in test) < 24 or sum(
        not item["answerable"] for item in test
    ) < 6:
        raise ValueError("Test benchmark needs at least 30 cases, 24 answerable, and 6 no-evidence")
    for category, minimum in TEST_CATEGORY_MINIMUMS.items():
        count = sum(item["answerable"] and item["category"] == category for item in test)
        if count < minimum:
            raise ValueError(f"Test benchmark needs at least {minimum} {category} cases; found {count}")


def _estimate_cost(model: str | None, usage: dict[str, int] | None) -> float | None:
    pricing = PRICING_USD_PER_MILLION.get(model or "")
    if pricing is None or usage is None:
        return None
    cached = min(usage.get("cached_input_tokens", 0), usage.get("input_tokens", 0))
    uncached = usage.get("input_tokens", 0) - cached
    amount = (
        uncached * pricing["input"]
        + cached * pricing["cached_input"]
        + usage.get("output_tokens", 0) * pricing["output"]
    ) / 1_000_000
    return round(amount, 10)


def _citation_locations(
    citation_ids: list[str],
    events: list[dict[str, Any]],
    source_url: str,
) -> list[dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("event") != "search":
            continue
        for result in event.get("results") or []:
            if isinstance(result, dict) and isinstance(result.get("id"), str):
                chunks[result["id"]] = result
    locations = []
    for citation_id in citation_ids:
        result = chunks.get(citation_id)
        if result is None:
            locations.append({"chunk_id": citation_id, "resolved": False})
            continue
        metadata = result.get("metadata") or {}
        locations.append(
            {
                "chunk_id": citation_id,
                "resolved": True,
                "source_url": source_url,
                "pdf_page": result.get("page"),
                "printed_page": metadata.get("printed_page"),
                "kind": metadata.get("kind"),
                "region": metadata.get("region"),
                "region_bbox": metadata.get("region_bbox"),
                "table_label": metadata.get("table_label"),
                "table_row_index": metadata.get("table_row_index"),
                "figure_label": metadata.get("figure_label"),
                "visual_type": metadata.get("visual_type"),
                "image_path": metadata.get("image_path"),
            }
        )
    return locations


def _save_run(
    output_dir: Path,
    run_manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "traces.json").write_text(
        json.dumps(traces, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_benchmark(
    benchmark_path: Path,
    split: str,
    output_dir: Path,
    *,
    top_k: int = 8,
    budget: int = 18000,
    max_searches: int = 3,
    index_dir: Path = DEFAULT_DATA_DIR / "chroma",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from agentic import OpenAIModelController, run_agent
    from advanced_retriever import AdvancedRetriever

    benchmark = _read_json(benchmark_path)
    if not isinstance(benchmark, dict):
        raise ValueError("Benchmark must be an object containing a questions array")
    validate_benchmark(benchmark)
    index_manifest_path = index_dir.parent / "index_manifest.json"
    if not index_manifest_path.is_file():
        raise FileNotFoundError("Advanced index manifest is missing; build the index first")
    index_manifest = _read_json(index_manifest_path)
    questions = [item for item in benchmark["questions"] if item.get("split") == split]
    if not questions:
        raise ValueError(f"No {split} questions found in {benchmark_path}")
    if len({str(item.get("id")) for item in questions}) != len(questions):
        raise ValueError(f"Duplicate question IDs in the {split} benchmark")
    if any(not isinstance(item.get("question"), str) or not item["question"].strip() for item in questions):
        raise ValueError("Every benchmark item needs a non-empty question")
    if top_k < 1 or budget < 1 or max_searches < 1:
        raise ValueError("top_k, budget, and max_searches must be positive")

    retriever = AdvancedRetriever(index_dir=index_dir)
    run_manifest = {
        "split": split,
        "benchmark": str(benchmark_path.resolve()),
        "benchmark_sha256": hashlib.sha256(benchmark_path.read_bytes()).hexdigest(),
        "source_url": index_manifest["source_url"],
        "source_sha256": index_manifest["source_sha256"],
        "embedding_model": index_manifest["embedding_model"],
        "retrieval": (
            "RRF fusion of dense Day 6 multilingual embeddings and TF-IDF, "
            "with explicit named-figure chunks promoted into the candidate context"
        ),
        "agent_model": None,
        "config": {
            "top_k": top_k,
            "budget_chars": budget,
            "max_searches": max_searches,
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gold_excluded_from_agent_inputs": True,
    }
    cases: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for question in questions:
        query_id = str(question["id"])
        case = {
            "id": query_id,
            "split": split,
            "category": str(question["category"]),
            "answerable": bool(question["answerable"]),
        }
        started = time.perf_counter()
        controller = None
        try:
            controller = OpenAIModelController()
            run_manifest["agent_model"] = controller.model
            result = run_agent(
                question["question"],
                controller,
                retriever,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget,
            )
            events = result.get("events", [])
            citations = result.get("citations", [])
            trace = {
                "query_id": query_id,
                "split": split,
                "category": case["category"],
                "question": question["question"],
                "status": result.get("status", "error"),
                "answer": result.get("answer"),
                "citations": citations,
                "resolved_citations": _citation_locations(
                    citations, events, index_manifest["source_url"]
                ),
                "events": events,
                "model": controller.model,
                "usage": controller.usage,
                "estimated_cost_usd": _estimate_cost(controller.model, controller.usage),
                "pricing_usd_per_1m_tokens": PRICING_USD_PER_MILLION.get(controller.model),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            trace = {
                "query_id": query_id,
                "split": split,
                "category": case["category"],
                "question": question["question"],
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "citations": [],
                "resolved_citations": [],
                "events": [],
                "model": getattr(controller, "model", None),
                "usage": getattr(controller, "usage", None),
                "estimated_cost_usd": _estimate_cost(
                    getattr(controller, "model", None),
                    getattr(controller, "usage", None),
                ),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        cases.append(case)
        traces.append(trace)
        _save_run(output_dir, run_manifest, cases, traces)
        print(
            f"{query_id}: {trace['status']} "
            f"({len(traces)}/{len(questions)}, {trace['elapsed_ms']:.0f} ms)"
        )
    run_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _save_run(output_dir, run_manifest, cases, traces)
    return cases, traces, run_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--budget", type=int, default=18000)
    parser.add_argument("--max-searches", type=int, default=3)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_DATA_DIR / "chroma")
    args = parser.parse_args()
    cases, traces, manifest = run_benchmark(
        args.benchmark,
        args.split,
        args.output,
        top_k=args.top_k,
        budget=args.budget,
        max_searches=args.max_searches,
        index_dir=args.index_dir,
    )
    print(
        json.dumps(
            {
                "cases": len(cases),
                "errors": sum(trace["status"] == "error" for trace in traces),
                "model": manifest.get("agent_model"),
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
