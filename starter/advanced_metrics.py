"""Aggregate manually graded advanced-benchmark traces without an LLM judge."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(
    questions: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    selected = [question for question in questions if question.get("split") == split]
    if not selected:
        raise ValueError(f"No {split} questions in benchmark")
    expected_ids = {str(question["id"]) for question in selected}
    for label, rows, key in (
        ("case", cases, "id"),
        ("trace", traces, "query_id"),
        ("review", reviews, "query_id"),
    ):
        ids = [str(row.get(key)) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate {label} IDs")
        if set(ids) != expected_ids:
            missing = sorted(expected_ids - set(ids))
            extra = sorted(set(ids) - expected_ids)
            raise ValueError(f"{label} IDs do not match benchmark; missing={missing}, extra={extra}")

    case_by_id = {str(row["id"]): row for row in cases}
    trace_by_id = {str(row["query_id"]): row for row in traces}
    review_by_id = {str(row["query_id"]): row for row in reviews}
    per_category: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    failures: list[dict[str, Any]] = []
    elapsed_values: list[float] = []
    total_cost = 0.0
    known_costs = 0
    system_errors = 0

    for question in selected:
        query_id = str(question["id"])
        case = case_by_id[query_id]
        trace = trace_by_id[query_id]
        review = review_by_id[query_id]
        category = str(question["category"])
        answerable = bool(question["answerable"])
        if case.get("category") != category or case.get("answerable") is not answerable:
            raise ValueError(f"{query_id}: case labels disagree with benchmark")
        if not isinstance(review.get("pass"), bool) or not str(review.get("reason", "")).strip():
            raise ValueError(f"{query_id}: review needs a boolean pass label and a reason")
        citation_support = review.get("citation_support", [])
        if not isinstance(citation_support, list) or any(type(value) is not bool for value in citation_support):
            raise ValueError(f"{query_id}: citation_support must be a boolean array")
        if len(citation_support) != len(trace.get("citations", [])):
            raise ValueError(f"{query_id}: citation_support must match the trace citations")

        passed = review["pass"]
        if trace.get("status") == "error":
            system_errors += 1
            if passed:
                raise ValueError(f"{query_id}: system errors cannot pass")
        elif answerable:
            if trace.get("status") != "answered" and passed:
                raise ValueError(f"{query_id}: abstention cannot pass an answerable question")
            locations = trace.get("resolved_citations", [])
            if passed and (
                not trace.get("citations")
                or not citation_support
                or not all(citation_support)
                or len(locations) != len(trace["citations"])
                or any(not item.get("resolved") for item in locations)
            ):
                raise ValueError(f"{query_id}: a passing answer needs supported, page-resolved citations")
        else:
            if passed and trace.get("status") != "abstain":
                raise ValueError(f"{query_id}: a no-evidence question passes only when the agent abstains")
            if not str(review.get("absence_check", "")).strip():
                raise ValueError(f"{query_id}: no-evidence review needs an absence-check description")

        group = per_category[category]
        group["total"] += 1
        group["passed"] += int(passed)
        if not passed:
            failures.append(
                {
                    "query_id": query_id,
                    "category": category,
                    "status": trace.get("status"),
                    "error": trace.get("error"),
                    "reason": review["reason"],
                }
            )
        elapsed = trace.get("elapsed_ms")
        if isinstance(elapsed, (int, float)):
            elapsed_values.append(float(elapsed))
        cost = trace.get("estimated_cost_usd")
        if isinstance(cost, (int, float)):
            total_cost += float(cost)
            known_costs += 1

    total = len(selected)
    passed_count = sum(group["passed"] for group in per_category.values())
    return {
        "split": split,
        "passed": passed_count,
        "total": total,
        "pass_rate": passed_count / total,
        "threshold": {
            "required_rate": 0.9,
            "required_passes": math.ceil(total * 0.9),
            "met": passed_count * 10 >= total * 9,
        },
        "per_category": {
            category: {
                **values,
                "pass_rate": values["passed"] / values["total"],
            }
            for category, values in sorted(per_category.items())
        },
        "system_errors": system_errors,
        "elapsed_ms": {
            "count": len(elapsed_values),
            "mean": statistics.mean(elapsed_values) if elapsed_values else None,
            "median": statistics.median(elapsed_values) if elapsed_values else None,
            "total": sum(elapsed_values) if elapsed_values else 0,
        },
        "estimated_cost_usd": round(total_cost, 10) if known_costs == total else None,
        "cost_cases_with_usage": known_costs,
        "failures": failures,
        "manual_grading_complete": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark = _read(args.benchmark)
    if not isinstance(benchmark, dict) or not isinstance(benchmark.get("questions"), list):
        raise ValueError("Benchmark must contain a questions array")
    result = aggregate(
        benchmark["questions"],
        _read(args.cases),
        _read(args.traces),
        _read(args.reviews),
        split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
