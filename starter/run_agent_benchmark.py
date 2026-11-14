#!/usr/bin/env python3
"""Attempt the Day 7 agent loop on the real Day 6 benchmark.

This script writes a trace for each benchmark question. If no real model is
configured, the trace records a provider error instead of silently fabricating
answers. That keeps evaluation honest and matches the lab rubric.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from agentic import OpenAIModelController, RealDay6Retriever, run_agent

STANDARD_PRICING_USD_PER_MILLION = {
    'gpt-4o-mini': {'input': 0.15, 'cached_input': 0.075, 'output': 0.60},
}


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    return payload.get('questions', payload)


def infer_answerable(question: dict[str, Any]) -> bool:
    if isinstance(question.get('expected_answer'), str) and question['expected_answer'].strip():
        return True
    if isinstance(question.get('gold'), list) and question['gold']:
        return True
    if isinstance(question.get('answerable'), bool):
        return question['answerable']
    return False


def make_case(question: dict[str, Any]) -> dict[str, Any]:
    return {
        'id': str(question['id']),
        'split': str(question.get('split', 'dev')),
        'answerable': infer_answerable(question),
    }


def make_review(question: dict[str, Any]) -> dict[str, Any]:
    qid = str(question['id'])
    if infer_answerable(question):
        return {
            'query_id': qid,
            'answer_grade': 0,
            'citation_support': [],
        }
    return {
        'query_id': qid,
        'abstention_correct': False,
        'citation_support': [],
    }


def estimate_cost_usd(model: str | None, usage: dict[str, int] | None) -> float | None:
    pricing = STANDARD_PRICING_USD_PER_MILLION.get(model or '')
    if pricing is None or usage is None:
        return None
    cached = min(usage.get('cached_input_tokens', 0), usage.get('input_tokens', 0))
    uncached = usage.get('input_tokens', 0) - cached
    cost = (
        uncached * pricing['input']
        + cached * pricing['cached_input']
        + usage.get('output_tokens', 0) * pricing['output']
    ) / 1_000_000
    return round(cost, 10)


def run_one(
    question: dict[str, Any],
    *,
    split: str,
    max_searches: int,
    top_k: int,
    budget: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    qid = str(question['id'])
    case = make_case(question)
    if case['split'] != split:
        raise ValueError(f'question {qid} has split {case["split"]} but caller requested {split}')

    started = time.perf_counter()
    controller = None
    try:
        controller = OpenAIModelController()
        retriever = RealDay6Retriever(strategy='B')
        result = run_agent(
            str(question['question']),
            controller,
            retriever,
            filters=question.get('filter'),
            max_searches=max_searches,
            top_k=top_k,
            budget=budget,
        )
        trace = {
            'query_id': qid,
            'status': result.get('status', 'error'),
            'answer': result.get('answer'),
            'citations': result.get('citations', []),
            'events': result.get('events', []),
            'model': controller.model,
            'usage': controller.usage,
            'estimated_cost_usd': estimate_cost_usd(controller.model, controller.usage),
            'pricing_usd_per_1m_tokens': STANDARD_PRICING_USD_PER_MILLION.get(controller.model),
            'config': {'max_searches': max_searches, 'top_k': top_k, 'budget_chars': budget},
            'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
        }
        return case, trace
    except Exception as exc:  # pragma: no cover - environment-dependent path
        trace = {
            'query_id': qid,
            'status': 'error',
            'error': str(exc),
            'events': [],
            'citations': [],
            'model': getattr(controller, 'model', None),
            'usage': getattr(controller, 'usage', None),
            'estimated_cost_usd': estimate_cost_usd(
                getattr(controller, 'model', None), getattr(controller, 'usage', None)
            ),
            'pricing_usd_per_1m_tokens': STANDARD_PRICING_USD_PER_MILLION.get(
                getattr(controller, 'model', '')
            ),
            'config': {'max_searches': max_searches, 'top_k': top_k, 'budget_chars': budget},
            'elapsed_ms': round((time.perf_counter() - started) * 1000, 2),
        }
        return case, trace


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the Day 7 agent against the Day 6 benchmark')
    parser.add_argument('--benchmark', type=Path, default=Path(r'D:\Python\VinAI\Lab6_L3A\K5-L3-Day06-ChuMaiHong-2A202603377-DataIngestionEngineering\starter\benchmark\questions.json'))
    parser.add_argument('--additional-questions', type=Path, help='Optional JSON array of Day 7-only questions to append without editing Day 6 files')
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-searches', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--budget', type=int, default=800)
    args = parser.parse_args()

    if args.max_searches < 1 or args.top_k < 1 or args.budget < 1:
        parser.error('--max-searches, --top-k, and --budget must be >= 1')

    all_questions = load_questions(args.benchmark)
    if args.additional_questions:
        extra_questions = json.loads(args.additional_questions.read_text(encoding='utf-8'))
        if not isinstance(extra_questions, list):
            raise ValueError('--additional-questions must contain a JSON array')
        all_questions.extend(extra_questions)
    if len({str(q['id']) for q in all_questions}) != len(all_questions):
        raise ValueError('duplicate question IDs across benchmark files')
    questions = [q for q in all_questions if q.get('split') == args.split]
    if not questions:
        raise ValueError(f'Không tìm thấy câu hỏi nào trong split {args.split}')

    cases: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for question in questions:
        case, trace = run_one(
            question,
            split=args.split,
            max_searches=args.max_searches,
            top_k=args.top_k,
            budget=args.budget,
        )
        cases.append(case)
        traces.append(trace)
        reviews.append(make_review(question))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.output / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (args.output / 'reviews.json').write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    print(f'Wrote {len(cases)} cases to {args.output}')
    print('Note: without OPENAI_API_KEY configured, all traces are recorded as status="error" instead of fabricating answers.')


if __name__ == '__main__':
    main()
