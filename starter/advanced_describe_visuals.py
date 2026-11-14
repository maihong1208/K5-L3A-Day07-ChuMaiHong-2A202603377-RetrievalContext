"""Create page-grounded visual descriptions with the configured vision-capable model."""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any


PRICING_USD_PER_MILLION = {
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "advanced" / "extracted"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _response_text(response: Any) -> str:
    if hasattr(response, "choices"):
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    raise ValueError("Vision model returned no readable text content")


def _decode_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict) or not isinstance(payload.get("visuals"), list):
        raise ValueError("Vision response must contain a 'visuals' array")
    for item in payload["visuals"]:
        if not isinstance(item, dict):
            raise ValueError("Each visual description must be a JSON object")
        if item.get("type") not in {"chart", "map", "diagram", "photograph", "other"}:
            raise ValueError("Each visual must have a supported type")
        if not isinstance(item.get("description"), str) or not item["description"].strip():
            raise ValueError("Each visual must have a non-empty description")
    return payload


def _estimate_cost(model: str, usage: dict[str, int]) -> float | None:
    prices = PRICING_USD_PER_MILLION.get(model)
    if prices is None:
        return None
    cached = min(usage.get("cached_input_tokens", 0), usage.get("input_tokens", 0))
    uncached = usage.get("input_tokens", 0) - cached
    amount = (
        uncached * prices["input"]
        + cached * prices["cached_input"]
        + usage.get("output_tokens", 0) * prices["output"]
    ) / 1_000_000
    return round(amount, 10)


def _totals(records: list[dict[str, Any]]) -> tuple[dict[str, int], float | None]:
    usage_total = {
        key: sum(
            int((record.get("usage") or {}).get(key, 0))
            for record in records
            if record.get("status") == "ok"
        )
        for key in ("requests", "input_tokens", "cached_input_tokens", "output_tokens")
    }
    costs = [
        record["estimated_cost_usd"]
        for record in records
        if record.get("status") == "ok" and isinstance(record.get("estimated_cost_usd"), (int, float))
    ]
    return usage_total, round(sum(costs), 10) if costs else None


def describe_pages(
    manifest_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    try:
        from agentic import OpenAIModelController
    except ImportError as exc:
        raise RuntimeError(
            "Run this script with the configured Day 7/Day 6 Python environment, "
            "which provides the OpenAI SDK and the project-local .env."
        ) from exc

    manifest = _read_json(manifest_path)
    pages = manifest.get("visual_pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("The extraction manifest has no visual pages")
    valid_page_numbers = {int(page["pdf_page"]) for page in pages}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, Any] = {}
    if resume and output_path.exists():
        existing = _read_json(output_path)
    records_by_page = {
        int(record["pdf_page"]): record
        for record in existing.get("pages", [])
        if isinstance(record, dict) and isinstance(record.get("pdf_page"), int)
        and int(record["pdf_page"]) in valid_page_numbers
    }

    controller = OpenAIModelController()
    selected_pages = pages[:limit] if limit is not None else pages
    for page in selected_pages:
        pdf_page = int(page["pdf_page"])
        previous = records_by_page.get(pdf_page)
        if previous and previous.get("status") == "ok":
            continue
        image_path = manifest_path.parent / str(page["image_path"])
        if not image_path.is_file():
            records_by_page[pdf_page] = {
                "pdf_page": pdf_page,
                "status": "error",
                "error": f"Rendered visual page is missing: {image_path}",
            }
            continue

        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
        figure_labels = page.get("figure_labels") or []
        captions = page.get("figure_captions") or []
        context = str(page.get("page_text_context") or "")[:2200]
        prompt = f"""Inspect this page image from a climate-science technical report.
Return only JSON with this shape:
{{"visuals":[{{"label":"figure/table label or page visual","type":"chart|map|diagram|photograph|other","description":"concise visually grounded description"}}]}}

Page anchors: PDF page {pdf_page}, printed page {page.get('printed_page')}.
Printed figure labels: {figure_labels or "none detected"}.
Extracted caption text: {captions or "none detected"}.
Nearby extracted page text (context only): {context}

Describe every meaningful information-bearing chart, map, diagram, or image on the page.
For charts, identify visible axes, units, series/legend, and direction or comparison only when legible.
For diagrams/maps, identify the elements and relationships or geographic scale visible.
Do not invent exact values, labels, causal claims, or content obscured in the image.
Use type "other" for a table or other visual that does not fit the categories.
If there is no information-bearing visual, return an empty visuals array."""

        started = time.perf_counter()
        usage_before = dict(controller.usage)
        try:
            response = controller.client.chat.completions.create(
                model=controller.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                temperature=0,
                max_tokens=700,
            )
            controller._record_usage(response)
            usage_delta = {
                key: controller.usage[key] - usage_before[key]
                for key in ("requests", "input_tokens", "cached_input_tokens", "output_tokens")
            }
            visuals = _decode_json(_response_text(response))["visuals"]
            records_by_page[pdf_page] = {
                "pdf_page": pdf_page,
                "printed_page": int(page["printed_page"]),
                "status": "ok",
                "image_path": str(page["image_path"]),
                "page_bbox": page.get("page_bbox"),
                "figure_labels": figure_labels,
                "figure_captions": captions,
                "visuals": visuals,
                "model": controller.model,
                "usage": usage_delta,
                "estimated_cost_usd": _estimate_cost(controller.model, usage_delta),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            records_by_page[pdf_page] = {
                "pdf_page": pdf_page,
                "printed_page": int(page["printed_page"]),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "image_path": str(page["image_path"]),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }

        records = sorted(records_by_page.values(), key=lambda item: int(item["pdf_page"]))
        usage_total, cost_total = _totals(records)
        result = {
            "source_url": manifest["document"]["source_url"],
            "document_sha256": manifest["document"]["sha256"],
            "model": controller.model,
            "pricing_usd_per_1m_tokens": PRICING_USD_PER_MILLION.get(controller.model),
            "usage_total": usage_total,
            "estimated_cost_usd": cost_total,
            "pages": records,
        }
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"PDF page {pdf_page}: {records_by_page[pdf_page]['status']} "
            f"({len(records_by_page)} of {len(pages)} pages recorded)"
        )

    records = sorted(records_by_page.values(), key=lambda item: int(item["pdf_page"]))
    usage_total, cost_total = _totals(records)
    result = {
        "source_url": manifest["document"]["source_url"],
        "document_sha256": manifest["document"]["sha256"],
        "model": controller.model,
        "pricing_usd_per_1m_tokens": PRICING_USD_PER_MILLION.get(controller.model),
        "usage_total": usage_total,
        "estimated_cost_usd": cost_total,
        "pages": records,
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA_DIR / "manifest.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA_DIR / "visual_descriptions.json")
    parser.add_argument("--limit", type=int, help="Process only the first N visual pages")
    parser.add_argument("--resume", action="store_true", help="Skip pages already described successfully")
    args = parser.parse_args()
    result = describe_pages(args.manifest, args.output, limit=args.limit, resume=args.resume)
    errors = [page for page in result["pages"] if page.get("status") != "ok"]
    print(
        json.dumps(
            {
                "model": result["model"],
                "pages_recorded": len(result["pages"]),
                "page_errors": len(errors),
                "usage_total": result["usage_total"],
                "estimated_cost_usd": result["estimated_cost_usd"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Descriptions: {args.output.resolve()}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
