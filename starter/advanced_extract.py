"""Extract page-aware text, tables, and visual page images for the advanced lab."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SOURCE_URL = "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_TS.pdf"
DEFAULT_PDF = Path(__file__).resolve().parent / "data" / "advanced" / "IPCC_AR6_WGI_TS.pdf"
DEFAULT_OUTPUT = DEFAULT_PDF.parent / "extracted"
FIGURE_CAPTION = re.compile(
    r"(?im)^\s*Figure\s+(TS\.\d+)(?:\s*\(continued\)|\s*\|)"
)
TABLE_CAPTION = re.compile(
    r"(?im)^\s*Table\s+(TS\.\d+)(?:\s*\(continued\))?\s*(?:\||:)"
)


def _clean_cell(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = re.sub(r"\s+", " ", str(value).replace("\xad", "")).strip()
    return cleaned or None


def chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 160) -> list[str]:
    """Split page text without dropping content, preserving a small overlap."""
    if max_chars < 1 or overlap < 0 or overlap >= max_chars:
        raise ValueError("max_chars must be positive and overlap must be smaller than it")
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            candidate = f"{current}\n\n{paragraph}" if current else paragraph
            if len(candidate) <= max_chars:
                current = candidate
            else:
                flush()
                current = paragraph
            continue

        flush()
        start = 0
        while start < len(paragraph):
            end = min(start + max_chars, len(paragraph))
            if end < len(paragraph):
                boundary = paragraph.rfind(" ", start + max_chars // 2, end)
                if boundary > start:
                    end = boundary
            piece = paragraph[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(paragraph):
                break
            start = max(start + 1, end - overlap)

    flush()
    return chunks


def _caption_text(text: str, match: re.Match[str], following: re.Pattern[str]) -> str:
    start = match.start()
    end = following.search(text, match.end())
    caption = text[start : end.start() if end else min(len(text), start + 1800)]
    return re.sub(r"\s+", " ", caption).strip()[:1800]


def _table_records(page: Any, page_text: str, page_number: int) -> tuple[list[dict[str, Any]], list[str]]:
    labels = sorted({match.group(1).upper() for match in TABLE_CAPTION.finditer(page_text)})
    if not labels:
        return [], []

    errors: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        tables = page.find_tables().tables
    except Exception as exc:
        return [], [f"PDF page {page_number}: table detection failed: {exc}"]

    for index, table in enumerate(tables, start=1):
        try:
            raw_rows = table.extract()
        except Exception as exc:
            errors.append(f"PDF page {page_number}, table {index}: extraction failed: {exc}")
            continue
        rows = [[_clean_cell(cell) for cell in row] for row in raw_rows]
        if not any(any(cell for cell in row) for row in rows):
            continue
        records.append(
            {
                "label": labels[min(index - 1, len(labels) - 1)],
                "table_index": index,
                "bbox": [round(float(value), 2) for value in table.bbox],
                "rows": rows,
            }
        )
    if not records:
        errors.append(f"PDF page {page_number}: captioned table was not extractable")
    return records, errors


def extract_pdf(
    pdf_path: Path,
    output_dir: Path,
    *,
    source_url: str = SOURCE_URL,
    printed_page_offset: int = 32,
    render_dpi: int = 115,
) -> dict[str, Any]:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for extraction. Install starter/requirements-advanced-extract.txt "
            "in an isolated Python environment."
        ) from exc

    pdf_path = pdf_path.resolve()
    output_dir = output_dir.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if render_dpi < 72 or render_dpi > 240:
        raise ValueError("render_dpi must be between 72 and 240")

    visual_dir = output_dir / "visual_pages"
    visual_dir.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open(pdf_path)
    all_pages: list[dict[str, Any]] = []
    visual_pages: list[dict[str, Any]] = []
    table_errors: list[str] = []
    low_text_pages: list[dict[str, int]] = []
    text_repairs: list[dict[str, Any]] = []
    text_encoding_warnings: list[dict[str, int]] = []
    total_characters = 0

    for index, page in enumerate(document):
        pdf_page = index + 1
        printed_page = pdf_page + printed_page_offset
        text = page.get_text("text", sort=True).replace("\xad", "")
        replacement_count = text.count("\ufffd")
        if replacement_count and "Table of Contents" in text:
            text = text.replace("\ufffd", " ")
            text_repairs.append(
                {
                    "pdf_page": pdf_page,
                    "replacement_characters_removed": replacement_count - text.count("\ufffd"),
                    "reason": "Decorative table-of-contents leader dots were undecodable font glyphs.",
                }
            )
        remaining_replacements = text.count("\ufffd")
        if remaining_replacements:
            text_encoding_warnings.append(
                {"pdf_page": pdf_page, "replacement_characters": remaining_replacements}
            )
        normalized = text.strip()
        figure_matches = list(FIGURE_CAPTION.finditer(text))
        figure_labels = list(dict.fromkeys(match.group(1).upper() for match in figure_matches))
        figure_captions = [
            _caption_text(text, match, FIGURE_CAPTION)
            for match in figure_matches
        ]
        table_labels = list(
            dict.fromkeys(match.group(1).upper() for match in TABLE_CAPTION.finditer(text))
        )
        table_records, errors = _table_records(page, text, pdf_page)
        table_errors.extend(errors)
        image_count = len(page.get_images())
        total_characters += len(normalized)
        if len(normalized) < 100:
            low_text_pages.append({"pdf_page": pdf_page, "characters": len(normalized)})

        is_content = bool(normalized or table_records or image_count or figure_labels)
        if len(normalized) < 10 and not (table_records or image_count or figure_labels):
            is_content = False
        record: dict[str, Any] = {
            "pdf_page": pdf_page,
            "printed_page": printed_page,
            "text": normalized,
            "text_characters": len(normalized),
            "figure_labels": figure_labels,
            "figure_captions": figure_captions,
            "table_labels": table_labels,
            "tables": table_records,
            "embedded_image_count": image_count,
            "page_bbox": [round(float(value), 2) for value in page.rect],
            "status": "content" if is_content else "blank_or_unreadable",
        }
        if not is_content:
            record["skip_reason"] = "Only a page number or negligible text was extracted; no visual or table content was found."
        all_pages.append(record)

        if figure_labels or table_labels or image_count:
            image_name = f"page-{pdf_page:03}.jpg"
            image_path = visual_dir / image_name
            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(render_dpi / 72, render_dpi / 72),
                alpha=False,
            )
            pixmap.save(image_path, jpg_quality=82)
            visual_pages.append(
                {
                    "pdf_page": pdf_page,
                    "printed_page": printed_page,
                    "image_path": str(Path("visual_pages") / image_name),
                    "figure_labels": figure_labels,
                    "figure_captions": figure_captions,
                    "table_labels": table_labels,
                    "page_bbox": record["page_bbox"],
                    "page_text_context": normalized[:2800],
                }
            )

    metadata = dict(document.metadata or {})
    document.close()
    manifest = {
        "document": {
            "title": metadata.get("title") or "Technical Summary",
            "author": metadata.get("author") or "IPCC AR6 Working Group I",
            "source_url": source_url,
            "rights_status": (
                "No explicit reuse license was found in the extracted PDF; the official report "
                "page could not be accessed from this environment, so reuse rights are unverified. "
                "Keep the source PDF, page renders, and full extracted text local; do not "
                "redistribute them without permission."
            ),
            "local_filename": pdf_path.name,
            "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
            "size_bytes": pdf_path.stat().st_size,
            "pdf_pages": len(all_pages),
            "printed_page_offset": printed_page_offset,
            "metadata": metadata,
        },
        "extraction": {
            "tool": "PyMuPDF",
            "tool_version": pymupdf.VersionBind,
            "render_dpi": render_dpi,
            "page_count_processed": len(all_pages),
            "total_extracted_characters": total_characters,
            "low_text_pages": low_text_pages,
            "text_repairs": text_repairs,
            "text_encoding_warnings": text_encoding_warnings,
            "table_errors": table_errors,
            "visual_page_count": len(visual_pages),
            "table_count": sum(len(page["tables"]) for page in all_pages),
            "status": "complete_with_warnings" if table_errors else "complete",
        },
        "pages": all_pages,
        "visual_pages": visual_pages,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-url", default=SOURCE_URL)
    parser.add_argument("--printed-page-offset", type=int, default=32)
    parser.add_argument("--render-dpi", type=int, default=115)
    args = parser.parse_args()
    result = extract_pdf(
        args.pdf,
        args.output,
        source_url=args.source_url,
        printed_page_offset=args.printed_page_offset,
        render_dpi=args.render_dpi,
    )
    print(json.dumps(result["extraction"], ensure_ascii=False, indent=2))
    print(f"Manifest: {args.output.resolve() / 'manifest.json'}")
    if result["extraction"]["table_errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
