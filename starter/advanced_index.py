"""Build an isolated Chroma index over text, extracted tables, and visual descriptions."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


STARTER_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = STARTER_DIR / "data" / "advanced" / "extracted"
DEFAULT_INDEX_DIR = DEFAULT_DATA_DIR / "chroma"
DAY6_ROOT = Path(
    r"D:\Python\VinAI\Lab6_L3A\K5-L3-Day06-ChuMaiHong-2A202603377-DataIngestionEngineering"
)
DAY6_STARTER = DAY6_ROOT / "starter"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SOURCE_NAME = "IPCC AR6 WGI Technical Summary (2021)"

if str(DAY6_STARTER) not in sys.path:
    sys.path.insert(0, str(DAY6_STARTER))

from src.embeddings import MyEmbedder  # noqa: E402
from src.models import Chunk  # noqa: E402
from src.store import ChromaVectorStore  # noqa: E402


def _read_object(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return result


def _header_names(rows: list[list[str | None]], header_count: int) -> list[str]:
    width = max((len(row) for row in rows), default=0)
    columns: list[list[str]] = [[] for _ in range(width)]
    for row in rows[:header_count]:
        carried = ""
        for column in range(width):
            value = row[column] if column < len(row) else None
            if value:
                carried = value
            if carried and carried not in columns[column]:
                columns[column].append(carried)
    return [
        " / ".join(parts) if parts else f"column {index + 1}"
        for index, parts in enumerate(columns)
    ]


def build_chunks(manifest: dict[str, Any], visual_descriptions: dict[str, Any]) -> list[dict[str, Any]]:
    from advanced_extract import chunk_text

    document = manifest.get("document")
    pages = manifest.get("pages")
    if not isinstance(document, dict) or not isinstance(pages, list):
        raise ValueError("Invalid extraction manifest")
    if visual_descriptions.get("document_sha256") != document.get("sha256"):
        raise ValueError("Visual descriptions were generated from a different source PDF")
    visual_by_page = {
        int(record["pdf_page"]): record
        for record in visual_descriptions.get("pages", [])
        if isinstance(record, dict) and isinstance(record.get("pdf_page"), int)
    }

    source_id = f"ipcc-ar6-wgi-ts-{str(document['sha256'])[:16]}"
    chunks: list[dict[str, Any]] = []
    for page in pages:
        pdf_page = int(page["pdf_page"])
        printed_page = int(page["printed_page"])
        page_text = str(page.get("text") or "")
        should_index_text = not (
            len(page_text) < 10
            and not page.get("tables")
            and not page.get("figure_labels")
            and not page.get("embedded_image_count")
        )
        for index, text in enumerate(chunk_text(page_text) if should_index_text else []):
            chunk_id = f"text-p{pdf_page:03}-c{index + 1:03}"
            chunks.append(
                {
                    "id": chunk_id,
                    "source_id": source_id,
                    "source": SOURCE_NAME,
                    "page": pdf_page,
                    "text": text,
                    "metadata": {
                        "kind": "text",
                        "printed_page": printed_page,
                        "chunk_index": index + 1,
                        "region": "full PDF page",
                    },
                }
            )

        for table in page.get("tables") or []:
            label = str(table.get("label") or "unlabeled table")
            rows = table.get("rows") or []
            if not rows:
                continue
            if label == "TS.1":
                header_count = 1
                header_rows = rows[:1]
            elif label == "TS.2":
                header_count = 2
                header_rows = rows[:2]
            elif label == "TS.3":
                header_count = 3
                header_rows = rows[:3]
            elif label == "TS.4" and pdf_page == 83:
                header_count = 2
                header_rows = rows[1:2]
            else:
                header_count = 3
                header_rows = rows[:3]
            headers = _header_names(header_rows, len(header_rows))
            bbox = json.dumps(table.get("bbox"), separators=(",", ":"))
            for row_index, row in enumerate(rows[header_count:], start=header_count + 1):
                cells = []
                for column, value in enumerate(row):
                    if value:
                        heading = headers[column] if column < len(headers) else f"column {column + 1}"
                        cells.append(f"{heading}: {value}")
                if not cells:
                    continue
                chunk_id = f"table-{label.lower()}-p{pdf_page:03}-r{row_index:03}"
                table_text = (
                    f"{label} extracted row {row_index} (PDF page {pdf_page}, "
                    f"printed page {printed_page}). " + " | ".join(cells)
                )
                chunks.append(
                    {
                        "id": chunk_id,
                        "source_id": source_id,
                        "source": SOURCE_NAME,
                        "page": pdf_page,
                        "text": table_text,
                        "metadata": {
                            "kind": "table",
                            "printed_page": printed_page,
                            "table_label": label,
                            "table_index": int(table.get("table_index", 1)),
                            "table_row_index": row_index,
                            "region_bbox": bbox,
                            "region": f"table {label}, extracted row {row_index}",
                        },
                    }
                )

        visual_record = visual_by_page.get(pdf_page)
        if visual_record is None:
            continue
        if visual_record.get("status") != "ok":
            raise ValueError(
                f"Visual description failed for PDF page {pdf_page}: "
                f"{visual_record.get('error', 'unknown error')}"
            )
        for visual_index, visual in enumerate(visual_record.get("visuals") or [], start=1):
            description = str(visual.get("description") or "").strip()
            if not description:
                continue
            label = str(visual.get("label") or "").strip()
            figure_labels = page.get("figure_labels") or []
            anchor = label or (figure_labels[0] if figure_labels else f"visual on PDF page {pdf_page}")
            caption = " ".join(page.get("figure_captions") or [])
            visual_text = (
                f"{anchor}. Visual type: {visual['type']}. "
                f"PDF page {pdf_page}, printed page {printed_page}. "
                f"Printed caption: {caption} "
                f"Model description: {description}"
            ).strip()
            chunks.append(
                {
                    "id": f"visual-p{pdf_page:03}-v{visual_index:02}",
                    "source_id": source_id,
                    "source": SOURCE_NAME,
                    "page": pdf_page,
                    "text": visual_text,
                    "metadata": {
                        "kind": "visual",
                        "printed_page": printed_page,
                        "visual_type": str(visual["type"]),
                        "figure_label": anchor,
                        "image_path": str(visual_record.get("image_path") or ""),
                        "region_bbox": json.dumps(
                            visual_record.get("page_bbox"), separators=(",", ":")
                        ),
                        "region": f"full-page image; {anchor}",
                    },
                }
            )

    ids = [chunk["id"] for chunk in chunks]
    if not chunks or len(ids) != len(set(ids)):
        raise ValueError("The index must contain non-empty, uniquely identified chunks")
    return chunks


def build_index(
    manifest_path: Path,
    visual_path: Path,
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> dict[str, Any]:
    manifest = _read_object(manifest_path)
    visual_descriptions = _read_object(visual_path)
    chunks = build_chunks(manifest, visual_descriptions)
    embedder = MyEmbedder(model_name=EMBEDDING_MODEL, device="cpu")
    vectors = embedder.embed_documents([chunk["text"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding model returned an unexpected number of vectors")

    typed_chunks = [
        Chunk(
            chunk_id=chunk["id"],
            document_id=chunk["source_id"],
            source=chunk["source"],
            page=chunk["page"],
            text=chunk["text"],
            metadata=chunk["metadata"],
        )
        for chunk in chunks
    ]
    store = ChromaVectorStore(persist_directory=index_dir, dimensions=embedder.dimensions)
    store.upsert("advanced", typed_chunks, vectors)
    counts = Counter(chunk["metadata"]["kind"] for chunk in chunks)
    index_manifest = {
        "source_url": manifest["document"]["source_url"],
        "source_sha256": manifest["document"]["sha256"],
        "source_id": chunks[0]["source_id"],
        "document_pages": manifest["document"]["pdf_pages"],
        "text_pages_indexed": sum(
            bool(page.get("text"))
            and not (
                len(str(page.get("text") or "")) < 10
                and not page.get("tables")
                and not page.get("figure_labels")
                and not page.get("embedded_image_count")
            )
            for page in manifest["pages"]
        ),
        "excluded_pages": [
            {
                "pdf_page": page["pdf_page"],
                "printed_page": page["printed_page"],
                "reason": page.get("skip_reason"),
            }
            for page in manifest["pages"]
            if page.get("status") != "content"
        ],
        "indexed_chunks": len(chunks),
        "chunks_by_kind": dict(sorted(counts.items())),
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "vector_store": "Chroma cosine",
        "collection": "day6_advanced",
        "isolated_persist_directory": str(index_dir.resolve()),
    }
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir.parent / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (index_dir.parent / "index_manifest.json").write_text(
        json.dumps(index_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return index_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA_DIR / "manifest.json")
    parser.add_argument(
        "--visual-descriptions",
        type=Path,
        default=DEFAULT_DATA_DIR / "visual_descriptions.json",
    )
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args()
    result = build_index(args.manifest, args.visual_descriptions, index_dir=args.index_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
