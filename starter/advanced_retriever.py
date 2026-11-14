"""Hybrid dense/lexical retrieval for the isolated AR6 advanced index."""
from __future__ import annotations

import json
import re
import sys
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
FIGURE_REFERENCE = re.compile(r"\bfigure\s+TS\.\d+(?:\s*\([a-z]\))?", re.IGNORECASE)

if str(DAY6_STARTER) not in sys.path:
    sys.path.insert(0, str(DAY6_STARTER))

from src.embeddings import MyEmbedder  # noqa: E402
from src.store import ChromaVectorStore  # noqa: E402

sys.path.insert(0, str(STARTER_DIR))
from baseline import TfidfIndex  # noqa: E402


def prioritize_named_figures(
    query: str,
    results: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    """Keep explicitly named figure evidence in the candidate context."""
    references = {
        re.sub(r"\s+", "", match.group()).casefold()
        for match in FIGURE_REFERENCE.finditer(query)
    }
    if not references:
        return results[:top_k]

    by_id = {str(result["id"]): result for result in results}
    anchors = []
    for chunk in chunks:
        label = str((chunk.get("metadata") or {}).get("figure_label") or "")
        normalized_label = re.sub(r"\s+", "", label).casefold()
        if normalized_label not in references:
            continue
        result = dict(by_id.get(str(chunk["id"]), chunk))
        result.setdefault("score", 0.0)
        result["figure_anchor"] = True
        anchors.append(result)

    anchored_ids = {str(result["id"]) for result in anchors}
    return (anchors + [result for result in results if str(result["id"]) not in anchored_ids])[:top_k]


def reciprocal_rank_fusion(
    dense: list[dict[str, Any]],
    lexical: list[dict[str, Any]],
    *,
    top_k: int,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    """Combine dense and TF-IDF rankings while retaining page/region metadata."""
    if top_k < 1 or rank_constant < 1:
        raise ValueError("top_k and rank_constant must be positive")
    fused: dict[str, dict[str, Any]] = {}
    for ranking_name, results in (("dense", dense), ("lexical", lexical)):
        for rank, result in enumerate(results, start=1):
            chunk_id = str(result["id"])
            entry = fused.setdefault(
                chunk_id,
                {
                    **result,
                    "score": 0.0,
                    "dense_rank": None,
                    "lexical_rank": None,
                    "dense_score": None,
                    "lexical_score": None,
                },
            )
            entry["score"] += 1.0 / (rank_constant + rank)
            entry[f"{ranking_name}_rank"] = rank
            entry[f"{ranking_name}_score"] = float(result.get("score", 0.0))
            if ranking_name == "dense":
                entry.update(
                    {
                        "text": result["text"],
                        "source_id": result["source_id"],
                        "source": result["source"],
                        "page": result["page"],
                        "metadata": result["metadata"],
                    }
                )
    return sorted(
        fused.values(),
        key=lambda item: (
            -item["score"],
            item["dense_rank"] is None,
            item["dense_rank"] or rank_constant,
            item["lexical_rank"] or rank_constant,
            item["id"],
        ),
    )[:top_k]


class AdvancedRetriever:
    """Query the new Chroma collection and a lexical view of the same chunks."""

    def __init__(
        self,
        *,
        index_dir: Path = DEFAULT_INDEX_DIR,
        chunks_path: Path = DEFAULT_DATA_DIR / "chunks.json",
        model_name: str = EMBEDDING_MODEL,
        device: str = "cpu",
    ) -> None:
        index_manifest_path = index_dir.parent / "index_manifest.json"
        if not index_manifest_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(
                "Advanced index is missing. Run starter/advanced_index.py first."
            )
        index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
        if index_manifest.get("embedding_model") != model_name:
            raise ValueError(
                f"Index uses {index_manifest.get('embedding_model')!r}, not {model_name!r}"
            )
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        if not isinstance(self.chunks, list) or not self.chunks:
            raise ValueError("Advanced index chunks must be a non-empty JSON array")
        self.embedder = MyEmbedder(model_name=model_name, device=device)
        if self.embedder.dimensions != index_manifest.get("embedding_dimensions"):
            raise ValueError("Embedding dimensions do not match the stored advanced index")
        self.store = ChromaVectorStore(
            persist_directory=index_dir,
            dimensions=self.embedder.dimensions,
        )
        self.lexical_index = TfidfIndex(self.chunks)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        candidate_count = max(top_k * 4, 20)
        dense_hits = self.store.search(
            "advanced",
            self.embedder.embed_query(query),
            top_k=candidate_count,
        )
        dense_by_id = {chunk["id"]: chunk for chunk in self.chunks}
        dense = []
        for hit in dense_hits:
            source_chunk = dense_by_id.get(str(hit["chunk_id"]))
            if source_chunk is None:
                raise RuntimeError(f"Chroma returned an unknown chunk ID: {hit['chunk_id']}")
            dense.append(
                {
                    **source_chunk,
                    "score": float(hit["score"]),
                }
            )
        lexical = self.lexical_index.search(query, top_k=candidate_count)
        results = reciprocal_rank_fusion(dense, lexical, top_k=candidate_count)
        results = prioritize_named_figures(
            query,
            results,
            self.chunks,
            top_k=candidate_count,
        )
        if filters:
            results = [
                result
                for result in results
                if all(
                    result["metadata"].get(key, result.get(key)) == value
                    for key, value in filters.items()
                )
            ]
        return results[:top_k]
