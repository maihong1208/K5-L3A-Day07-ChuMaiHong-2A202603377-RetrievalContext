r"""Real Day 6 connector for Day 7.

This module connects to the actual Day 6 project in:
    D:\Python\VinAI\Lab6_L3A\K5-L3-Day06-ChuMaiHong-2A202603377-DataIngestionEngineering
without modifying any Day 6 source files.

It reuses the same embedding model and Chroma vector store that Day 6 already
built, then exposes the interface expected by Day 7's `Retriever.search`.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

DAY6_ROOT = Path(r"D:\Python\VinAI\Lab6_L3A\K5-L3-Day06-ChuMaiHong-2A202603377-DataIngestionEngineering")
DAY6_STARTER = DAY6_ROOT / "starter"
DAY6_SRC = DAY6_STARTER / "src"
DAY6_CHROMA = DAY6_STARTER / "results" / "chroma"

if str(DAY6_STARTER) not in sys.path:
    sys.path.insert(0, str(DAY6_STARTER))
if str(DAY6_SRC) not in sys.path:
    sys.path.insert(0, str(DAY6_SRC))

try:
    from src.embeddings import MyEmbedder  # type: ignore
    from src.store import ChromaVectorStore  # type: ignore
except Exception as exc:  # pragma: no cover - import-time failure indicates Day 6 is missing
    raise RuntimeError(
        "Cannot import Day 6 code from the original project. "
        f"Expected folder: {DAY6_ROOT}. Make sure the Day 6 project exists and the "
        "embedding dependencies are installed."
    ) from exc

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DIMENSIONS = 384
DEFAULT_STRATEGY = "B"


def normalize_day6_filters(filters: Any) -> Any:
    """Map Day 6 benchmark filters into the Chroma filter schema.

    Day 6 benchmark questions use records like:
        {"field": "section", "operator": "eq", "value": "LLM01:2025"}
    while Chroma expects a direct equality mapping like:
        {"section": "LLM01:2025"}
    """
    if filters is None:
        return None
    if isinstance(filters, Mapping):
        if {"field", "operator", "value"}.issubset(filters.keys()):
            field = filters.get("field")
            value = filters.get("value")
            if field is not None and value is not None:
                return {str(field): value}
        normalized: dict[str, Any] = {}
        for key, value in filters.items():
            if key in {"field", "operator"}:
                continue
            normalized[str(key)] = normalize_day6_filters(value)
        return normalized
    if isinstance(filters, (list, tuple)):
        return [normalize_day6_filters(item) for item in filters]
    return filters


@lru_cache(maxsize=4)
def get_day6_embedder(model_name: str = DEFAULT_MODEL, device: str = "cpu") -> MyEmbedder:
    """Instantiate the same local sentence-transformers embedder used by Day 6."""
    return MyEmbedder(model_name=model_name, device=device)


@lru_cache(maxsize=4)
def get_day6_store(dimensions: int = DEFAULT_DIMENSIONS) -> ChromaVectorStore:
    """Open the persistent Chroma index produced by Day 6."""
    return ChromaVectorStore(persist_directory=DAY6_CHROMA, dimensions=dimensions)


def normalize_day6_hit(hit: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(hit.get("metadata") or {})
    page = metadata.get("page")
    source_id = metadata.get("document_id")
    source = metadata.get("source")

    return {
        "id": str(hit.get("chunk_id") or hit.get("id") or "unknown-chunk"),
        "text": str(hit.get("text") or ""),
        "source_id": str(source_id) if source_id is not None else "unknown-doc",
        "source": str(source) if source is not None else "unknown-source",
        "page": int(page) if page is not None else 0,
        "metadata": metadata,
        "score": float(hit.get("score", 0.0)),
    }


def search_day6(
    query: str,
    *,
    strategy: str = DEFAULT_STRATEGY,
    filters: dict[str, Any] | None = None,
    top_k: int = 3,
    model_name: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Query the real Day 6 vector store and convert hits to the Day 7 schema."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    embedder = get_day6_embedder(model_name=model_name, device=device)
    store = get_day6_store(dimensions=embedder.dimensions)
    query_vector = embedder.embed_query(query)
    normalized_filters = normalize_day6_filters(filters)
    raw_hits = store.search(strategy=strategy, query_vector=query_vector, top_k=top_k, filters=normalized_filters)
    return [normalize_day6_hit(hit) for hit in raw_hits]


class Day6Retriever:
    """Simple Day 7-compatible retriever that wraps the existing Day 6 Chroma index."""

    def __init__(
        self,
        *,
        strategy: str = DEFAULT_STRATEGY,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
    ) -> None:
        self.strategy = strategy
        self.model_name = model_name
        self.device = device

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        return search_day6(
            query,
            strategy=self.strategy,
            filters=filters,
            top_k=top_k,
            model_name=self.model_name,
            device=self.device,
        )


if __name__ == "__main__":
    sample = "Theo OWASP LLM01, prompt injection làm thay đổi điều gì ở LLM?"
    print("Day 6 project:", DAY6_ROOT)
    results = search_day6(sample, strategy="B", top_k=3)
    for item in results:
        print({
            "id": item["id"],
            "page": item["page"],
            "source": item["source"],
            "score": round(item["score"], 4),
            "text_preview": item["text"][:180].replace("\n", " "),
        })
