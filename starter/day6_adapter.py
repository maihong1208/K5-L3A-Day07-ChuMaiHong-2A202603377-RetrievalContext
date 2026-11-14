"""Template adapter to connect the Day 6 vector store to the Day 7 agent.

This file intentionally does NOT modify the Day 6 codebase. It only wraps the
existing Day 6 embedding model + vector store into the `Retriever.search` contract
expected by `starter/agentic.py`.

Usage example:
    from day6_adapter import Day6Adapter

    retriever = Day6Adapter(
        embedding_model=my_embedding_model,
        vector_store=my_vector_store,
        chunks=my_day6_chunks,
    )
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


REQUIRED_CHUNK_FIELDS = {
    "id",
    "text",
    "source_id",
    "source",
    "page",
    "metadata",
}


def validate_day6_output(chunks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return normalized chunk list, raising a clear error if Day 6 output is invalid.

    This helper checks only compatibility with the Day 7 contract. It does not
    alter the original Day 6 implementation or data structure.
    """
    normalized: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, Mapping):
            raise TypeError(f"Chunk #{idx} must be a mapping/dictionary.")

        missing = sorted(REQUIRED_CHUNK_FIELDS - set(chunk.keys()))
        if missing:
            raise ValueError(
                f"Chunk #{idx} missing required fields for Day 7: {missing}. "
                "Expected at least: id, text, source_id, source, page, metadata."
            )

        item = dict(chunk)
        item["id"] = str(item["id"])
        item["source_id"] = str(item.get("source_id") or item.get("document_id", "unknown"))
        item["source"] = str(item.get("source") or item.get("source_name") or item["source_id"])
        item["page"] = int(item.get("page") or 0)
        item["metadata"] = item.get("metadata") or {}
        if not isinstance(item["metadata"], dict):
            raise TypeError(f"Chunk #{idx} field 'metadata' must be a dict.")
        item["text"] = str(item["text"])
        normalized.append(item)

    return normalized


class Day6Adapter:
    """Adapter that wraps an existing Day 6 embedder + vector store.

    The constructor accepts any object that supports:
      - encode(query: str) -> list[float] or np.ndarray
      - query(vector: list[float], top_k: int, filter: dict[str, Any] | None = None)

    The adapter keeps Day 6 unchanged and exposes only the contract required by
    `starter/agentic.py`.
    """

    def __init__(
        self,
        *,
        embedding_model: Any,
        vector_store: Any,
        chunks: Iterable[Mapping[str, Any]] | None = None,
        id_field: str = "id",
        source_id_field: str = "source_id",
        source_field: str = "source",
        page_field: str = "page",
        text_field: str = "text",
        metadata_field: str = "metadata",
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.id_field = id_field
        self.source_id_field = source_id_field
        self.source_field = source_field
        self.page_field = page_field
        self.text_field = text_field
        self.metadata_field = metadata_field
        self.chunks = validate_day6_output(chunks) if chunks is not None else []

    def _embed(self, query: str) -> list[float]:
        if self.embedding_model is None:
            raise ValueError("embedding_model is required for Day 6 retrieval.")

        # Common patterns across embedding libraries:
        # - model.encode(query)
        # - model.embed_query(query)
        # - model.generate_embeddings([query])[0]
        for method_name in ("encode", "embed_query", "encode_query"):
            method = getattr(self.embedding_model, method_name, None)
            if callable(method):
                value = method(query)
                if isinstance(value, list):
                    return [float(x) for x in value]
                if hasattr(value, "tolist"):
                    return [float(x) for x in value.tolist()]
                return [float(x) for x in value]

        raise AttributeError(
            "embedding_model must expose one of: encode(), embed_query(), encode_query()."
        )

    def _normalize_hit(self, hit: Mapping[str, Any]) -> dict[str, Any]:
        record = dict(hit)

        record_id = record.get(self.id_field) or record.get("chunk_id") or record.get("chunkId")
        if record_id is None:
            raise ValueError("Vector-store hit is missing a chunk ID.")

        source_id = record.get(self.source_id_field) or record.get("document_id") or record.get("doc_id")
        source = record.get(self.source_field) or record.get("source_name") or source_id
        page = record.get(self.page_field) or record.get("page_number") or 0
        text = record.get(self.text_field) or record.get("content") or record.get("chunk_text")
        metadata = record.get(self.metadata_field) or record.get("meta") or {}

        score = record.get("score")
        if score is None:
            score = record.get("similarity")
        if score is None:
            score = record.get("distance")
        if score is None:
            score = 0.0

        return {
            "id": str(record_id),
            "text": str(text),
            "source_id": str(source_id) if source_id is not None else "unknown",
            "source": str(source) if source is not None else "unknown",
            "page": int(page) if page is not None else 0,
            "metadata": metadata if isinstance(metadata, dict) else {"raw": metadata},
            "score": float(score),
        }

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int) -> list[dict[str, Any]]:
        """Query the real Day 6 vector store and return chunks in the Day 7 format."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if top_k < 1:
            raise ValueError("top_k must be >= 1.")

        query_vector = self._embed(query)

        # This is intentionally generic: every vector store has a slightly different API.
        # Common patterns:
        #   vector_store.search(query_vector, top_k=top_k, filter=filters)
        #   vector_store.query(query_vector, top_k=top_k, filter=filters)
        #   vector_store.similarity_search_by_vector(query_vector, top_k=top_k, filter=filters)
        #   vector_store.search_by_vector(query_vector, top_k=top_k, filter=filters)
        for method_name in (
            "search",
            "query",
            "similarity_search_by_vector",
            "search_by_vector",
        ):
            method = getattr(self.vector_store, method_name, None)
            if callable(method):
                raw_hits = method(query_vector, top_k=top_k, filter=filters)
                break
        else:
            # Fallback if vector_store exposes a lower-level function expecting a direct vector.
            if hasattr(self.vector_store, "query_vector"):
                raw_hits = self.vector_store.query_vector(query_vector, top_k=top_k, filter=filters)
            else:
                raise AttributeError(
                    "vector_store must expose a query/search method such as search(), query(), "
                    "similarity_search_by_vector(), or search_by_vector()."
                )

        if raw_hits is None:
            return []

        # In many vector DBs, the hits are already dict-like objects; in others they are tuples.
        if isinstance(raw_hits, list):
            hits = raw_hits[:top_k]
        else:
            hits = list(raw_hits)[:top_k]

        results: list[dict[str, Any]] = []
        for hit in hits:
            if isinstance(hit, Mapping):
                normalized = self._normalize_hit(hit)
            else:
                # Some stores return (id, payload, score) or a custom object.
                if isinstance(hit, tuple) and len(hit) >= 2:
                    record = hit[1]
                    if isinstance(record, Mapping):
                        normalized = self._normalize_hit(record)
                        normalized["score"] = float(hit[2]) if len(hit) > 2 else normalized["score"]
                    else:
                        raise TypeError("Unsupported tuple result from vector store.")
                else:
                    raise TypeError("Unsupported result type from vector store.")
            results.append(normalized)

        # Optional: preserve local chunk metadata for debugging and stronger schema guarantees.
        if self.chunks:
            by_id = {chunk["id"]: chunk for chunk in self.chunks}
            for item in results:
                if item["id"] in by_id:
                    chunk = by_id[item["id"]]
                    item["source_id"] = str(chunk.get("source_id") or item["source_id"])
                    item["source"] = str(chunk.get("source") or item["source"])
                    item["page"] = int(chunk.get("page") or item["page"])
                    item["metadata"] = {**chunk.get("metadata", {}), **item.get("metadata", {})}
                    item["text"] = str(chunk.get("text") or item["text"])

        return results


__all__ = ["Day6Adapter", "validate_day6_output"]
