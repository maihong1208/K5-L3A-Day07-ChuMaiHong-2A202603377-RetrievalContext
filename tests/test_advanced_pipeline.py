import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))

from advanced_extract import chunk_text
from advanced_index import build_chunks
from advanced_retriever import prioritize_named_figures
from advanced_run import validate_benchmark


class AdvancedPipelineTests(unittest.TestCase):
    def test_benchmark_meets_required_counts_and_has_distinct_question_text(self):
        benchmark_path = Path(__file__).resolve().parents[1] / "starter" / "advanced_benchmark.json"
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))

        validate_benchmark(benchmark)

        dev = [item for item in benchmark["questions"] if item["split"] == "dev"]
        test = [item for item in benchmark["questions"] if item["split"] == "test"]
        self.assertEqual(len(dev), 18)
        self.assertEqual(len(test), 30)

    def test_explicit_figure_reference_promotes_its_visual_chunk(self):
        chunks = [
            {"id": "figure", "metadata": {"figure_label": "Figure TS.9 (a)"}, "text": "figure data"},
            {"id": "other", "metadata": {"kind": "text"}, "text": "other data"},
        ]
        results = [{"id": "other", "score": 0.1}]

        ranked = prioritize_named_figures(
            "Read Figure TS.9(a) and identify its axes.",
            results,
            chunks,
            top_k=2,
        )

        self.assertEqual([item["id"] for item in ranked], ["figure", "other"])
        self.assertTrue(ranked[0]["figure_anchor"])

    def test_chunking_covers_long_page_text_with_overlap(self):
        source = " ".join(f"token{i}" for i in range(120))

        chunks = chunk_text(source, max_chars=64, overlap=12)

        source_tokens = set(re.findall(r"\w+", source))
        chunk_tokens = {
            token
            for chunk in chunks
            for token in re.findall(r"\w+", chunk)
        }
        self.assertTrue(chunks)
        self.assertLessEqual(max(map(len, chunks)), 64)
        self.assertTrue(source_tokens.issubset(chunk_tokens))

    def test_chunking_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("some text", max_chars=10, overlap=10)

    def test_index_chunks_keep_table_and_visual_page_provenance(self):
        digest = "a" * 64
        manifest = {
            "document": {
                "sha256": digest,
                "source_url": "https://example.test/report.pdf",
                "pdf_pages": 112,
            },
            "pages": [
                {
                    "pdf_page": 33,
                    "printed_page": 65,
                    "text": "Table TS.1 observed warming and attribution.",
                    "tables": [
                        {
                            "label": "TS.1",
                            "table_index": 1,
                            "bbox": [20, 40, 500, 700],
                            "rows": [
                                ["Indicator", "Observed change"],
                                ["Warming", "0.9 to 1.2 degrees C"],
                            ],
                        }
                    ],
                }
            ],
        }
        visual_descriptions = {
            "document_sha256": digest,
            "pages": [
                {
                    "pdf_page": 33,
                    "status": "ok",
                    "image_path": "visual_pages/page-033.jpg",
                    "page_bbox": [0, 0, 595, 842],
                    "visuals": [
                        {
                            "label": "Figure TS.1",
                            "type": "chart",
                            "description": "Lines show warming across scenarios.",
                        }
                    ],
                }
            ],
        }

        chunks = build_chunks(manifest, visual_descriptions)

        table_chunk = next(chunk for chunk in chunks if chunk["metadata"]["kind"] == "table")
        visual_chunk = next(chunk for chunk in chunks if chunk["metadata"]["kind"] == "visual")
        self.assertEqual(table_chunk["page"], 33)
        self.assertEqual(table_chunk["metadata"]["printed_page"], 65)
        self.assertEqual(table_chunk["metadata"]["table_label"], "TS.1")
        self.assertIn("0.9 to 1.2 degrees C", table_chunk["text"])
        self.assertEqual(visual_chunk["metadata"]["visual_type"], "chart")
        self.assertIn("page-033.jpg", visual_chunk["metadata"]["image_path"])

    def test_index_rejects_visuals_with_failed_generation(self):
        manifest = {
            "document": {"sha256": "b" * 64, "source_url": "https://example.test/report.pdf"},
            "pages": [{"pdf_page": 1, "printed_page": 33, "text": "A page."}],
        }
        visual_descriptions = {
            "document_sha256": "b" * 64,
            "pages": [{"pdf_page": 1, "status": "error", "error": "provider unavailable"}],
        }

        with self.assertRaisesRegex(ValueError, "provider unavailable"):
            build_chunks(manifest, visual_descriptions)


if __name__ == "__main__":
    unittest.main()
