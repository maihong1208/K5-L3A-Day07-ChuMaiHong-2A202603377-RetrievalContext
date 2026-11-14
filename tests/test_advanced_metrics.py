import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "starter"))

from advanced_metrics import aggregate


class AdvancedMetricsTests(unittest.TestCase):
    def test_system_error_stays_in_denominator_and_exact_90_percent_passes(self):
        questions = [
            {
                "id": f"Q{index:02}",
                "split": "test",
                "category": "text",
                "answerable": True,
            }
            for index in range(10)
        ]
        cases = [
            {"id": item["id"], "category": "text", "answerable": True}
            for item in questions
        ]
        traces = []
        reviews = []
        for index, item in enumerate(questions):
            is_error = index == 9
            traces.append(
                {
                    "query_id": item["id"],
                    "status": "error" if is_error else "answered",
                    "citations": [] if is_error else ["chunk-1"],
                    "resolved_citations": []
                    if is_error
                    else [{"resolved": True}],
                    "elapsed_ms": 100,
                }
            )
            reviews.append(
                {
                    "query_id": item["id"],
                    "pass": not is_error,
                    "reason": "System error." if is_error else "Answer and citation verified.",
                    "citation_support": [] if is_error else [True],
                }
            )

        result = aggregate(questions, cases, traces, reviews, split="test")

        self.assertEqual((result["passed"], result["total"]), (9, 10))
        self.assertTrue(result["threshold"]["met"])
        self.assertEqual(result["system_errors"], 1)

    def test_no_evidence_case_requires_documented_absence_check(self):
        question = {
            "id": "N01",
            "split": "test",
            "category": "no_evidence",
            "answerable": False,
        }
        case = {"id": "N01", "category": "no_evidence", "answerable": False}
        trace = {
            "query_id": "N01",
            "status": "abstain",
            "citations": [],
            "resolved_citations": [],
        }
        review = {
            "query_id": "N01",
            "pass": True,
            "reason": "The agent abstained without making an unsupported claim.",
            "citation_support": [],
        }

        with self.assertRaisesRegex(ValueError, "absence-check"):
            aggregate([question], [case], [trace], [review], split="test")


if __name__ == "__main__":
    unittest.main()
