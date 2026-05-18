import tempfile
import unittest

from pdf_rag import (
    CombinedGuidelineRAGRetriever,
    GuidelineRAGRetriever,
    TextChunk,
    build_risk_query,
)


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search_for_risk(self, risk):
        self.calls.append(risk)
        return self.results


class GuidelineRAGRetrieverTest(unittest.TestCase):
    def test_build_risk_query_adds_high_risk_acs_terms(self):
        query = build_risk_query({"label_zh": "高风险", "label_en": "High Risk"})

        self.assertIn("high risk", query)
        self.assertIn("acute coronary syndrome", query)
        self.assertIn("invasive", query)

    def test_build_risk_query_adds_low_risk_follow_up_terms(self):
        query = build_risk_query({"label_zh": "低风险", "label_en": "Low Risk"})

        self.assertIn("low risk", query)
        self.assertIn("discharge", query)
        self.assertIn("follow-up", query)

    def test_search_for_high_risk_prefers_invasive_acs_guidance(self):
        retriever = GuidelineRAGRetriever.from_chunks(
            [
                TextChunk(page=3, text="Low-risk chest pain patients may be considered for discharge and follow-up."),
                TextChunk(
                    page=8,
                    text=(
                        "Patients with high-risk acute coronary syndrome should receive "
                        "guideline-directed management and early invasive evaluation."
                    ),
                ),
            ],
            top_k=1,
        )

        results = retriever.search_for_risk({"label_zh": "高风险", "label_en": "High Risk"})

        self.assertEqual(results[0]["page"], 8)
        self.assertIn("early invasive evaluation", results[0]["text"])
        self.assertGreater(results[0]["score"], 0)

    def test_search_for_low_risk_prefers_discharge_and_follow_up_guidance(self):
        retriever = GuidelineRAGRetriever.from_chunks(
            [
                TextChunk(page=4, text="High-risk ACS patients require invasive evaluation and intensive monitoring."),
                TextChunk(
                    page=15,
                    text=(
                        "Low-risk patients may be eligible for discharge with outpatient "
                        "follow-up after appropriate evaluation."
                    ),
                ),
            ],
            top_k=1,
        )

        results = retriever.search_for_risk({"label_zh": "低风险", "label_en": "Low Risk"})

        self.assertEqual(results[0]["page"], 15)
        self.assertIn("outpatient follow-up", results[0]["text"])

    def test_cache_path_uses_pdf_location(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = f"{temp_dir}/guideline.pdf"
            retriever = GuidelineRAGRetriever(pdf_path)

            self.assertTrue(str(retriever.cache_path).endswith("guideline.rag_cache.json"))

    def test_combined_retriever_merges_and_sorts_multiple_guidelines(self):
        risk = {"label_zh": "高风险", "label_en": "High Risk"}
        first = FakeRetriever(
            [{"page": 3, "text": "ACC high-risk acute coronary syndrome guidance", "score": 2.0}]
        )
        second = FakeRetriever(
            [{"page": 9, "text": "ESC chronic coronary syndrome risk guidance", "score": 5.0}]
        )
        combined = CombinedGuidelineRAGRetriever([first, second], top_k=2)

        results = combined.search_for_risk(risk)

        self.assertEqual(first.calls, [risk])
        self.assertEqual(second.calls, [risk])
        self.assertEqual([result["source"] for result in results], ["Guideline 2", "Guideline 1"])
        self.assertEqual([result["page"] for result in results], [9, 3])

    def test_combined_retriever_uses_configured_source_names(self):
        first = FakeRetriever([{"page": 1, "text": "ACC", "score": 1.0}])
        second = FakeRetriever([{"page": 2, "text": "ESC", "score": 2.0}])
        combined = CombinedGuidelineRAGRetriever(
            [
                ("2025 ACC/AHA ACS Guideline", first),
                ("2024 ESC CCS Guideline", second),
            ],
            top_k=2,
        )

        results = combined.search_for_risk({"label_zh": "高风险", "label_en": "High Risk"})

        self.assertEqual(results[0]["source"], "2024 ESC CCS Guideline")
        self.assertEqual(results[1]["source"], "2025 ACC/AHA ACS Guideline")

    def test_combined_retriever_keeps_at_least_one_result_from_each_source(self):
        first = FakeRetriever(
            [
                {"page": 1, "text": "ACC best", "score": 10.0},
                {"page": 2, "text": "ACC second", "score": 9.0},
                {"page": 3, "text": "ACC third", "score": 8.0},
            ]
        )
        second = FakeRetriever([{"page": 4, "text": "ESC best", "score": 1.0}])
        combined = CombinedGuidelineRAGRetriever(
            [
                ("2025 ACC/AHA ACS Guideline", first),
                ("2024 ESC CCS Guideline", second),
            ],
            top_k=3,
        )

        results = combined.search_for_risk({"label_zh": "高风险", "label_en": "High Risk"})

        self.assertEqual(len(results), 3)
        self.assertIn("2024 ESC CCS Guideline", [result["source"] for result in results])
        self.assertEqual(results[0]["source"], "2025 ACC/AHA ACS Guideline")


if __name__ == "__main__":
    unittest.main()
