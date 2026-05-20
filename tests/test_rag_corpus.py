import json
import tempfile
import unittest
from pathlib import Path

from rag_store import RagCorpusStore


class RagCorpusStoreTest(unittest.TestCase):
    def test_discover_documents_finds_pdf_in_category_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir)
            guidelines = corpus_dir / "guidelines"
            guidelines.mkdir(parents=True)
            pdf_path = guidelines / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")

            store = RagCorpusStore(corpus_dir=corpus_dir, retrieval_mode="lexical", top_k=2)
            documents = store.discover_documents()

            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].category, "guidelines")

    def test_build_and_search_lexical_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir)
            inbox = corpus_dir / "inbox"
            inbox.mkdir(parents=True)
            text_path = inbox / "note.md"
            text_path.write_text(
                "High-risk acute coronary syndrome requires early invasive evaluation.",
                encoding="utf-8",
            )

            store = RagCorpusStore(corpus_dir=corpus_dir, retrieval_mode="lexical", top_k=1)
            store.rebuild_index()

            results = store.search(
                "high risk acute coronary syndrome invasive evaluation",
                top_k=1,
            )

            self.assertEqual(len(results), 1)
            self.assertIn("early invasive evaluation", results[0]["text"])

    def test_search_for_prediction_prefers_guidelines_category(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_dir = Path(temp_dir)
            (corpus_dir / "guidelines").mkdir(parents=True)
            (corpus_dir / "methods").mkdir(parents=True)
            (corpus_dir / "guidelines" / "esc.md").write_text(
                "High-risk acute coronary syndrome requires guideline-directed invasive evaluation.",
                encoding="utf-8",
            )
            (corpus_dir / "methods" / "rag.md").write_text(
                "Retrieval augmented generation benchmark methodology only.",
                encoding="utf-8",
            )

            store = RagCorpusStore(corpus_dir=corpus_dir, retrieval_mode="lexical", top_k=2)
            store.rebuild_index()
            results = store.search_for_prediction(
                0.4,
                {"label_zh": "高风险", "label_en": "High Risk"},
            )

            self.assertTrue(any(item["category"] == "guidelines" for item in results))


if __name__ == "__main__":
    unittest.main()
