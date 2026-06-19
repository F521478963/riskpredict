import unittest

from llm_pipeline import (
    JUDGMENT_LABELS,
    PROMPT_FILES,
    ClinicalAssistantPipeline,
    load_prompt_config,
    normalize_judgment_mode,
)


class JudgmentModeTest(unittest.TestCase):
    def test_normalize_judgment_mode_defaults_to_rag_only(self):
        self.assertEqual(normalize_judgment_mode(None), "rag_only")
        self.assertEqual(normalize_judgment_mode("invalid"), "rag_only")

    def test_all_four_modes_are_registered(self):
        self.assertEqual(set(PROMPT_FILES.keys()), set(JUDGMENT_LABELS.keys()))
        self.assertEqual(len(PROMPT_FILES), 4)

    def test_load_zero_prompt_mode(self):
        config = load_prompt_config("zero_prompt")
        self.assertEqual(config["judgment_label"], "Zero-Shot Prompt")
        self.assertFalse(config.get("use_system_prompt", True))
        self.assertIn("Predicted value", config["user_template"])

    def test_load_simple_prompt_mode(self):
        config = load_prompt_config("simple_prompt")
        self.assertEqual(config["judgment_label"], "Simple Prompt")
        self.assertIn("3–5 short paragraphs", config["user_template"])

    def test_load_rag_only_prompt_disallows_web_style_supplement(self):
        config = load_prompt_config("rag_only")
        self.assertEqual(config["judgment_label"], "RAG-Only")
        self.assertIn("Do not search the web", config["system"])
        self.assertIn("### Clinical Assessment", config["user_template"])
        self.assertIn("### RAG Key Points", config["user_template"])
        self.assertIn("Ridge-RF", config["user_template"])
        self.assertIn("This assessment only suggests", config["user_template"])
        self.assertNotIn("Assessment & Management Recommendations", config["user_template"])
        self.assertIn("Do not discuss triage", config["user_template"])
        self.assertIn("do not use [Snippet #]", config["user_template"])
        self.assertIn("uncertainty or limitation lists", config["user_template"])
        self.assertIn("triage settings", config["system"])

    def test_load_combined_prompt_allows_model_supplement(self):
        config = load_prompt_config("combined")
        self.assertEqual(config["judgment_label"], "Combined Analysis")
        self.assertIn("[Model]", config["user_template"])
        self.assertNotIn("Do not search the web", config["system"])
        self.assertIn("### Clinical Assessment", config["user_template"])
        self.assertIn("### RAG Key Points & Model Supplement", config["user_template"])
        self.assertIn("Ridge-RF", config["user_template"])
        self.assertIn("This assessment only suggests", config["user_template"])
        self.assertNotIn("triage and disposition", config["user_template"])
        self.assertNotIn("### RAG Summary", config["user_template"])
        self.assertIn("Do not discuss triage", config["user_template"])
        self.assertIn("do not use [Snippet #]", config["user_template"])
        self.assertIn("uncertainty or limitation lists", config["user_template"])
        self.assertIn("triage settings", config["system"])
        self.assertGreaterEqual(int(config["generation"]["max_tokens"]), 8192)

    def test_format_guideline_context_for_benchmark_modes(self):
        snippets = [
            {
                "source": "Test Guideline",
                "page": 1,
                "category": "guidelines",
                "text": "Evaluate with ECG.",
            }
        ]
        zero_ctx = ClinicalAssistantPipeline._format_guideline_context(
            snippets, "zero_prompt"
        )
        simple_ctx = ClinicalAssistantPipeline._format_guideline_context(
            snippets, "simple_prompt"
        )
        self.assertIn("Reference materials:", zero_ctx)
        self.assertIn("1. Test Guideline", zero_ctx)
        self.assertNotIn("[Snippet1]", zero_ctx)
        self.assertIn("Reference materials:", simple_ctx)

    def test_judgment_labels_cover_all_modes(self):
        self.assertEqual(JUDGMENT_LABELS["zero_prompt"], "Zero-Shot Prompt")
        self.assertEqual(JUDGMENT_LABELS["simple_prompt"], "Simple Prompt")
        self.assertEqual(JUDGMENT_LABELS["rag_only"], "RAG-Only")
        self.assertEqual(JUDGMENT_LABELS["combined"], "Combined Analysis")


if __name__ == "__main__":
    unittest.main()
