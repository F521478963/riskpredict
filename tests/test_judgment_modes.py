import unittest

from llm_pipeline import (
    JUDGMENT_LABELS,
    load_prompt_config,
    normalize_judgment_mode,
)


class JudgmentModeTest(unittest.TestCase):
    def test_normalize_judgment_mode_defaults_to_rag_only(self):
        self.assertEqual(normalize_judgment_mode(None), "rag_only")
        self.assertEqual(normalize_judgment_mode("invalid"), "rag_only")

    def test_load_rag_only_prompt_disallows_web_style_supplement(self):
        config = load_prompt_config("rag_only")
        self.assertEqual(config["judgment_label"], "仅RAG判断")
        self.assertIn("不要联网查询", config["system"])

    def test_load_combined_prompt_allows_model_supplement(self):
        config = load_prompt_config("combined")
        self.assertEqual(config["judgment_label"], "综合判断")
        self.assertIn("[模型]", config["user_template"])
        self.assertNotIn("不要联网查询", config["system"])
        self.assertIn("### 综合判断", config["user_template"])
        self.assertIn("推荐分诊与处置路径", config["user_template"])
        self.assertGreaterEqual(int(config["generation"]["max_tokens"]), 8192)

    def test_judgment_labels_cover_both_modes(self):
        self.assertEqual(JUDGMENT_LABELS["rag_only"], "仅RAG判断")
        self.assertEqual(JUDGMENT_LABELS["combined"], "综合判断")


if __name__ == "__main__":
    unittest.main()
