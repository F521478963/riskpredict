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
        self.assertEqual(config["judgment_label"], "0提示词模式")
        self.assertFalse(config.get("use_system_prompt", True))
        self.assertIn("预测值", config["user_template"])

    def test_load_simple_prompt_mode(self):
        config = load_prompt_config("simple_prompt")
        self.assertEqual(config["judgment_label"], "简易提示词")
        self.assertIn("3–5 段话", config["user_template"])

    def test_load_rag_only_prompt_disallows_web_style_supplement(self):
        config = load_prompt_config("rag_only")
        self.assertEqual(config["judgment_label"], "仅RAG判断")
        self.assertIn("不要联网查询", config["system"])
        self.assertIn("### 综合判断", config["user_template"])
        self.assertIn("### RAG 要点", config["user_template"])
        self.assertIn("体表高光谱无创筛查模型（Ridge-RF）", config["user_template"])
        self.assertIn("本次评估仅提示潜在冠脉病变可能性", config["user_template"])
        self.assertNotIn("建议评估与管理措施", config["user_template"])
        self.assertIn("全文不要写分诊场景", config["user_template"])
        self.assertIn("不得出现 [片段#]", config["user_template"])
        self.assertNotIn("不确定性、局限性", config["user_template"])
        self.assertIn("分诊场景、监护级别", config["system"])

    def test_load_combined_prompt_allows_model_supplement(self):
        config = load_prompt_config("combined")
        self.assertEqual(config["judgment_label"], "综合判断")
        self.assertIn("[模型]", config["user_template"])
        self.assertNotIn("不要联网查询", config["system"])
        self.assertIn("### 综合判断", config["user_template"])
        self.assertIn("### RAG 要点与模型补充", config["user_template"])
        self.assertIn("体表高光谱无创筛查模型（Ridge-RF）", config["user_template"])
        self.assertIn("本次评估仅提示潜在冠脉病变可能性", config["user_template"])
        self.assertNotIn("推荐分诊与处置路径", config["user_template"])
        self.assertNotIn("### RAG 要点摘要", config["user_template"])
        self.assertIn("全文不要写分诊场景", config["user_template"])
        self.assertIn("不得出现 [片段#]", config["user_template"])
        self.assertNotIn("不确定性、局限性", config["user_template"])
        self.assertIn("分诊场景、监护级别", config["system"])
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
        self.assertIn("参考资料", zero_ctx)
        self.assertIn("1. Test Guideline", zero_ctx)
        self.assertNotIn("[片段1]", zero_ctx)
        self.assertIn("参考资料", simple_ctx)

    def test_judgment_labels_cover_all_modes(self):
        self.assertEqual(JUDGMENT_LABELS["zero_prompt"], "0提示词模式")
        self.assertEqual(JUDGMENT_LABELS["simple_prompt"], "简易提示词")
        self.assertEqual(JUDGMENT_LABELS["rag_only"], "仅RAG判断")
        self.assertEqual(JUDGMENT_LABELS["combined"], "综合判断")


if __name__ == "__main__":
    unittest.main()
