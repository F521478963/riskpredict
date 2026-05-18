import unittest

from ai_analysis import DEEPSEEK_BASE_URL, DeepSeekAnalyzer, build_feature_summary


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, captured):
        self.captured = captured

    def create(self, **kwargs):
        self.captured["completion_kwargs"] = kwargs
        return FakeCompletion("AI 风险分析报告 / AI Risk Analysis Report")


class FakeChat:
    def __init__(self, captured):
        self.completions = FakeCompletions(captured)


class FakeClient:
    def __init__(self, captured):
        self.chat = FakeChat(captured)


class FakeGuidelineRetriever:
    def __init__(self, snippets):
        self.snippets = snippets
        self.calls = []

    def search_for_risk(self, risk):
        self.calls.append(risk)
        return self.snippets


class DeepSeekAnalyzerTest(unittest.TestCase):
    def test_build_feature_summary_pairs_labels_with_values(self):
        fields = [
            {"label_zh": "面部光谱均值1", "label_en": "Face spectrum mean 1"},
            {"label_zh": "右耳纹理均值71", "label_en": "Right ear texture mean 71"},
        ]

        summary = build_feature_summary(fields, [0.12, 0.34])

        self.assertIn("面部光谱均值1 / Face spectrum mean 1: 0.12", summary)
        self.assertIn("右耳纹理均值71 / Right ear texture mean 71: 0.34", summary)

    def test_analyzer_skips_api_when_key_is_missing(self):
        analyzer = DeepSeekAnalyzer(api_key=None)

        result = analyzer.analyze(
            fields=[],
            values=[],
            prediction=0.9,
            risk={"label_zh": "高风险", "label_en": "High Risk"},
        )

        self.assertIsNone(result["content"])
        self.assertIn("未配置", result["error"])

    def test_analyzer_calls_deepseek_with_openai_sdk_contract(self):
        captured = {}

        def fake_client_factory(api_key, base_url, timeout, verify_ssl):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["timeout"] = timeout
            captured["verify_ssl"] = verify_ssl
            return FakeClient(captured)

        analyzer = DeepSeekAnalyzer(api_key="test-key", client_factory=fake_client_factory)
        result = analyzer.analyze(
            fields=[{"label_zh": "面部光谱均值1", "label_en": "Face spectrum mean 1"}],
            values=[0.1],
            prediction=0.9,
            risk={"label_zh": "高风险", "label_en": "High Risk"},
        )

        self.assertEqual(result["content"], "AI 风险分析报告 / AI Risk Analysis Report")
        self.assertIsNone(result["error"])
        self.assertEqual(captured["api_key"], "test-key")
        self.assertEqual(captured["base_url"], DEEPSEEK_BASE_URL)
        self.assertTrue(captured["verify_ssl"])
        self.assertEqual(captured["completion_kwargs"]["model"], "deepseek-v4-flash")
        self.assertFalse(captured["completion_kwargs"]["stream"])
        prompt = captured["completion_kwargs"]["messages"][1]["content"]
        self.assertIn("冠状动脉功能学状态", prompt)
        self.assertIn("只允许依据【本地指南检索片段】", prompt)
        self.assertIn("必须引用预测值", prompt)
        self.assertIn("对应风险等级的建议治疗措施", prompt)
        self.assertNotIn("【指标-结果对应分析】", prompt)

    def test_analyzer_includes_local_guideline_context_for_risk(self):
        captured = {}

        def fake_client_factory(api_key, base_url, timeout, verify_ssl):
            return FakeClient(captured)

        retriever = FakeGuidelineRetriever(
            [
                {
                    "page": 12,
                    "text": "High-risk ACS patients should be evaluated promptly with guideline-directed management.",
                    "score": 3.5,
                }
            ]
        )
        risk = {"label_zh": "高风险", "label_en": "High Risk"}
        analyzer = DeepSeekAnalyzer(
            api_key="test-key",
            client_factory=fake_client_factory,
            guideline_retriever=retriever,
        )

        result = analyzer.analyze(fields=[], values=[], prediction=0.92, risk=risk)

        self.assertIsNone(result["error"])
        self.assertEqual(retriever.calls, [risk])
        prompt = captured["completion_kwargs"]["messages"][1]["content"]
        self.assertIn("【本地指南检索片段】", prompt)
        self.assertIn("Page 12", prompt)
        self.assertIn("High-risk ACS patients should be evaluated promptly", prompt)
        self.assertIn("只允许依据上述本地指南片段", prompt)

    def test_analyzer_prompt_restricts_deepseek_to_local_rag_treatment_advice(self):
        captured = {}

        def fake_client_factory(api_key, base_url, timeout, verify_ssl):
            return FakeClient(captured)

        retriever = FakeGuidelineRetriever(
            [
                {
                    "source": "2024 ESC CCS Guideline",
                    "page": 88,
                    "text": "Low-risk patients may receive guideline-directed medical therapy and outpatient follow-up.",
                    "score": 6.0,
                }
            ]
        )
        analyzer = DeepSeekAnalyzer(
            api_key="test-key",
            client_factory=fake_client_factory,
            guideline_retriever=retriever,
        )

        analyzer.analyze(
            fields=[],
            values=[],
            prediction=0.42,
            risk={"label_zh": "低风险", "label_en": "Low Risk"},
        )

        messages = captured["completion_kwargs"]["messages"]
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        self.assertIn("只能依据用户提供的本地 RAG 指南片段回答", system_prompt)
        self.assertIn("不要联网查询", system_prompt)
        self.assertIn("不要使用未出现在本地片段中的医学常识补充", system_prompt)
        self.assertIn("【本地指南检索片段】", user_prompt)
        self.assertIn("2024 ESC CCS Guideline", user_prompt)
        self.assertIn("只允许依据【本地指南检索片段】", user_prompt)
        self.assertIn("对应风险等级的建议治疗措施", user_prompt)
        self.assertIn("### 建议治疗措施", user_prompt)
        self.assertIn("### 本地依据", user_prompt)
        self.assertNotIn("面部光谱", user_prompt)

    def test_analyzer_can_disable_ssl_verification_for_local_development(self):
        captured = {}

        def fake_client_factory(api_key, base_url, timeout, verify_ssl):
            captured["verify_ssl"] = verify_ssl
            return FakeClient(captured)

        analyzer = DeepSeekAnalyzer(
            api_key="test-key",
            verify_ssl=False,
            client_factory=fake_client_factory,
        )
        result = analyzer.analyze(
            fields=[],
            values=[],
            prediction=0.9,
            risk={"label_zh": "高风险", "label_en": "High Risk"},
        )

        self.assertEqual(result["content"], "AI 风险分析报告 / AI Risk Analysis Report")
        self.assertFalse(captured["verify_ssl"])


if __name__ == "__main__":
    unittest.main()
