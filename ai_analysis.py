DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def build_feature_summary(fields, values):
    lines = []
    for field, value in zip(fields, values):
        lines.append(f"- {field['label_zh']} / {field['label_en']}: {value}")
    return "\n".join(lines)


class DeepSeekAnalyzer:
    def __init__(
        self,
        api_key,
        model=DEFAULT_DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        timeout=30,
        verify_ssl=True,
        client_factory=None,
        guideline_retriever=None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.client_factory = client_factory or self._default_client_factory
        self.guideline_retriever = guideline_retriever

    def analyze(self, fields, values, prediction, risk):
        if not self.api_key:
            return {
                "content": None,
                "error": "未配置 DEEPSEEK_API_KEY，已跳过 AI 风险分析。",
            }

        try:
            client = self.client_factory(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                verify_ssl=self.verify_ssl,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个谨慎的医学 AI 辅助分析助手。"
                            "你只能依据用户提供的本地 RAG 指南片段回答。"
                            "不要联网查询，不要使用未出现在本地片段中的医学常识补充。"
                            "如果本地片段不足以支持具体治疗措施，必须明确说明依据不足。"
                            "不要声称可以替代医生诊断，不要编造检查结果或治疗方案。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._build_prompt(
                            fields,
                            values,
                            prediction,
                            risk,
                            guideline_snippets=self._retrieve_guideline_snippets(risk),
                        ),
                    },
                ],
                temperature=0.2,
                stream=False,
            )
            content = response.choices[0].message.content.strip()
            return {"content": content, "error": None}
        except Exception as exc:
            return {"content": None, "error": f"AI 风险分析暂不可用：{exc}"}

    def _retrieve_guideline_snippets(self, risk):
        if not self.guideline_retriever:
            return []

        try:
            return self.guideline_retriever.search_for_risk(risk)
        except Exception:
            return []

    def _build_prompt(self, fields, values, prediction, risk, guideline_snippets=None):
        guideline_context = self._format_guideline_context(guideline_snippets or [])
        return f"""
你必须只允许依据【本地指南检索片段】回答当前风险等级对应的建议治疗措施。

场景：冠状动脉功能学状态风险辅助分析。该报告用于根据一个已经训练好的 SVR 模型输出，给出非诊断性的、基于本地指南片段的治疗措施建议。

【模型输出】
- 预测值 / Predicted Value: {prediction:.6f}
- 风险分级 / Risk Level: {risk['label_zh']} / {risk['label_en']}
- 判定规则：预测值 >= 0.8 为 Low Risk；预测值 < 0.8 为 High Risk。

{guideline_context}

【强制要求】
1. 必须引用预测值 {prediction:.6f}、阈值 0.8、风险等级 {risk['label_zh']} / {risk['label_en']}，且风险解释必须与上述判定规则一致。
2. 只允许依据上述本地指南片段和【本地指南检索片段】回答，不要联网查询。
3. 不要使用未出现在本地片段中的医学常识补充，不要输出本地片段未支持的治疗措施。
4. 只回答“对应风险等级的建议治疗措施”，不要分析面部、耳部或图像指标如何支持结果。
5. 如果本地指南片段不足以支持具体治疗措施，请明确写出“本地指南片段不足以支持具体治疗措施”，不要自行补全。
6. 不要编造患者症状、病史、检查结果、冠脉狭窄位置、FFR 数值、药物剂量或临床诊断。

【必须输出以下固定章节】
### 风险等级
用 1-2 句话说明预测值、阈值 0.8、当前风险等级，以及该输出不是临床诊断。

### 建议治疗措施
仅根据本地指南片段，按要点列出与 {risk['label_zh']} / {risk['label_en']} 对应的治疗、评估或管理措施。每一点都应能从本地片段找到依据。

### 本地依据
列出你使用的本地指南来源、页码和对应原文要点。若没有足够依据，说明依据不足。
""".strip()

    @staticmethod
    def _format_guideline_context(snippets):
        if not snippets:
            return "【本地指南检索片段】\n未检索到可用的本地指南片段。"

        lines = ["【本地指南检索片段】", "只允许依据上述本地指南片段提出对应风险等级的建议治疗措施；如片段不足，请明确说明依据不足。"]
        for index, snippet in enumerate(snippets, start=1):
            source = snippet.get("source", "本地指南")
            page = snippet.get("page", "未知")
            text = " ".join(str(snippet.get("text", "")).split())
            if not text:
                continue
            lines.append(f"{index}. {source} Page {page}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _default_client_factory(api_key, base_url, timeout, verify_ssl):
        from openai import OpenAI

        if verify_ssl:
            return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        import httpx

        http_client = httpx.Client(verify=False, timeout=timeout)
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            http_client=http_client,
        )
