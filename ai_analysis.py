from clinical_query import RISK_THRESHOLD
from llm_config import (
    DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    get_deepseek_timeout_seconds,
)
from llm_pipeline import (
    ClinicalAssistantPipeline,
    JUDGMENT_LABELS,
    load_prompt_config,
    normalize_judgment_mode,
)
from rag_store import get_default_corpus_store


def build_feature_summary(fields, values):
    lines = []
    for field, value in zip(fields, values):
        lines.append(f"- {field['label_en']}: {value}")
    return "\n".join(lines)


class DeepSeekAnalyzer:
    def __init__(
        self,
        api_key,
        model=DEFAULT_DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        timeout=None,
        verify_ssl=True,
        client_factory=None,
        guideline_retriever=None,
        corpus_store=None,
        pipeline=None,
        rag_mode=None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout if timeout is not None else get_deepseek_timeout_seconds()
        self.verify_ssl = verify_ssl
        self.client_factory = client_factory
        self.guideline_retriever = guideline_retriever
        self.corpus_store = corpus_store or get_default_corpus_store()
        self.rag_mode = rag_mode
        self.pipeline = pipeline or ClinicalAssistantPipeline(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout=timeout,
            verify_ssl=verify_ssl,
            corpus_store=self.corpus_store,
            client_factory=client_factory,
            rag_mode=rag_mode,
        )

    def analyze(
        self,
        fields,
        values,
        prediction,
        risk,
        patient_context=None,
        judgment_mode="rag_only",
    ):
        del fields, values
        result = self.pipeline.run(
            prediction=prediction,
            risk=risk,
            patient_context=patient_context,
            judgment_mode=judgment_mode,
        )
        if result.get("error"):
            return {
                "content": result.get("content"),
                "error": result["error"],
                "snippets": result.get("snippets", []),
                "mode": result.get("mode"),
                "judgment_mode": result.get("judgment_mode"),
                "judgment_label": result.get("judgment_label"),
            }

        return {
            "content": result.get("content"),
            "error": None,
            "snippets": result.get("snippets", []),
            "mode": result.get("mode"),
            "judgment_mode": result.get("judgment_mode"),
            "judgment_label": result.get("judgment_label"),
        }

    @staticmethod
    def _default_client_factory(api_key, base_url, timeout, verify_ssl):
        from llm_pipeline import _default_openai_client_factory

        return _default_openai_client_factory(api_key, base_url, timeout, verify_ssl)


__all__ = [
    "DEEPSEEK_BASE_URL",
    "DEFAULT_DEEPSEEK_MODEL",
    "RISK_THRESHOLD",
    "JUDGMENT_LABELS",
    "DeepSeekAnalyzer",
    "build_feature_summary",
    "load_prompt_config",
    "normalize_judgment_mode",
]
