import os
from pathlib import Path

import httpx

from clinical_query import RISK_THRESHOLD
from llm_config import (
    DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    build_openai_timeout,
    format_llm_error,
    get_deepseek_timeout_seconds,
)
from rag_store import RagCorpusStore, get_default_corpus_store


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_JUDGMENT_MODE = "rag_only"
PROMPT_FILES = {
    "zero_prompt": PROMPTS_DIR / "clinical_assistant_zero.yaml",
    "simple_prompt": PROMPTS_DIR / "clinical_assistant_simple.yaml",
    "rag_only": PROMPTS_DIR / "clinical_assistant_v1.yaml",
    "combined": PROMPTS_DIR / "clinical_assistant_v2_combined.yaml",
}
JUDGMENT_LABELS = {
    "zero_prompt": "Zero-Shot Prompt",
    "simple_prompt": "Simple Prompt",
    "rag_only": "RAG-Only",
    "combined": "Combined Analysis",
}
RAG_ONLY_FALLBACK_MODES = frozenset({"rag_only"})


def load_prompt_config(judgment_mode=DEFAULT_JUDGMENT_MODE, path=None):
    if path:
        prompt_path = Path(path)
    else:
        mode = normalize_judgment_mode(judgment_mode)
        prompt_path = PROMPT_FILES[mode]

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required. Install dependencies from requirements.txt."
        ) from exc

    data = yaml.safe_load(prompt_path.read_text(encoding="utf-8")) or {}
    data["path"] = str(prompt_path)
    data.setdefault("judgment_mode", normalize_judgment_mode(judgment_mode))
    data.setdefault(
        "judgment_label",
        JUDGMENT_LABELS.get(data["judgment_mode"], data["judgment_mode"]),
    )
    return data


def normalize_judgment_mode(mode):
    value = (mode or DEFAULT_JUDGMENT_MODE).strip().lower()
    if value in PROMPT_FILES:
        return value
    return DEFAULT_JUDGMENT_MODE


class ClinicalAssistantPipeline:
    def __init__(
        self,
        api_key,
        model=DEFAULT_DEEPSEEK_MODEL,
        base_url=DEEPSEEK_BASE_URL,
        timeout=None,
        verify_ssl=True,
        corpus_store=None,
        prompt_config=None,
        judgment_mode=DEFAULT_JUDGMENT_MODE,
        client_factory=None,
        rag_mode=None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout if timeout is not None else get_deepseek_timeout_seconds()
        self.verify_ssl = verify_ssl
        self.corpus_store = corpus_store or get_default_corpus_store()
        self.judgment_mode = normalize_judgment_mode(judgment_mode)
        self.prompt_config = prompt_config or load_prompt_config(self.judgment_mode)
        self.client_factory = client_factory
        self.rag_mode = (rag_mode or os.environ.get("LLM_RAG_MODE", "rag")).lower()

    def run(self, prediction, risk, patient_context=None, judgment_mode=None):
        mode = normalize_judgment_mode(judgment_mode or self.judgment_mode)
        prompt_config = load_prompt_config(mode)
        label = prompt_config.get("judgment_label", JUDGMENT_LABELS[mode])

        if not self.api_key:
            return {
                "content": None,
                "error": "DEEPSEEK_API_KEY is not configured; AI risk analysis was skipped.",
                "snippets": [],
                "mode": self.rag_mode,
                "judgment_mode": mode,
                "judgment_label": label,
            }

        snippets = []
        if self.rag_mode != "no_rag":
            try:
                snippets = self.corpus_store.search_for_prediction(
                    prediction=prediction,
                    risk=risk,
                    patient_context=patient_context,
                )
            except Exception as exc:
                return {
                    "content": None,
                    "error": f"RAG retrieval failed: {exc}",
                    "snippets": [],
                    "mode": self.rag_mode,
                    "judgment_mode": mode,
                    "judgment_label": label,
                }

        if self.rag_mode != "no_rag" and not snippets and mode in RAG_ONLY_FALLBACK_MODES:
            return {
                "content": _format_fallback(prompt_config, prediction, risk),
                "error": None,
                "snippets": [],
                "mode": self.rag_mode,
                "judgment_mode": mode,
                "judgment_label": label,
            }

        try:
            content = self._generate(prompt_config, prediction, risk, snippets, mode)
            return {
                "content": content,
                "error": None,
                "snippets": snippets,
                "mode": self.rag_mode,
                "judgment_mode": mode,
                "judgment_label": label,
            }
        except Exception as exc:
            return {
                "content": None,
                "error": format_llm_error(exc),
                "snippets": snippets,
                "mode": self.rag_mode,
                "judgment_mode": mode,
                "judgment_label": label,
            }

    def _generate(self, prompt_config, prediction, risk, snippets, judgment_mode):
        client = self._create_client()
        messages = []
        system_text = (prompt_config.get("system") or "").strip()
        if prompt_config.get("use_system_prompt", True) and system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append(
            {
                "role": "user",
                "content": self._build_user_prompt(
                    prompt_config, prediction, risk, snippets, judgment_mode
                ),
            }
        )
        generation = prompt_config.get("generation", {})
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=float(generation.get("temperature", 0.2)),
            max_tokens=int(generation.get("max_tokens", 2048)),
            stream=False,
        )
        return response.choices[0].message.content.strip()

    def _build_user_prompt(self, prompt_config, prediction, risk, snippets, judgment_mode):
        template = prompt_config["user_template"]
        return template.format(
            prediction=prediction,
            risk_label_en=risk.get("label_en", ""),
            threshold=RISK_THRESHOLD,
            guideline_context=self._format_guideline_context(snippets, judgment_mode),
        ).strip()

    @staticmethod
    def _format_guideline_context(snippets, judgment_mode):
        if not snippets:
            if judgment_mode == "combined":
                return (
                    "[Local RAG Snippets]\n"
                    "No usable local guideline snippets were retrieved "
                    "(you may rely primarily on [Model] supplement)."
                )
            if judgment_mode in ("zero_prompt", "simple_prompt"):
                return "Reference materials: none"
            return (
                "[Local Guideline Snippets]\n"
                "No usable local guideline snippets were retrieved."
            )

        if judgment_mode in ("zero_prompt", "simple_prompt"):
            lines = ["Reference materials:"]
        elif judgment_mode == "combined":
            header = (
                "[Local RAG Snippets]\n"
                "Read the snippets below first, then combine with [Model] knowledge; "
                "cite using [Snippet #]."
            )
            lines = [header]
        else:
            header = (
                "[Local Guideline Snippets]\n"
                "Base recommendations only on the snippets below; cite with [Snippet #]."
            )
            lines = [header]

        for index, snippet in enumerate(snippets, start=1):
            source = snippet.get("source", "Local corpus")
            page = snippet.get("page", "n/a")
            category = snippet.get("category", "")
            text = " ".join(str(snippet.get("text", "")).split())
            if not text:
                continue
            prefix = f"{source}"
            if category:
                prefix = f"{prefix} ({category})"
            if judgment_mode in ("zero_prompt", "simple_prompt"):
                lines.append(f"{index}. {prefix} p.{page}: {text}")
            else:
                lines.append(f"[Snippet{index}] {prefix} Page {page}: {text}")
        return "\n".join(lines)

    def _create_client(self):
        if self.client_factory:
            return self.client_factory(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                verify_ssl=self.verify_ssl,
            )

        return _default_openai_client_factory(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            verify_ssl=self.verify_ssl,
        )


def _format_fallback(prompt_config, prediction, risk):
    template = prompt_config.get("fallback_no_snippets", "")
    return template.format(
        prediction=prediction,
        risk_label_en=risk.get("label_en", ""),
    ).strip()


def _default_openai_client_factory(api_key, base_url, timeout, verify_ssl):
    from openai import OpenAI

    timeout_config = (
        timeout if isinstance(timeout, httpx.Timeout) else build_openai_timeout(timeout)
    )

    if verify_ssl:
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_config)

    http_client = httpx.Client(verify=False, timeout=timeout_config)
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_config,
        http_client=http_client,
    )
