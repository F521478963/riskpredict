import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+|\d+(?:\.\d+)?")


@dataclass(frozen=True)
class TextChunk:
    page: int
    text: str


def build_risk_query(risk):
    from clinical_query import build_risk_query as _build_risk_query

    return _build_risk_query(risk)


class GuidelineRAGRetriever:
    def __init__(self, pdf_path, top_k=4, chunk_size=1400, chunk_overlap=220):
        self.pdf_path = Path(pdf_path)
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cache_path = self.pdf_path.with_suffix(".rag_cache.json")
        self._chunks = None

    @classmethod
    def from_chunks(cls, chunks, top_k=4):
        retriever = cls("__in_memory__.pdf", top_k=top_k)
        retriever._chunks = list(chunks)
        return retriever

    def search_for_risk(self, risk):
        query = build_risk_query(risk)
        return self.search(query)

    def search(self, query):
        chunks = self._load_chunks()
        if not chunks:
            return []

        query_tokens = Counter(_tokenize(query))
        scored = []
        for chunk in chunks:
            score = _score_chunk(query, query_tokens, chunk.text)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "page": chunk.page,
                "text": _compact_text(chunk.text),
                "score": round(score, 4),
            }
            for score, chunk in scored[: self.top_k]
        ]

    def _load_chunks(self):
        if self._chunks is not None:
            return self._chunks

        self._chunks = self._load_cache()
        if self._chunks is not None:
            return self._chunks

        self._chunks = self._extract_chunks_from_pdf()
        self._write_cache(self._chunks)
        return self._chunks

    def _load_cache(self):
        if not self.cache_path.exists():
            return None

        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        cached_source = Path(str(data.get("source", ""))).name
        if cached_source and cached_source != self.pdf_path.name:
            return None

        cached_chunk_size = data.get("chunk_size")
        cached_chunk_overlap = data.get("chunk_overlap")
        if (
            cached_chunk_size is not None
            and cached_chunk_size != self.chunk_size
        ) or (
            cached_chunk_overlap is not None
            and cached_chunk_overlap != self.chunk_overlap
        ):
            return None

        return [TextChunk(**chunk) for chunk in data.get("chunks", [])]

    def _write_cache(self, chunks):
        if not self.pdf_path.exists():
            return

        payload = {
            "source": str(self.pdf_path),
            "source_mtime": self._source_mtime(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunks": [asdict(chunk) for chunk in chunks],
        }
        try:
            self.cache_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            return

    def _source_mtime(self):
        try:
            return self.pdf_path.stat().st_mtime
        except OSError:
            return None

    def _extract_chunks_from_pdf(self):
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("缺少 PyMuPDF，请先安装 requirements.txt 中的 PyMuPDF。") from exc

        if not self.pdf_path.exists():
            return []

        chunks = []
        with fitz.open(self.pdf_path) as document:
            for page_index, page in enumerate(document, start=1):
                text = _compact_text(page.get_text("text"))
                chunks.extend(
                    TextChunk(page=page_index, text=chunk)
                    for chunk in _split_text(text, self.chunk_size, self.chunk_overlap)
                )
        return chunks


class CombinedGuidelineRAGRetriever:
    def __init__(self, retrievers, top_k=4):
        self.retrievers = [
            item if isinstance(item, tuple) else (f"Guideline {index}", item)
            for index, item in enumerate(retrievers, start=1)
        ]
        self.top_k = top_k

    def search_for_risk(self, risk):
        grouped_results = []
        for source_name, retriever in self.retrievers:
            source_results = []
            for result in retriever.search_for_risk(risk):
                enriched = dict(result)
                enriched.setdefault("source", source_name)
                source_results.append(enriched)
            source_results.sort(key=lambda result: result.get("score", 0), reverse=True)
            if source_results:
                grouped_results.append(source_results)

        selected = [source_results[0] for source_results in grouped_results]
        remaining = [
            result
            for source_results in grouped_results
            for result in source_results[1:]
        ]
        remaining.sort(key=lambda result: result.get("score", 0), reverse=True)

        selected.extend(remaining[: max(self.top_k - len(selected), 0)])
        selected.sort(key=lambda result: result.get("score", 0), reverse=True)
        return selected[: self.top_k]


def _split_text(text, chunk_size, chunk_overlap):
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - chunk_overlap, start + 1)
    return [chunk for chunk in chunks if chunk]


def _score_chunk(query, query_tokens, text):
    text_normalized = text.lower()
    text_tokens = Counter(_tokenize(text))
    if not text_tokens:
        return 0.0

    overlap = 0.0
    for token, query_count in query_tokens.items():
        if token in text_tokens:
            overlap += min(query_count, text_tokens[token]) * (1.0 + math.log1p(len(token)))

    phrase_bonus = 0.0
    for phrase in _important_phrases(query):
        if phrase in text_normalized:
            phrase_bonus += 4.0

    return overlap + phrase_bonus


def _important_phrases(query):
    phrases = []
    for phrase in (
        "high risk",
        "low risk",
        "acute coronary syndrome",
        "invasive evaluation",
        "guideline-directed management",
        "follow-up",
        "discharge",
        "serial troponin",
    ):
        if phrase in query.lower():
            phrases.append(phrase)
    return phrases


def _tokenize(text):
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _compact_text(text):
    return " ".join(str(text).split())
