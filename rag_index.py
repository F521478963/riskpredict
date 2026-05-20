import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from clinical_query import build_clinical_query


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+|\d+(?:\.\d+)?")
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INDEX_VERSION = 2


@dataclass(frozen=True)
class IndexedChunk:
    chunk_id: int
    doc_id: str
    source: str
    category: str
    page: int
    text: str
    rel_path: str


def extract_chunks_from_pdf(pdf_path, chunk_size, chunk_overlap):
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("缺少 PyMuPDF，请先安装 requirements.txt 中的 PyMuPDF。") from exc

    chunks = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            text = _compact_text(page.get_text("text"))
            for piece in _split_text(text, chunk_size, chunk_overlap):
                chunks.append((page_index, piece))
    return chunks


def extract_chunks_from_text_file(text_path, chunk_size, chunk_overlap):
    text = _compact_text(text_path.read_text(encoding="utf-8", errors="ignore"))
    return [(1, piece) for piece in _split_text(text, chunk_size, chunk_overlap)]


class CorpusIndex:
    def __init__(
        self,
        index_dir,
        retrieval_mode="hybrid",
        top_k=DEFAULT_TOP_K,
        lexical_weight=0.35,
    ):
        self.index_dir = Path(index_dir)
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.lexical_weight = lexical_weight
        self.meta_path = self.index_dir / "meta.json"
        self.chunks_path = self.index_dir / "chunks.json"
        self.embeddings_path = self.index_dir / "embeddings.npy"
        self._chunks = None
        self._embeddings = None
        self._embedding_model = None
        self._meta = None

    @property
    def is_built(self):
        return self.meta_path.exists() and self.chunks_path.exists()

    def build(self, documents, chunk_size, chunk_overlap, embedding_model):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        indexed_chunks = []
        chunk_id = 0

        for document in documents:
            if not document.enabled:
                continue

            if document.path.suffix.lower() == ".pdf":
                raw_chunks = extract_chunks_from_pdf(
                    document.path, chunk_size, chunk_overlap
                )
            else:
                raw_chunks = extract_chunks_from_text_file(
                    document.path, chunk_size, chunk_overlap
                )

            for page, text in raw_chunks:
                indexed_chunks.append(
                    IndexedChunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        source=document.title,
                        category=document.category,
                        page=page,
                        text=text,
                        rel_path=document.rel_path,
                    )
                )
                chunk_id += 1

        embeddings, model_name, embedding_enabled = self._build_embeddings(
            indexed_chunks, embedding_model
        )

        meta = {
            "version": INDEX_VERSION,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_count": len(indexed_chunks),
            "document_count": len([doc for doc in documents if doc.enabled]),
            "embedding_model": model_name,
            "embedding_enabled": embedding_enabled,
        }

        self.meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.chunks_path.write_text(
            json.dumps([asdict(chunk) for chunk in indexed_chunks], ensure_ascii=False),
            encoding="utf-8",
        )

        if embeddings is not None:
            import numpy as np

            np.save(self.embeddings_path, embeddings)
        elif self.embeddings_path.exists():
            self.embeddings_path.unlink()

        self._chunks = indexed_chunks
        self._embeddings = embeddings
        self._meta = meta
        return meta

    def search(self, query, top_k=None, categories=None):
        chunks = self._load_chunks()
        if not chunks:
            return []

        top_k = top_k or self.top_k
        filtered = chunks
        if categories:
            allowed = {category.lower() for category in categories}
            filtered = [chunk for chunk in chunks if chunk.category.lower() in allowed]
            if not filtered:
                filtered = chunks

        lexical_scores = self._lexical_scores(query, filtered)
        embedding_scores = self._embedding_scores(query, filtered)

        combined = []
        for index, chunk in enumerate(filtered):
            lexical = lexical_scores.get(chunk.chunk_id, 0.0)
            embedding = embedding_scores.get(chunk.chunk_id, 0.0)
            if self.retrieval_mode == "lexical":
                score = lexical
            elif self.retrieval_mode == "embedding":
                score = embedding
            else:
                if embedding_scores:
                    score = (1.0 - self.lexical_weight) * embedding + self.lexical_weight * lexical
                else:
                    score = lexical

            if score > 0:
                combined.append((score, chunk))

        combined.sort(key=lambda item: item[0], reverse=True)
        diversified = _diversify_by_source(combined, top_k)
        return [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "source": chunk.source,
                "category": chunk.category,
                "page": chunk.page,
                "text": _compact_text(chunk.text),
                "score": round(score, 4),
                "rel_path": chunk.rel_path,
            }
            for score, chunk in diversified
        ]

    def search_for_prediction(self, prediction, risk, patient_context=None, top_k=None):
        query = build_clinical_query(prediction, risk, patient_context)
        guideline_first = self.search(
            query,
            top_k=top_k,
            categories=["guidelines"],
        )
        if len(guideline_first) >= (top_k or self.top_k):
            return guideline_first[: top_k or self.top_k]

        remaining = (top_k or self.top_k) - len(guideline_first)
        other = self.search(query, top_k=remaining, categories=None)
        seen = {item["chunk_id"] for item in guideline_first}
        merged = list(guideline_first)
        for item in other:
            if item["chunk_id"] in seen:
                continue
            merged.append(item)
            seen.add(item["chunk_id"])
            if len(merged) >= (top_k or self.top_k):
                break
        return merged

    def _load_chunks(self):
        if self._chunks is not None:
            return self._chunks

        if not self.chunks_path.exists():
            self._chunks = []
            return self._chunks

        data = json.loads(self.chunks_path.read_text(encoding="utf-8"))
        self._chunks = [IndexedChunk(**item) for item in data]
        return self._chunks

    def _load_meta(self):
        if self._meta is not None:
            return self._meta

        if not self.meta_path.exists():
            self._meta = {}
            return self._meta

        self._meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return self._meta

    def _load_embeddings(self):
        if self._embeddings is not None:
            return self._embeddings

        meta = self._load_meta()
        if not meta.get("embedding_enabled") or not self.embeddings_path.exists():
            self._embeddings = None
            return None

        import numpy as np

        self._embeddings = np.load(self.embeddings_path)
        return self._embeddings

    def _lexical_scores(self, query, chunks):
        query_tokens = Counter(_tokenize(query))
        scores = {}
        for chunk in chunks:
            score = _score_chunk(query, query_tokens, chunk.text)
            if score > 0:
                scores[chunk.chunk_id] = score
        return scores

    def _embedding_scores(self, query, chunks):
        embeddings = self._load_embeddings()
        if embeddings is None or self.retrieval_mode == "lexical":
            return {}

        query_vector = self._encode_query(query)
        if query_vector is None:
            return {}

        import numpy as np

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        matrix = embeddings[chunk_ids]
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return {}

        denom = np.linalg.norm(matrix, axis=1) * query_norm
        denom[denom == 0] = 1.0
        similarities = matrix @ query_vector / denom
        return {
            chunk.chunk_id: float(similarities[index])
            for index, chunk in enumerate(chunks)
        }

    def _encode_query(self, query):
        model = self._get_embedding_model()
        if model is None:
            return None
        vector = model.encode([query], normalize_embeddings=True)
        return vector[0]

    def _get_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model

        meta = self._load_meta()
        model_name = meta.get("embedding_model") or DEFAULT_EMBEDDING_MODEL
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            return None

        self._embedding_model = SentenceTransformer(model_name)
        return self._embedding_model

    def _build_embeddings(self, chunks, embedding_model):
        if not chunks:
            return None, embedding_model, False

        import os

        if os.environ.get("RAG_SKIP_EMBEDDINGS", "").lower() in {"1", "true", "yes"}:
            return None, embedding_model, False

        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            return None, embedding_model, False

        try:
            model = SentenceTransformer(embedding_model)
            texts = [chunk.text for chunk in chunks]
            embeddings = model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(embeddings), embedding_model, True
        except Exception:
            return None, embedding_model, False


def _diversify_by_source(scored_chunks, top_k):
    if not scored_chunks:
        return []

    selected = []
    seen_sources = set()
    leftovers = []

    for score, chunk in scored_chunks:
        if chunk.source not in seen_sources:
            selected.append((score, chunk))
            seen_sources.add(chunk.source)
        else:
            leftovers.append((score, chunk))
        if len(selected) >= top_k:
            return selected

    for item in leftovers:
        selected.append(item)
        if len(selected) >= top_k:
            break
    return selected


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
        "noninvasive screening",
        "coronary angiography",
    ):
        if phrase in query.lower():
            phrases.append(phrase)
    return phrases


def _tokenize(text):
    return [token.lower() for token in TOKEN_RE.findall(text)]


def _compact_text(text):
    return " ".join(str(text).split())
