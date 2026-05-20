import json
import os
from dataclasses import dataclass
from pathlib import Path

from rag_index import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TOP_K,
    CorpusIndex,
)


CORPUS_CATEGORIES = ("guidelines", "screening", "methods", "papers", "inbox")
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
DEFAULT_CORPUS_DIRNAME = "rag_corpus"


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    title: str
    category: str
    rel_path: str
    path: Path
    enabled: bool = True
    weight: float = 1.0


class RagCorpusStore:
    def __init__(
        self,
        corpus_dir=None,
        retrieval_mode=None,
        top_k=None,
        lexical_weight=0.35,
    ):
        base_dir = Path(__file__).resolve().parent
        self.corpus_dir = Path(
            corpus_dir or os.environ.get("RAG_CORPUS_DIR", base_dir / DEFAULT_CORPUS_DIRNAME)
        )
        self.index_dir = self.corpus_dir / ".rag_index"
        self.retrieval_mode = retrieval_mode or os.environ.get("RAG_RETRIEVAL_MODE", "hybrid")
        self.top_k = int(os.environ.get("RAG_TOP_K", top_k or DEFAULT_TOP_K))
        self.lexical_weight = lexical_weight
        self.manifest = self._load_manifest()
        self._index = None

    @property
    def index(self):
        if self._index is None:
            self._index = CorpusIndex(
                self.index_dir,
                retrieval_mode=self.retrieval_mode,
                top_k=self.top_k,
                lexical_weight=self.lexical_weight,
            )
        return self._index

    def discover_documents(self):
        manifest_docs = {
            item["path"]: item
            for item in self.manifest.get("documents", [])
            if item.get("path")
        }
        discovered = []

        for category in CORPUS_CATEGORIES:
            category_dir = self.corpus_dir / category
            if not category_dir.exists():
                continue

            for path in sorted(category_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.name.startswith("."):
                    continue
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                rel_path = path.relative_to(self.corpus_dir).as_posix()
                override = manifest_docs.get(rel_path, {})
                if override.get("enabled") is False:
                    continue

                title = override.get("title") or _title_from_path(path, category)
                doc_id = override.get("doc_id") or rel_path
                discovered.append(
                    CorpusDocument(
                        doc_id=doc_id,
                        title=title,
                        category=category,
                        rel_path=rel_path,
                        path=path.resolve(),
                        enabled=True,
                        weight=float(override.get("weight", 1.0)),
                    )
                )

        return discovered

    def rebuild_index(self, force=False):
        documents = self.discover_documents()
        if not documents:
            raise RuntimeError(
                f"RAG 资料库为空：请向 {self.corpus_dir} 下的子目录放入 PDF/TXT/MD 后重试。"
            )

        defaults = self.manifest.get("defaults", {})
        chunk_size = int(defaults.get("chunk_size", DEFAULT_CHUNK_SIZE))
        chunk_overlap = int(defaults.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP))
        embedding_model = defaults.get("embedding_model", DEFAULT_EMBEDDING_MODEL)

        if force and self.index_dir.exists():
            for path in self.index_dir.glob("*"):
                if path.is_file():
                    path.unlink()

        meta = self.index.build(
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
        )
        meta["documents"] = [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "category": doc.category,
                "rel_path": doc.rel_path,
            }
            for doc in documents
        ]
        (self.index_dir / "documents.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return meta

    def ensure_index(self, auto_build=True):
        if self.index.is_built:
            return self.index

        if not auto_build:
            raise RuntimeError(
                f"未找到 RAG 索引，请先运行: python scripts/rebuild_rag_index.py"
            )

        return self.rebuild_index()

    def search(self, query, top_k=None, categories=None):
        self.ensure_index()
        return self.index.search(query, top_k=top_k, categories=categories)

    def search_for_prediction(self, prediction, risk, patient_context=None, top_k=None):
        self.ensure_index()
        return self.index.search_for_prediction(
            prediction,
            risk,
            patient_context=patient_context,
            top_k=top_k,
        )

    def search_for_risk(self, risk, prediction=None):
        """Backward-compatible hook for tests expecting risk-only retrieval."""
        from clinical_query import build_risk_query

        if prediction is not None:
            return self.search_for_prediction(prediction, risk)

        self.ensure_index()
        return self.index.search(build_risk_query(risk), top_k=self.top_k)

    def status(self):
        documents = self.discover_documents()
        built = self.index.is_built
        meta = {}
        if built and (self.index_dir / "meta.json").exists():
            meta = json.loads((self.index_dir / "meta.json").read_text(encoding="utf-8"))

        return {
            "corpus_dir": str(self.corpus_dir),
            "index_built": built,
            "document_count": len(documents),
            "documents": [
                {
                    "title": doc.title,
                    "category": doc.category,
                    "rel_path": doc.rel_path,
                }
                for doc in documents
            ],
            "meta": meta,
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
        }

    def _load_manifest(self):
        manifest_path = self.corpus_dir / "manifest.yaml"
        if not manifest_path.exists():
            return {"defaults": {}, "documents": []}

        try:
            import yaml
        except ImportError:
            return {"defaults": {}, "documents": []}

        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        data.setdefault("defaults", {})
        data.setdefault("documents", [])
        return data


def get_default_corpus_store():
    return RagCorpusStore()


def _title_from_path(path, category):
    stem = path.stem.replace("_", " ").strip()
    category_labels = {
        "guidelines": "Guideline",
        "screening": "Screening",
        "methods": "Methods",
        "papers": "Paper",
        "inbox": "Inbox",
    }
    prefix = category_labels.get(category, category)
    return f"{prefix}: {stem}"
