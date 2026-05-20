#!/usr/bin/env python3
"""Rebuild the RAG corpus index after adding or updating documents."""

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_store import RagCorpusStore


def main():
    parser = argparse.ArgumentParser(description="Rebuild RAG corpus index.")
    parser.add_argument(
        "--corpus-dir",
        default=None,
        help="Path to rag_corpus (default: ./rag_corpus or RAG_CORPUS_DIR)",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "embedding", "lexical"],
        default=None,
        help="Retrieval mode override",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing index files before rebuild",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Build lexical-only index (no HuggingFace download)",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print corpus status without rebuilding",
    )
    args = parser.parse_args()

    if args.skip_embeddings:
        import os

        os.environ["RAG_SKIP_EMBEDDINGS"] = "1"
        if args.mode is None:
            args.mode = "lexical"

    store = RagCorpusStore(corpus_dir=args.corpus_dir, retrieval_mode=args.mode)

    if args.status_only:
        print(json.dumps(store.status(), ensure_ascii=False, indent=2))
        return 0

    meta = store.rebuild_index(force=args.force)
    print("RAG index rebuilt successfully.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
