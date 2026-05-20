#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_store import RagCorpusStore


if __name__ == "__main__":
    store = RagCorpusStore()
    print(json.dumps(store.status(), ensure_ascii=False, indent=2))
