#!/usr/bin/env python3
"""Offline evaluation for RAG retrieval and optional LLM generation."""

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_store import RagCorpusStore


def load_cases(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("gold cases file must be a JSON array")
    return data


def retrieval_hit(snippets, keywords):
    blob = " ".join(snippet.get("text", "") for snippet in snippets).lower()
    return any(keyword.lower() in blob for keyword in keywords)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default=str(Path(__file__).with_name("gold_cases.json")),
        help="Path to gold cases JSON",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only evaluate retrieval hits",
    )
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        cases_path = Path(__file__).with_name("gold_cases.sample.json")

    cases = load_cases(cases_path)
    store = RagCorpusStore()
    store.ensure_index(auto_build=True)

    hits = 0
    for case in cases:
        snippets = store.search_for_prediction(
            prediction=case["prediction"],
            risk=case["risk"],
        )
        keywords = case.get("expected_keywords", [])
        ok = retrieval_hit(snippets, keywords) if keywords else bool(snippets)
        hits += int(ok)
        print(
            f"{case['id']}: retrieval_hit={ok} snippets={len(snippets)} "
            f"top_source={snippets[0]['source'] if snippets else 'N/A'}"
        )

    accuracy = hits / len(cases) if cases else 0.0
    print(f"Retrieval accuracy: {hits}/{len(cases)} = {accuracy:.1%}")

    if args.retrieval_only:
        return 0

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Set DEEPSEEK_API_KEY for generation evaluation.")
        return 1

    from ai_analysis import DeepSeekAnalyzer

    analyzer = DeepSeekAnalyzer(api_key=api_key, corpus_store=store)
    for case in cases:
        result = analyzer.analyze(
            fields=[],
            values=[],
            prediction=case["prediction"],
            risk=case["risk"],
        )
        print(f"{case['id']}: generation_error={result.get('error')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
