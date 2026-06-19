#!/usr/bin/env python3
"""Generate SHAP interpretability figures for Ridge-RF prediction models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shap_analysis import DEFAULT_OUTPUT_DIR, run_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP analysis for Ridge-RF models.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for SHAP figures and CSV outputs.",
    )
    parser.add_argument(
        "--max-background",
        type=int,
        default=100,
        help="Maximum background samples for SHAP LinearExplainer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = run_all(args.output_dir, max_background=args.max_background)
    print(f"SHAP analysis completed. Output directory: {args.output_dir.resolve()}")
    print("Classification top features:")
    for item in reports["classification"]["top_features"]:
        print(f"  - {item['feature']}: {item['mean_abs_shap']:.4f}")
    for branch_report in reports["regression"]:
        print(f"{branch_report['title_en']} top feature: {branch_report['top_features'][0]['feature']}")


if __name__ == "__main__":
    main()
