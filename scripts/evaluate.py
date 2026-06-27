#!/usr/bin/env python
"""Stage 3 — Evaluation.

Parses the frozen LLM's free-form answers into YES/NO predictions and reports
accuracy / precision / recall / F1 against the ground-truth labels. Works for
both POPE (accuracy is the headline metric) and VSR (recall / F1 matter most).

Example
-------
    python scripts/evaluate.py \
        --input outputs/final_results.csv \
        --pred-col llm_caption --label-col answer
"""

import argparse
import os
import sys

import pandas as pd

# Allow running as `python scripts/evaluate.py` from the repo root without
# installing the package first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capstone.eval import compute_metrics, normalize_label, parse_yes_no


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAPSTONE predictions")
    parser.add_argument("--input", default="outputs/final_results.csv")
    parser.add_argument("--pred-col", default="llm_caption",
                        help="Column holding raw LLM output")
    parser.add_argument("--label-col", default="answer",
                        help="Column holding the ground-truth label")
    parser.add_argument("--relation-col", default="relation",
                        help="Optional column for per-relation VSR breakdown")
    parser.add_argument("--by-relation", action="store_true",
                        help="Also print a per-relation-type breakdown (VSR)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    preds = [parse_yes_no(t) for t in df[args.pred_col]]
    labels = [normalize_label(v) for v in df[args.label_col]]

    overall = compute_metrics(preds, labels)
    print("\n=== Overall ===")
    print(f"  Samples evaluated : {overall['evaluated']} (skipped {overall['skipped']})")
    print(f"  Accuracy          : {overall['accuracy'] * 100:.2f}%")
    print(f"  Precision         : {overall['precision'] * 100:.2f}%")
    print(f"  Recall            : {overall['recall'] * 100:.2f}%")
    print(f"  F1 Score          : {overall['f1'] * 100:.2f}%")
    print(f"  TP/TN/FP/FN       : {overall['tp']}/{overall['tn']}/{overall['fp']}/{overall['fn']}")

    if args.by_relation and args.relation_col in df.columns:
        print("\n=== By relation type ===")
        df = df.assign(_pred=preds, _label=labels)
        for rel, group in df.groupby(args.relation_col):
            m = compute_metrics(group["_pred"].tolist(), group["_label"].tolist())
            print(f"  {str(rel):<20} n={m['evaluated']:<5} "
                  f"acc={m['accuracy'] * 100:5.1f}%  f1={m['f1'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
