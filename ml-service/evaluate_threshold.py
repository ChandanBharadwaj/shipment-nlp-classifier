"""
Threshold evaluation with train / validation / test split awareness.

Workflow
--------
1. centroid_builder.py builds centroids from split='train' rows only.

2. Tune threshold on the validation set:
       python evaluate_threshold.py --split validation
   Sweeps all thresholds, prints metrics, and recommends the best one by F1.

3. Confirm on the held-out test set (run ONCE, with the threshold chosen in step 2):
       python evaluate_threshold.py --split test --threshold 0.35
   Evaluates at that single threshold and prints the final production metrics.

   WARNING: do not sweep thresholds against the test set — that leaks the test
   set into your threshold selection. Always tune on validation, confirm on test.

Data source
-----------
Primary:  shipment_labels table, filtered by the `split` column.
          Supports multi-label: rows are grouped by shipment_id so a shipment
          with two category labels becomes one sample with expected={cat1, cat2}.
Fallback: --csv path/to/file.csv  (columns: cargo, commodity, expected_labels)
          expected_labels is comma-separated, e.g. "electronics,perishables"
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

from classifier import load_centroids, load_keywords, predict
from db import get_connection

THRESHOLDS_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_from_db(conn, split: str) -> list[dict]:
    """
    Load labeled samples from the shipment_labels table for a given split.

    Groups by shipment_id to handle multi-label cases: a shipment that appears
    twice (two category rows) becomes one sample with expected={cat1, cat2}.

    Returns:
        List of dicts with keys: cargo, commodity, expected (set of str)
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT shipment_id,
                   MAX(cargo_text)                     AS cargo,
                   MAX(commodity_text)                 AS commodity,
                   array_agg(category_name ORDER BY category_name) AS expected_labels
            FROM   shipment_labels
            WHERE  split = %s
            GROUP  BY shipment_id
            ORDER  BY shipment_id
        """, (split,))
        rows = cur.fetchall()

    if not rows:
        raise ValueError(
            f"No rows found for split='{split}' in shipment_labels.\n"
            f"Run python init_db.py to apply schema and seed data first."
        )

    return [
        {"cargo": row[1], "commodity": row[2], "expected": set(row[3])}
        for row in rows
    ]


def load_from_csv(path: str) -> list[dict]:
    """
    Load samples from a CSV file.

    Columns: cargo, commodity, expected_labels  (expected_labels comma-separated)
    """
    samples = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected = {c.strip() for c in row["expected_labels"].split(",") if c.strip()}
            samples.append({
                "cargo":     row["cargo"],
                "commodity": row["commodity"],
                "expected":  expected,
            })
    if not samples:
        raise ValueError(f"No samples loaded from {path}.")
    return samples


# ── Metrics ────────────────────────────────────────────────────────────────────

def evaluate_at(
    samples: list[dict],
    centroids: dict,
    keywords: dict,
    threshold: float,
) -> dict:
    """
    Compute micro-averaged precision, recall, and F1 at a single threshold.

    Also tracks per-category TP/FP/FN for a breakdown table.
    """
    total_tp = total_fp = total_fn = 0
    per_cat: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for sample in samples:
        result    = predict(sample["cargo"], sample["commodity"], centroids, keywords, threshold)
        predicted = set(result["categories"])
        expected  = sample["expected"]

        tp = predicted & expected
        fp = predicted - expected
        fn = expected  - predicted

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        for cat in tp:
            per_cat[cat]["tp"] += 1
        for cat in fp:
            per_cat[cat]["fp"] += 1
        for cat in fn:
            per_cat[cat]["fn"] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "n_samples": len(samples),
        "tp":        total_tp,
        "fp":        total_fp,
        "fn":        total_fn,
        "per_category": dict(per_cat),
    }


def _print_per_category(per_cat: dict, threshold: float) -> None:
    """Print a per-category precision/recall breakdown."""
    print(f"\n  Per-category breakdown at threshold={threshold:.2f}:")
    print(f"  {'category':<20}  {'tp':>4}  {'fp':>4}  {'fn':>4}  {'prec':>6}  {'rec':>6}")
    print("  " + "-" * 55)
    for cat in sorted(per_cat):
        s   = per_cat[cat]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"  {cat:<20}  {tp:>4}  {fp:>4}  {fn:>4}  {p:>6.3f}  {r:>6.3f}")


# ── Modes ──────────────────────────────────────────────────────────────────────

def run_validation_sweep(samples: list[dict], centroids: dict, keywords: dict) -> float:
    """
    Sweep all thresholds on the validation set and recommend the best by F1.
    Returns the recommended threshold.
    """
    print(f"\n{'threshold':>10}  {'precision':>10}  {'recall':>8}  {'f1':>8}  "
          f"{'tp':>5}  {'fp':>5}  {'fn':>5}  {'n':>5}")
    print("-" * 68)

    results = []
    for t in THRESHOLDS_SWEEP:
        r = evaluate_at(samples, centroids, keywords, t)
        results.append(r)
        marker = ""
        print(
            f"{r['threshold']:>10.2f}  "
            f"{r['precision']:>10.4f}  "
            f"{r['recall']:>8.4f}  "
            f"{r['f1']:>8.4f}  "
            f"{r['tp']:>5}  "
            f"{r['fp']:>5}  "
            f"{r['fn']:>5}  "
            f"{r['n_samples']:>5}"
            f"{marker}"
        )

    best = max(results, key=lambda x: x["f1"])
    print(f"\n{'─' * 68}")
    print(f"  Best by F1: threshold={best['threshold']:.2f}  "
          f"precision={best['precision']:.4f}  recall={best['recall']:.4f}  "
          f"f1={best['f1']:.4f}")
    _print_per_category(best["per_category"], best["threshold"])
    print(f"\n  → Recommended threshold: {best['threshold']:.2f}")
    print(f"  → Next step: confirm on the test set:")
    print(f"       python evaluate_threshold.py --split test --threshold {best['threshold']:.2f}")

    return best["threshold"]


def run_test_confirmation(
    samples: list[dict],
    centroids: dict,
    keywords: dict,
    threshold: float,
) -> None:
    """
    Evaluate at a single threshold on the test set.
    This is the final held-out evaluation — run once after validation tuning.
    """
    r = evaluate_at(samples, centroids, keywords, threshold)

    print(f"\n{'─' * 68}")
    print(f"  FINAL TEST EVALUATION at threshold={threshold:.2f}")
    print(f"  (held-out set — {r['n_samples']} samples)")
    print(f"{'─' * 68}")
    print(f"  Precision : {r['precision']:.4f}")
    print(f"  Recall    : {r['recall']:.4f}")
    print(f"  F1        : {r['f1']:.4f}")
    print(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}")
    _print_per_category(r["per_category"], threshold)
    print(f"\n{'─' * 68}")
    print(f"  Production threshold: {threshold:.2f}")
    print(f"  Set this as the default in classifier.py:")
    print(f"      def predict(..., threshold: float = {threshold:.2f}):")
    print(f"  Or pass it per-request:")
    print(f'      {{"threshold": {threshold:.2f}, ...}}')


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate classification thresholds using train/validation/test splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune threshold on validation set (run after centroid_builder.py)
  python evaluate_threshold.py --split validation

  # Confirm on test set with threshold chosen from validation
  python evaluate_threshold.py --split test --threshold 0.35

  # Use an external CSV instead of the DB split
  python evaluate_threshold.py --csv test_data.csv
        """,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--split",
        choices=["validation", "test"],
        help="Which DB split to evaluate. 'validation' sweeps all thresholds; "
             "'test' evaluates at --threshold (requires --threshold).",
    )
    source.add_argument(
        "--csv",
        metavar="PATH",
        help="Path to CSV file (columns: cargo, commodity, expected_labels). "
             "Sweeps all thresholds — use only when you don't have DB splits.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        metavar="FLOAT",
        help="Required when --split test. The threshold chosen from --split validation.",
    )

    args = parser.parse_args()

    if args.split == "test" and args.threshold is None:
        parser.error(
            "--threshold is required when using --split test.\n"
            "Run --split validation first to choose a threshold, then pass it here."
        )

    conn = None
    try:
        conn      = get_connection()
        centroids = load_centroids(conn)
        keywords  = load_keywords(conn)

        if args.csv:
            samples = load_from_csv(args.csv)
            label   = f"CSV ({args.csv})"
        else:
            samples = load_from_db(conn, args.split)
            label   = f"split='{args.split}'"

        print(f"Centroids loaded : {len(centroids)} categories")
        print(f"Samples loaded   : {len(samples)} ({label})")

        if args.csv or args.split == "validation":
            run_validation_sweep(samples, centroids, keywords)
        else:
            run_test_confirmation(samples, centroids, keywords, args.threshold)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
