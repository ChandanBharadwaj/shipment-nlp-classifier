"""
Threshold evaluation with train / validation / test split awareness.

Workflow
--------
1. centroid_builder.py builds multi-centroids from split='train' rows.
2. fit_calibration.py fits Platt scaling on split='validation'.
3. Tune threshold on the validation set:
       python evaluate_threshold.py --split validation
   Sweeps all thresholds, prints metrics, and recommends the best one by F1.
4. Confirm on held-out test set (run ONCE, with the threshold from step 3):
       python evaluate_threshold.py --split test --threshold 0.45

   WARNING: do not sweep thresholds against the test set — that leaks the test
   set into your threshold selection.

Outputs
-------
- Micro precision/recall/F1 at each swept threshold (validation) or at one
  threshold (test).
- Per-category TP/FP/FN breakdown.
- Confusion matrix on FPs and FNs: "when the model said X, what did analysts
  actually expect?" — identifies which category pairs are getting confused.

Data source
-----------
Primary:  shipment_labels table, filtered by the `split` column.
Fallback: --csv path/to/file.csv  (columns: cargo, commodity, expected_labels)
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

from classifier import (
    load_category_config,
    load_centroids,
    load_keywords,
    predict,
)
from db import get_connection

THRESHOLDS_SWEEP = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_from_db(conn, split: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT shipment_id,
                   MAX(cargo_text)                                 AS cargo,
                   MAX(commodity_text)                             AS commodity,
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
    category_config: dict,
    threshold: float,
) -> dict:
    """
    Compute micro-averaged precision/recall/F1 + per-category breakdown +
    confusion matrix at a single threshold.

    Confusion matrix records, for each wrong prediction, the (predicted, expected)
    pair so you can see e.g. "machinery was called metals 4 times".
    """
    total_tp = total_fp = total_fn = 0
    per_cat: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    confusion: dict[tuple[str, str], int] = defaultdict(int)

    for sample in samples:
        result    = predict(
            sample["cargo"],
            sample["commodity"],
            centroids,
            keywords,
            category_config,
            threshold=threshold,
        )
        predicted = set(result["categories"])
        expected  = sample["expected"]

        tp = predicted & expected
        fp = predicted - expected
        fn = expected  - predicted

        total_tp += len(tp)
        total_fp += len(fp)
        total_fn += len(fn)

        for cat in tp: per_cat[cat]["tp"] += 1
        for cat in fp: per_cat[cat]["fp"] += 1
        for cat in fn: per_cat[cat]["fn"] += 1

        # Confusion: for every FP x FN pair on the same sample, count one
        # "said X, meant Y" confusion. When a sample has only FPs (no FN),
        # map each FP to every expected label (missed).
        if fp and fn:
            for p in fp:
                for e in fn:
                    confusion[(p, e)] += 1
        elif fp:
            for p in fp:
                for e in expected:
                    confusion[(p, e)] += 1
        elif fn:
            for e in fn:
                # Model said nothing or something unrelated → "nothing" column.
                confusion[("<none>", e)] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "threshold":    threshold,
        "precision":    round(precision, 4),
        "recall":       round(recall,    4),
        "f1":           round(f1,        4),
        "n_samples":    len(samples),
        "tp":           total_tp,
        "fp":           total_fp,
        "fn":           total_fn,
        "per_category": dict(per_cat),
        "confusion":    dict(confusion),
    }


def _print_per_category(per_cat: dict, threshold: float) -> None:
    print(f"\n  Per-category breakdown at threshold={threshold:.2f}:")
    print(f"  {'category':<20}  {'tp':>4}  {'fp':>4}  {'fn':>4}  {'prec':>6}  {'rec':>6}")
    print("  " + "-" * 55)
    for cat in sorted(per_cat):
        s   = per_cat[cat]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"  {cat:<20}  {tp:>4}  {fp:>4}  {fn:>4}  {p:>6.3f}  {r:>6.3f}")


def _print_confusion(confusion: dict[tuple[str, str], int], top_n: int = 15) -> None:
    """Print the top-N most common (predicted, expected) confusions."""
    if not confusion:
        print("\n  No confusions to report — perfect predictions.")
        return
    items = sorted(confusion.items(), key=lambda kv: -kv[1])
    print(f"\n  Top {min(top_n, len(items))} confusions (predicted -> expected, count):")
    for (pred, exp), count in items[:top_n]:
        print(f"    {pred:<20} -> {exp:<20}  {count}")


# ── Modes ──────────────────────────────────────────────────────────────────────

def run_validation_sweep(samples: list[dict], centroids: dict, keywords: dict, category_config: dict) -> float:
    print(f"\n{'threshold':>10}  {'precision':>10}  {'recall':>8}  {'f1':>8}  "
          f"{'tp':>5}  {'fp':>5}  {'fn':>5}  {'n':>5}")
    print("-" * 68)

    results = []
    for t in THRESHOLDS_SWEEP:
        r = evaluate_at(samples, centroids, keywords, category_config, t)
        results.append(r)
        print(
            f"{r['threshold']:>10.2f}  {r['precision']:>10.4f}  "
            f"{r['recall']:>8.4f}  {r['f1']:>8.4f}  "
            f"{r['tp']:>5}  {r['fp']:>5}  {r['fn']:>5}  {r['n_samples']:>5}"
        )

    best = max(results, key=lambda x: x["f1"])
    print(f"\n{'-' * 68}")
    print(f"  Best by F1: threshold={best['threshold']:.2f}  "
          f"precision={best['precision']:.4f}  recall={best['recall']:.4f}  "
          f"f1={best['f1']:.4f}")
    _print_per_category(best["per_category"], best["threshold"])
    _print_confusion(best["confusion"])
    print(f"\n  -> Recommended threshold: {best['threshold']:.2f}")
    print(f"  -> Next step: confirm on the test set:")
    print(f"       python evaluate_threshold.py --split test --threshold {best['threshold']:.2f}")

    return best["threshold"]


def run_test_confirmation(samples: list[dict], centroids: dict, keywords: dict,
                          category_config: dict, threshold: float) -> None:
    r = evaluate_at(samples, centroids, keywords, category_config, threshold)
    print(f"\n{'-' * 68}")
    print(f"  FINAL TEST EVALUATION at threshold={threshold:.2f}")
    print(f"  (held-out set — {r['n_samples']} samples)")
    print(f"{'-' * 68}")
    print(f"  Precision : {r['precision']:.4f}")
    print(f"  Recall    : {r['recall']:.4f}")
    print(f"  F1        : {r['f1']:.4f}")
    print(f"  TP={r['tp']}  FP={r['fp']}  FN={r['fn']}")
    _print_per_category(r["per_category"], threshold)
    _print_confusion(r["confusion"])


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate classification thresholds using train/validation/test splits.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split", choices=["validation", "test"])
    source.add_argument("--csv", metavar="PATH")
    parser.add_argument("--threshold", type=float, metavar="FLOAT")
    args = parser.parse_args()

    if args.split == "test" and args.threshold is None:
        parser.error("--threshold is required when using --split test.")

    conn = None
    try:
        conn = get_connection()
        centroids       = load_centroids(conn)
        keywords        = load_keywords(conn)
        category_config = load_category_config(conn)

        if args.csv:
            samples = load_from_csv(args.csv)
            label   = f"CSV ({args.csv})"
        else:
            samples = load_from_db(conn, args.split)
            label   = f"split='{args.split}'"

        sub = sum(len(v) for v in centroids.values())
        calibrated = any(c.get("platt_a") is not None for c in category_config.values())
        print(f"Centroids loaded : {len(centroids)} categories ({sub} sub-centroids)")
        print(f"Calibration      : {'fit' if calibrated else 'NOT FIT (run fit_calibration.py)'}")
        print(f"Samples loaded   : {len(samples)} ({label})")

        if args.csv or args.split == "validation":
            run_validation_sweep(samples, centroids, keywords, category_config)
        else:
            run_test_confirmation(samples, centroids, keywords, category_config, args.threshold)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
