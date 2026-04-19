"""
2D calibration: grid-search (threshold × margin_delta), report F1 heatmap.

Why a dedicated script instead of piping evaluate_threshold.py in a loop:
every ``predict()`` call embeds the sample text, which dominates wall time.
Threshold and margin_delta are post-embedding decisions, so we can embed
each sample once, cache the per-category final_scores, then sweep the grid
by re-applying only the decision logic. A 9×7 = 63-cell sweep runs in the
time of one full evaluation.

Usage:
    python calibrate.py                          # validation split, default grid
    python calibrate.py --split test             # (confirmation only — DO NOT tune)
    python calibrate.py --csv labeled.csv        # custom labeled CSV
    python calibrate.py --thresholds 0.40 0.45 0.50 --deltas 0.04 0.06 0.08 0.10

The output is a compact F1 heatmap plus the top-k (threshold, delta) pairs
by F1, with precision/recall alongside each.

Scope:
- This script calibrates the TWO runtime knobs (threshold, margin_delta).
- KEYWORD_SATURATION_ALPHA requires re-running keyword_score per sample, so
  sweeping it would defeat the cache. If you want to re-tune alpha, change
  it in config.py (or via env) and rerun calibrate.py — cheap enough.
- MIN_KEYWORD_COSINE is offline (prune_keywords.py), not in scope here.
- Cross-encoder is not engaged during calibration — the reranker operates
  AFTER the margin rule has already chosen candidates, so calibrating the
  margin rule without it is the principled default. Use evaluate_threshold.py
  with RERANKER_ENABLED=true to measure the uplift.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from classifier import (
    load_category_config,
    load_centroids,
    load_chapter_titles,
    load_keywords,
    predict,
)
from config import MARGIN_DELTA_DEFAULT, UNCLASSIFIED_THRESHOLD_DEFAULT
from db import get_connection
from evaluate_threshold import load_from_csv, load_from_db

DEFAULT_THRESHOLDS = [0.35, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55]
DEFAULT_DELTAS     = [0.02, 0.04, 0.06, 0.08, 0.10, 0.15]


def _build_cache(
    samples: list[dict], centroids: dict, keywords: dict,
    category_config: dict, chapter_titles: dict,
) -> list[tuple[set, dict]]:
    """For each sample, run predict once at a permissive threshold so that
    every category's final_score is reported, then keep only
    ``(expected_labels, {cat: final_score})`` — we discard everything else.

    We pass a very low threshold (0.0) and MARGIN_DELTA_DEFAULT=1.0 so that
    nothing is filtered during cache-build; the filtering is what we're
    sweeping over.
    """
    cache: list[tuple[set, dict]] = []
    for i, sample in enumerate(samples, 1):
        if i % 25 == 0 or i == len(samples):
            print(f"  embedding {i}/{len(samples)}…", file=sys.stderr)
        r = predict(
            sample["cargo"],
            sample["commodity"],
            centroids,
            keywords,
            category_config,
            threshold=0.0,
            unclassified_threshold=0.0,
            chapter_titles=chapter_titles,
            margin_delta=1.0,
        )
        flat = {cat: s["final_score"] for cat, s in (r.get("scores") or {}).items()}
        cache.append((sample["expected"], flat))
    return cache


def _score_grid_cell(
    cache: list[tuple[set, dict]],
    threshold: float,
    margin_delta: float,
    unclassified_threshold: float,
) -> tuple[float, float, float, int, int, int]:
    """Apply (threshold, margin_delta) decision logic to cached scores;
    return (precision, recall, f1, tp, fp, fn)."""
    unc = min(unclassified_threshold, threshold)
    total_tp = total_fp = total_fn = 0
    for expected, scores in cache:
        if not scores:
            predicted: set = set()
        else:
            max_score = max(scores.values())
            if max_score < unc:
                predicted = set()
            elif max_score < threshold:
                top = max(scores, key=lambda c: scores[c])
                predicted = {top}
            else:
                cutoff = max_score - margin_delta
                predicted = {
                    c for c, v in scores.items()
                    if v >= threshold and v >= cutoff
                }
        total_tp += len(predicted & expected)
        total_fp += len(predicted - expected)
        total_fn += len(expected  - predicted)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    return precision, recall, f1, total_tp, total_fp, total_fn


def _print_heatmap(
    thresholds: list[float], deltas: list[float],
    f1_grid: dict[tuple[float, float], float],
) -> None:
    print()
    print(f"{'F1 heatmap':<12}  " + "  ".join(f"δ={d:<5.2f}" for d in deltas))
    print("-" * (12 + 9 * len(deltas) + 4))
    for t in thresholds:
        cells = [f"{f1_grid[(t, d)]:>6.3f}" for d in deltas]
        print(f"thr={t:<7.2f}   " + "   ".join(cells))


def run(
    samples: list[dict], centroids: dict, keywords: dict, category_config: dict,
    chapter_titles: dict, thresholds: list[float], deltas: list[float],
    unclassified_threshold: float, top_k: int,
) -> None:
    print(f"Building score cache for {len(samples)} samples…", file=sys.stderr)
    cache = _build_cache(samples, centroids, keywords, category_config, chapter_titles)

    f1_grid: dict[tuple[float, float], float] = {}
    cells = []
    for t in thresholds:
        for d in deltas:
            p, r, f1, tp, fp, fn = _score_grid_cell(cache, t, d, unclassified_threshold)
            f1_grid[(t, d)] = f1
            cells.append({
                "threshold": t, "delta": d,
                "precision": p, "recall": r, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn,
            })

    _print_heatmap(thresholds, deltas, f1_grid)

    top = sorted(cells, key=lambda c: -c["f1"])[:top_k]
    print(f"\nTop {top_k} cells by F1:")
    print(f"{'#':>3}  {'threshold':>9}  {'delta':>6}  {'precision':>10}  "
          f"{'recall':>8}  {'f1':>8}  {'tp':>5}  {'fp':>5}  {'fn':>5}")
    print("-" * 74)
    for i, c in enumerate(top, 1):
        print(f"{i:>3}  {c['threshold']:>9.3f}  {c['delta']:>6.3f}  "
              f"{c['precision']:>10.4f}  {c['recall']:>8.4f}  {c['f1']:>8.4f}  "
              f"{c['tp']:>5}  {c['fp']:>5}  {c['fn']:>5}")
    best = top[0]
    print(f"\nRecommended: CLASSIFY_THRESHOLD={best['threshold']:.2f}  "
          f"MARGIN_DELTA={best['delta']:.2f}")
    print("Set those in config.py (or via env) and rerun on the test split to confirm:")
    print(f"  python evaluate_threshold.py --split test "
          f"--threshold {best['threshold']:.2f} --margin-delta {best['delta']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--split", choices=["validation", "test"], default="validation")
    source.add_argument("--csv", metavar="PATH")
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--deltas",     type=float, nargs="+", default=DEFAULT_DELTAS)
    ap.add_argument("--unclassified-threshold", type=float,
                    default=UNCLASSIFIED_THRESHOLD_DEFAULT,
                    help="Kept constant during the sweep — rarely worth tuning.")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    if args.split == "test":
        print("WARNING: tuning on the test split leaks information. "
              "Prefer --split validation.", file=sys.stderr)

    conn = None
    try:
        conn = get_connection()
        centroids       = load_centroids(conn)
        keywords        = load_keywords(conn)
        category_config = load_category_config(conn)
        chapter_titles  = load_chapter_titles(conn)

        if args.csv:
            samples = load_from_csv(args.csv)
        else:
            samples = load_from_db(conn, args.split)

        print(f"Calibrating on {len(samples)} samples "
              f"({len(args.thresholds)}×{len(args.deltas)} grid = "
              f"{len(args.thresholds) * len(args.deltas)} cells)\n")
        print("Default baseline in config.py: "
              f"threshold=0.45, margin_delta={MARGIN_DELTA_DEFAULT}")

        run(samples, centroids, keywords, category_config, chapter_titles,
            args.thresholds, args.deltas, args.unclassified_threshold, args.top_k)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
