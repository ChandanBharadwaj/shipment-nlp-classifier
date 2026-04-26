"""
Property P1 verification — no keyword is removed by the rebalancer.

The CCTR plan's Property P1 says: "every keyword stays attached to its
right HS chapter." The mechanism is the rebalancer
(diagnostics.rebalance_keyword_weights). This file verifies the
mechanism in isolation, plus a couple of related anchor/modifier
stickiness invariants.

The tests don't touch the database — they exercise the rebalancer's
pure-Python `_new_weight` formula and the load/loop logic against
in-memory fixtures. The DB-side verification (full corpus row-count
audit) lives in the build pipeline's smoke run; this is the per-PR gate.
"""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from diagnostics.rebalance_keyword_weights import _new_weight  # noqa: E402
from config import CCTR_WEIGHT_CAP, CCTR_WEIGHT_FLOOR  # noqa: E402


# ── Bounds & monotonicity ────────────────────────────────────────────────────

def test_new_weight_clamped_to_floor() -> None:
    """Even at cosine=0 (totally orthogonal), weight never drops below the floor."""
    w = _new_weight(0.10, cosine=0.0, floor=CCTR_WEIGHT_FLOOR, cap=CCTR_WEIGHT_CAP)
    assert w >= CCTR_WEIGHT_FLOOR
    assert w <= CCTR_WEIGHT_CAP


def test_new_weight_clamped_to_cap() -> None:
    """Even at cosine=1 (perfect alignment), weight never exceeds the cap."""
    w = _new_weight(1.00, cosine=1.0, floor=CCTR_WEIGHT_FLOOR, cap=CCTR_WEIGHT_CAP)
    assert w <= CCTR_WEIGHT_CAP
    assert w >= CCTR_WEIGHT_FLOOR


def test_new_weight_neutral_cosine_is_close_to_old() -> None:
    """Cosine 0.5 is the neutral point — the formula `(cosine-0.5)*0.5` is 0 there,
    so new_weight ≈ old_weight (modulo clamping)."""
    w = _new_weight(0.50, cosine=0.5, floor=CCTR_WEIGHT_FLOOR, cap=CCTR_WEIGHT_CAP)
    assert w == pytest.approx(0.50, abs=1e-6)


def test_new_weight_monotonic_in_cosine() -> None:
    """For a fixed old_weight inside the bounds, new_weight is monotonically
    non-decreasing in cosine. This is what makes the pull-toward-bounds
    behavior intuitive — higher alignment cannot lower the weight."""
    old = 0.5
    prev = _new_weight(old, cosine=0.0, floor=CCTR_WEIGHT_FLOOR, cap=CCTR_WEIGHT_CAP)
    for cos in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        cur = _new_weight(old, cosine=cos, floor=CCTR_WEIGHT_FLOOR, cap=CCTR_WEIGHT_CAP)
        assert cur >= prev - 1e-9, f"non-monotonic at cosine={cos}: {prev:.4f} → {cur:.4f}"
        prev = cur


# ── P1 — no row deletion in default mode ─────────────────────────────────────
#
# This test substitutes a fake DB connection layer to drive the rebalancer
# end-to-end without Postgres. We verify two things:
#   1. After a default-mode run, the row set is exactly the input row set
#      (same primary keys; only weights change).
#   2. Rows with signal_class IN ('anchor', 'modifier') have unchanged
#      weights — the stickiness rule.

class _FakeCursor:
    """Captures executemany() so the test can assert which rows were touched."""
    def __init__(self) -> None:
        self.executions: list[tuple[str, list]] = []
    def execute(self, sql, params=()):
        self.executions.append(("execute", sql, list(params)))
    def executemany(self, sql, rows):
        self.executions.append(("executemany", sql, list(rows)))
    def __enter__(self): return self
    def __exit__(self, *a): pass


class _FakeConn:
    def __init__(self) -> None:
        self.cursor_obj = _FakeCursor()
        self.committed = False
    def cursor(self): return self.cursor_obj
    def commit(self): self.committed = True


def test_p1_no_rows_deleted_in_default_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the rebalancer in default (non-strict) mode against a tiny fixture
    and assert that no DELETE statements were issued."""
    import numpy as np
    from diagnostics import rebalance_keyword_weights as rb

    # Fixture — three rows under "toys", one of them an anchor.
    fixture_rows = [
        # (id, category, hs, kw, weight, signal_class)
        (1, "toys", "95", "doll",          1.0,  "anchor"),       # sticky
        (2, "toys", "95", "battery",       0.5,  "signal"),       # rebalance candidate
        (3, "toys", "95", "toy",           1.0,  "modifier"),     # sticky
    ]

    # Centroid for (toys, 95) — direction matters less than that the dot
    # product produces a believable cosine; we'll pin it via the embed mock.
    fake_centroid_vec = np.array([1.0, 0.0, 0.0])
    fake_centroids = {"toys": [("95", fake_centroid_vec)]}

    # Stub the DB-touching helpers so no Postgres is required.
    monkeypatch.setattr(rb, "_load_keyword_rows", lambda conn, category_filter=None: fixture_rows)
    monkeypatch.setattr(rb, "load_centroids", lambda conn: fake_centroids)
    # embed_texts returns one row per keyword with cosine 0.5 to the centroid
    # (so non-sticky weights stay roughly the same).
    def fake_embed(texts):
        # Each keyword's vector has [0.5, ?, ?] so dot with [1,0,0] = 0.5.
        return np.array([[0.5, 0.5, np.sqrt(0.5)] for _ in texts])
    monkeypatch.setattr(rb, "embed_texts", fake_embed)

    # Capture the connection used in --apply path.
    captured: dict = {}
    class _CtxConn:
        def __enter__(self_inner):
            captured["conn"] = _FakeConn()
            return captured["conn"]
        def __exit__(self_inner, *a): pass
    monkeypatch.setattr(rb, "pooled_connection", lambda: _CtxConn())

    # Run with --apply so we can assert what statements the rebalancer issues.
    rb.run(apply=True, strict=False)

    conn = captured["conn"]
    sql_kinds = [op[0] for op in conn.cursor_obj.executions]
    # Default mode: no SELECT/DELETE, only UPDATEs.
    for kind, sql, *_rest in conn.cursor_obj.executions:
        if kind == "executemany":
            assert "DELETE" not in sql.upper(), f"P1 violation: rebalancer issued DELETE: {sql!r}"
            assert "UPDATE" in sql.upper(), f"unexpected statement: {sql!r}"


def test_p1_anchors_and_modifiers_stay_sticky(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even when their cosine is technically rebalance-eligible, anchor and
    modifier rows must not be touched by the default-mode rebalancer."""
    import numpy as np
    from diagnostics import rebalance_keyword_weights as rb

    fixture_rows = [
        (10, "toys", "95", "anchor_kw",   1.0, "anchor"),
        (11, "toys", "95", "modifier_kw", 1.0, "modifier"),
        (12, "toys", "95", "signal_kw",   0.5, "signal"),
    ]
    fake_centroids = {"toys": [("95", np.array([1.0, 0.0, 0.0]))]}
    monkeypatch.setattr(rb, "_load_keyword_rows", lambda conn, category_filter=None: fixture_rows)
    monkeypatch.setattr(rb, "load_centroids", lambda conn: fake_centroids)
    monkeypatch.setattr(rb, "embed_texts",
                        lambda texts: np.array([[0.9, 0.1, 0.0] for _ in texts]))  # cosine ~0.9

    captured = {}
    class _CtxConn:
        def __enter__(self_inner):
            captured["conn"] = _FakeConn()
            return captured["conn"]
        def __exit__(self_inner, *a): pass
    monkeypatch.setattr(rb, "pooled_connection", lambda: _CtxConn())

    rb.run(apply=True, strict=False)

    conn = captured["conn"]
    updates_seen: list[int] = []
    for kind, sql, params in conn.cursor_obj.executions:
        if kind == "executemany" and "UPDATE" in sql.upper():
            for new_w, row_id in params:
                updates_seen.append(row_id)

    assert 10 not in updates_seen, "anchor row was rebalanced — stickiness broken"
    assert 11 not in updates_seen, "modifier row was rebalanced — stickiness broken"
    # Signal row (id=12) MAY have been updated; we don't pin its presence here
    # because the dampened formula at cosine 0.9 may or may not move it past
    # the 1e-4 epsilon threshold. The point of THIS test is the sticky rule.


# ── Strict mode parity (legacy rollback) ─────────────────────────────────────

def test_strict_mode_can_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators retain a path to legacy delete behaviour for emergency
    rollback. The fact that --strict CAN delete is the point — but it's the
    only path that does, and it's never used in CI."""
    import numpy as np
    from diagnostics import rebalance_keyword_weights as rb

    fixture_rows = [
        (20, "toys", "95", "battery", 0.5, "signal"),
    ]
    monkeypatch.setattr(rb, "_load_keyword_rows", lambda conn, category_filter=None: fixture_rows)
    monkeypatch.setattr(rb, "load_centroids",
                        lambda conn: {"toys": [("95", np.array([1.0, 0.0, 0.0]))]})
    # cosine 0.0 — well below MIN_KEYWORD_COSINE_DEFAULT (0.30) so --strict deletes.
    monkeypatch.setattr(rb, "embed_texts",
                        lambda texts: np.array([[0.0, 1.0, 0.0] for _ in texts]))

    captured = {}
    class _CtxConn:
        def __enter__(self_inner):
            captured["conn"] = _FakeConn()
            return captured["conn"]
        def __exit__(self_inner, *a): pass
    monkeypatch.setattr(rb, "pooled_connection", lambda: _CtxConn())

    rb.run(apply=True, strict=True)

    conn = captured["conn"]
    delete_seen = any(
        kind == "executemany" and "DELETE" in sql.upper()
        for kind, sql, *_ in conn.cursor_obj.executions
    )
    assert delete_seen, "--strict mode should delete cosine<min_cosine single-token rows"
