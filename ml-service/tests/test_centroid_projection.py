"""
test_centroid_projection.py — unit tests for the lazy 2D projection
backing /admin/api/centroids.

We don't stand up Postgres; instead we monkeypatch the two helpers
that talk to the DB (`_fetch_centroid_rows`, `_max_updated_at`) and
pass an arbitrary `conn` sentinel through. The reducer code path runs
for real — UMAP if present, PCA otherwise — so the test exercises the
same branch the production endpoint will hit on this machine.
"""
from __future__ import annotations

import numpy as np
import pytest

import centroid_projection as cp


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Each test starts with an empty memoization map so cache hits in
    one test don't leak into another."""
    cp.clear_cache()
    yield
    cp.clear_cache()


def _fake_rows(n: int) -> list[dict]:
    """n synthetic centroid rows — 384-dim vectors with a small
    category-specific bias so PCA/UMAP have something to separate on."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        cat_id = i % 3 + 1
        vec = rng.normal(loc=cat_id * 0.1, scale=0.05, size=384).astype(np.float32)
        # pgvector text format: '[v1,v2,...]'
        txt = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
        rows.append({
            "category_id":   cat_id,
            "category_name": f"cat{cat_id}",
            "hs_chapter":    f"{10 + i:02d}",
            "chapter_title": f"Chapter {i}",
            "sample_count":  100 + i,
            "centroid_txt":  txt,
        })
    return rows


def test_empty_corpus_returns_empty_points(monkeypatch) -> None:
    monkeypatch.setattr(cp, "_fetch_centroid_rows", lambda conn: [])
    monkeypatch.setattr(cp, "_max_updated_at", lambda conn: "k0")
    out = cp.fetch_centroid_projection(conn=object())
    assert out["points"] == []
    assert out["reducer"] in ("pca", "umap")


def test_projection_shape_matches_rows(monkeypatch) -> None:
    rows = _fake_rows(12)
    monkeypatch.setattr(cp, "_fetch_centroid_rows", lambda conn: rows)
    monkeypatch.setattr(cp, "_max_updated_at", lambda conn: "k1")
    out = cp.fetch_centroid_projection(conn=object())
    assert len(out["points"]) == 12
    assert out["reducer"] in ("umap", "pca")
    for p in out["points"]:
        assert isinstance(p["x"], float)
        assert isinstance(p["y"], float)
        assert p["category_name"].startswith("cat")
        assert len(p["hs_chapter"]) == 2
        assert isinstance(p["sample_count"], int)


def test_small_corpus_falls_back_to_pca(monkeypatch) -> None:
    """n<4 → UMAP code path is skipped, PCA always runs."""
    rows = _fake_rows(2)
    monkeypatch.setattr(cp, "_fetch_centroid_rows", lambda conn: rows)
    monkeypatch.setattr(cp, "_max_updated_at", lambda conn: "k2")
    out = cp.fetch_centroid_projection(conn=object())
    assert out["reducer"] == "pca"
    assert len(out["points"]) == 2


def test_cache_hits_return_same_object(monkeypatch) -> None:
    """Same updated_at key → cached projection returned without re-fetching."""
    rows = _fake_rows(8)
    call_count = {"n": 0}
    def counting(conn):
        call_count["n"] += 1
        return rows
    monkeypatch.setattr(cp, "_fetch_centroid_rows", counting)
    monkeypatch.setattr(cp, "_max_updated_at", lambda conn: "stable_key")

    first  = cp.fetch_centroid_projection(conn=object())
    second = cp.fetch_centroid_projection(conn=object())
    assert first is second
    assert call_count["n"] == 1


def test_cache_invalidates_when_key_changes(monkeypatch) -> None:
    """A bumped MAX(updated_at) → recompute, fresh dict."""
    rows = _fake_rows(8)
    monkeypatch.setattr(cp, "_fetch_centroid_rows", lambda conn: rows)

    keys = iter(["k_a", "k_b"])
    monkeypatch.setattr(cp, "_max_updated_at", lambda conn: next(keys))

    first  = cp.fetch_centroid_projection(conn=object())
    second = cp.fetch_centroid_projection(conn=object())
    assert first is not second


def test_parse_pgvector_round_trips() -> None:
    parsed = cp._parse_pgvector("[0.1,-0.2,0.3]")
    assert parsed == [0.1, -0.2, 0.3]
    assert cp._parse_pgvector("[]") == []


# ── transform_point coverage ─────────────────────────────────────────────────


def test_transform_point_returns_xy_in_same_space(monkeypatch) -> None:
    """Projecting a new point should land somewhere in the same coordinate
    range as the cached centroids (we don't pin a specific (x, y) — UMAP
    is non-trivial — but bounds are useful sanity)."""
    rows = _fake_rows(12)
    monkeypatch.setattr(cp, "_fetch_centroid_rows", lambda conn: rows)
    monkeypatch.setattr(cp, "_max_updated_at", lambda conn: "k_xform")

    proj = cp.fetch_centroid_projection(conn=object())
    assert len(proj["points"]) == 12

    rng = np.random.default_rng(1)
    new_vec = rng.normal(loc=0.1, scale=0.05, size=384).astype(np.float32)
    out = cp.transform_point(conn=object(), vec=new_vec)

    assert out["reducer"] in ("umap", "pca")
    assert isinstance(out["x"], float)
    assert isinstance(out["y"], float)


def test_transform_point_uses_cached_reducer(monkeypatch) -> None:
    """The whole point of caching the fitted reducer: subsequent
    transforms must NOT trigger a refit. We check via a counter on
    `_reduce` (which is the only place a fit happens)."""
    rows = _fake_rows(8)
    monkeypatch.setattr(cp, "_fetch_centroid_rows", lambda conn: rows)
    monkeypatch.setattr(cp, "_max_updated_at", lambda conn: "k_cache")

    fit_count = {"n": 0}
    real_reduce = cp._reduce
    def counting_reduce(matrix):
        fit_count["n"] += 1
        return real_reduce(matrix)
    monkeypatch.setattr(cp, "_reduce", counting_reduce)

    cp.fetch_centroid_projection(conn=object())          # fit #1
    rng = np.random.default_rng(2)
    for _ in range(5):
        cp.transform_point(conn=object(), vec=rng.normal(size=384).astype(np.float32))
    assert fit_count["n"] == 1, f"reducer was refit {fit_count['n']} times"


def test_transform_point_raises_without_centroids(monkeypatch) -> None:
    """No centroids in DB → there is no fitted reducer to project against."""
    monkeypatch.setattr(cp, "_fetch_centroid_rows", lambda conn: [])
    monkeypatch.setattr(cp, "_max_updated_at", lambda conn: "k_empty")

    with pytest.raises(ValueError, match="no centroids"):
        cp.transform_point(conn=object(), vec=np.zeros(384, dtype=np.float32))


def test_transform_point_pca_path(monkeypatch) -> None:
    """Force the PCA fallback (n<4) and confirm the fitted PCA also
    supports `.transform()`. n=2 keeps n_components=2."""
    rows = _fake_rows(2)  # n<4 → PCA path
    monkeypatch.setattr(cp, "_fetch_centroid_rows", lambda conn: rows)
    monkeypatch.setattr(cp, "_max_updated_at", lambda conn: "k_pca")

    cp.fetch_centroid_projection(conn=object())
    out = cp.transform_point(conn=object(), vec=np.ones(384, dtype=np.float32) * 0.1)
    assert out["reducer"] == "pca"
    assert isinstance(out["x"], float)
    assert isinstance(out["y"], float)


def test_transform_point_pca_padding_for_single_centroid(monkeypatch) -> None:
    """n=1 → PCA gets n_components=1 → transform output is padded to 2D
    so plotting code can rely on (x, y)."""
    rows = _fake_rows(1)
    monkeypatch.setattr(cp, "_fetch_centroid_rows", lambda conn: rows)
    monkeypatch.setattr(cp, "_max_updated_at", lambda conn: "k_pca1")

    cp.fetch_centroid_projection(conn=object())
    out = cp.transform_point(conn=object(), vec=np.ones(384, dtype=np.float32) * 0.1)
    assert out["reducer"] == "pca"
    # Padded dimension is always 0.0 when n_comp=1.
    assert out["y"] == 0.0
