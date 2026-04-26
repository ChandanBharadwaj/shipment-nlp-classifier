"""
test_admin_routes.py — Commit 2 verification for the /admin/api router.

Tests use FastAPI's TestClient against a stripped-down app that mounts
*only* the admin router, so the global startup lifespan in main.py
(which loads centroids, keywords, the cross-encoder, etc.) doesn't
run. Each test monkeypatches the DB-shaped helper functions directly,
so no Postgres is required.

The boundary we care about here is "router behaves like a router" —
query-param plumbing, response shape, validation 422s, 404s. The SQL
helpers themselves are exercised against a real DB by the integration
suite that runs against docker-compose.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import admin_routes


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A fresh FastAPI app per test with the admin router mounted and
    pooled_connection stubbed to a sentinel — every helper is
    monkeypatched per-test so the sentinel never reaches real psycopg2."""
    app = FastAPI()
    app.include_router(admin_routes.router)

    class _SentinelConn:
        """If a test forgets to patch a helper, this surfaces the gap
        as a clear AttributeError instead of an opaque DB connection
        failure."""
        def __getattr__(self, name):
            raise AssertionError(
                f"unpatched helper reached pooled_connection.{name} — "
                "test must monkeypatch admin_routes.fetch_*"
            )

    class _CtxConn:
        def __enter__(self):  return _SentinelConn()
        def __exit__(self, *a): return False

    monkeypatch.setattr(admin_routes, "pooled_connection", lambda: _CtxConn())
    return TestClient(app)


# ── /overview ─────────────────────────────────────────────────────────────────

def test_overview_returns_expected_shape(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(admin_routes, "fetch_overview", lambda conn: {
        "total_keywords":   2_400,
        "total_collisions": 12,
        "last_audit_ts":    "2026-04-25T10:00:00+00:00",
        "counts": {
            "by_class":    [{"signal_class": "anchor", "n": 50}],
            "by_category": [{"category": "toys", "n": 200}],
            "by_chapter":  [{"hs_chapter": "95", "n": 200}],
        },
        "top_polysemous": [{"keyword": "battery", "n_chapters": 4, "chapters": ["30","85","87","95"]}],
    })
    r = client.get("/admin/api/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["total_keywords"] == 2_400
    assert body["total_collisions"] == 12
    assert body["counts"]["by_class"][0]["signal_class"] == "anchor"
    assert body["top_polysemous"][0]["keyword"] == "battery"


# ── /categories ───────────────────────────────────────────────────────────────

def test_categories_returns_chapters_per_category(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(admin_routes, "fetch_categories", lambda conn: [{
        "id": 1, "name": "toys", "display_name": "Toys",
        "semantic_weight": 0.8, "keyword_weight": 0.2, "threshold": 0.45,
        "chapters": [{"hs_chapter": "95", "chapter_title": "Toys, games...",
                      "is_primary": True, "keyword_count": 200}],
    }])
    r = client.get("/admin/api/categories")
    assert r.status_code == 200
    cats = r.json()
    assert len(cats) == 1
    assert cats[0]["chapters"][0]["hs_chapter"] == "95"


# ── /keywords ─────────────────────────────────────────────────────────────────

def test_keywords_passes_filters_through(client: TestClient, monkeypatch) -> None:
    seen: dict = {}
    def fake(conn, category, chapter, signal_class, search, limit, offset):
        seen.update(category=category, chapter=chapter, signal_class=signal_class,
                    search=search, limit=limit, offset=offset)
        return {"rows": [], "total": 0, "limit": limit, "offset": offset}
    monkeypatch.setattr(admin_routes, "fetch_keywords", fake)

    r = client.get("/admin/api/keywords",
                   params={"category": "toys", "chapter": "95",
                           "signal_class": "anchor", "search": "doll",
                           "limit": 50, "offset": 10})
    assert r.status_code == 200
    assert seen == {"category": "toys", "chapter": "95",
                    "signal_class": "anchor", "search": "doll",
                    "limit": 50, "offset": 10}


def test_keywords_rejects_bad_signal_class(client: TestClient) -> None:
    r = client.get("/admin/api/keywords", params={"signal_class": "banana"})
    assert r.status_code == 422


def test_keywords_clamps_limit(client: TestClient) -> None:
    r = client.get("/admin/api/keywords", params={"limit": 0})
    assert r.status_code == 422  # ge=1
    r = client.get("/admin/api/keywords", params={"limit": 10_000})
    assert r.status_code == 422  # le=1000


# ── /keywords/by-token/{token} ────────────────────────────────────────────────

def test_keyword_by_token_returns_polysemy_view(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(admin_routes, "fetch_keyword_by_token",
                        lambda conn, token: {"token": token, "rows": [
                            {"category_name": "electronics", "hs_chapter": "85",
                             "keyword": token, "weight": 1.0, "signal_class": "signal",
                             "id": 1, "category_id": 1, "source": "manual",
                             "notes": None, "created_at": None},
                            {"category_name": "automotive",  "hs_chapter": "87",
                             "keyword": token, "weight": 0.7, "signal_class": "signal",
                             "id": 2, "category_id": 2, "source": "manual",
                             "notes": None, "created_at": None},
                        ]})
    r = client.get("/admin/api/keywords/by-token/battery")
    assert r.status_code == 200
    body = r.json()
    assert body["token"] == "battery"
    assert {row["hs_chapter"] for row in body["rows"]} == {"85", "87"}


# ── /collisions ───────────────────────────────────────────────────────────────

def test_collisions_returns_registry(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(admin_routes, "fetch_collisions", lambda conn: [{
        "id": 1, "token": "gun", "home_chapters": ["93", "95"],
        "risk_tier": "high", "resolution": [{"if": "modifier_hit:toy", "then": "force_chapter:95"}],
        "owner": "jane", "test_case": "test_gun_toy_modifier",
        "notes": None, "created_at": None,
    }])
    r = client.get("/admin/api/collisions")
    assert r.status_code == 200
    rows = r.json()
    assert rows[0]["token"] == "gun"
    assert rows[0]["risk_tier"] == "high"


def test_collision_404_for_unknown_token(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(admin_routes, "fetch_collision_by_token", lambda conn, token: None)
    r = client.get("/admin/api/collisions/nonexistent")
    assert r.status_code == 404


def test_collision_returns_one(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(admin_routes, "fetch_collision_by_token",
                        lambda conn, token: {"token": "battery", "risk_tier": "low",
                                             "home_chapters": ["85","87","95","30"],
                                             "resolution": [], "owner": None,
                                             "test_case": None, "notes": None,
                                             "created_at": None, "id": 1})
    r = client.get("/admin/api/collisions/battery")
    assert r.status_code == 200
    assert r.json()["risk_tier"] == "low"


# ── /audit-log ────────────────────────────────────────────────────────────────

def test_audit_log_filters_passed_through(client: TestClient, monkeypatch) -> None:
    seen: dict = {}
    def fake(conn, table, actor, operation, limit, offset):
        seen.update(table=table, actor=actor, operation=operation,
                    limit=limit, offset=offset)
        return {"rows": [], "total": 0, "limit": limit, "offset": offset}
    monkeypatch.setattr(admin_routes, "fetch_audit_log", fake)

    r = client.get("/admin/api/audit-log",
                   params={"table": "category_keywords", "actor": "jane",
                           "operation": "insert", "limit": 25, "offset": 5})
    assert r.status_code == 200
    assert seen == {"table": "category_keywords", "actor": "jane",
                    "operation": "insert", "limit": 25, "offset": 5}


def test_audit_log_rejects_bad_table(client: TestClient) -> None:
    r = client.get("/admin/api/audit-log", params={"table": "shipments"})
    assert r.status_code == 422


def test_audit_log_rejects_bad_operation(client: TestClient) -> None:
    r = client.get("/admin/api/audit-log", params={"operation": "drop"})
    assert r.status_code == 422


# ── /discover ─────────────────────────────────────────────────────────────────

def test_discover_returns_candidates(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(admin_routes, "run_discover", lambda conn, min_chapters, min_weight: {
        "candidates": [
            {"keyword": "core", "n_chapters": 4, "home_chapters": "28,85,87,95",
             "categories": "chemicals,electronics,automotive,toys",
             "max_weight": "0.50", "n_rows": 4, "suggested_tier": "low"},
        ],
        "n_already_registered": 12,
        "min_chapters": min_chapters,
        "min_weight": min_weight,
    })
    r = client.post("/admin/api/discover", params={"min_chapters": 3, "min_weight": 0.3})
    assert r.status_code == 200
    body = r.json()
    assert body["candidates"][0]["keyword"] == "core"
    assert body["min_chapters"] == 3
    assert body["min_weight"] == 0.3


def test_discover_param_bounds(client: TestClient) -> None:
    r = client.post("/admin/api/discover", params={"min_chapters": 1})
    assert r.status_code == 422  # ge=2
    r = client.post("/admin/api/discover", params={"min_weight": 1.5})
    assert r.status_code == 422  # le=1.0


# ── Sanity: helpers are exposed at module level for monkeypatching ────────────

def test_helpers_are_module_attributes() -> None:
    """Every test above patches admin_routes.fetch_*; if a helper got
    inlined into a route handler this catches it before the real
    suite runs."""
    for name in (
        "fetch_overview", "fetch_categories", "fetch_keywords",
        "fetch_keyword_by_token", "fetch_collisions",
        "fetch_collision_by_token", "fetch_audit_log", "run_discover",
    ):
        assert hasattr(admin_routes, name), f"{name} missing"
        assert callable(getattr(admin_routes, name))
