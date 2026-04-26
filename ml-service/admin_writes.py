"""
admin_writes.py — single source of truth for governed writes to
``token_collisions`` and ``category_keywords``.

Both the CLI (``scripts/apply_collision_change.py``) and the HTTP admin
router (``admin_routes.py``) call into this module so the audit-log
discipline stays in one place. Per CCTR §5c (governance), every change
to either table must be:

  1. Tied to an actor and a free-text reason (ticket optional).
  2. Recorded atomically in ``keyword_audit_log`` (before/after JSONB).
  3. Pinned by a pytest case (caller's responsibility — neither the CLI
     nor the route runs pytest itself; that's the job of
     ``scripts/audit_collisions.py``).

The functions here do NOT manage transactions. Callers pass a
``psycopg2`` connection; ``apply_change()`` runs every statement on that
connection and either commits at the end or leaves the rollback to the
caller's ``with conn:`` context. This makes the route's
"insert-then-reload-in-process" pattern work — if the reload step fails
the caller can roll back the whole thing.

The CLI's previous direct-SQL helpers used to live in
``scripts/apply_collision_change.py`` (~lines 77–147 pre-refactor). They
were lifted here verbatim so CLI behaviour is byte-for-byte unchanged.
"""
from __future__ import annotations

import json
from typing import Any

GOVERNED_TABLES = ("token_collisions", "category_keywords")
GOVERNED_ACTIONS = ("insert", "update", "delete")


# ── Audit-log primary-key shaping ──────────────────────────────────────────────

def row_pk(table: str, row: dict[str, Any] | None) -> str:
    """Compute the human-readable primary-key string we stamp into
    ``keyword_audit_log.row_pk``. ``token_collisions`` uses ``token``;
    ``category_keywords`` uses the composite identity that
    ``uniq_category_keywords_cctr`` enforces."""
    if not row:
        return "<empty>"
    if table == "token_collisions":
        return row.get("token", "<no-token>")
    if table == "category_keywords":
        parts = [
            str(row.get("category_id", "?")),
            str(row.get("hs_chapter", "?")),
            str(row.get("keyword", "?")),
            str(row.get("signal_class", "?")),
        ]
        return ":".join(parts)
    return "<unknown-table>"


# ── Low-level statement helpers ────────────────────────────────────────────────

def _select_row(cur, table: str, where: str) -> dict | None:
    """SELECT the row matching ``where`` and return it as a dict, or
    None if no match. ``where`` is raw SQL (no leading 'WHERE')."""
    cur.execute(f"SELECT row_to_json(t) FROM {table} t WHERE {where}")
    row = cur.fetchone()
    return row[0] if row else None


def _audit(
    cur,
    table: str,
    op: str,
    pk: str,
    before: dict | None,
    after: dict | None,
    actor: str,
    ticket: str | None,
    reason: str,
) -> None:
    """Append one row to ``keyword_audit_log``. The caller is
    responsible for the surrounding transaction."""
    cur.execute(
        """
        INSERT INTO keyword_audit_log
            (table_name, operation, row_pk, before, after, actor, ticket, reason)
        VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s)
        """,
        (
            table, op, pk,
            json.dumps(before) if before is not None else None,
            json.dumps(after)  if after  is not None else None,
            actor, ticket, reason,
        ),
    )


def _apply_insert(cur, table: str, payload: dict) -> dict:
    """Build a parameterised INSERT from the payload; return the
    inserted row as a dict (via ``row_to_json``).

    JSONB-typed columns (``resolution`` on ``token_collisions``) need
    explicit ``json.dumps``; arrays (``home_chapters``) round-trip
    through psycopg2 natively as Python lists.
    """
    cols = list(payload.keys())
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(cols)
    sql = (
        f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
        f"RETURNING row_to_json({table})"
    )
    values: list[Any] = []
    for c in cols:
        v = payload[c]
        if isinstance(v, (dict, list)) and c == "resolution":
            values.append(json.dumps(v))
        else:
            values.append(v)
    cur.execute(sql, values)
    return cur.fetchone()[0]


def _apply_update(
    cur, table: str, where: str, set_clause: str
) -> tuple[dict | None, dict | None]:
    """Run UPDATE; return (before, after) as dicts."""
    before = _select_row(cur, table, where)
    cur.execute(f"UPDATE {table} SET {set_clause} WHERE {where}")
    after = _select_row(cur, table, where)
    return before, after


def _apply_delete(cur, table: str, where: str) -> dict | None:
    """Run DELETE; return the row that was deleted (None if nothing
    matched)."""
    before = _select_row(cur, table, where)
    cur.execute(f"DELETE FROM {table} WHERE {where}")
    return before


# ── Umbrella entry point ───────────────────────────────────────────────────────

def apply_change(
    conn,
    *,
    table: str,
    action: str,
    payload: dict | None = None,
    where: str | None = None,
    set_clause: str | None = None,
    actor: str,
    ticket: str | None = None,
    reason: str,
) -> dict:
    """Execute one governed change AND its audit-log entry on ``conn``.

    Validates inputs, runs the statement, writes the matching
    ``keyword_audit_log`` row, and commits the connection. Returns
    ``{'pk': ..., 'before': ..., 'after': ...}``.

    The caller may instead wrap this in their own ``with conn:`` block
    (no explicit commit) when they need to atomically chain the change
    with follow-up work like an in-process loader reload — pass
    ``conn`` from a context manager and rely on the outer ``with`` to
    commit/rollback. If ``conn.autocommit`` is False we leave the
    commit to the caller in that case; if True the per-statement
    commits already cover us. To keep the simple CLI flow working we
    also call ``conn.commit()`` here when no outer transaction is in
    progress (psycopg2 connections start a transaction lazily, so
    calling commit at the end of a fresh connection is harmless and
    matches the prior CLI behaviour).

    Parameters
    ----------
    table       : 'category_keywords' | 'token_collisions'
    action      : 'insert' | 'update' | 'delete'
    payload     : (insert) row body as a dict
    where       : (update / delete) raw WHERE clause without 'WHERE'
    set_clause  : (update) raw SET clause without 'SET'
    actor       : person or system making the change. Free text.
    ticket      : ticket / PR / change id. Optional.
    reason      : one-line justification. Required.

    Raises
    ------
    ValueError on bad input shape.
    """
    if table not in GOVERNED_TABLES:
        raise ValueError(f"unknown governed table: {table!r}")
    if action not in GOVERNED_ACTIONS:
        raise ValueError(f"unknown action: {action!r}")
    if action == "insert" and not payload:
        raise ValueError("insert requires a payload dict")
    if action in ("update", "delete") and not where:
        raise ValueError(f"{action} requires a where clause")
    if action == "update" and not set_clause:
        raise ValueError("update requires a set clause")
    if not actor:
        raise ValueError("actor is required")
    if not reason:
        raise ValueError("reason is required")

    before: dict | None = None
    after: dict | None = None

    with conn.cursor() as cur:
        if action == "insert":
            after = _apply_insert(cur, table, payload or {})
        elif action == "update":
            before, after = _apply_update(cur, table, where or "", set_clause or "")
        else:  # delete
            before = _apply_delete(cur, table, where or "")

        pk = row_pk(table, after or before)
        _audit(
            cur, table, action, pk,
            before=before, after=after,
            actor=actor, ticket=ticket, reason=reason,
        )

    # End-of-call commit so the simple CLI usage (one fresh connection,
    # one change, one commit) keeps working. Callers that want atomic
    # multi-step flows (e.g. the HTTP route's "write then in-process
    # reload" pattern) should still wrap in their own ``with conn:``;
    # psycopg2 treats a commit-on-already-committed connection as a
    # no-op so the extra call is safe either way.
    conn.commit()
    return {"pk": pk, "before": before, "after": after}
