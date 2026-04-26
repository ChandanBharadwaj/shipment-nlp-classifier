"""
apply_collision_change.py — the only sanctioned write path to
token_collisions and category_keywords.

Per the CCTR plan §5c (governance), every change to either table must
be:
  1. Tied to an owner and ticket / justification.
  2. Recorded in keyword_audit_log (before/after JSONB, actor, reason).
  3. Pinned by a pytest case (caller's responsibility — the script
     verifies a name was supplied but doesn't run pytest itself; that's
     audit_collisions.py's job).

Both this CLI and the FastAPI admin router (``ml-service/admin_routes.py``)
delegate to ``ml-service/admin_writes.py`` so the audit-log discipline
lives in one place. CLI behaviour (args, output, exit codes) is
unchanged from the pre-refactor version.

Direct SQL on the governed tables still works (we don't add a trigger
because the audit log is intentionally an *additional* signal, not a
security boundary), but every change merged via this script — or the
admin UI — is automatically auditable.

Usage examples:

    # Add a new collision row.
    python -m scripts.apply_collision_change \
        --table token_collisions \
        --action insert \
        --json '{"token":"toy soldier","home_chapters":["95","93"], ...}' \
        --actor jane.doe \
        --ticket CCTR-37 \
        --reason "discover_collisions surfaced new candidate"

    # Update an existing keyword row's weight.
    python -m scripts.apply_collision_change \
        --table category_keywords \
        --action update \
        --where 'category_id=17 AND hs_chapter=''95'' AND keyword=''battery'' AND signal_class=''signal''' \
        --set 'weight=0.35' \
        --actor jane.doe \
        --ticket CCTR-42 \
        --reason "rebalancer recommended; manual override after triage"

    # Delete a row (rare; usually only a typo cleanup).
    python -m scripts.apply_collision_change \
        --table token_collisions \
        --action delete \
        --where 'token=''typo_word''' \
        --actor jane.doe \
        --ticket CCTR-99 \
        --reason "duplicate of correct row created in CCTR-98"

Notes
-----
- All changes run inside a single transaction with the audit log row.
  If either fails, both roll back.
- --where uses raw SQL because the legitimate change set is small
  enough to read carefully in review. The script does NOT sanitize
  against injection — callers must be trusted (this is a governance
  tool, not a public endpoint).
- For inserts, --json is the row body. The audit-log primary-key string
  is computed from the row body automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The shared write helpers live in ``ml-service/admin_writes.py``. The
# folder name has a hyphen so it can't be imported as a package name —
# put it on sys.path the same way diagnostics/* do.
_ML_SERVICE = Path(__file__).resolve().parent.parent / "ml-service"
if str(_ML_SERVICE) not in sys.path:
    sys.path.insert(0, str(_ML_SERVICE))

from admin_writes import GOVERNED_TABLES, apply_change  # noqa: E402


def _connect(url: str):
    import psycopg2
    return psycopg2.connect(url)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=os.environ.get("DATABASE_URL"))
    ap.add_argument("--table", required=True, choices=GOVERNED_TABLES)
    ap.add_argument("--action", required=True, choices=("insert", "update", "delete"))
    ap.add_argument("--actor", required=True,
                    help="Person or system making the change. Free text; "
                         "convention is 'firstname.lastname' or 'cron:job-name'.")
    ap.add_argument("--ticket", default=None,
                    help="Ticket / PR / change id. Required by CI for "
                         "non-emergency changes.")
    ap.add_argument("--reason", required=True,
                    help="One-line justification recorded in keyword_audit_log.reason.")

    # Per-action body
    ap.add_argument("--json", default=None,
                    help="(insert) JSON body of the new row.")
    ap.add_argument("--where", default=None,
                    help="(update/delete) raw WHERE clause without 'WHERE'.")
    ap.add_argument("--set", dest="set_clause", default=None,
                    help="(update) raw SET clause without 'SET'.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would run; don't write.")
    args = ap.parse_args()

    if not args.db:
        print("[error] --db or DATABASE_URL required", file=sys.stderr)
        return 2

    if args.action == "insert" and not args.json:
        print("[error] --json required for insert", file=sys.stderr); return 2
    if args.action in ("update", "delete") and not args.where:
        print("[error] --where required for update/delete", file=sys.stderr); return 2
    if args.action == "update" and not args.set_clause:
        print("[error] --set required for update", file=sys.stderr); return 2

    if args.dry_run:
        print(f"[dry-run] table={args.table} action={args.action} actor={args.actor} "
              f"ticket={args.ticket} reason={args.reason!r}")
        if args.json:    print(f"[dry-run] body: {args.json}")
        if args.where:   print(f"[dry-run] where: {args.where}")
        if args.set_clause: print(f"[dry-run] set: {args.set_clause}")
        return 0

    payload = json.loads(args.json) if args.json else None

    conn = _connect(args.db)
    try:
        result = apply_change(
            conn,
            table=args.table,
            action=args.action,
            payload=payload,
            where=args.where,
            set_clause=args.set_clause,
            actor=args.actor,
            ticket=args.ticket,
            reason=args.reason,
        )
        print(
            f"[ok] {args.action} on {args.table} (pk={result['pk']}) "
            f"by {args.actor} ticket={args.ticket}"
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
