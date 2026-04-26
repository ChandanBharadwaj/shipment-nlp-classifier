"""
audit_collisions.py — CCTR governance CI gate.

For every row in `token_collisions`, parse its `test_case` column (which
names a pytest function in ml-service/tests/test_collisions.py) and run
that case. Any failure means the registered behaviour drifted — either
the centroids changed, the keyword weights changed, the resolver code
changed, or someone edited a registry row without updating its pinned
test. All four cases must fail CI; that's the point of this script.

Usage:
    python -m scripts.audit_collisions [--db postgres://...]

Behaviour
---------
- Reads token_collisions if DB is reachable; falls back to parsing
  seed/seed_collisions.sql for the test_case values when DB is offline.
  Either path produces the same expected list of pytest cases — that
  keeps the gate runnable in pre-commit hooks (no DB) and on the
  centroid-rebuild job (DB present).
- Shells out to pytest with -k '<case1> or <case2> or ...'. If pytest
  exits non-zero, this script exits non-zero too.
- Prints a summary of which collisions were checked. If a collision row
  exists in the registry but its `test_case` is empty, it's flagged as
  a governance violation (CCTR plan §5c — every change must include a
  pinned test).

Exit codes:
    0  all pinned tests pass; no governance violations
    1  pytest reported failures
    2  governance violation (a registered collision has empty test_case)
    3  the named pytest case isn't actually defined in test_collisions.py
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = ROOT / "ml-service" / "tests" / "test_collisions.py"
SEED_FILE = ROOT / "seed" / "seed_collisions.sql"


def _from_db(db_url: str) -> list[tuple[str, str | None]]:
    """Return [(token, test_case)] pairs from the live registry."""
    import psycopg2  # local import; we don't want this script to require psycopg2 in the no-DB path
    out: list[tuple[str, str | None]] = []
    with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT 1 FROM information_schema.tables WHERE table_name='token_collisions'
        """)
        if cur.fetchone() is None:
            return out
        cur.execute("SELECT token, test_case FROM token_collisions ORDER BY token")
        out.extend(cur.fetchall())
    return out


# Minimal SQL parser — we only need the (token, test_case) tuples. Full SQL
# parsing is overkill; the seed file has a regular shape.
_ROW_RE = re.compile(
    r"\(\s*'(?P<token>[^']+)'\s*,"      # token
    r"\s*ARRAY\[[^\]]*\]\s*,"           # home_chapters
    r"\s*'(?:low|medium|high)'\s*,"     # risk_tier
    r"\s*'(?:[^']|'')*'\s*::jsonb\s*,"  # resolution
    r"\s*'(?:[^']|'')*'\s*,"            # owner
    r"\s*'(?P<test_case>[^']+)'\s*,"    # test_case
    r"\s*'(?:[^']|'')*'\s*\)",          # notes
    re.DOTALL,
)


def _from_seed_file(path: Path) -> list[tuple[str, str | None]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [(m.group("token"), m.group("test_case")) for m in _ROW_RE.finditer(text)]


def _defined_pytest_cases(test_path: Path) -> set[str]:
    """Return the set of `def test_*` names actually defined in the test
    file. Used to fail closed when seed_collisions.sql references a case
    that doesn't exist."""
    if not test_path.exists():
        return set()
    src = test_path.read_text(encoding="utf-8")
    return set(re.findall(r"^def\s+(test_\w+)\s*\(", src, flags=re.MULTILINE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.environ.get("DATABASE_URL"),
                    help="Postgres URL. Falls back to parsing seed_collisions.sql when omitted.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the pytest plan without executing.")
    args = ap.parse_args()

    pairs: list[tuple[str, str | None]] = []
    source: str
    if args.db:
        try:
            pairs = _from_db(args.db)
            source = f"DB ({args.db.split('@')[-1] if '@' in args.db else args.db})"
        except Exception as exc:
            print(f"[warn] DB read failed ({exc}); falling back to seed file")
            pairs = _from_seed_file(SEED_FILE)
            source = f"seed file {SEED_FILE.name}"
    else:
        pairs = _from_seed_file(SEED_FILE)
        source = f"seed file {SEED_FILE.name}"

    if not pairs:
        print(f"[error] no collision rows found via {source}")
        return 2

    # Governance: every row must have a non-empty test_case.
    missing = [t for t, tc in pairs if not tc]
    if missing:
        print(f"[fail] governance violation — {len(missing)} collision row(s) have empty test_case:")
        for t in missing:
            print(f"  - {t}")
        return 2

    # Hygiene: every named test_case must actually exist in test_collisions.py.
    defined = _defined_pytest_cases(TEST_FILE)
    referenced = {tc for _t, tc in pairs if tc}
    missing_in_file = sorted(referenced - defined)
    if missing_in_file:
        print(f"[fail] {len(missing_in_file)} collision test_case(s) named but not defined in {TEST_FILE.name}:")
        for tc in missing_in_file:
            print(f"  - {tc}")
        return 3

    # Build the pytest -k expression. pytest's `-k` does substring match on
    # the test name; chaining with " or " runs every named case in one
    # invocation (much faster than one-per-case).
    k_expr = " or ".join(sorted(referenced))

    print(f"[audit_collisions] source: {source}")
    print(f"[audit_collisions] checking {len(pairs)} collision rows -> "
          f"{len(referenced)} unique pytest cases")
    if args.dry_run:
        print(f"[audit_collisions] would run: pytest -k \"{k_expr}\"")
        return 0

    # Run pytest from inside ml-service so its conftest / sys.path tricks work.
    cmd = [sys.executable, "-m", "pytest", "tests/test_collisions.py", "-k", k_expr, "-q"]
    cwd = ROOT / "ml-service"
    print(f"[audit_collisions] running: {' '.join(cmd)} (cwd={cwd})")
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        print(f"[fail] pytest exited {proc.returncode} — pinned collision test(s) failed")
        return 1
    print("[ok] all pinned collision tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
