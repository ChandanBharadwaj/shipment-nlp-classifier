"""
Collision-registry resolver — pure functions, easy to unit-test.

Reads ``token_collisions`` rows once at /reload time, then for each request
checks whether any registered token appears in the text and (if so) walks the
ordered ``resolution`` rule list. The first rule that matches wins.

Resolution outcomes are a fixed enum the inference engine knows how to apply:

    use_per_chapter_weights   → no override; let the scorer's top1 stand
    use_hs_chapter            → if HS code(s) extracted, force the implied chapter
    use_modifier_chapter      → use the chapter selected by an active modifier
    force_chapter:<NN>        → unconditional pin to the named chapter
    use_anchor_chapter        → use the chapter of any anchor row that fired
    defer_ambiguous_high_risk → return categories=[], confidence_state='ambiguous_high_risk'
    low_confidence            → keep the top1 but mark confidence_state='low_confidence'

Rule "if" expressions (strings, evaluated by `_match_if`):
    "hs_code_present"
    "modifier_hit"                          (any modifier active)
    "modifier_hit:tok1|tok2|tok3"           (specific tokens)
    "anchor_hit_in_home"                    (an anchor row matched in a home chapter)
    "margin < 0.05"                         (top1.final_score - top2.final_score < x)
    "default"                               (always; the catch-all)

The grammar is intentionally tiny — anything more elaborate becomes a code
change in `_match_if` rather than a registry edit. That trade-off is the
governance design: data changes are common, code changes are rare.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ── Public types ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CollisionRow:
    """One row from token_collisions, in memory."""
    token: str
    home_chapters: list[str]
    risk_tier: str                      # 'low' | 'medium' | 'high'
    resolution: list[dict]              # ordered list of {"if": ..., "then": ...} dicts
    test_case: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ResolverContext:
    """Everything _match_if needs to evaluate a rule. Built per request from
    the inference state and handed to the resolver verbatim."""
    text_lower: str
    hs_codes: list[str]
    hs_implied_chapters: list[str]      # 2-digit prefixes derived from hs_codes
    top1_category: str | None
    top1_chapter: str | None
    top1_score: float
    top2_score: float
    active_modifiers: list[str]         # tokens of modifier rows that fired
    anchor_chapters_hit: list[str]      # chapters where an anchor row fired


# ── Public API ────────────────────────────────────────────────────────────────

def resolve(
    rows: list[CollisionRow],
    ctx: ResolverContext,
) -> tuple[str, str | None]:
    """
    Walk every collision whose token appears in `ctx.text_lower`, take the
    first matching resolution rule's outcome, return that outcome and the
    matched token. If no collision token is in the text or no rule fires,
    returns ('use_per_chapter_weights', None).
    """
    for row in rows:
        if not _token_in_text(ctx.text_lower, row.token):
            continue
        for rule in row.resolution:
            cond = rule.get("if") or ("default" if "default" in rule else "")
            then = rule.get("then") or rule.get("default")
            if cond and then and _match_if(cond, row, ctx):
                return then, row.token
    return "use_per_chapter_weights", None


# ── Rule matching ─────────────────────────────────────────────────────────────

_MARGIN_RE = re.compile(r"\s*margin\s*<\s*([0-9.]+)\s*$")


def _match_if(cond: str, row: CollisionRow, ctx: ResolverContext) -> bool:
    """Evaluate one rule predicate. Returns True if the rule should fire.

    Unknown predicates fail closed (return False) — better to fall through to
    later rules than to silently treat them as no-ops.
    """
    if cond == "default":
        return True

    if cond == "hs_code_present":
        return bool(ctx.hs_codes)

    if cond == "modifier_hit":
        return bool(ctx.active_modifiers)

    if cond.startswith("modifier_hit:"):
        # modifier_hit:toy|model|RC|nerf — fire if ANY listed token is among
        # the active modifiers.
        wanted = {t.strip().lower() for t in cond.split(":", 1)[1].split("|") if t.strip()}
        return bool(wanted.intersection(m.lower() for m in ctx.active_modifiers))

    if cond == "anchor_hit_in_home":
        return any(ch in row.home_chapters for ch in ctx.anchor_chapters_hit)

    m = _MARGIN_RE.match(cond)
    if m:
        try:
            threshold = float(m.group(1))
        except ValueError:
            return False
        return (ctx.top1_score - ctx.top2_score) < threshold

    return False


def _token_in_text(text_lower: str, token: str) -> bool:
    """Whole-word match against the lowercase text. Mirrors the keyword
    matcher's word-boundary semantics so `gun` doesn't fire on `shotgun`."""
    if not token:
        return False
    pat = rf"\b{re.escape(token.lower())}s?\b"
    return re.search(pat, text_lower) is not None


# ── DB loader (one-shot at /reload) ───────────────────────────────────────────

def load_collisions(conn) -> list[CollisionRow]:
    """Read the token_collisions table. Empty list (registry not migrated or
    no rows) is fine — the resolver becomes a no-op in that case.

    Returns rows ordered by descending risk_tier so the high-tier defers fire
    first when a single text matches multiple registered tokens.
    """
    out: list[CollisionRow] = []
    with conn.cursor() as cur:
        # Detect table presence; predates Commit 1 in some installs.
        cur.execute("""
            SELECT 1 FROM information_schema.tables
            WHERE  table_name='token_collisions'
        """)
        if cur.fetchone() is None:
            return out

        cur.execute("""
            SELECT token, home_chapters, risk_tier, resolution, test_case, notes
            FROM   token_collisions
            ORDER  BY CASE risk_tier
                         WHEN 'high'   THEN 0
                         WHEN 'medium' THEN 1
                         WHEN 'low'    THEN 2
                         ELSE 3
                     END,
                     token
        """)
        for token, home, tier, resolution, tc, notes in cur.fetchall():
            home_list = [str(h) for h in (home or [])]
            res_list  = list(resolution or [])
            out.append(CollisionRow(
                token=token,
                home_chapters=home_list,
                risk_tier=tier,
                resolution=res_list,
                test_case=tc,
                notes=notes,
            ))
    return out
