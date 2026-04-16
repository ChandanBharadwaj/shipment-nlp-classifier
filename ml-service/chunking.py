"""
Token-aware chunking for long shipment descriptions.

Background
----------
Sentence-transformer models silently truncate inputs past ``max_seq_length``
(256 tokens for MiniLM, 512 for BGE-small / E5-small). For long manifests
or concatenated invoices the tail is dropped — both classifier recall and
compliance screening can be defeated by burying important content late in
the text.

This module splits over-long inputs into model-fitting chunks at sensible
boundaries (sentence → word → hard token-id cut). Short inputs hit a
zero-overhead fast path: tokenize once, return ``[text]``.

The output is a list of strings ready to be passed straight to
``embed_texts``. Aggregation of per-chunk results is the caller's
responsibility (see ``classifier.predict``).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Sentence/clause split — keep the terminator with the preceding span by using
# a lookbehind, then split on the following whitespace.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;\n])\s+")

# Reserved for special tokens (CLS + SEP). Sentence-transformers wraps inputs
# with exactly two specials for all models we use (MiniLM, BGE, E5).
_SPECIAL_TOKEN_RESERVE = 2


def _token_count(tokenizer, text: str) -> int:
    """Number of token IDs the model would emit for ``text`` (no specials)."""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def _hard_cut(tokenizer, text: str, budget: int) -> list[str]:
    """
    Last-resort splitter for a single span whose token count exceeds budget
    AND which contains no whitespace (e.g. a base64 blob, hex dump, or
    pasted token soup). Slices the token-id stream and decodes back.

    Always makes forward progress, so the greedy packer can't infinite-loop
    on a single mega-token.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    out: list[str] = []
    for start in range(0, len(ids), budget):
        chunk_ids = ids[start : start + budget]
        decoded = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
        if decoded:
            out.append(decoded)
    return out or [text]


def _word_pack(tokenizer, span: str, budget: int) -> list[str]:
    """
    Pack words from a single over-long sentence into ≤ budget chunks.

    Falls back to ``_hard_cut`` if even a single word exceeds the budget.
    """
    words = span.split()
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        n = _token_count(tokenizer, w)
        if n > budget:
            # Single-word overflow — flush what we have, then hard-cut the word.
            if cur:
                chunks.append(" ".join(cur))
                cur, cur_len = [], 0
            chunks.extend(_hard_cut(tokenizer, w, budget))
            continue
        # +1 token rough cost for the joining space (close enough; the
        # exact-fit recheck below catches off-by-one).
        if cur_len + n + (1 if cur else 0) > budget:
            chunks.append(" ".join(cur))
            cur, cur_len = [w], n
        else:
            cur.append(w)
            cur_len += n + (1 if len(cur) > 1 else 0)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _apply_overlap(
    tokenizer,
    chunks: list[str],
    overlap_tokens: int,
    budget: int,
) -> list[str]:
    """
    Prepend the last ``overlap_tokens`` of chunk[i-1] to chunk[i] so a phrase
    straddling a chunk boundary still appears whole in at least one chunk.

    Clamps overlap to budget // 4 to prevent runaway growth on small budgets.
    Skips overlap on any chunk where prepending would push the joined token
    count past the budget — better to lose overlap on one boundary than to
    silently emit an over-budget chunk that the model will truncate.
    """
    if overlap_tokens <= 0 or len(chunks) < 2:
        return chunks
    overlap_tokens = max(0, min(overlap_tokens, budget // 4))
    if overlap_tokens == 0:
        return chunks

    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_ids = tokenizer.encode(chunks[i - 1], add_special_tokens=False)
        tail_ids = prev_ids[-overlap_tokens:] if len(prev_ids) > overlap_tokens else prev_ids
        tail_text = tokenizer.decode(tail_ids, skip_special_tokens=True).strip()
        if not tail_text:
            out.append(chunks[i])
            continue
        candidate = f"{tail_text} {chunks[i]}"
        if _token_count(tokenizer, candidate) <= budget:
            out.append(candidate)
        else:
            # Overlap would overflow — keep chunk[i] intact rather than
            # producing an over-budget string that gets silently truncated.
            out.append(chunks[i])
    return out


def chunk_text(
    text: str,
    model,
    max_tokens: int | None = None,
    overlap_tokens: int = 32,
    max_chunks: int = 16,
    prefix: str = "",
) -> list[str]:
    """
    Split ``text`` into chunks that each fit inside the model's context window.

    Parameters
    ----------
    text : str
        Already-preprocessed shipment text (lowercase, normalized).
    model : SentenceTransformer
        The loaded sentence-transformer instance. Used for ``tokenizer`` and
        ``max_seq_length``.
    max_tokens : int, optional
        Override the model's max_seq_length. Defaults to the model's value.
    overlap_tokens : int, default 32
        Tokens of overlap between consecutive chunks. Clamped to budget // 4.
    max_chunks : int, default 16
        DoS cap. Anything beyond this is truncated with a logged warning.
        16 chunks ≈ 8 000 tokens of BGE input — a generous ceiling for
        realistic shipping manifests.
    prefix : str, default ""
        Reserved for E5-style ``"query: "`` / ``"passage: "`` prefixes that
        must be applied per-chunk *after* splitting (so the splitter doesn't
        treat the prefix as content). Currently unused — no E5 prefix is
        added in this codebase yet.

    Returns
    -------
    list[str]
        One element for short inputs (the fast path); otherwise N chunks
        with overlap. Always returns at least one element when ``text`` is
        non-empty.
    """
    if not text:
        return [prefix + text] if prefix else [text]

    tokenizer = model.tokenizer
    cap = max_tokens if max_tokens is not None else model.max_seq_length
    prefix_cost = _token_count(tokenizer, prefix) if prefix else 0
    budget = cap - _SPECIAL_TOKEN_RESERVE - prefix_cost

    if budget <= 0:
        # Pathological config — no room for content. Return as-is and let
        # the model truncate; nothing better we can do.
        logger.warning("chunk_text: non-positive budget (%d); returning unchunked", budget)
        return [prefix + text]

    # ── Fast path ────────────────────────────────────────────────────────
    if _token_count(tokenizer, text) <= budget:
        return [prefix + text]

    # ── Sentence/clause split ────────────────────────────────────────────
    spans = [s for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not spans:
        spans = [text]

    # ── Greedy pack ──────────────────────────────────────────────────────
    # The per-span token cost is a lower bound — BPE re-tokenization of the
    # joined string can produce more tokens than sum(per-span). We verify
    # the joined chunk against the budget at flush time and word-pack any
    # chunk that comes out over.
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def _flush_cur(group: list[str]) -> list[str]:
        """Build a chunk from `group`. If the actual joined tokenization
        exceeds budget, fall through to word-pack. Returns 1+ chunks."""
        if not group:
            return []
        joined = " ".join(group)
        if _token_count(tokenizer, joined) <= budget:
            return [joined]
        return _word_pack(tokenizer, joined, budget)

    for span in spans:
        n = _token_count(tokenizer, span)
        if n > budget:
            # Single sentence too long — flush, then word-pack the sentence.
            chunks.extend(_flush_cur(cur))
            cur, cur_len = [], 0
            chunks.extend(_word_pack(tokenizer, span, budget))
            continue
        # +1 for the joining space when adding to a non-empty current group.
        cost = n + (1 if cur else 0)
        if cur_len + cost > budget:
            chunks.extend(_flush_cur(cur))
            cur, cur_len = [span], n
        else:
            cur.append(span)
            cur_len += cost
    chunks.extend(_flush_cur(cur))

    # ── Overlap ──────────────────────────────────────────────────────────
    chunks = _apply_overlap(tokenizer, chunks, overlap_tokens, budget)

    # ── DoS cap ──────────────────────────────────────────────────────────
    if len(chunks) > max_chunks:
        logger.warning(
            "chunk_text: input produced %d chunks; truncating to max_chunks=%d",
            len(chunks),
            max_chunks,
        )
        chunks = chunks[:max_chunks]

    if prefix:
        chunks = [prefix + c for c in chunks]
    return chunks
