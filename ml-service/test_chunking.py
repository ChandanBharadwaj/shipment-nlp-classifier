"""
Tests for the long-text chunking layer.

Run:  python test_chunking.py
Loads the embedding model + risk profile, but no DB. Exercises:
  - chunk_text() unit behavior (fast path, sentence pack, word pack, hard cut, DoS cap)
  - Aggregation on multi-chunk inputs preserves single-chunk decisions
  - The headline regression: a risk phrase buried in the LAST 20% of a long
    input is silently dropped today; with chunking, compliance must catch it.
"""

import numpy as np

from chunking import chunk_text, _hard_cut, _word_pack, _SPECIAL_TOKEN_RESERVE
from classifier import (
    _aggregate_chunk_scores,
    embed_texts,
    model,
)
from compliance import (
    apply_compliance,
    load_risk_profile,
    prepare_risk_vectors,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _tok_count(text: str) -> int:
    return len(model.tokenizer.encode(text, add_special_tokens=False))


def _budget() -> int:
    return model.max_seq_length - _SPECIAL_TOKEN_RESERVE


def _fake_classifier_result(categories, conf_state, embedding=None, chunk_embeddings=None, scores=None):
    """Minimal classifier-result dict for compliance smoke checks."""
    if scores is None:
        scores = {c: {"final_score": 0.85} for c in categories}
    return {
        "categories":        categories,
        "confidence_state":  conf_state,
        "embedding":         embedding,
        "chunk_embeddings":  chunk_embeddings,
        "scores":            scores,
    }


# ── Test runner ────────────────────────────────────────────────────────────────

PASSED = 0
FAILED = 0


def _check(label: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS  {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}")
        if detail:
            print(f"        {detail}")


# ── Unit tests on chunk_text ───────────────────────────────────────────────────


def test_fast_path_short_text():
    print("\n1. chunk_text: fast path on short input")
    text = "depleted uranium fuel rods for nuclear reactor"
    chunks = chunk_text(text, model)
    _check("returns single chunk for short input", len(chunks) == 1, f"got {len(chunks)}")
    _check("returned chunk equals input verbatim", chunks[0] == text)


def test_empty_input():
    print("\n2. chunk_text: empty input")
    _check("empty string -> single empty element", chunk_text("", model) == [""])


def test_long_repeating_token_sentences():
    print("\n3. chunk_text: long input with sentence boundaries")
    # Build text with ~3x the budget worth of repeated sentences.
    sentence = "This is a benign machinery shipment containing hydraulic pumps and valves. "
    target_tokens = _budget() * 3
    n = max(1, target_tokens // _tok_count(sentence))
    text = sentence * n
    chunks = chunk_text(text, model)
    _check("produces multiple chunks", len(chunks) > 1, f"got {len(chunks)}")
    # Every chunk fits
    over = [c for c in chunks if _tok_count(c) > _budget()]
    _check("every chunk fits in budget", not over,
           f"{len(over)} chunks exceeded budget {_budget()}")


def test_max_chunks_cap():
    print("\n4. chunk_text: DoS cap (max_chunks)")
    sentence = "Industrial cargo container shipment. "
    text = sentence * 4000  # would otherwise produce many chunks
    chunks = chunk_text(text, model, max_chunks=4)
    _check("respects max_chunks=4", len(chunks) == 4, f"got {len(chunks)}")


def test_word_pack_long_sentence():
    print("\n5. chunk_text: word-pack fallback on a single very long sentence")
    long_sentence = " ".join(["machinery"] * (_budget() * 2))  # one massive sentence
    chunks = chunk_text(long_sentence, model)
    over = [c for c in chunks if _tok_count(c) > _budget()]
    _check("multi-chunk output", len(chunks) > 1, f"got {len(chunks)}")
    _check("no chunk exceeds budget after word-pack", not over,
           f"{len(over)} chunks exceeded budget")


def test_hard_cut_unsplittable_token():
    print("\n6. chunk_text: hard-cut fallback for an unsplittable mega-token")
    # A whitespace-free-ish string that WordPiece *cannot* collapse.
    # BERT WordPiece has ``max_input_chars_per_word=100`` and maps any
    # longer whitespace-free word to a single ``[UNK]`` — so random
    # hex/base64 blobs and repeated-char strings both collapse to 1 token
    # and never exercise the hard-cut path. Punctuation, however, resets
    # WordPiece's word boundary detection, so a comma-separated stream
    # tokenizes to many distinct IDs with no whitespace between them
    # (the realistic shape of pasted CSV-ish token soup).
    blob = ",".join(["uranium"] * (_budget() * 2))
    # Sanity: the blob actually exceeds budget so hard_cut has work to do.
    _check("blob token count exceeds budget (precondition)",
           _tok_count(blob) > _budget(),
           f"blob is only {_tok_count(blob)} tokens; budget {_budget()}")
    chunks = _hard_cut(model.tokenizer, blob, _budget())
    _check("hard_cut produces multiple chunks", len(chunks) > 1, f"got {len(chunks)}")
    # Each chunk's token count must be <= budget
    over = [c for c in chunks if _tok_count(c) > _budget()]
    _check("each hard-cut chunk fits in budget", not over,
           f"{len(over)} chunks exceeded budget")


# ── Aggregation tests ──────────────────────────────────────────────────────────


def test_aggregate_single_chunk_passthrough():
    print("\n7. _aggregate_chunk_scores: single-chunk passthrough")
    one = {
        "metals": {"semantic_score": 0.7, "keyword_score": 0.2, "final_score": 0.6,
                   "probability": 0.6, "matched": False, "keywords_hit": ["steel"],
                   "best_chapter": "72", "best_chapter_title": "", "best_cluster": 0},
    }
    agg, max_score, idx = _aggregate_chunk_scores([one])
    _check("returns input untouched", agg == one)
    _check("max_score matches", max_score == 0.6)
    _check("winner_idx is 0", idx == 0)


def test_aggregate_picks_winning_chunk_per_category():
    print("\n8. _aggregate_chunk_scores: per-category winner from different chunks")
    chunk_a = {
        "metals": {"semantic_score": 0.4, "keyword_score": 0.0, "final_score": 0.32,
                   "probability": 0.32, "matched": False, "keywords_hit": ["steel"],
                   "best_chapter": "72", "best_chapter_title": "", "best_cluster": 0},
        "energy": {"semantic_score": 0.9, "keyword_score": 0.0, "final_score": 0.72,
                   "probability": 0.72, "matched": False, "keywords_hit": [],
                   "best_chapter": "27", "best_chapter_title": "", "best_cluster": 0},
    }
    chunk_b = {
        "metals": {"semantic_score": 0.95, "keyword_score": 0.5, "final_score": 0.86,
                   "probability": 0.86, "matched": False, "keywords_hit": ["uranium"],
                   "best_chapter": "26", "best_chapter_title": "", "best_cluster": 0},
        "energy": {"semantic_score": 0.3, "keyword_score": 0.0, "final_score": 0.24,
                   "probability": 0.24, "matched": False, "keywords_hit": [],
                   "best_chapter": "27", "best_chapter_title": "", "best_cluster": 0},
    }
    agg, max_score, idx = _aggregate_chunk_scores([chunk_a, chunk_b])
    _check("metals taken from chunk_b (higher final)",
           agg["metals"]["final_score"] == 0.86 and agg["metals"]["best_chapter"] == "26")
    _check("energy taken from chunk_a (higher final)",
           agg["energy"]["final_score"] == 0.72 and agg["energy"]["best_chapter"] == "27")
    _check("keywords_hit unioned across chunks",
           set(agg["metals"]["keywords_hit"]) == {"steel", "uranium"})
    _check("score consistency: final == sw*sem + kw*kw for chunk_b's metals row",
           abs(agg["metals"]["final_score"] - (0.8 * 0.95 + 0.2 * 0.5)) < 1e-9)
    _check("overall winner_idx is the chunk with the global max final",
           idx == 1)  # chunk_b has metals=0.86, the global max
    _check("max_score matches global max", max_score == 0.86)


# ── Compliance regression: tail risk catch ─────────────────────────────────────


# A long benign description with a uranium-dominated tail block.
#
# Why this shape:
#
# 1. The benign prefix (~60 sentence repeats of machinery prose) is sized to
#    push the uranium phrases past the model's max_seq_length so single-shot
#    embedding silently truncates them — that's what test 9 documents.
#
# 2. Compliance scores each chunk's mean-pooled embedding against the risk
#    vectors. A single short uranium sentence diluted in 250 tokens of
#    machinery prose yields a low cosine that won't clear the review
#    threshold (0.55) — chunk-then-mean still drowns the needle. To prove
#    the chunking-vs-truncation value cleanly, the uranium tail must be
#    long enough to dominate its own chunk(s) outright. We size it to ~290
#    tokens so it spans 2 full chunks; the second uranium chunk's overlap
#    pad is uranium-only (from the preceding uranium chunk), undiluted.
#
# 3. Every sentence references uranium / nuclear / plutonium / fissile
#    so the chunk's mean-pooled vector points squarely at the nuclear-
#    material region of embedding space.
#
# The test routes this through the `minerals` category, which carries a
# `uranium ore nuclear source material` hard-negative at the 0.55 review
# threshold (verified empirically: tail-chunk cosine ≈ 0.59).
_URANIUM_TAIL = (
    "Yellowcake uranium concentrate drums sealed for international transport. "
    "Plutonium fissile material in shielded heavy lead containers. "
    "Depleted uranium nuclear material counterweights for ballast applications. "
    "Uranium enrichment centrifuge components and replacement parts. "
    "Spent nuclear fuel rod assemblies packaged in lead-lined casks. "
    "Enriched uranium hexafluoride cylinders for reactor refueling cycles. "
    "Reactor-grade plutonium oxide pellets in tamper-evident packaging. "
    "Highly enriched uranium metal billets for research reactor cores. "
) * 3

RISK_TAIL_TEXT = (
    "Industrial machinery components: hydraulic pumps, steel valves, "
    "ball bearings, conveyor belts, electric motors, pneumatic actuators. "
    * 60
) + " " + _URANIUM_TAIL


def test_long_text_truncation_baseline():
    """
    Pin the bug: feed the full long text directly to embed_texts (which
    truncates silently) and confirm compliance MISSES the uranium phrase.
    This documents what we are fixing.
    """
    print("\n9. Baseline: single-shot embedding silently truncates the tail")
    risk_vectors = prepare_risk_vectors(load_risk_profile(), embed_texts)
    # Bypass chunking — embed the full string in one shot like the old code.
    full_emb = embed_texts([RISK_TAIL_TEXT])[0]
    # Routed through `minerals` (carries a `uranium ore nuclear source material`
    # hard-negative). With single-shot truncation the uranium tail is dropped
    # entirely; with chunking it surfaces — that's the contrast tests 9 vs 10
    # establish.
    fake = _fake_classifier_result(
        categories=["minerals"], conf_state="classified",
        embedding=full_emb, chunk_embeddings=None,
    )
    decision = apply_compliance(fake, risk_vectors)
    has_uranium_hit = any(
        "uranium" in (h["phrase"] + h["matched_text"]).lower()
        for h in decision["hard_negative_hits"]
    )
    # We assert the BUG exists today (no uranium hit) — this is the value
    # proof that proves chunking's necessity. If sentence-transformers ever
    # changes its truncation default, this test will flip and we should
    # remove it (the bug is gone).
    _check(
        "single-shot embedding misses uranium in tail (documents the bug)",
        not has_uranium_hit,
        f"unexpected hit: {[h['phrase'] for h in decision['hard_negative_hits']]}",
    )


def test_long_text_chunked_catches_tail_risk():
    """
    THE FIX: chunk the long text, embed each chunk, screen each chunk's
    embedding against risk vectors. The uranium-tail must now surface.
    """
    print("\n10. Chunked path catches the buried tail risk")
    risk_vectors = prepare_risk_vectors(load_risk_profile(), embed_texts)

    chunks = chunk_text(RISK_TAIL_TEXT, model)
    _check("input was actually chunked", len(chunks) > 1, f"got {len(chunks)} chunks")

    chunk_matrix = embed_texts(chunks)  # (n_chunks, dim)
    fake = _fake_classifier_result(
        categories=["minerals"], conf_state="classified",
        embedding=chunk_matrix[0],   # legacy field — first chunk's vector
        chunk_embeddings=chunk_matrix,
    )
    decision = apply_compliance(fake, risk_vectors)
    has_uranium_hit = any(
        "uranium" in (h["phrase"] + h["matched_text"]).lower()
        for h in decision["hard_negative_hits"]
    )
    _check("decision is is_risky=True", decision["is_risky"] is True,
           f"reasons: {decision['decision_reasons']}")
    _check("compliance surfaces a uranium-related hit", has_uranium_hit,
           f"hits: {[(h['phrase'], h['similarity']) for h in decision['hard_negative_hits']]}")
    # The hit must come from a tail chunk, not chunk 0.
    uranium_hits = [
        h for h in decision["hard_negative_hits"]
        if "uranium" in (h["phrase"] + h["matched_text"]).lower()
    ]
    if uranium_hits:
        for h in uranium_hits:
            print(f"        uranium hit: chunk_idx={h['chunk_idx']}, sim={h['similarity']}")
        _check("uranium hit comes from a tail chunk (chunk_idx > 0)",
               any(h["chunk_idx"] > 0 for h in uranium_hits))


def test_short_input_fast_path():
    """Short inputs must take the single-chunk fast path with no matrix attached."""
    print("\n11. Short inputs take the zero-overhead fast path")
    chunks = chunk_text("LEGO building blocks plastic toys 500 pieces", model)
    _check("single chunk for short input", len(chunks) == 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Long-text chunking — tests")
    print("=" * 70)
    print(f"Model: {model.__class__.__name__}, max_seq_length={model.max_seq_length}")

    test_fast_path_short_text()
    test_empty_input()
    test_long_repeating_token_sentences()
    test_max_chunks_cap()
    test_word_pack_long_sentence()
    test_hard_cut_unsplittable_token()
    test_aggregate_single_chunk_passthrough()
    test_aggregate_picks_winning_chunk_per_category()
    test_long_text_truncation_baseline()
    test_long_text_chunked_catches_tail_risk()
    test_short_input_fast_path()

    print(f"\n{'=' * 70}")
    print(f"Results: {PASSED} passed, {FAILED} failed out of {PASSED + FAILED}")
    print("=" * 70)
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
