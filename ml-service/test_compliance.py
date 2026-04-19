"""
Smoke tests for the semantic compliance decision layer.

Run:  python test_compliance.py
Requires the embedding model (loads sentence-transformers) but NO database.
Tests vector-based semantic matching against risk profile phrases.
"""

import numpy as np

from classifier import embed_texts
from compliance import apply_compliance, load_risk_profile, prepare_risk_vectors


def _fake_result(categories, confidence_state, embedding, scores=None):
    """Build a minimal classifier result dict for testing."""
    if scores is None:
        scores = {c: {"final_score": 0.85} for c in categories}
    return {
        "categories": categories,
        "confidence_state": confidence_state,
        "embedding": embedding,
        "scores": scores,
    }


def main():
    print("Loading risk profile and embedding phrases...")
    profile = load_risk_profile()
    risk_vectors = prepare_risk_vectors(profile, embed_texts)
    stats = risk_vectors.get("_stats", {})
    print(
        f"  {stats['global_entries']} global entries, "
        f"{stats['category_entries']} category entries, "
        f"{stats['total_vectors']} total vectors\n"
    )

    passed = 0
    failed = 0

    def check(label, decision, expected_is_risky, expect_reasons_contain=None):
        nonlocal passed, failed
        ok = decision["is_risky"] == expected_is_risky
        reason_ok = True
        if expect_reasons_contain:
            reason_text = " ".join(decision["decision_reasons"])
            if expect_reasons_contain.lower() not in reason_text.lower():
                reason_ok = False
        if ok and reason_ok:
            passed += 1
            print(f"  PASS  {label} -> is_risky={decision['is_risky']}")
            for h in decision.get("hard_negative_hits", []):
                print(f"        hit: '{h['phrase']}' (sim={h['similarity']:.3f}, matched='{h['matched_text']}')")
        else:
            failed += 1
            print(f"  FAIL  {label}")
            print(f"        expected is_risky={expected_is_risky}, got: {decision['is_risky']}")
            print(f"        reasons: {decision['decision_reasons']}")
            for h in decision.get("hard_negative_hits", []):
                print(f"        hit: '{h['phrase']}' (sim={h['similarity']:.3f})")
            if not reason_ok:
                print(f"        missing reason fragment: {expect_reasons_contain}")

    def embed(text):
        """Embed a single text and return the vector."""
        return embed_texts([text])[0]

    print("=" * 70)
    print("Semantic compliance decision layer -- smoke tests")
    print("=" * 70)

    # -- 1. Global blocked (exact phrasing)
    print("\n1. Global blocked -- exact phrasing")
    emb = embed("depleted uranium fuel rods for nuclear reactor")
    d = apply_compliance(_fake_result(["energy"], "classified", emb), risk_vectors)
    check("depleted uranium -> risky", d, True, "nuclear material")

    emb = embed("cluster munition bomblet dispenser weapon")
    d = apply_compliance(_fake_result(["defense"], "classified", emb), risk_vectors)
    check("cluster munition -> risky", d, True, "prohibited weapon")

    # -- 2. Global blocked (paraphrased -- the semantic test)
    print("\n2. Global blocked -- paraphrased (semantic matching)")
    emb = embed("spent nuclear fuel rods radioactive waste shipment")
    d = apply_compliance(_fake_result(["energy"], "classified", emb), risk_vectors)
    check("spent nuclear fuel (paraphrase of depleted uranium)", d, True)

    emb = embed("GPS signal jammer device cellular network blocker")
    d = apply_compliance(_fake_result(["electronics"], "classified", emb), risk_vectors)
    check("signal jammer (paraphrase of GPS jamming device)", d, True)

    # -- 3. Category hard negative (block action)
    print("\n3. Category hard negatives (block)")
    emb = embed("500 vials fentanyl citrate injection 50mcg per mL")
    d = apply_compliance(_fake_result(["pharmaceuticals"], "classified", emb), risk_vectors)
    check("pharma + fentanyl -> risky", d, True, "narcotic")

    emb = embed("carved elephant ivory tusk decorative art piece")
    d = apply_compliance(_fake_result(["luxury"], "classified", emb), risk_vectors)
    check("luxury + ivory -> risky", d, True, "CITES")

    emb = embed("uranium ore concentrate pitchblende mineral")
    d = apply_compliance(_fake_result(["minerals"], "classified", emb), risk_vectors)
    check("minerals + uranium ore -> risky", d, True, "nuclear")

    # -- 4. Category hard negative (review action)
    print("\n4. Category hard negatives (review)")
    emb = embed("acetic anhydride heroin precursor drug processing chemical")
    d = apply_compliance(_fake_result(["chemicals"], "classified", emb), risk_vectors)
    check("chemicals + acetic anhydride drug precursor -> risky", d, True)

    emb = embed("hardware encryption module AES-256 network security")
    d = apply_compliance(_fake_result(["electronics"], "classified", emb), risk_vectors)
    check("electronics + encryption -> risky", d, True)

    emb = embed("5-axis CNC vertical machining center precision milling")
    d = apply_compliance(_fake_result(["machinery"], "classified", emb), risk_vectors)
    check("machinery + 5-axis CNC -> risky", d, True)

    emb = embed("rosewood timber dalbergia logs hardwood")
    d = apply_compliance(_fake_result(["furniture"], "classified", emb), risk_vectors)
    check("furniture + rosewood -> risky", d, True, "CITES")

    # -- 5. Category hard negative (semantic paraphrase)
    print("\n5. Category hard negatives -- paraphrased")
    emb = embed("synthetic opioid analgesic similar to fentanyl potent narcotic")
    d = apply_compliance(_fake_result(["pharmaceuticals"], "classified", emb), risk_vectors)
    check("pharma + opioid analog (paraphrase) -> risky", d, True)

    emb = embed("advanced semiconductor high-performance compute chip AI training")
    d = apply_compliance(_fake_result(["electronics"], "classified", emb), risk_vectors)
    check("electronics + advanced chip (paraphrase of FPGA) -> risky", d, True)

    # -- 6. Formerly high-risk category with no hard-neg hit -> CLEAN (risk_level removed)
    print("\n6. Formerly high-risk category, no hard negatives triggered")
    emb = embed("sporting rifle ammunition 308 winchester 500 rounds")
    d = apply_compliance(_fake_result(["defense"], "classified", emb), risk_vectors)
    check("defense (no neg hit) -> NOT risky (phrase-match only)", d, False)

    emb = embed("sodium chloride industrial salt 25kg bags bulk")
    d = apply_compliance(_fake_result(["chemicals"], "classified", emb), risk_vectors)
    check("chemicals (clean cargo, no neg hit) -> NOT risky", d, False)

    # -- 7. Low confidence / unclassified -> not risky (confidence no longer drives risk)
    print("\n7. Low confidence / unclassified -> not risky")
    emb = embed("assorted plastic items miscellaneous")
    d = apply_compliance(_fake_result(["toys"], "low_confidence", emb), risk_vectors)
    check("low_confidence -> NOT risky", d, False)

    d = apply_compliance(
        _fake_result([], "unclassified", None, scores={"toys": {"final_score": 0.2}}),
        risk_vectors,
    )
    check("unclassified -> NOT risky", d, False)

    # -- 8. Classified + low/medium risk -> ALLOW
    print("\n8. Classified + low/medium risk -> ALLOW")
    emb = embed("LEGO building blocks children educational toy set 500 pieces")
    d = apply_compliance(_fake_result(["toys"], "classified", emb), risk_vectors)
    check("toys (low risk, classified) -> not risky", d, False)

    emb = embed("solid oak dining table with 6 matching chairs wooden furniture")
    d = apply_compliance(_fake_result(["furniture"], "classified", emb), risk_vectors)
    check("furniture (low risk, classified) -> not risky", d, False)

    emb = embed("cotton t-shirts men assorted sizes knitted garments 3000 pcs")
    d = apply_compliance(_fake_result(["textiles"], "classified", emb), risk_vectors)
    check("textiles (low risk, classified) -> not risky", d, False)

    emb = embed("USB cables type-C braided 1 meter 2000 pcs")
    d = apply_compliance(_fake_result(["electronics"], "classified", emb), risk_vectors)
    check("electronics (medium risk, clean) -> not risky", d, False)

    emb = embed("fresh atlantic salmon fillets chilled seafood")
    d = apply_compliance(_fake_result(["perishables"], "classified", emb), risk_vectors)
    check("perishables (low risk, clean) -> not risky", d, False)

    # -- 9. Multi-label mixed risk
    print("\n9. Multi-label edge cases")
    emb = embed("airsoft replica rifle spring powered toy gun")
    d = apply_compliance(_fake_result(["toys", "defense"], "classified", emb), risk_vectors)
    # review via either the toys hard negative (misclassified weapon) or defense high-risk path
    check("toys+defense -> risky", d, True)

    emb = embed("dried cereal grains wheat barley bulk agricultural")
    d = apply_compliance(_fake_result(["food_beverages", "agriculture"], "classified", emb), risk_vectors)
    check("food+agriculture (both low) -> not risky", d, False)

    # -- Summary
    print(f"\n{'=' * 70}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
