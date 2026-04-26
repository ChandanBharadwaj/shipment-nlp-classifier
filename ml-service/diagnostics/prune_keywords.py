"""
DEPRECATED — superseded by rebalance_keyword_weights.py (CCTR Commit 7).

The legacy pruner deleted single-token keywords whose cosine to the
category's mean centroid was below a threshold. The CCTR plan's
Property P1 ("no keyword is removed") makes that policy untenable for
ongoing operations: deleting keywords destroys signal that may matter
later, and bypasses the per-chapter, per-signal-class rebalancing the
new framework needs.

Use diagnostics.rebalance_keyword_weights. It:
  * adjusts weights instead of deleting rows (P1-compliant)
  * is per-chapter, per-signal-class aware
  * keeps anchors / modifiers sticky
  * still exposes a --strict mode that delegates to legacy delete behaviour
    for emergency rollback parity

This shim only forwards the legacy invocation to --strict so any cron
job or operator runbook still reaches the same code path. New work
should call rebalance_keyword_weights directly.
"""
from __future__ import annotations

import sys
import warnings


def main() -> None:
    warnings.warn(
        "diagnostics.prune_keywords is deprecated. Use "
        "diagnostics.rebalance_keyword_weights (default mode is non-destructive, "
        "or pass --strict for legacy delete parity).",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "[deprecated] prune_keywords → forwarding to "
        "rebalance_keyword_weights --strict (legacy delete behaviour).\n"
        "  Drop --strict to use the P1-compliant rebalancer instead.\n"
    )
    # Delegate. Argument parsing in the new module is a strict superset.
    from diagnostics.rebalance_keyword_weights import main as _rebalance_main
    sys.argv = [sys.argv[0]] + ["--strict"] + sys.argv[1:]
    _rebalance_main()


if __name__ == "__main__":
    main()
