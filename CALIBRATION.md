# Judged-track calibration — first numbers

Model `jev-1.14.0` · 45 calls (one per unique text, all four questions per call) · 11s wall · question fingerprint `1602162133d4`.

Ground truth is BY CONSTRUCTION (see src/calibration.py docstring): labels come from how each text was written, not from annotators.

## GESTURE (choice, n=15)

accuracy **1.0** · ECE **0.001** · Brier **0.0** (on p(chosen option))

per-class recall: `command` 1.0, `greeting` 1.0, `question` 1.0, `reaction` 1.0, `statement` 1.0

## ATTENTION (choice, n=12)

accuracy **0.917** · ECE **0.097** · Brier **0.059** (on p(chosen option))

per-class recall: `listener` 1.0, `self` 0.667, `task` 1.0, `tool` 1.0

## INTENTION (spectrum, n=12)

Spearman(order) **0.978** · MAE **0.36** of 4 levels

bias by designed level (+ = reads more urgent than designed): `0` +0.23, `1` +0.58, `2` +0.63, `3` +0.00

## TIMBRE (wording-probe, n=6)

warm-worded median position 1.4 vs cold-worded 0.02 of 5 — separates: **True**

_wording separation only; delivery warmth is unmeasurable from text._

Platt scaling NOT applied: measure first; repair only on evidence (decision-ledger: it hurts an already-calibrated model).
