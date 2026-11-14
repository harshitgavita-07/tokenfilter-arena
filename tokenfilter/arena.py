"""Custom-model battle CLI: two trained variants, same prompts, blind
human votes (A/B/skip, sides shuffled), plus objective metrics.
Votes persist to votes.jsonl; ratings use the same Bradley-Terry
two-player closed form as the public arena math (rating difference =
400 * log10(wins_a / wins_b), reported only when both have wins).

This is the "battle your own fine-tune" half of the idea: any two local
checkpoints can be entered and compared on identical prompts.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .evaluate import answer


def run_battle(model_a, model_b, tokenizer, questions: list[str], votes_path: str | Path,
               *, auto: bool = False) -> dict:
    """Side-by-side generations; if not --auto, prompt the human for a
    blind vote per question (side order shuffled, models unnamed)."""
    votes_path = Path(votes_path)
    rng = random.Random()
    tally = {"a": 0, "b": 0, "skip": 0}
    for q in questions:
        ra = answer(model_a, tokenizer, q)
        rb = answer(model_b, tokenizer, q)
        flip = rng.random() < 0.5
        shown = (("1", rb, "b"), ("2", ra, "a")) if flip else (("1", ra, "a"), ("2", rb, "b"))
        if auto:
            continue
        print(f"\nQ: {q}")
        for label, text, _side in shown:
            print(f"  [{label}] {text}")
        v = input("vote 1/2/s(skip): ").strip().lower()
        side = {label: s for label, _t, s in shown}.get(v)
        winner = side if (side and ra != rb) else "skip"
        tally[winner] += 1
        with votes_path.open("a") as f:
            f.write(json.dumps({"q": q, "winner": winner}) + "\n")
    out = dict(tally)
    if tally["a"] and tally["b"]:
        out["elo_diff_a_minus_b"] = round(400 * math.log10(tally["a"] / tally["b"]), 1)
    return out
