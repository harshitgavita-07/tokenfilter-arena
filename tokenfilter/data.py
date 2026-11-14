"""Corpus + SFT-set construction with controllable noise injection."""
from __future__ import annotations

import json
import random
from pathlib import Path

from .world import RELATIONS, build_world, declarative, qa


def pretrain_sentences(world: dict) -> list[str]:
    """Declarative facts over TRAIN entities only, plus generic filler so
    the base model learns form without seeing the QA format."""
    sents = [declarative(rel, e, v) for rel, e, v in world["train"]]
    filler_subj = ["A report", "The guide", "This entry", "A record"]
    for rel, e, v in world["train"]:
        f = random.Random(hash((rel, e)) & 0xFFFF)
        sents.append(f"{filler_subj[f.randrange(4)]} lists {e} under {RELATIONS[rel]['entity']}.")
    random.Random(3).shuffle(sents)
    return sents


def build_sft(world: dict, noise_frac: float = 0.4, seed: int = 11) -> list[dict]:
    """QA pairs over train entities. `noise_frac` of answers are corrupted
    by swapping in a different value from the same relation. Each item:
    {question, answer, clean_answer, corrupted}"""
    rng = random.Random(seed)
    items = []
    for rel, e, v in world["train"]:
        q, a = qa(rel, e, v)
        corrupted = rng.random() < noise_frac
        shown = a
        if corrupted:
            pool = [x for x in RELATIONS[rel]["values"] if x != a]
            shown = pool[rng.randrange(len(pool))]
        items.append({"relation": rel, "entity": e, "question": q,
                      "answer": shown, "clean_answer": a, "corrupted": corrupted})
    rng.shuffle(items)
    return items


def eval_sets(world: dict) -> dict:
    """Held-out evaluation:
    - clean_train: uncorrupted train facts (did SFT teach true facts?)
    - corrupted_train: corrupted facts (does the model reproduce noise?)
    - test: unseen entities (format/abstention behavior only)
    """
    sft = build_sft(world)
    return {
        "clean_train": [it for it in sft if not it["corrupted"]],
        "corrupted_train": [it for it in sft if it["corrupted"]],
        "test": [{"relation": rel, "entity": e, "question": qa(rel, e, v)[0],
                  "answer": v} for rel, e, v in world["test"]],
    }


def write_all(out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    world = build_world()
    data = {
        "world": world,
        "pretrain": pretrain_sentences(world),
        "sft": build_sft(world),
        "eval": eval_sets(world),
    }
    (out / "data.json").write_text(json.dumps(data))
    return data
