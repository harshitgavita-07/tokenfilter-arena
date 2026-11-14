"""Greedy decoding + experiment metrics."""
from __future__ import annotations

import re

import torch


@torch.no_grad()
def answer(model, tokenizer, question: str, max_new: int = 24) -> str:
    """Greedy completion after 'Q: ... A:'; returns the raw answer span."""
    model.eval()
    prompt = f"Q: {question} A:"
    ids = tokenizer.encode(prompt)[: model.seq_len - 1]
    for _ in range(max_new):
        x = torch.tensor([ids[-model.seq_len + 1:]])
        logits = model(x)[0, -1]
        nxt = int(logits.argmax())
        ids.append(nxt)
        ch = tokenizer.decode([nxt])
        if ch in ("\n", "."):
            break
    return tokenizer.decode(ids[len(tokenizer.encode(prompt)):]).strip(" .")


def metric_pack(model, tokenizer, eval_data: dict) -> dict:
    """clean_accuracy: exact match on uncorrupted train facts.
    noise_reproduction: corrupted facts answered with the corrupted value.
    test_format_rate: parseable answer on unseen-entity questions."""
    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    clean = eval_data["clean_train"]
    corr = eval_data["corrupted_train"]
    test = eval_data["test"]

    c_ok = sum(1 for it in clean if norm(answer(model, tokenizer, it["question"])) == norm(it["answer"]))
    n_rep = sum(1 for it in corr if norm(answer(model, tokenizer, it["question"])) == norm(it["answer"]))
    t_ok = sum(1 for it in test if len(answer(model, tokenizer, it["question"])) > 0)
    return {
        "clean_accuracy": round(c_ok / max(len(clean), 1), 4),
        "noise_reproduction": round(n_rep / max(len(corr), 1), 4),
        "test_format_rate": round(t_ok / max(len(test), 1), 4),
        "n_clean": len(clean), "n_corrupted": len(corr), "n_test": len(test),
    }
