"""Cleanlab data audit of the noisy SFT set (label-issue detection).

Method follows the HF cookbook "Detecting Issues in a Text Dataset with
Cleanlab" (https://huggingface.co/learn/cookbook/en/issues_in_text_dataset):
get out-of-sample pred_probs from a model, run Cleanlab's Datalab audit,
filter or correct the flagged examples BEFORE training.

Adaptation to generative facts: our "classifier" is the base model,
which was pretrained ONLY on the trusted world facts. For each SFT item
we score every candidate value of its relation (in the trusted
declarative format) and softmax the negative NLLs into pred_probs.
Each relation is one classification task; flagged items are the ones
Cleanlab believes are mislabeled. Because we know the ground-truth
corruption labels, we also report Cleanlab's precision/recall.
"""
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tokenfilter.tinygpt import CharTokenizer, TinyGPT, sequence_batch, per_token_losses
from tokenfilter.world import RELATIONS, declarative

RUNS = pathlib.Path("runs")


def value_nll(model, tok, rel, entity, value):
    s = declarative(rel, entity, value)
    x, y, w = sequence_batch(tok, [s], 64)
    losses = per_token_losses(model, x, y, w)[0].tolist()
    prefix = s[: len(s) - len(value) - 1]
    vals = [losses[i - 1] for i in range(len(prefix), len(s)) if 0 < i <= len(losses)]
    return sum(vals) / max(len(vals), 1)


def item_rel_ent(question):
    for rel, spec in RELATIONS.items():
        pre, post = spec["question"].split("{e}")
        if question.startswith(pre) and question.endswith(post):
            return rel, question[len(pre): len(question) - len(post) if post else None]
    raise ValueError(question)


def main():
    data = json.loads((RUNS / "data.json").read_text())
    tok = CharTokenizer(data["pretrain"] + [f"Q: {i['question']} A: {i['answer']}." for i in data["sft"]])
    base = TinyGPT(tok.vocab_size, seq_len=64)
    base.load_state_dict(torch.load(RUNS / "base.pt", weights_only=True))
    base.eval()

    # per-item pred_probs over its relation's candidate values
    items = []
    for it in data["sft"]:
        rel, ent = item_rel_ent(it["question"])
        values = RELATIONS[rel]["values"]
        nlls = [value_nll(base, tok, rel, ent, v) for v in values]
        m = min(nlls)
        probs = [math.exp(-(l - m)) for l in nlls]
        z = sum(probs)
        items.append({"rel": rel, "entity": ent, "values": values,
                      "answer": it["answer"], "corrupted": it["corrupted"],
                      "pred_probs": [p / z for p in probs]})

    # Cleanlab label-issue detection, one classification task per relation.
    # (Datalab's label check requires every candidate class to appear in the
    # observed labels, which breaks on relations with unused values; the
    # underlying confident-learning call handles unused classes directly.)
    from cleanlab.filter import find_label_issues
    import numpy as np
    flagged = set()
    for rel in sorted(RELATIONS):
        idxs = [i for i, it in enumerate(items) if it["rel"] == rel]
        values = RELATIONS[rel]["values"]
        labels = np.array([values.index(items[i]["answer"]) for i in idxs])
        pred_probs = np.array([items[i]["pred_probs"] for i in idxs])
        mask = find_label_issues(labels=labels, pred_probs=pred_probs)
        for m, i in zip(mask, idxs):
            if m:
                flagged.add(i)

    tp = sum(1 for i in flagged if items[i]["corrupted"])
    fp = sum(1 for i in flagged if not items[i]["corrupted"])
    fn = sum(1 for i, it in enumerate(items) if it["corrupted"] and i not in flagged)
    stats = {"flagged": len(flagged), "corrupted_total": sum(i["corrupted"] for i in items),
             "clean_total": sum(not i["corrupted"] for i in items),
             "true_positive": tp, "false_positive": fp, "false_negative": fn,
             "precision": round(tp / max(tp + fp, 1), 4),
             "recall": round(tp / max(tp + fn, 1), 4),
             "method": "cleanlab find_label_issues (confident learning); pred_probs from base "
                       "model scoring candidate values in trusted declarative format"}
    out = {"stats": stats,
           "flagged_items": sorted(flagged),
           "suggested_labels": [RELATIONS[it["rel"]]["values"][
               max(range(len(it["pred_probs"])), key=lambda j: it["pred_probs"][j])] for it in items]}
    (RUNS / "audit.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
