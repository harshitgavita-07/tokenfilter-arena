"""Phased experiment runner (each phase fits a short CPU budget).
Phases: pretrain -> sft_all -> sft_filtered -> eval -> finalize"""
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tokenfilter.evaluate import metric_pack
from tokenfilter.scoring import select_mask
from tokenfilter.tinygpt import CharTokenizer, TinyGPT, per_token_losses, sequence_batch
from tokenfilter.train import train

import os
KEEP_FRAC = float(os.environ.get("KEEP_FRAC", "0.7"))
SEQ_LEN = 64
RUNS = pathlib.Path("runs")


def sft_text(it):
    return f"Q: {it['question']} A: {it['answer']}."


def load_all():
    data = json.loads((RUNS / "data.json").read_text())
    ref = json.loads((RUNS / "ref_scores.json").read_text())
    tok = CharTokenizer(data["pretrain"] + [sft_text(i) for i in data["sft"]])
    return data, ref, tok


def weight_list(it, r, mode):
    text = sft_text(it)
    ans_w = [1] * len(r["char_positions"]) if mode == "all" else \
        select_mask(r["ref_losses"], r["base_losses"], KEEP_FRAC, mode=mode)
    w = [0.0] * len(text)
    # boundary fix: always train the space right after "A:" (char
    # span_start-1) so decoding starts on-distribution; without it the
    # first generated char is off-distribution and cascades.
    if 0 < r["span_start"] - 1 <= len(w):
        w[r["span_start"] - 2] = 1.0
    for c, m in zip(r["char_positions"], ans_w):
        if c - 1 < len(w):
            w[c - 1] = float(m)
    return w


def phase_pretrain_a():
    data, ref, tok = load_all()
    torch.manual_seed(0)
    base = TinyGPT(tok.vocab_size, seq_len=SEQ_LEN)
    h1 = train(base, tok, data["pretrain"], steps=400, seq_len=SEQ_LEN, seed=1)
    torch.save(base.state_dict(), RUNS / "base.pt")
    print("pretrain half-1 losses:", h1)


def phase_pretrain_b():
    data, ref, tok = load_all()
    base = TinyGPT(tok.vocab_size, seq_len=SEQ_LEN)
    base.load_state_dict(torch.load(RUNS / "base.pt", weights_only=True))
    h2 = train(base, tok, data["pretrain"], steps=400, seq_len=SEQ_LEN, seed=1)
    torch.save(base.state_dict(), RUNS / "base.pt")
    print("pretrain half-2 losses:", h2)
    for it, r in zip(data["sft"], ref):
        x, y, w = sequence_batch(tok, [sft_text(it)], SEQ_LEN)
        losses = per_token_losses(base, x, y, w)[0].tolist()
        r["base_losses"] = [round(losses[c - 1], 4) if 0 < c <= len(losses) else 0.0
                            for c in r["char_positions"]]
    (RUNS / "ref_scores.json").write_text(json.dumps(ref))
    print("pretrain half-2 losses:", h2)


def phase_sft(name, mode):
    data, ref, tok = load_all()
    model = TinyGPT(tok.vocab_size, seq_len=SEQ_LEN)
    model.load_state_dict(torch.load(RUNS / "base.pt", weights_only=True))
    weights = [weight_list(it, r, mode) for it, r in zip(data["sft"], ref)]
    h = train(model, tok, [sft_text(i) for i in data["sft"]], steps=400,
              seq_len=SEQ_LEN, seed=2, weight_texts=weights, lr=1e-3)
    suffix = "" if (name == "sft_all" or KEEP_FRAC == 0.7) else f"_k{int(KEEP_FRAC*100)}"
    torch.save(model.state_dict(), RUNS / f"{name}{suffix}.pt")
    print(f"{name} losses:", h)


def phase_sft_audit(kind):
    """Train on the cleanlab-audited SFT set. kind=drop: remove flagged
    items; kind=fix: replace flagged answers with the audit's suggestion."""
    data, ref, tok = load_all()
    audit = json.loads((RUNS / "audit.json").read_text())
    flagged = set(audit["flagged_items"])
    suggested = audit["suggested_labels"]
    model = TinyGPT(tok.vocab_size, seq_len=SEQ_LEN)
    model.load_state_dict(torch.load(RUNS / "base.pt", weights_only=True))
    texts, weights = [], []
    for i, it in enumerate(data["sft"]):
        ans = it["answer"]
        if i in flagged:
            if kind == "drop":
                continue
            ans = suggested[i]
        text = f"Q: {it['question']} A: {ans}."
        a0 = text.index(" A: ") + 4
        w = [0.0] * len(text)
        for c in range(a0 - 2, len(text) - 1):  # space after "A:" .. final "."
            w[c] = 1.0
        texts.append(text)
        weights.append(w)
    h = train(model, tok, texts, steps=400, seq_len=SEQ_LEN, seed=2,
              weight_texts=weights, lr=1e-3)
    torch.save(model.state_dict(), RUNS / f"sft_audit_{kind}.pt")
    print(f"sft_audit_{kind} on {len(texts)} items, losses:", h)


def _audit_weights(text):
    a0 = text.index(" A: ") + 4
    w = [0.0] * len(text)
    for c in range(a0 - 2, len(text) - 1):
        w[c] = 1.0
    return w


def phase_sft_ablation(kind):
    """Controls isolating cleanlab vs trusted-reference vs data reduction.
    refonly: correct ALL items to the reference argmax (no cleanlab gate).
    randdrop: drop as many RANDOM items as cleanlab flagged."""
    import random as _r
    data, ref, tok = load_all()
    audit = json.loads((RUNS / "audit.json").read_text())
    suggested = audit["suggested_labels"]
    n_flag = len(audit["flagged_items"])
    drop = set(_r.Random(3).sample(range(len(data["sft"])), n_flag)) if kind == "randdrop" else set()
    model = TinyGPT(tok.vocab_size, seq_len=SEQ_LEN)
    model.load_state_dict(torch.load(RUNS / "base.pt", weights_only=True))
    texts, weights = [], []
    for i, it in enumerate(data["sft"]):
        if i in drop:
            continue
        ans = suggested[i] if kind == "refonly" else it["answer"]
        text = f"Q: {it['question']} A: {ans}."
        texts.append(text)
        weights.append(_audit_weights(text))
    h = train(model, tok, texts, steps=400, seq_len=SEQ_LEN, seed=2,
              weight_texts=weights, lr=1e-3)
    torch.save(model.state_dict(), RUNS / f"sft_{kind}.pt")
    print(f"sft_{kind} on {len(texts)} items, losses:", h)


def phase_eval_one(name):
    data, ref, tok = load_all()
    suffix = "" if (name in ("base", "sft_all") or KEEP_FRAC == 0.7) else f"_k{int(KEEP_FRAC*100)}"
    model = TinyGPT(tok.vocab_size, seq_len=SEQ_LEN)
    model.load_state_dict(torch.load(RUNS / f"{name}{suffix}.pt", weights_only=True))
    key = ("base_no_sft" if name == "base" else name) + suffix
    path = RUNS / f"metrics_{key}.json"
    res = metric_pack(model, tok, data["eval"])
    path.write_text(json.dumps(res, indent=2))
    print(key, json.dumps(res))


def phase_finalize():
    data, ref, tok = load_all()
    metrics = {}
    for key in ["base_no_sft", "sft_all", "sft_filtered", "sft_random", "sft_audit_drop", "sft_audit_fix", "sft_refonly", "sft_randdrop"]:
        f = RUNS / f"metrics_{key}.json"
        if f.exists():
            metrics[key] = json.loads(f.read_text())
    dc = kc = dl = kl = 0
    for it, r in zip(data["sft"], ref):
        for m in select_mask(r["ref_losses"], r["base_losses"], KEEP_FRAC, mode="excess"):
            if it["corrupted"]:
                dc, kc = dc + (m == 0), kc + (m == 1)
            else:
                dl, kl = dl + (m == 0), kl + (m == 1)
    metrics["_filter_stats"] = {
        "keep_frac": KEEP_FRAC, "mode": "excess",
        "corrupted_tokens_dropped_rate": round(dc / max(dc + kc, 1), 4),
        "clean_tokens_dropped_rate": round(dl / max(dl + kl, 1), 4)}
    model = TinyGPT(tok.vocab_size, seq_len=SEQ_LEN)
    metrics["_config"] = {"model_params": model.n_params(), "seq_len": SEQ_LEN,
                          "pretrain_steps": 800, "sft_steps": 400,
                          "noise_frac": 0.4,
                          "reference": "HuggingFaceTB/SmolLM2-135M (bf16, inference-only)"}
    audit = json.loads((RUNS / "audit.json").read_text()) if (RUNS / "audit.json").exists() else None
    if audit:
        metrics["_audit_stats"] = audit["stats"]
    (RUNS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["_filter_stats"]))


if __name__ == "__main__":
    phase = sys.argv[1]
    {"pretrain_a": phase_pretrain_a, "pretrain_b": phase_pretrain_b,
     "sft_all": lambda: phase_sft("sft_all", "all"),
     "sft_filtered": lambda: phase_sft("sft_filtered", "excess"),
     "sft_random": lambda: phase_sft("sft_random", "random"),
     "eval_base": lambda: phase_eval_one("base"), "eval_sft_all": lambda: phase_eval_one("sft_all"), "eval_sft_filtered": lambda: phase_eval_one("sft_filtered"),
     "eval_sft_random": lambda: phase_eval_one("sft_random"),
     "sft_audit_drop": lambda: phase_sft_audit("drop"), "sft_audit_fix": lambda: phase_sft_audit("fix"),
     "eval_sft_audit_drop": lambda: phase_eval_one("sft_audit_drop"),
     "eval_sft_audit_fix": lambda: phase_eval_one("sft_audit_fix"),
     "sft_refonly": lambda: phase_sft_ablation("refonly"),
     "sft_randdrop": lambda: phase_sft_ablation("randdrop"),
     "eval_sft_refonly": lambda: phase_eval_one("sft_refonly"),
     "eval_sft_randdrop": lambda: phase_eval_one("sft_randdrop"),
     "finalize": phase_finalize}[phase]()
