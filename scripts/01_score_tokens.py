"""Score SFT answer tokens with the reference model (SmolLM2-135M, an
open HF model, inference-only). Per-CHARACTER reference losses are cached
so the training phase never loads the reference model (1GB RAM budget).

Approximation, documented: BPE tokens are mapped to the characters they
cover; each covered char gets that token's NLL. The answer span is the
text after ' A: '."""
import json
import sys
import pathlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    data = json.loads(pathlib.Path("runs/data.json").read_text())
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M", dtype=torch.bfloat16)
    model.eval()

    out = []
    for it in data["sft"]:
        prompt = f"Q: {it['question']} A:"
        full = f"{prompt} {it['answer']}."
        enc = tok(full, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"]).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        ids = enc["input_ids"][0].tolist()
        # NLL of token t at position t (predicted from <t)
        nll = [0.0] + [-float(logp[t - 1, ids[t]]) for t in range(1, len(ids))]
        span_start = len(prompt) + 1  # char index where answer begins
        char_loss = {}
        for t, ((s, e), l) in enumerate(zip(offsets, nll)):
            if t == 0 or s == e:
                continue
            if e <= span_start:
                continue
            for c in range(max(s, span_start), e):
                char_loss[c] = l
        ans_chars = sorted(char_loss)
        out.append({"entity": it["entity"], "corrupted": it["corrupted"],
                    "answer": it["answer"], "span_start": span_start,
                    "char_positions": ans_chars,
                    "ref_losses": [round(char_loss[c], 4) for c in ans_chars]})
    pathlib.Path("runs/ref_scores.json").write_text(json.dumps(out))
    corr = [o for o in out if o["corrupted"]]
    clean = [o for o in out if not o["corrupted"]]
    import statistics as st
    mc = st.mean(l for o in clean for l in o["ref_losses"])
    mn = st.mean(l for o in corr for l in o["ref_losses"])
    print(f"scored {len(out)} items; mean ref NLL clean={mc:.3f} corrupted={mn:.3f}")


if __name__ == "__main__":
    main()
