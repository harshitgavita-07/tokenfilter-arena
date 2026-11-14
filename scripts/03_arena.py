"""Battle two trained variants on the same prompts with blind votes.
Usage: python3 scripts/03_arena.py runs/sft_all.pt runs/sft_filtered.pt [--auto]"""
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tokenfilter.arena import run_battle
from tokenfilter.tinygpt import CharTokenizer, TinyGPT


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    auto = "--auto" in sys.argv
    data = json.loads(pathlib.Path("runs/data.json").read_text())
    tokenizer = CharTokenizer(data["pretrain"] + [f"Q: {i['question']} A: {i['answer']}." for i in data["sft"]])
    models = []
    for p in (a_path, b_path):
        m = TinyGPT(tokenizer.vocab_size, seq_len=64)
        m.load_state_dict(torch.load(p, weights_only=True))
        models.append(m)
    qs = [it["question"] for it in data["eval"]["clean_train"][:20]]
    if auto:
        for q in qs[:5]:
            from tokenfilter.evaluate import answer
            print(f"Q: {q}\n  A({pathlib.Path(a_path).stem}): {answer(models[0], tokenizer, q)}\n  B({pathlib.Path(b_path).stem}): {answer(models[1], tokenizer, q)}")
        return
    out = run_battle(models[0], models[1], tokenizer, qs, "runs/votes.jsonl")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
