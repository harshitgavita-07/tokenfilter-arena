import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tokenfilter.data import build_sft, pretrain_sentences
from tokenfilter.scoring import select_mask
from tokenfilter.tinygpt import CharTokenizer, TinyGPT, per_token_losses, sequence_batch
from tokenfilter.train import train
from tokenfilter.world import RELATIONS, build_world


def test_world_disjoint_and_typed():
    w = build_world()
    te = {e for _, e, _ in w["train"]}
    assert not te & {e for _, e, _ in w["test"]}
    for rel, e, v in w["train"]:
        assert v in RELATIONS[rel]["values"]


def test_sft_corruption_same_relation_values():
    w = build_world()
    sft = build_sft(w)
    bad = [i for i in sft if i["corrupted"]]
    assert 0.3 < len(bad) / len(sft) < 0.5
    for i in bad:
        assert i["answer"] != i["clean_answer"]
        assert i["answer"] in RELATIONS[i["relation"]]["values"]


def test_select_mask_fraction_and_mode():
    ref = [1.0, 5.0, 0.5, 3.0]
    base = [2.0, 5.0, 0.5, 1.0]
    m = select_mask(ref, base, 0.5, mode="excess")
    assert sum(m) == 2
    # excess scores: [1, 0, 0, -2] -> keep tokens 0 and 1(or 2, tie)
    assert m[0] == 1
    m2 = select_mask(ref, None, 0.5, mode="reference")
    assert sum(m2) == 2 and m2[2] == 1 and m2[0] == 1  # lowest ref NLL kept


def test_training_reduces_masked_loss_on_answer_span():
    # the boundary case that bit us: "A:" + space + value must all be learnable
    text = "Q: What country is Tandenis in? A: France."
    tok = CharTokenizer([text])
    m = TinyGPT(tok.vocab_size, d=64, layers=1, seq_len=64)
    w = [0.0] * len(text)
    span = text.index(" A:") + 3  # first answer char (the space)
    for c in range(span, len(text)):
        w[c - 1] = 1.0
    train(m, tok, [text], steps=80, seq_len=64, batch=4, lr=3e-3, weight_texts=[w], log_every=999)
    x, y, ww = sequence_batch(tok, [text], 64)
    losses = per_token_losses(m, x, y, ww)[0].tolist()
    assert max(losses[span - 1:len(text) - 1]) < 0.5


def test_tokenizer_roundtrip():
    t = CharTokenizer(["hello world"])
    assert t.decode(t.encode("hello")) == "hello"


def test_select_mask_random_equal_budget_deterministic():
    ref = [1.0, 0.5, 2.0, 3.0, 0.1, 1.5]
    m1 = select_mask(ref, None, 0.5, mode="random")
    m2 = select_mask(ref, None, 0.5, mode="random")
    assert m1 == m2 and sum(m1) == 3 and len(m1) == 6
