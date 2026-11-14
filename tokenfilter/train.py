"""Training loops: base pretraining and SFT with optional token masks."""
from __future__ import annotations

import math
import random

import torch

from .tinygpt import masked_loss, sequence_batch


def train(model, tokenizer, texts: list[str], *, steps: int, seq_len: int = 64,
          batch: int = 64, lr: float = 3e-3, weight_texts: list[list[float]] | None = None,
          seed: int = 0, log_every: int = 200) -> list[float]:
    """Train on texts. If weight_texts is given, it must parallel `texts`
    with per-character target weights (len == len(text)); masked tokens
    contribute zero to the loss."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * s / max(steps, 1))))
    model.train()
    history = []
    for step in range(steps):
        idxs = [rng.randrange(len(texts)) for _ in range(batch)]
        x, y, w = sequence_batch(tokenizer, [texts[i] for i in idxs], seq_len)
        if weight_texts is not None:
            for r, i in enumerate(idxs):
                wt = weight_texts[i][: len(y[r])]
                if wt:
                    w[r, : len(wt)] = torch.tensor(wt, dtype=torch.float)
        loss = masked_loss(model(x), y, w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % log_every == 0:
            history.append(round(float(loss), 4))
    return history
