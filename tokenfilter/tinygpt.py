"""A tiny char-level GPT trained from scratch (torch, CPU).

The trainable subject of the experiment. Deliberately small (~0.5M
params) so full training runs in minutes on a laptop CPU with <1GB RAM.
Char-level keeps the tokenizer trivial and exact-match evaluation clean.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharTokenizer:
    def __init__(self, texts: list[str]):
        chars = sorted({c for t in texts for c in t})
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}  # 0 = pad
        self.itos = {i: c for c, i in self.stoi.items()}
        self.pad = 0

    @property
    def vocab_size(self) -> int:
        return len(self.stoi) + 1

    def encode(self, s: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids if i != self.pad)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, seq_len: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.register_buffer("mask", torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), 1))

    def forward(self, x):
        T = x.shape[1]
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=self.mask[:T, :T])
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab: int, d: int = 128, layers: int = 2, heads: int = 4, seq_len: int = 64):
        super().__init__()
        self.seq_len = seq_len
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq_len, d)
        self.blocks = nn.ModuleList([Block(d, heads, seq_len) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def sequence_batch(tokenizer: CharTokenizer, texts: list[str], seq_len: int):
    """Pack texts into (x, y, w) tensors: input ids, next-token targets,
    and a per-target weight (1 = train on this token, 0 = mask)."""
    x = torch.zeros(len(texts), seq_len, dtype=torch.long)
    y = torch.zeros(len(texts), seq_len, dtype=torch.long)
    w = torch.zeros(len(texts), seq_len, dtype=torch.float)
    for i, t in enumerate(texts):
        ids = tokenizer.encode(t)[: seq_len + 1]
        if len(ids) < 2:
            continue
        xi, yi = ids[:-1], ids[1:]
        x[i, : len(xi)] = torch.tensor(xi)
        y[i, : len(yi)] = torch.tensor(yi)
        w[i, : len(yi)] = 1.0
    return x, y, w


def masked_loss(logits, y, w):
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    loss = loss.reshape(y.shape)
    denom = w.sum().clamp_min(1.0)
    return (loss * w).sum() / denom


@torch.no_grad()
def per_token_losses(model: TinyGPT, x, y, w):
    """Per-target-token NLL (masked positions return 0). Used for
    RHO-1-style excess scoring: score = L_ref - L_model."""
    model.eval()
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none")
    return (loss.reshape(y.shape) * w)
