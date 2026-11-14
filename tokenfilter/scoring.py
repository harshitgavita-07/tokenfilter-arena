"""Reference-model token scoring and selective masks (RHO-1 style).

RHO-1 (Lin et al. 2024, arXiv:2404.07965, "Not All Tokens Are What You
Need") scores training tokens with a reference model and trains only on
high-scoring ones. We implement two published variants:

- reference: keep tokens the REFERENCE model finds likely (low NLL)
- excess:    keep tokens by (L_reference - L_model), RHO-1's Selective
             Language Modeling score - tokens the reference knows but
             the model does not yet
- random:    ablation baseline - keep the SAME number of tokens, chosen
             uniformly at random (deterministic seed), so the filter is
             compared against an equal token budget, not just full SFT

Only answer tokens are scored; prompt tokens always stay masked (they
are context, not training signal).
"""
from __future__ import annotations


def select_mask(ref_losses: list[float], model_losses: list[float] | None,
                keep_frac: float, mode: str = "excess") -> list[int]:
    """Return a 0/1 mask over scored tokens. keep_frac of tokens kept by
    score; the rest are masked out of the loss."""
    if mode == "random":
        import hashlib, random
        seed = int(hashlib.md5(repr(ref_losses).encode()).hexdigest()[:8], 16)
        n = len(ref_losses)
        n_keep = max(1, round(n * keep_frac))
        keep = set(random.Random(seed).sample(range(n), n_keep))
        return [1 if i in keep else 0 for i in range(n)]
    if mode == "reference":
        scores = [-l for l in ref_losses]          # lower ref loss -> higher score
    elif mode == "excess":
        assert model_losses is not None
        scores = [m - r for r, m in zip(ref_losses, model_losses)]  # RHO-1 excess
    else:
        raise ValueError(f"unknown mode {mode}")
    n_keep = max(1, round(len(scores) * keep_frac))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = set(order[:n_keep])
    return [1 if i in keep else 0 for i in range(len(scores))]
