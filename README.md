# tokenfilter-arena

A small, fully local, $0 MVP of an idea for Design Arena
(https://www.designarena.ai):

1. **Training-time token filter** - RHO-1-style selective fine-tuning
   (Lin et al. 2024, arXiv:2404.07965): a reference model scores each
   training token and only the selected ones contribute to the loss.
   The reference scorer here is a real open Hugging Face model
   (SmolLM2-135M, bf16, **inference-only - it is never trained or
   fine-tuned**). The model being trained is a tiny 420k-parameter
   char-level GPT **trained from scratch**, so the whole experiment
   runs on 2 CPU cores in under 1GB RAM.
2. **Custom-model battle** - `scripts/03_arena.py` battles any two
   trained variants on the same prompts with blind human votes and a
   Bradley-Terry-style rating: the "bring your own fine-tune" loop in
   miniature. Pretrained checkpoints ship in `runs/`, so the battle
   works straight from a fresh clone without retraining.

## What the experiment measures

A controlled synthetic world (300 train + 60 test entities, 6 relation
types, disjoint name sets) where ground truth is known. ~40% of the
SFT answers are deliberately corrupted (wrong value, same relation).
Two headline metrics:

- **clean_accuracy**: exact-match rate on true facts (did SFT teach them?)
- **noise_reproduction**: exact-match rate on corrupted facts (did the
  model memorize the noise?)

## Measured results (this repo's own run, 2026-11-14)

| arm | tokens kept | clean_accuracy | noise_reproduction |
|---|---|---|---|
| base (no SFT)             | -    | 0.0%  | 0.0%  |
| sft_all                   | 100% | 100%  | 100%  |
| sft_random k=0.70 (baseline) | 70% | 60.4% | 56.8% |
| sft_filtered k=0.85       | 85%  | 47.3% | 61.0% |
| sft_filtered k=0.70       | 70%  | 30.8% | 33.9% |
| sft_filtered k=0.50       | 50%  | 8.8%  | 8.5%  |

Raw numbers in `runs/metrics.json` + `runs/metrics_*.json`.

Honest reading - this is a **negative result at toy scale**, reported
as measured:

- Unfiltered SFT memorizes everything (clean and corrupted: 100%/100%).
- Any token budget reduces memorization, but the RHO-1-style excess
  filter does **not** beat an equal-budget random mask here: at k=0.70
  the filter suppresses noise more than random (33.9% vs 56.8%) but
  also loses more clean accuracy (30.8% vs 60.4%).
- Direct selectivity check (`metrics.json -> _filter_stats`): the
  filter drops corrupted tokens 28.6% of the time vs clean tokens
  28.8% - essentially no clean/corrupted discrimination.
- Expected cause: the reference model cannot know our synthetic
  entities, so its clean-vs-corrupted signal is weak (mean reference
  NLL 5.63 clean vs 6.09 corrupted). RHO-1's published gains come from
  real data where the reference signal is far stronger.

What the MVP demonstrates: an end-to-end harness (reference scoring ->
selective SFT -> falsifiable metrics -> blind human battle loop) that
runs for $0 and can *measure* whether filtering helps - including
catching when it doesn't. It is NOT evidence that the filter improves
real models, and no training-efficiency or hallucination-reduction
benefit is claimed.

## Try it (WSL / Linux / macOS)

Prereqs: Python 3.10+ and pip (on Windows use WSL:
`wsl --install`, then `sudo apt update && sudo apt install -y python3-pip`).

```bash
git clone https://github.com/harshitgavita-07/tokenfilter-arena.git
cd tokenfilter-arena
pip install torch numpy pytest        # CPU wheels are enough
python3 -m pytest tests/ -q           # 6 tests
python3 scripts/03_arena.py runs/sft_all.pt runs/sft_filtered.pt   # blind battle, ships with checkpoints
```

Full rerun from scratch (adds `transformers` for the reference scorer):

```bash
pip install transformers
python3 scripts/00_gen_data.py        # build the world + corpora
python3 scripts/01_score_tokens.py    # SmolLM2-135M reference scoring (inference-only)
python3 scripts/exp.py pretrain_a && python3 scripts/exp.py pretrain_b
python3 scripts/exp.py sft_all
KEEP_FRAC=0.7 python3 scripts/exp.py sft_filtered
KEEP_FRAC=0.7 python3 scripts/exp.py sft_random      # equal-budget baseline
python3 scripts/exp.py eval_sft_all
KEEP_FRAC=0.7 python3 scripts/exp.py eval_sft_filtered
KEEP_FRAC=0.7 python3 scripts/exp.py eval_sft_random
python3 scripts/exp.py finalize
```

## Limits

- Toy scale on purpose: 420k-param char-level model, synthetic world.
  Numbers demonstrate the harness, not a capability claim.
- Reference scoring maps BPE tokens to covered characters (documented
  approximation); only answer tokens are scored, prompts stay masked.
- The arena's rating is the two-player Bradley-Terry closed form - a
  voting aid, not a benchmark.
