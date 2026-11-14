# tokenfilter-arena

A small, fully local, $0 MVP of an idea for Design Arena
(https://www.designarena.ai):

1. **Training-time token filter** - RHO-1-style selective fine-tuning
   (Lin et al. 2024, arXiv:2404.07965): a reference model scores each
   training token, and only the useful ones contribute to the loss.
   Here the reference scorer is a real open Hugging Face model
   (SmolLM2-135M, inference-only) and the trainable subject is a tiny
   GPT trained from scratch, so the whole experiment runs on 2 CPU
   cores in under 1GB RAM.
2. **Custom-model battle** - `scripts/03_arena.py` battles any two
   trained variants on the same prompts with blind human votes and
   Bradley-Terry-style ratings: the "bring your own fine-tune" loop in
   miniature.

## What the experiment measures

A controlled synthetic world (300 train + 60 test entities, 6 relation
types) where ground truth is known. 40% of the SFT answers are
deliberately corrupted (wrong value from the same relation). Three
questions, three metrics:

- clean_accuracy: does SFT teach the true facts? (exact match)
- noise_reproduction: does the model memorize the corrupted facts?
- the filter's job: keep the first high while pushing the second down.

## Measured results (this repo's own run, 2026-11-14)

| arm | tokens kept | clean_accuracy | noise_reproduction |
|---|---|---|---|
| base (no SFT)      | -    | 0.0%  | 0.0%  |
| sft_all            | 100% | 99.5% | 100%  |
| sft_filtered k=0.85| 85%  | 54.4% | 53.4% |
| sft_filtered k=0.70| 70%  | 23.6% | 29.7% |
| sft_filtered k=0.50| 50%  | 2.8%  | 6.8%  |

Raw numbers in `runs/metrics.json`; reproduce with the phases below.

Honest reading: the mechanism works directionally - filtering always
suppresses noise memorization, and at k=0.5 noise drops to 7% - but in
this toy setup the clean/noise trade-off is steep. That is expected
here and is NOT a result about real models: the reference model cannot
know our synthetic entities, so its clean-vs-corrupted signal is weak
(mean reference NLL 5.62 clean vs 6.09 corrupted). RHO-1's published
gains come from real data where the reference signal is far stronger.
What this MVP proves: the filter pipeline, the measurement harness, and
the battle loop run end-to-end and quantify the trade-off.

## Run it

```bash
pip install torch transformers pytest numpy   # CPU is enough
python3 scripts/00_gen_data.py        # build the world + corpora
python3 scripts/01_score_tokens.py    # reference-model scoring (HF, inference-only)
python3 scripts/exp.py pretrain_a     # base model, part 1
python3 scripts/exp.py pretrain_b     # base model, part 2
python3 scripts/exp.py sft_all        # arm: train on everything
KEEP_FRAC=0.7 python3 scripts/exp.py sft_filtered   # arm: filtered
python3 scripts/exp.py eval_sft_all
KEEP_FRAC=0.7 python3 scripts/exp.py eval_sft_filtered
python3 scripts/exp.py finalize       # filter selectivity stats
python3 scripts/03_arena.py runs/sft_all.pt runs/sft_filtered.pt   # blind battle
python3 -m pytest tests/ -q           # 5 tests
```

## Limits

- Toy scale on purpose: 420k-param char-level model, synthetic world;
  numbers demonstrate the mechanism and harness, not capability claims.
  "No hallucination" is not claimed or achievable; what is measurable
  is noise memorization rate.
- Reference scoring maps BPE tokens to covered characters
  (documented approximation); the answer span excludes the prompt.
- The shipped run's test split had a 3-entity name overlap with train
  (fixed in `world.py` for fresh runs; it only touched the
  non-headline test_format_rate metric, not clean_accuracy or
  noise_reproduction, which use train entities by design).
- The arena's Elo difference is the two-player Bradley-Terry closed
  form; it is a voting aid, not a benchmark.
