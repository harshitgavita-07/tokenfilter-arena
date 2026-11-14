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

## v2: Cleanlab data audit (2026-11-14)

The token-level filter above failed because no model can tell a made-up
fact is wrong from the text alone. v2 audits the training data AGAINST A
TRUSTED REFERENCE before training, following the HF cookbook "Detecting
Issues in a Text Dataset with Cleanlab"
(https://huggingface.co/learn/cookbook/en/issues_in_text_dataset) with
the cleanlab library (https://github.com/cleanlab/cleanlab):

1. Trusted reference = the pretrain corpus (true facts only), which the
   shared base model memorized. It scores every candidate value of each
   SFT item in the trusted declarative format -> pred_probs. (Measured:
   in this format the same base model separates clean vs corrupted
   answers strongly - NLL ~0.00-0.07 vs ~1.1-2.3 - while in Q/A format
   its answer-span NLL does not separate them at all, P(corrupted >
   clean) = 0.50 over all 300 items. Same model, same facts, two
   formats.)
2. cleanlab confident learning (`find_label_issues`) flags likely
   mislabeled items, one classification task per relation.
3. Retrain after dropping or correcting flagged items.

### What the metrics actually measure

Memorization of the model's own training data, not generalization:
- clean_accuracy (n=182): exact match on the UNCORRUPTED SFT facts
  (train entities).
- noise_reproduction (n=118): exact match on the CORRUPTED SFT facts,
  scored against the corrupted value the model was shown.
- The two sets are disjoint items (one SFT item per entity), so an arm
  can score 100% on both by memorizing everything (sft_all does).
- Held-out 60 test entities are used for test_format_rate only; they
  are never trained or audited on (entity names disjoint, verified).
- Fixed split + seeds: world seed 7, pretrain seed 1, SFT seed 2; the
  exact benchmark is the committed runs/data.json; decoding is greedy.

### Audit quality (measured against known injected noise)

cleanlab flagged 94 of 300 items: 93 of 118 corrupted caught, 1 clean
item wrongly flagged -> precision 98.9%, recall 78.8%. Every flagged
corrupted item's suggested label was the true value; the one clean
flag (Isismon) was "corrected" to a wrong value, which is exactly the
1-fact clean-accuracy loss in the fix arm.

### Results (one run, one eval)

| arm | clean_accuracy | noise_reproduction |
|---|---|---|
| sft_all (no audit)                       | 100%  | 100%  |
| sft_random k=0.70 (token budget)         | 60.4% | 56.8% |
| sft_filtered k=0.70 (token-level RHO-1)  | 30.8% | 33.9% |
| sft_randdrop (drop 94 RANDOM items)      | 73.1% | 68.6% |
| sft_refonly (correct ALL to reference)   | 81.3% | 2.5%  |
| **sft_audit_drop (cleanlab drop)**       | **99.5%** | **27.1%** |
| **sft_audit_fix (cleanlab + correct flagged)** | **99.5%** | **21.2%** |

What each control isolates:
- Data reduction alone (randdrop) fails: random removal keeps ~69% of
  the noise and loses ~27% of the clean facts.
- Reference access alone (refonly) is the best noise-killer (2.5%) but
  it "corrects" clean items wherever the reference is unsure, losing
  ~19pt of clean accuracy.
- cleanlab's contribution on top of the reference: its confident-learning
  gate restricts corrections to high-confidence disagreements, keeping
  clean accuracy at 99.5% - the price is the residual 21.2% noise,
  exactly the 25 corrupted items it missed (25/118, fully accounted).

### Scope honesty

- This is detection of INJECTED noise in a controlled world with a
  known trusted reference. It is not evidence about unknown errors in
  real datasets, where the hard part is choosing what to trust as the
  reference (vetted corpus, unit tests for code, human-verified labels).
- The reference had no special access: every arm starts from the same
  base.pt pretrained on the trusted corpus; only the audit pipeline
  uses it to check labels.
- All numbers are from the single committed run (runs/metrics.json);
  no arm was retried or cherry-picked.

### Run the audit yourself

```bash
pip install cleanlab
python3 scripts/04_audit.py                 # prints precision/recall vs known noise
python3 scripts/exp.py sft_audit_fix        # retrain on cleanlab-corrected data
python3 scripts/exp.py eval_sft_audit_fix
python3 scripts/03_arena.py runs/sft_all.pt runs/sft_audit_fix.pt
```

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
