# 1. Baseline: the "caveman" model (CB)

## Description

The baseline is deliberately naive about cricket: **one network is asked to learn every mapping by itself**. It never hears the
words "keeper", "bowler", "form" or "probability of playing". It gets raw history and must output one number per squad member,
the expected fantasy score E[Y].

```
raw history of each candidate (up to L tokens, all dated before the cutoff c)
  -> Transformer history encoder (per candidate)            -> one vector per candidate
  -> set decoder over both squads + a fixture token          -> candidates see each other (competition for places)
  -> Ŷ_i = E[Y_i]    -> exact XI optimiser
```

- **Tokens.** One token per past appearance (16 raw counts: runs, balls, 4s, 6s, wickets, dots, catches, …, plus the realised
  fantasy points) and one per match of the fixture team the player *missed* in the last 365 days (so the model can see who is
  a regular and who was dropped). Each token carries the match context (format, event, venue, opponent, team) and its age.
- **Fixture token.** Format, gender, team type, event, venue, month, scheduled overs and log(1 + Δ), so the model knows how
  stale its history is.
- **Loss.** Squared error on Y (0 for benched players). Its minimiser is E[Y | inputs] = p·μ, so the product of "plays" and
  "points if plays" is learned without ever being named.
- **What it keeps that is not modelling:** the history cutoff and staleness (data correctness), pseudo-squads as candidate sets
  (the only source of zeros), roles for the output only (the PS needs them), the exact ILP.
- Two history lengths were pre-registered: **CB64** and **CB128** tokens, each trained with two seeds and averaged.

## Code

| file | role |
|---|---|
| [`config.py`](config.py) | the fixed budget (d = 64, 2 + 2 layers, 4 heads, AdamW 1e-3, cosine, EMA 0.999, ≤ 20 epochs, patience 4), paths, seeding |
| [`ingestion.py`](ingestion.py) | player-match and team-match tables → numpy token store; vocabularies and scaling from the training fold only |
| [`preprocessing.py`](preprocessing.py) | candidate sets (pseudo-squads, padded to 20 per team in training), Δ per fixture and epoch, token tensors, batching |
| [`scoring_model.py`](scoring_model.py) | the network: token embedding, Transformer encoder, set decoder, output head; batch prediction |
| [`decision_model.py`](decision_model.py) | Ŷ → exact XI (shared ILP); captured/oracle on validation |
| [`pipeline.py`](pipeline.py) | training with a checkpoint every epoch and resume, packaging, test-time prediction, smoke and overfit checks |
| [`tests/test_baseline.py`](../../tests/test_baseline.py) | T1 causality + time travel (388 candidates), T1c absence rule, T4 permutation equivariance, T7 cold start |

```bash
python src/baseline/pipeline.py train --arm CB128 --split F24 --seed 0 --device cuda:0   # one run (checkpoints, resumes)
python src/train.py --approach baseline --device cuda                                   # the two-seed FINAL model
python src/evaluate.py --model-path model_artifacts/baseline_cb128_2025-12-31.pkl --test-file data/mock/test_fixtures.json --output results/baseline/mock_predictions_cb128.csv
```

Trained on Kaggle (2× T4): 12 runs (CB64/CB128 × F24/F25/FINAL × 2 seeds) in 3.4 h, 80–140 s per epoch, 13–20 epochs each.

## Results

Test years, headline = mean over Δ ∈ {0, 90, 180, 270, 365}; differences are captured points per fixture with 95% paired bootstrap.

| | 2024 | 2025 |
|---|---|---|
| CB64 captured/oracle | 0.6143 | 0.6229 |
| **CB128 captured/oracle** (chosen on validation, rule R2) | **0.6165** | **0.6254** |
| CB128 − graph base 3a | −5.34 [−8.26, −2.49] | −1.93 [−4.74, +0.90] |
| CB128 − CB64 | +1.83 [−0.11, +3.81] | +1.67 [−0.21, +3.53] |
| MAE / RMSE on squad rows | 24.75 / 33.91 | 24.22 / 33.67 |
| within-fixture Spearman | 0.318 | 0.341 |
| predicted team total vs realised | −13.1% | −15.7% |

By staleness (2025): 0.662 at Δ = 0, 0.620 at 90, 0.619 at 180, 0.614 at 270, 0.611 at 365. Almost all of its gap to 3a is at
Δ = 0; at Δ = 365 it is +3.2 points per fixture [−0.5, +7.1] **ahead** of 3a (descriptive).

![training curves](../../results/baseline/training_curves.png)

**Reading.** The caveman model learns the task (0.54 → 0.65 on validation) but does not beat the engineered gradient-boosted
model, and it under-predicts team totals by 13–16%: nothing tells it that exactly 11 players take the field. Its relative
strength at large Δ (where "played the last match" is stale for everyone) is the first hint of what the refined approach
exploits. Pre-registered expectation was "CB ties 3a": held in 2025, lost in 2024.

Full numbers: [`results/experiments/E22_final/`](../../results/experiments/E22_final/) (`decision.md`, `metrics.json`, `bootstrap.json`).
