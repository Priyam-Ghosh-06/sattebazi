# 2. Refined approach (ADV)

## Description

The refined approach keeps the baseline's inputs and training data but gives the network the structure of the problem: a memory
that fades in real time, separate "who plays" and "points if plays" heads, and an exact model of how a squad is cut to eleven.
Every piece is a flag, so each one's contribution can be measured against the baseline.

| component | what it adds | where |
|---|---|---|
| **A0a** continuous-time memory (CT-WKV) | each channel forgets per appearance *and* per day; with constant keys it reduces exactly to the EWMA bank of the graph model (test T2). Staleness lowers the memory's confidence, not its estimate (T6) | `scoring_model.py` |
| **A0b** factorised heads | a utility u (who plays) and μ (points if plays), bound to capacities (keeper, bowling option, batting depth) as u ⊙ (1 + R c) | `scoring_model.py` |
| **A2** context | fixture context (format, venue, event, month, overs, Δ) rescales every candidate (FiLM) | `scoring_model.py` |
| **A1** squad attention | own-team and opponent attention on the μ path (a missing opener pushes No. 3 up the order) | `scoring_model.py` |
| **A3 decision model** | the exact constrained choice over each squad: P(XI) ∝ exp(Σ u_i + φ_keep(#keepers) + φ_bowl(#bowling options)) with exactly n_XI players; the normaliser by dynamic programming, P(plays) = ∂ log Z / ∂u_i by autograd. A benched backup keeper is not pushed to "never plays": when the first choice is absent, the keeper potential makes him play | `decision_model.py` |
| **A5** privileged training on time | a month-by-month transition of the player state, learned from the real matches inside the staleness gap (available in training only), used at test time to roll the December state forward to the fixture date | `scoring_model.py`, `pipeline.py` (P2, P3) |
| **A6** distillation | a one-pass student trained on a mix of labels and the roll-out's predictions | `pipeline.py` (P4) |

Prediction: E[Y_i] = π_i · μ_i, with π from the choice layer. Training runs in four phases: P1 encoder + decoder (choice loss +
L2 on μ), P2 the transition (encoder frozen), P3 the decoder on rolled-forward states, P4 the students; each seed then picks its
final predictor (direct / roll-out / student) on validation.

**Not built:** the innings simulator (A4 in the design), which needs over-by-over bowling sequences that the ingestion does not
extract, and the overseas-player potential. Both remain in [`docs/design/ADVANCED.md`](../../docs/design/ADVANCED.md).

## Code

| file | role |
|---|---|
| [`config.py`](config.py) | budget of SPEC §14.2 (L = 128, lr 2e-3, P1 ≤ 15 epochs, P2 6, P3 ≤ 6, P4 λ grid), counters of the choice layer |
| [`ingestion.py`](ingestion.py) | the baseline token store + bench tokens (season rule, cutoff-independent), capacities as of c, the XI-size rule (11 or 12) |
| [`preprocessing.py`](preprocessing.py) | per-fixture tensors without cutoff-relative inputs (one encoder pass gives the state at any cutoff), batching by team |
| [`scoring_model.py`](scoring_model.py) | CT-WKV encoder, utility/μ decoder with A1/A2, transition g, student |
| [`decision_model.py`](decision_model.py) | A3: log Z by DP with saturating counters, the choice loss, exact marginals; `sum_to_n` (the τ shift it replaces) |
| [`pipeline.py`](pipeline.py) | phases P1–P4 with a checkpoint every epoch and resume, prediction of every predictor, packaging, smoke, overfit |
| [`tests/test_refined.py`](../../tests/test_refined.py) | T1, T2 (EWMA containment, err 1.7e-16), T6, T11 (gap tokens never reach the X₁ path), T14 (DP = brute force over 364 subsets, err 6e-15), T16 gauge, T4, T7 |

```bash
python src/refined/pipeline.py train --split F25 --seed 0 --device cuda:0
python src/train.py --approach refined --device cuda        # FINAL two-seed model (not trained: ADV did not pass its rule)
```

Trained on Kaggle (2× T4): 4 runs (F24/F25 × 2 seeds) in 2.9 h, P1 epochs 230–310 s.

## Results

| | 2024 | 2025 |
|---|---|---|
| ADV captured/oracle | 0.6234 | **0.6326** |
| ADV − graph base 3a | −0.90 [−3.52, +1.63] | +2.82 [−0.06, +5.61] |
| **ADV − baseline CB128** | **+4.45 [+1.77, +7.22]** | **+4.76 [+2.15, +7.40]** |
| A3 choice layer − τ shift (same weights) | +0.49 [−0.35, +1.27] | −0.47 [−1.27, +0.37] |
| A5 roll-out − direct | +0.91 [−0.07, +1.94] | +0.17 [−0.76, +1.08] |
| A6 student − direct | −17.54 [−20.34, −14.64] | −14.02 [−16.84, −11.37] |
| MAE / RMSE on squad rows | 25.90 / 36.78 | 25.53 / 38.33 |
| within-fixture Spearman | 0.341 | 0.359 |

ADV − 3a by staleness (descriptive):

| Δ (days) | 0 | 90 | 180 | 270 | 365 |
|---|---|---|---|---|---|
| 2024 | −6.72 | −0.33 | −1.38 | +1.42 | +2.52 |
| 2025 | −2.44 | +3.32 | **+5.58** | +2.14 | **+5.50** |

![training phases](../../results/refined/training_phases.png)
![component ladder](../../results/refined/component_ladder.png)

**Reading.**
- Structure pays: the same inputs with the refined architecture beat the learn-everything baseline by ~4.5 points per fixture
  in both years.
- Against the gradient-boosted 3a it ties (rule: B0s kept). The whole of 3a's lead is at Δ = 0, where "played the team's last
  match" is still fresh; as history goes stale (the regime of the hidden set) ADV leads, significantly at Δ = 180 and 365 in 2025.
- The exact choice layer ties the τ shift on captured points: the ranking of who is most likely to play barely changes. The
  roll-forward (A5) leans positive and was picked on validation by 3 of 4 seeds. The student failed badly: it saw neither the
  squad nor the choice layer, so it could not learn competition for places.
- ADV's expected points are well ordered but badly calibrated in level in 2025 (validation MSE ≈ 2,100 points² vs ≈ 1,180 for
  the others, see the overall validation figure): captured/oracle only uses the order.

Full numbers: [`results/experiments/E23_adv/`](../../results/experiments/E23_adv/) (`decision.md`, `metrics.json`, `bootstrap.json`, `by_delta_descriptive.json`).
