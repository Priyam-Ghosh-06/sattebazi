# Evaluation protocol

## Splits (walk-forward, SPEC §13.3)

| fold | training fixtures | validation (early stopping) | test (scored once) |
|---|---|---|---|
| F24 | 2019-01-01 → 2023-09-30 (5,376) | 2023-10-01 → 2023-12-31 (300) | 2024 (1,776 fixtures) |
| F25 | 2019-01-01 → 2024-09-30 (7,028) | 2024-10-01 → 2024-12-31 (424) | 2025 (1,801 fixtures) |
| FINAL | 2019-01-01 → 2025-09-30 | 2025-10-01 → 2025-12-31 | the hidden set |

Features always use all history before each row's cutoff (from 2005). Candidates are the season pseudo-squads.

## Staleness

History stops at 2025-12-31 and the hidden fixtures are later. Every fixture is therefore evaluated five times, with history cut
Δ ∈ {0, 90, 180, 270, 365} days before it; the **headline** is the equal-weight mean over the five. In training, Δ is drawn
per fixture from 0.1·δ₀ + 0.9·U[1, 450] days.

## Metrics

- **Team level (decides everything):** captured points of the chosen XI, the oracle XI's points, captured/oracle, captured
  minus a random valid XI, and the share of picked players who actually played.
- **Player level (the PS asks for them):** MAE and RMSE on squad rows (benched = 0) and on played rows, and the
  within-fixture Spearman rank correlation.
- **Probability model:** log-loss and Brier score of p (graph approach).
- **Comparisons:** paired bootstrap (2,000 resamples of fixtures) of the per-fixture captured difference, each fixture averaged
  over the five Δ.

**Which metric suits fantasy.** The task is to choose 11 of ~30, and the hidden set is scored on the chosen XI's actual points,
so captured points (and rank correlation inside a fixture) are the right yardstick. MAE is the wrong one to optimise: most
squad rows are benched zeros, per-match points are mostly noise (ICC ≈ 0.13), and the MAE-optimal prediction is the conditional
median, which is 0 for anyone with P(play) < 0.5 and picks worse teams (measured: a direct L1 model lost 6.9 points per match
[−11.2, −2.7] against the mean model). Every loss aimed at the ranking itself (LambdaRank, ListNet, SPO+) tied or lost too,
which is what you expect when the plain mean is already the right target.

## Decision rules (pre-registered, written before any test number)

- **E22** (SPEC §13): baseline CB (history length 64 or 128, chosen on validation) against the graph base 3a; replace 3a only
  if the 95% interval of CB − 3a is above 0 in **both** 2024 and 2025.
- **E23** (SPEC §14): refined ADV against 3a, same bar. Written after E22's numbers were seen, so reported as post-hoc to E22.
- **E24** (SPEC §15): 3b (graph features) against 3a, same bar.
- **R4, no second look:** no rerun, retune or reselection after a test number is seen.
