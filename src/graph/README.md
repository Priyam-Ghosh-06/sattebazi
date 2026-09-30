# 3. Graph approach

## Description

The graph approach has two layers.

**3a: the base model (this is the submitted model).** A gradient-boosted, fully engineered version of the p·μ decomposition:

- **p = P(plays | squad)**: LightGBM logistic model on squad rows (benched = 0), then a per-team shift τ so that the
  probabilities of each squad add up to the XI size.
- **μ = E[points | plays]**: LightGBM L2 model on the rows of players who played.
- **score = p·μ** → the exact XI optimiser.
- **Features** (all strictly before the cutoff c): form as a bank of exponentially weighted averages (half-lives 3, 10, 30
  appearances) plus career means, opportunity (batting position, balls faced and bowled), appearance shares for this team,
  where a player ranks inside his own squad, fantasy-role evidence, the opponent's and his own squad's strength and a shrunk
  venue mean.
- **Staleness-aware:** every row is built as of c = date − Δ (one Δ per training fixture, the Δ grid for evaluation), day
  gaps are measured to the fixture date and Δ itself is a feature. Gate G7 rebuilds rows from truncated history and requires
  exact equality.

**3b: the batter × bowler graph.** Every Cricsheet delivery is an edge between a batter and a bowler (645k pair-match records,
2.87M balls). A player's points depend on whom he faces, so:

1. **Edges carry conditions.** For runs and for dismissals separately, the rate of pair (i, j) is
   `n_ij · exp(β + a_i + b_j + u_i · v_j)`: batter quality a_i, bowler quality b_j, and a pair term u_i · v_j.
2. **Generalisation.** The pair term is a product of two short learned vectors (rank 4), so a batter who does well against a
   *kind* of bowler does well against every bowler of that kind, including ones he never faced. The bowler vectors group
   bowlers by type without any label (Cricsheet has no bowling-style field). Priors shrink thin pairs.
3. **Causal snapshots.** The factorisation is refitted every month on deliveries dated before the snapshot (100 snapshots,
   2017-10 → 2026-01). A row with cutoff c uses the latest snapshot ≤ c. Gate G8 checks it, including a bit-identical refit
   from truncated data.
4. **Pre-toss use.** Nobody knows who will bowl to whom, so each candidate gets the expected effect against the opponent squad,
   weighted by how likely each opponent is to play and bowl (or bat): 10 features (`x_mu_*`) added to 3a.

## Code

| file | role |
|---|---|
| [`preprocessing.py`](preprocessing.py) | staleness-aware feature table (one copy per fixture and Δ) + gate G7 |
| [`scoring_model.py`](scoring_model.py) | the LightGBM p and μ models, early stopping on the last 3 months, the τ shift, prediction |
| [`ingestion.py`](ingestion.py) | the delivery graph: per (match, batter, bowler) balls faced, runs, bowler-credited dismissals |
| [`graph_model.py`](graph_model.py) | Poisson factorisation, monthly snapshots, opponent-expected graph features (training and test time), gate G8 |
| [`experiments.py`](experiments.py) | CLI: `ingest`, `snapshots`, `features`, `g8`, `folds` (3a on F24/F25, the E22 reference), `e24` (3b vs 3a) |
| decision | score = p·μ → the shared exact ILP (`src/shared/optimization/selection.py`) |
| [`../experiments/`](../experiments/) | how 3a was built: E2 noise ceiling, E3 baselines, E4 memory length, E5 losses, E6 p-model, E8–E15 structure and context (log: `results/experiments/LOG.md`) |

```bash
python src/graph/preprocessing.py && python src/graph/preprocessing.py --g7                     # 3a features + G7
python src/train.py --train-start 2000-01-01 --train-end 2025-12-31                              # 3a (submitted), ~1 min CPU
for s in ingest snapshots features g8; do python src/graph/experiments.py $s; done              # the graph (~15 min CPU)
python src/train.py --approach graph --graph-features --out model_artifacts/graph_features_2025-12-31.pkl   # 3b
python src/graph/experiments.py e24                                                               # 3b vs 3a backtest
```

## Results

| | 2024 | 2025 |
|---|---|---|
| **3a captured/oracle** (submitted) | **0.6245** | 0.6284 |
| 3a captured / oracle / random XI (points) | 437.0 / 702.4 / 317.1 | 437.9 / 700.4 / 314.1 |
| 3a share of picked players who played | 85.7% | 86.2% |
| 3a MAE / RMSE, squad rows | 24.87 / 33.17 | 24.70 / 33.08 |
| 3a MAE / RMSE, played rows (μ) | 26.44 / 34.44 | 26.44 / 34.47 |
| 3a within-fixture Spearman | 0.346 | 0.361 |
| 3a p model: log-loss / Brier | 0.487 / 0.161 | 0.485 / 0.161 |
| 3b captured/oracle | 0.6236 | **0.6308** |
| **3b − 3a** | −0.77 [−2.10, +0.62] | **+1.59 [+0.23, +2.92]** |

3a by staleness (2025): 0.678 at Δ = 0, 0.622 at 90, 0.618 at 180, 0.617 at 270, 0.607 at 365. By segment (2025):
women's 0.682 vs men's 0.601, international 0.664 vs club 0.589.

![curves and gain](../../results/graph/curves_and_gain.png)
![graph feature gain](../../results/graph/graph_feature_gain.png)

**Reading.**
- 3a is the strongest model at Δ = 0 and on average, and it is the submission; it trains in about a minute on a CPU.
- The graph adds a significant +1.6 points per fixture in 2025 (growing from +1.4 at Δ = 0 to +2.0 at Δ = 365) and ties in
  2024, so under the pre-registered two-year rule 3a stays. Most of the graph's gain is opponent quality ("whom you face");
  the pure pair terms (`x_mu_*_int`) carry about a quarter as much gain.
- **Next for the graph pipeline:** a graph neural network where edge conditions depend on the squad context, phase-specific
  edges (powerplay / death overs), and the graph features inside the refined model, since both help most at large Δ.

Full numbers: [`results/experiments/E22_final/`](../../results/experiments/E22_final/) (3a as the E22 reference) and
[`results/experiments/E24_matchups/`](../../results/experiments/E24_matchups/).
