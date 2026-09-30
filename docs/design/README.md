# Design history

These documents were written during the hackathon, in this order, and are kept unchanged as the record of what was planned,
pre-registered and decided. File names inside them refer to the working layout before the repository was organised by
approach:

| name in these documents | now |
|---|---|
| `src/ingest.py`, `points.py`, `roles.py`, `squads.py`, `causal.py`, `features.py`, `context.py` | `src/shared/ingestion/`, `src/shared/preprocessing/` |
| `src/selection.py` | `src/shared/optimization/selection.py` |
| `src/cb.py` (CB) | `src/baseline/` |
| `src/adv.py` (ADV) | `src/refined/` |
| `src/train.py` helpers, `src/stale.py` (B0s), `src/matchups.py` (E24) | `src/graph/` |
| `src/final_run.py` | `src/shared/evaluation/` (+ `src/graph/experiments.py folds`) |
| `src/backtest.py`, `noise_ceiling.py`, `kalman_fit.py` | `src/experiments/` |
| `src/kaggle_run.py`, `output/…` | `src/run_kaggle.py`, `model_artifacts/…`, `results/…` |
| B0 / B0s | graph approach base (3a) |
| E24 matchups | graph approach 3b |

| document | contents |
|---|---|
| `SPEC.md` | the specification; §13 the final run, §14 the refined run, §15 the graph run, each pre-registered |
| `PIPELINE.md` | the pipeline as presented, open questions |
| `BASELINE.md`, `ADVANCED.md` | the baseline and refined designs with their propositions and tests |
| `DECISIONS.md` | every interpretive choice: alternatives, reason, flag |
| `EXPERIMENTS.md` | the exploratory experiments E2–E15 |
| `THOUGHT_LOG.md` | the author's reasoning, entry by entry |
