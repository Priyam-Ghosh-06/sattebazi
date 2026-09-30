#!/usr/bin/env bash
# End to end: data download -> shared preprocessing + gates -> graph-approach features + gates -> train -> hidden-set CSV
# (-> optional LLM overviews). Every stage exits non-zero when its gate fails, and set -e stops the run there.
#   bash run.sh                         CPU only; trains and submits the graph approach's base model (3a, the submitted model)
#   GRAPH_FEATURES=1 bash run.sh        also builds the batter x bowler graph and trains 3b
#   NEURAL=1 bash run.sh                also trains the baseline (CB) and refined (ADV) models (GPU; src/run_kaggle.py runs them on Kaggle)
#   LLM=1 bash run.sh                   also writes open-weight LLM overviews of the predicted XIs
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
TRAIN_START=${TRAIN_START:-2019-01-01}  # feature history always starts in 2005; 2011/2015 training starts were no better
TRAIN_END=${TRAIN_END:-2025-12-31}
TEST_FILE=${TEST_FILE:-data/raw/test_fixtures.json}
MODEL=model_artifacts/model_${TRAIN_END}.pkl
RAW=data/raw/cricsheet

# 0. data: download Cricsheet only if the raw folder is empty (per-year archives, split by gender)
if [ -z "$(ls -A $RAW 2>/dev/null)" ]; then
  mkdir -p $RAW data/raw/registry tmp_dl
  for g in male female; do
    for y in $(seq 2001 2025); do
      if curl -fsSL "https://cricsheet.org/downloads/${y}_${g}_json.zip" -o "tmp_dl/${y}_${g}.zip"; then
        mkdir -p "$RAW/$g/$y" && unzip -q -o "tmp_dl/${y}_${g}.zip" -d "$RAW/$g/$y"
      fi
    done
  done
  curl -fsSL https://cricsheet.org/register/people.csv -o data/raw/registry/people.csv
  curl -fsSL https://cricsheet.org/register/names.csv -o data/raw/registry/names.csv
  rm -rf tmp_dl
fi
$PY -c "import pandas, numpy, pyarrow, lightgbm, scipy"   # parameter / dependency check
mkdir -p data/interim data/processed data/audit model_artifacts results

# 1. shared spine (all approaches): ingestion, fantasy points, roles, pseudo-squads, causal features + gates G1-G6
$PY src/shared/ingestion/ingest.py --train-end "$TRAIN_END"
$PY src/shared/preprocessing/points.py
$PY tests/test_points.py
$PY src/shared/preprocessing/roles.py
$PY src/shared/preprocessing/squads.py
$PY src/shared/preprocessing/features.py --train-end "$TRAIN_END"
$PY tests/test_causality.py
$PY tests/test_select.py

# 2. graph approach: staleness-aware features + gate G7 (and, optionally, the batter x bowler graph + gate G8)
$PY src/graph/preprocessing.py --train-start "$TRAIN_START" --train-end "$TRAIN_END"
$PY src/graph/preprocessing.py --g7
if [ "${GRAPH_FEATURES:-0}" = "1" ]; then
  for step in ingest snapshots features g8; do $PY src/graph/experiments.py $step; done
  $PY src/train.py --approach graph --graph-features --train-start "$TRAIN_START" --train-end "$TRAIN_END" \
      --out "model_artifacts/graph_features_${TRAIN_END}.pkl"
fi

# 3. the submitted model: graph approach 3a (LightGBM p·μ), trained on everything up to the cutoff
$PY src/train.py --train-start "$TRAIN_START" --train-end "$TRAIN_END" --out "$MODEL"

# 4. optional neural approaches (GPU recommended; each seed checkpoints every epoch and resumes)
if [ "${NEURAL:-0}" = "1" ]; then
  $PY tests/test_baseline.py && $PY tests/test_refined.py
  $PY src/train.py --approach baseline --train-start "$TRAIN_START" --train-end "$TRAIN_END" --out "model_artifacts/baseline_cb128_${TRAIN_END}.pkl"
  $PY src/train.py --approach refined --train-start "$TRAIN_START" --train-end "$TRAIN_END" --out "model_artifacts/refined_adv_${TRAIN_END}.pkl"
fi

# 5. hidden-set predictions (PS CSV), then the optional LLM overviews
if [ -f "$TEST_FILE" ]; then
  $PY src/evaluate.py --model-path "$MODEL" --test-file "$TEST_FILE" --output results/hidden_predictions.csv
  if [ "${LLM:-0}" = "1" ]; then
    $PY src/shared/llm/team_overview.py --predictions results/hidden_predictions.csv --output results/hidden_team_overviews.txt
  fi
else
  echo "no $TEST_FILE yet: skipping hidden-set predictions (the mock set: data/mock/test_fixtures.json)"
fi
