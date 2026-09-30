# model_artifacts

| file | approach | how it is made |
|---|---|---|
| `model_2025-12-31.pkl` | **3a graph base, the submitted model** (LightGBM p and μ) | `python src/train.py --train-start 2000-01-01 --train-end 2025-12-31` |
| `graph_features_2025-12-31.pkl` | 3b, + batter × bowler graph features | `python src/train.py --approach graph --graph-features --out ...` |
| `baseline_cb128_2025-12-31.pkl` | 1, CB128, two seeds (packaged from the FINAL Kaggle runs) | `python src/train.py --approach baseline --device cuda --out ...` |
| `checkpoints/<run>/done.json` | training history of every Kaggle run (per-epoch losses and validation scores) | written by the pipelines |
| `checkpoints/<run>/best.pt` | best EMA weights of each run | written by the pipelines |

The refined approach has no FINAL artifact: it did not pass its pre-registered rule, so its FINAL runs were not trained
(`python src/train.py --approach refined --device cuda` trains them). Every `.pkl` is read by `src/evaluate.py`.
