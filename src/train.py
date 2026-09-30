"""Training entry point (PS CLI).

python src/train.py --train-start 2019-01-01 --train-end 2025-12-31 --device cuda
    trains the final model, 3c (the refined net + the batter x bowler graph features, two seeds, warm-started from the
    refined runs; GPU) -> model_artifacts/model_<end>.pkl

python src/train.py --approach graph                        3a: the trees (LightGBM p·μ, CPU, about a minute) -> model_trees_<end>.pkl
python src/train.py --approach graph --graph-features       3b: the trees + the graph features -> model_trees_graph_<end>.pkl
python src/train.py --approach baseline --device cuda       1: CB, two seeds (a finished run with the same dates is reused)
python src/train.py --approach refined --device cuda        2: ADV, two seeds (same reuse and resume behaviour)

The last 3 months before --train-end are the early-stopping window. Feature tables are built by run.sh;
the training data never contains a match dated after --train-end.
"""
import argparse, json, os, pickle, random, sys

import numpy as np
import pandas as pd

random.seed(0); np.random.seed(0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import scoring_model as T  # noqa: E402

KEY = ["match_id", "team", "player_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start", default="2019-01-01")
    ap.add_argument("--train-end", default="2025-12-31")
    ap.add_argument("--approach", default="final", choices=["final", "graph", "baseline", "refined"],
                    help="final = 3c = refined + graph features (the final model)")
    ap.add_argument("--graph-features", action="store_true", help="graph: add the batter x bowler graph features (3b)")
    ap.add_argument("--features", default="data/processed/stale_features.parquet")
    ap.add_argument("--graph-feature-table", default="data/processed/matchup_features.parquet")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-context", action="store_true", help="graph: drop the opponent/venue context features (E14)")
    ap.add_argument("--cb-arm", default="CB128", help="baseline: CB64 or CB128 (history length; CB128 won R2)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="baseline/refined: the ensemble's seeds")
    ap.add_argument("--device", default="auto", help="baseline/refined: cpu, cuda, cuda:0, ...")
    a = ap.parse_args()
    if a.approach == "final":
        a.approach, a.graph_features = "refined", True
        tag = ""
    else:
        tag = {"graph": "trees_graph_" if a.graph_features else "trees_", "baseline": "baseline_", "refined": "refined_"}[a.approach]
    out = a.out or f"model_artifacts/model_{tag}{a.train_end}.pkl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    vf = T.valid_start(a.train_end)
    dates = (a.train_start, vf, (pd.Timestamp(a.train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), None)
    if a.approach == "baseline":
        from baseline.pipeline import package, train_run
        runs = [train_run(a.cb_arm, dates, s, a.device) for s in a.seeds]
        pickle.dump(package(runs, a.train_start, a.train_end), open(out, "wb"))
        print(json.dumps({"out": out, "approach": "baseline", "arm": a.cb_arm, "runs": runs}))
        return
    if a.approach == "refined":
        from refined.config import ADVM, HP
        from refined.pipeline import package, train_run
        if a.graph_features:  # the final model, 3c: refined + graph features (E25), warm-started from the refined runs
            HP.update(ADVM)
        runs = [train_run(dates, s, a.device) for s in a.seeds]
        pickle.dump(package(runs, a.train_start, a.train_end), open(out, "wb"))
        print(json.dumps({"out": out, "approach": "3c final (refined + graph)" if a.graph_features else "refined", "runs": runs}))
        return
    X = pd.read_parquet(a.features)
    X = X[(X.part == "train") & (X.date >= a.train_start) & (X.date <= a.train_end)]
    extra = ()
    if a.graph_features:
        G = pd.read_parquet(a.graph_feature_table).drop(columns=["c_date", "mu_snapshot"])
        X = X.merge(G, on=KEY, how="left", validate="one_to_one")
        extra = ("x_mu_",)
    assert X.date.max() <= a.train_end, "leakage guard: a training fixture after --train-end"
    model = T.fit_pipeline(T.prep(X), use_context=not a.no_context, valid_from=vf, extra=extra)
    model.update(arch="b0s", graph_features=bool(a.graph_features), train_start=a.train_start, train_end=a.train_end, valid_from=vf)
    pickle.dump(model, open(out, "wb"))
    print(json.dumps({"out": out, "approach": "graph" + (" + graph features" if a.graph_features else ""),
                      "mu_iters": model["mu"].best_iteration, "p_iters": model["p"].best_iteration}))


if __name__ == "__main__":
    main()
