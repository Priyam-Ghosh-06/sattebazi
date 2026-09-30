"""Evaluation entry point (PS CLI): fixtures (two squads) -> the approach's expected points per candidate -> exact ILP XI -> CSV.

python src/evaluate.py --model-path model_artifacts/model_2025-12-31.pkl --test-file data/raw/test_fixtures.json --output results/hidden_predictions.csv
The model file says which approach made it: graph (LightGBM p·μ, optionally + graph features), baseline (CB) or refined (ADV).
"""
import argparse, csv, os, pickle, random, sys

import numpy as np
import pandas as pd

random.seed(0); np.random.seed(0)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph import scoring_model as T  # noqa: E402
from shared.ingestion.fixtures import ROLE_MAP, Resolver, load_fixtures, team_ctx  # noqa: E402
from shared.optimization.selection import select_xi  # noqa: E402
from shared.preprocessing import context, features as F  # noqa: E402
from shared.preprocessing.causal import cutoff  # noqa: E402

ARCH = {"cb": "baseline.pipeline", "adv": "refined.pipeline"}


def predict_fixtures(model, fixtures, P, M, registry_dir, log=print):
    P = P[P.date <= model["train_end"]]
    res = Resolver(P, registry_dir)
    rows, meta, info, resolved = [], [], {}, {}
    for i, fx in enumerate(fixtures):
        mid = f"fx{i}"
        t1, s1 = res.team(fx["team1"]); t2, s2 = res.team(fx["team2"])
        resolved[mid] = t1
        ctx = team_ctx(P, M, t1, fx["date"])
        for k in ("gender", "team_type", "match_type"):
            ctx[k] = fx.get(k) or ctx[k]
        stat = {}
        for team, raw_team, squad, other in [(t1, fx["team1"], fx["squad1"], t2), (t2, fx["team2"], fx["squad2"], t1)]:
            for n in squad:
                pid, how = res.player(n, team)
                stat[how] = stat.get(how, 0) + 1
                given = ROLE_MAP.get(str(fx["roles"].get(n, "")).strip().lower())
                rows.append({"match_id": mid, "team": team, "team_display": raw_team, "player_id": pid, "name": n,
                             "date": fx["date"], "opp": other, "given_role": given, "venue": fx.get("venue"), **ctx})
        unres = stat.get("unresolved", 0)
        log(f"[resolve] {fx['date']} {fx['team1']} ({s1}) vs {fx['team2']} ({s2}): {stat}" + (f"  <-- {unres} UNRESOLVED (cold start)" if unres else ""))
        c, delta = cutoff(fx["date"], model["train_end"])
        info[mid] = {**ctx, "date": fx["date"], "delta": delta, "event": fx.get("event"), "venue": fx.get("venue")}
        meta.append((mid, fx))
    Q = pd.DataFrame(rows)
    Q["side"] = (Q.team != Q.match_id.map({m: resolved[m] for m in resolved})).astype(int)
    X = F.build(P, Q[["match_id", "team", "player_id", "date"]])  # history ends at train_end, so this is the state as of c
    X = pd.concat([X, Q[["name", "team_display", "given_role", "gender", "team_type", "match_type", "sched_overs", "balls_per_over", "opp", "venue"]].reset_index(drop=True)], axis=1)
    if any(c.startswith(T.CONTEXT_PREFIX) for c in model.get("cols", [])):
        X = context.add_match_context(X.assign(in_base=True), P, M)
    X["role"] = np.where(X.given_role.notna(), X.given_role, X.role)  # organiser-provided role wins
    X = T.prep(X)
    if model.get("arch") in ARCH:  # E[Y] from the two-seed ensemble (refined: each seed's validation-chosen predictor)
        import importlib
        X["score"] = importlib.import_module(ARCH[model["arch"]]).predict_rows(model, Q, info, P, M)
        X["mu"], X["p"] = np.nan, np.nan
    else:
        X["delta"] = X.match_id.map({m: v["delta"] for m, v in info.items()}).astype(float)  # staleness input (B0s)
        if model.get("graph_features"):  # graph approach 3b: batter x bowler graph features from the last snapshot <= c
            from graph.graph_model import add_graph_features
            X = add_graph_features(X, model["train_end"])
        mu, p = T.predict(model, X)
        X["mu"], X["p"], X["score"] = mu, p, mu * p
    out = []
    for mid, fx in meta:
        g = X[X.match_id == mid].reset_index(drop=True)
        idx, ok = select_xi(g.score.to_numpy(), g.role.to_numpy(), g.team.to_numpy(), g.player_id.to_numpy())
        if not ok:
            log(f"[select] {mid}: role bounds infeasible, fell back to team constraint only")
        pick = g.iloc[idx].sort_values("score", ascending=False)
        for rank, r in enumerate(pick.itertuples(), 1):
            out.append({"match_date": fx["date"], "team1": fx["team1"], "team2": fx["team2"], "rank": rank, "player": r.name,
                        "player_team": r.team_display, "role": "BAT" if r.role == "UNK" else r.role,
                        "predicted_points": round(float(r.score), 2), "points_if_plays": round(float(r.mu), 2),
                        "p_plays": round(float(r.p), 3)})
    return pd.DataFrame(out), X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--history", default="data/processed/player_match.parquet")
    ap.add_argument("--matches", default="data/interim/matches.parquet")
    ap.add_argument("--registry-dir", default="data/raw/registry")
    a = ap.parse_args()
    model = pickle.load(open(a.model_path, "rb"))
    P = pd.read_parquet(a.history); M = pd.read_parquet(a.matches)
    M = M[M.date <= model["train_end"]]
    fx = load_fixtures(a.test_file)
    out, _ = predict_fixtures(model, fx, P, M, a.registry_dir)
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    out.to_csv(a.output, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")  # same bytes on Windows and Linux
    print(f"wrote {a.output}: {len(fx)} fixtures, {len(out)} rows")


if __name__ == "__main__":
    main()
