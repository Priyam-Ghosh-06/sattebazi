"""SPEC §12.1 / §13: staleness-aware features for B0s + gate G7.

Every row is a (fixture copy, candidate): features are computed as of the cutoff c = fixture date − Δ with the
same code as B0 (features.build, context.add_match_context; the as-of key is c instead of the match date), then
the day-gap features are measured to the fixture date (+Δ) and Δ itself is an input.
  copy 'train': every base-squad fixture dated >= --train-start, one Δ per fixture (causal.draw_delta(fixture id)).
  copy 'g<Δ>':  fixtures dated >= --grid-start at every Δ of the grid (validation windows and test years).
python src/stale.py [--train-start 2019-01-01 --grid-start 2023-10-01 --train-end 2025-12-31]
python src/stale.py --g7     (gate G7: rebuild 8 random fixture copies from history truncated at c; exact equality)
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from shared.preprocessing import context, features as F  # noqa: E402
from shared.preprocessing.causal import DELTAS, draw_delta  # noqa: E402

OUT = "data/processed/stale_features.parquet"


def queries(Sq, train_start, grid_start):
    """Base-squad candidate rows, one copy per (fixture, Δ). match_id = '<fixture>@<Δ>' keeps copies apart."""
    b = Sq[Sq.in_base & (Sq.date >= train_start)][["match_id", "team", "player_id", "date", "played", "in_base"]]
    b = b.rename(columns={"match_id": "fx_id", "date": "fx_date"})
    tr = b.assign(part="train", delta=b.fx_id.map({m: draw_delta(m) for m in b.fx_id.unique()}))
    g = b[b.fx_date >= grid_start]
    Q = pd.concat([tr] + [g.assign(part=f"g{d}", delta=d) for d in DELTAS], ignore_index=True)
    Q["match_id"] = Q.fx_id + "@" + Q.delta.astype(str) + "@" + Q.part
    return Q


def build(P, M, Q):
    """Q: [match_id, fx_id, team, player_id, fx_date, delta, played, in_base, copy]. Features as of c = fx_date − Δ."""
    Q = Q.reset_index(drop=True)
    c = (pd.to_datetime(Q.fx_date) - pd.to_timedelta(Q.delta, unit="D")).dt.strftime("%Y-%m-%d")
    X = F.build(P, Q[["match_id", "team", "player_id"]].assign(date=c.to_numpy()))
    d = Q.delta.to_numpy(float)
    X["days_since"] = X.days_since + d  # last appearance -> fixture date
    X["team_days_since"] = X.team_days_since + d
    X["delta"] = d
    for k in ("fx_id", "fx_date", "played", "in_base", "part"):
        X[k] = Q[k].to_numpy()
    ctx = M[["match_id", "venue", "event", *F.CTX]].rename(columns={"match_id": "fx_id"})  # the fixture's own pre-match context
    X = X.merge(ctx, on="fx_id", how="left")
    tgt = P[["match_id", "team", "player_id", "total"]].rename(columns={"match_id": "fx_id"})
    X = X.merge(tgt, on=["fx_id", "team", "player_id"], how="left")
    X["y"] = X.total.fillna(0)
    teams = M.set_index("match_id")[["team1", "team2"]].reindex(X.fx_id)  # the opponent is a fixture fact, not history
    X["opp"] = np.where(teams.team1.to_numpy() == X.team.to_numpy(), teams.team2.to_numpy(), teams.team1.to_numpy())
    X = context.add_match_context(X, P, M)  # X.date is still c here: venue history as of c
    X["c_date"] = X.date
    X["date"] = X.fx_date
    return X


def g7(P, M, X, n_copies=8, seed=0):
    """Gate G7: for random fixture copies with Δ > 0, rebuild from history truncated at c and require exact equality."""
    assert (X.feat_source_max_date.isna() | (X.feat_source_max_date < X.c_date)).all(), "a source is dated >= c"
    rng = np.random.default_rng(seed)
    ids = np.sort(X.loc[(X.delta > 0) & (X.fx_date >= "2020-01-01"), "match_id"].unique())
    cols = [c for c in F.feature_cols(X) if X[c].dtype != object] + ["role"]
    bad, rows = [], 0
    for mid in rng.choice(ids, n_copies, replace=False):
        full = X[X.match_id == mid].sort_values(["team", "player_id"]).reset_index(drop=True)
        c = full.c_date.iloc[0]
        Q = full[["match_id", "fx_id", "team", "player_id", "fx_date", "delta", "played", "in_base", "part"]]
        t = build(P[P.date < c], M, Q).sort_values(["team", "player_id"]).reset_index(drop=True)
        a, b = full[cols], t[cols]
        num = [k for k in cols if k != "role"]
        same = np.isclose(a[num].to_numpy(float), b[num].to_numpy(float), equal_nan=True, rtol=1e-9, atol=1e-9).all(0)
        diff = [k for k, s in zip(num, same) if not s] + ([] if (a.role == b.role).all() else ["role"])
        if diff:
            bad.append({"copy": mid, "cols": diff})
        rows += len(full)
    return {"rows_checked": rows, "copies": n_copies, "mismatches": bad, "pass": not bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start", default="2019-01-01")
    ap.add_argument("--grid-start", default="2023-10-01")
    ap.add_argument("--train-end", default="2025-12-31")
    ap.add_argument("--player-match", default="data/processed/player_match.parquet")
    ap.add_argument("--squads", default="data/processed/squads.parquet")
    ap.add_argument("--matches", default="data/interim/matches.parquet")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--audit-dir", default="data/audit")
    ap.add_argument("--g7", action="store_true", help="run gate G7 on an existing --out")
    a = ap.parse_args()
    P = pd.read_parquet(a.player_match)
    M = pd.read_parquet(a.matches)
    assert (P.date <= a.train_end).all(), "leakage guard: post-cutoff rows in history"
    if a.g7:
        au = g7(P, M, pd.read_parquet(a.out))
        json.dump(au, open(os.path.join(a.audit_dir, "g7_audit.json"), "w"), indent=1)
        print(json.dumps(au))
        sys.exit(0 if au["pass"] else 1)
    Q = queries(pd.read_parquet(a.squads), a.train_start, a.grid_start)
    X = build(P, M, Q)
    ok = bool((X.feat_source_max_date.isna() | (X.feat_source_max_date < X.c_date)).all())
    X.to_parquet(a.out, index=False)
    au = {"rows": len(X), "copies": X.part.value_counts().to_dict(), "fixtures_train": int(X[X.part == "train"].fx_id.nunique()),
          "delta_train": X[X.part == "train"].drop_duplicates("fx_id").delta.describe().round(1).to_dict(),
          "source_date_strictly_before_cutoff": ok, "pass": ok}
    json.dump(au, open(os.path.join(a.audit_dir, "stale_audit.json"), "w"), indent=1)
    print(json.dumps(au))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
