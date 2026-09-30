"""Stage 6: strictly causal features for query rows (match, team, player) -> processed/features.parquet.

build(P, Q): P = played history (player_match with Φ), Q = query rows [match_id, team, player_id, date, ...ctx].
Every feature comes from rows of P with date < Q.date (causal.asof). The same code serves training,
backtests, the time-travel test, and test-time fixtures.
python src/features.py
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # src/
from shared.preprocessing.causal import asof, state_table  # noqa: E402
from shared.preprocessing import roles as rl  # noqa: E402

HALFLIVES = (3, 10, 30)
CTX = ["gender", "team_type", "match_type", "sched_overs", "balls_per_over"]


def player_state(P):
    df = P.reset_index(drop=True).copy()
    df["pos"] = df.bat_position.replace(0, 11)
    df["bowl12"] = (df.legal_balls_bowled >= 12).astype(float)
    df["big"] = (df.total >= 60).astype(float)
    df["r25"] = (df.runs >= 25).astype(float)
    df["w2"] = (df.wickets >= 2).astype(float)
    win = {"last5_total": ("total", 5, "mean"), "car_total": ("total", None, "mean"),
           "car_bat": ("pts_bat", None, "mean"), "car_bowl": ("pts_bowl", None, "mean"), "car_field": ("pts_field", None, "mean"),
           "pos10": ("pos", 10, "median"), "bf10": ("balls_faced", 10, "mean"), "lb10": ("legal_balls_bowled", 10, "mean"),
           "bowl12_10": ("bowl12", 10, "mean"), "sd20_total": ("total", 20, "std"), "big20": ("big", 20, "mean"),
           "r25_20": ("r25", 20, "mean"), "w2_20": ("w2", 20, "mean"), "runs20": ("runs", 20, "mean"),
           "wk20": ("wickets", 20, "mean"), "ct20": ("catches", 20, "mean")}
    ew = {f"ewm{h}_{c}": (c, h) for h in HALFLIVES for c in ("total", "pts_bat", "pts_bowl", "pts_field")}
    S = state_table(df, "player_id", win, ew)
    S["last_date"] = S.date
    return S


def team_app_features(P, Q):
    """Appearance-for-this-team features over the team's own match sequence (strictly earlier dates)."""
    TM = P.drop_duplicates(["match_id", "team"]).groupby(["team", "date"]).size().rename("n").reset_index()
    TM["tcount"] = TM.groupby("team").n.cumsum()  # team matches up to and incl. this date (unique per team-date)
    # k = number of team matches strictly before the query date
    k = asof(Q[["team", "date"]], TM[["team", "date", "tcount"]].rename(columns={"tcount": "k"}), "team")["k"].fillna(0).to_numpy()
    A = P[["team", "player_id", "date"]].merge(TM[["team", "date", "tcount"]], on=["team", "date"])
    A["key"] = A.team + "|" + A.player_id
    A = A.sort_values(["date", "tcount"], kind="stable")
    A["idx"] = A.tcount  # team-match index of this appearance (ties on same day -> last index; ok, strict date below)
    A["cnt"] = A.groupby("key").cumcount() + 1
    A["src_date"] = A.date
    key = (Q.team + "|" + Q.player_id).to_numpy()
    base = asof(pd.DataFrame({"key": key, "date": Q.date.to_numpy()}), A[["key", "date", "cnt", "idx", "src_date"]], "key")
    cnt_now = base.cnt.fillna(0).to_numpy()
    out = pd.DataFrame(index=Q.index)
    out["team_apps"] = cnt_now
    out["team_matches_before"] = k
    out["team_days_since"] = (pd.to_datetime(Q.date).to_numpy() - pd.to_datetime(base.src_date).to_numpy()) / np.timedelta64(1, "D")
    out["team_gap_matches"] = k - base.idx.fillna(-1e3).to_numpy()  # 1 = played the previous team match
    # count of appearances with idx > k-N: cnt_now - cnt(idx <= k-N)
    Ai = A[["key", "idx", "cnt"]].astype({"idx": float}).sort_values(["idx", "cnt"], kind="stable")
    for N in (3, 5, 10):
        q = pd.DataFrame({"key": key, "idx": np.maximum(k - N, 0) + 0.5, "_o": np.arange(len(key))}).sort_values("idx", kind="stable")
        c = pd.merge_asof(q, Ai, on="idx", by="key", direction="backward").sort_values("_o").cnt.fillna(0).to_numpy()
        out[f"share{N}"] = np.where(k > 0, (cnt_now - c) / np.minimum(N, np.maximum(k, 1)), np.nan)
    return out


def build(P, Q, role_cfg=None):
    """Features for query rows Q from played history P (only rows with date < each query's date are used)."""
    role_cfg = role_cfg or rl.DEFAULTS
    Q = Q.reset_index(drop=True)
    S = player_state(P)
    X = asof(Q[["match_id", "team", "player_id", "date"]], S, "player_id")
    X["days_since"] = (pd.to_datetime(X.date) - pd.to_datetime(X.last_date)).dt.days
    X = X.drop(columns=["last_date"]).rename(columns={"src_date": "src_date_player"})
    T = team_app_features(P, Q)
    X = pd.concat([X, T], axis=1)
    R = rl.infer(Q[["match_id", "team", "player_id", "date"]], rl.role_state(P, role_cfg["window"]), **role_cfg)
    X["role"] = R.role.to_numpy()
    for c in ["bshare_w", "pos_w", "catch_w", "stump_w"]:
        X["role_" + c] = R[c].to_numpy()
    # candidate-set (squad-relative) features: where the player stands among the squad they compete with
    grp = [X.match_id, X.team]
    X["squad_n"] = X.groupby(grp).player_id.transform("size")
    X["rank_share10"] = X.groupby(grp).share10.rank(ascending=False, method="average")
    X["rank_ewm10"] = X.groupby(grp).ewm10_total.rank(ascending=False, method="average")
    X["role_rank_share10"] = X.groupby([X.match_id, X.team, X.role]).share10.rank(ascending=False, method="average")
    X["role_n_in_squad"] = X.groupby([X.match_id, X.team, X.role]).player_id.transform("size")
    X["feat_source_max_date"] = X.src_date_player  # team features come from the same strict asof
    return X


def feature_cols(X):
    """Model inputs only: no ids, no targets, no free-text context."""
    drop = {"match_id", "team", "player_id", "date", "src_date_player", "feat_source_max_date", "role",
            "played", "in_base", "total", "pts_bat", "pts_bowl", "pts_field", "runs", "wickets", "y",
            "venue", "event", "opp", "gender", "team_type", "match_type", "n_violations",
            "fx_id", "fx_date", "c_date", "part"}
    return [c for c in X.columns if c not in drop]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-match", default="data/processed/player_match.parquet")
    ap.add_argument("--squads", default="data/processed/squads.parquet")
    ap.add_argument("--out", default="data/processed/features.parquet")
    ap.add_argument("--audit-dir", default="data/audit")
    ap.add_argument("--train-end", default="2025-12-31")
    a = ap.parse_args()
    P = pd.read_parquet(a.player_match)
    assert (P.date <= a.train_end).all(), "leakage guard: post-cutoff rows in history"
    Sq = pd.read_parquet(a.squads)
    X = build(P, Sq[["match_id", "team", "player_id", "date"]])
    M = pd.read_parquet("data/interim/matches.parquet", columns=["match_id", "venue", "event", *CTX])
    tgt = P[["match_id", "team", "player_id", "total", "pts_bat", "pts_bowl", "pts_field", "runs", "wickets"]]
    X = X.merge(Sq[["match_id", "team", "player_id", "played", "in_base"]], on=["match_id", "team", "player_id"]) \
         .merge(M, on="match_id").merge(tgt, on=["match_id", "team", "player_id"], how="left")
    X["y"] = X.total.fillna(0)  # benched players score 0
    opp = P.drop_duplicates(["match_id", "team"])[["match_id", "team", "opp"]]
    X = X.merge(opp, on=["match_id", "team"], how="left")
    ok = (X.feat_source_max_date.isna() | (X.feat_source_max_date < X.date)).all()
    X.to_parquet(a.out, index=False)
    au = {"rows": len(X), "cols": len(X.columns), "source_date_strictly_before_match": bool(ok),
          "null_rate": X[feature_cols(X)].select_dtypes("number").isna().mean().round(3).to_dict()}
    au["pass"] = au["source_date_strictly_before_match"]
    json.dump(au, open(os.path.join(a.audit_dir, "features_audit.json"), "w"), indent=1)
    print(json.dumps({k: au[k] for k in ("rows", "cols", "pass")}))
    sys.exit(0 if au["pass"] else 1)


if __name__ == "__main__":
    main()
