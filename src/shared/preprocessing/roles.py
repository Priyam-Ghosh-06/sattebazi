"""Stage 4: causal role inference (WK/BAT/AR/BOWL/UNK) + gate G4.

python src/roles.py --in data/processed/player_match.parquet --out data/processed/roles.parquet
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # src/
from shared.preprocessing.causal import asof, state_table  # noqa: E402

BOUNDS = {"WK": (1, 4), "BAT": (3, 6), "AR": (1, 4), "BOWL": (3, 6)}
DEFAULTS = dict(window=30, wk_catch_rate=1.2, wk_top_min=0.3, bowl_share=0.25, bowl_pos=8.0)


def role_state(P, window=30):
    df = P.reset_index(drop=True).copy()
    df["bshare"] = df.legal_balls_bowled / (np.ceil(df.sched_overs / 5) * df.balls_per_over)
    df["pos"] = df.bat_position.replace(0, 11)  # did not bat -> tail
    return state_table(df, "player_id", {
        "stump_w": ("stumpings", window, "sum"), "catch_w": ("catches", window, "mean"),
        "bshare_w": ("bshare", window, "mean"), "pos_w": ("pos", window, "median"),
        "stump_career": ("stumpings", None, "sum")}, {})


def infer(Q, S, group=("match_id", "team"), wk_catch_rate=1.2, wk_top_min=0.3, bowl_share=0.25, bowl_pos=8, **_):
    """Role for each query row (player at a date) from the state strictly before that date.
    group: rows competing for the 'most keeper-like' slot (the XI in history, the squad at test time)."""
    df = asof(Q[["player_id", "date", *group]], S[["player_id", "date", "n_apps", "stump_w", "catch_w", "bshare_w", "pos_w", "stump_career"]], "player_id")
    df["wk_score"] = (df.stump_career.fillna(0) > 0) * 1.0 + df.catch_w.fillna(0) + df.stump_w.fillna(0)
    top = df.wk_score.eq(df.groupby(list(group)).wk_score.transform("max"))
    wk = (df.stump_w >= 1) | (df.catch_w >= wk_catch_rate) | (top & (df.wk_score >= wk_top_min))
    bowler = df.bshare_w >= bowl_share
    df["role"] = np.select([df.n_apps.isna(), wk, bowler & (df.pos_w >= bowl_pos), bowler], ["UNK", "WK", "BOWL", "AR"], "BAT")
    return df


def feasible(counts, wildcard=0):
    """Can some 11 from these role counts meet the composition bounds (UNK as wildcards)?"""
    need = sum(max(0, BOUNDS[r][0] - counts.get(r, 0)) for r in BOUNDS)
    cap = sum(min(counts.get(r, 0), BOUNDS[r][1]) for r in BOUNDS)
    return need <= wildcard and cap + wildcard >= 11


def gate(R):
    rows = []
    for mid, g in R.groupby("match_id", sort=False):
        c = g.role.value_counts().to_dict()
        rows.append({"match_id": mid, "n_unk": c.get("UNK", 0), "feasible_strict": feasible(c),
                     "feasible_wild": feasible(c, c.get("UNK", 0)), "n_wk": c.get("WK", 0)})
    G = pd.DataFrame(rows)
    known = G[G.n_unk == 0]
    per_xi = R[R.role != "UNK"].groupby(["match_id", "team"]).role.value_counts().unstack(fill_value=0)
    return {
        "matches": len(G), "matches_no_unk": len(known),
        "feasible_rate_no_unk_matches": float(known.feasible_strict.mean()),
        "feasible_rate_unk_wildcard_all": float(G.feasible_wild.mean()),
        "matches_zero_wk_no_unk": float((known.n_wk == 0).mean()),
        "per_team_role_mean": per_xi.mean().round(2).to_dict(),
        "per_team_wk_count_dist": per_xi.get("WK", pd.Series(dtype=int)).value_counts(normalize=True).sort_index().round(3).head(6).to_dict(),
        "role_share": R.role.value_counts(normalize=True).round(3).to_dict(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/player_match.parquet")
    ap.add_argument("--out", default="data/processed/roles.parquet")
    ap.add_argument("--audit-dir", default="data/audit")
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in DEFAULTS}
    P = pd.read_parquet(a.inp)
    S = role_state(P, a.window)
    R = infer(P[["match_id", "team", "player_id", "date"]], S, **cfg)
    R[["match_id", "team", "player_id", "role"]].to_parquet(a.out, index=False)
    au = gate(R)
    au["config"] = cfg
    au["pass"] = au["feasible_rate_no_unk_matches"] >= 0.99
    json.dump(au, open(os.path.join(a.audit_dir, "roles_audit.json"), "w"), indent=1, default=str)
    print(json.dumps(au, indent=1, default=str))
    sys.exit(0 if au["pass"] else 1)


if __name__ == "__main__":
    main()
