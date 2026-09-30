"""Stage 3: Φ, the PS points function, vectorised over player_match_raw.

python src/points.py --in data/interim/player_match_raw.parquet --out data/processed/player_match.parquet
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

COMPONENTS = ["participation", "bat_runs", "bat_boundary", "bat_milestone", "bat_duck", "bat_sr",
              "bowl_dots", "bowl_wkts", "bowl_lbwb", "bowl_haul", "bowl_maiden", "bowl_econ", "field"]
GROUPS = {"bat": ["bat_runs", "bat_boundary", "bat_milestone", "bat_duck", "bat_sr"],
          "bowl": ["bowl_dots", "bowl_wkts", "bowl_lbwb", "bowl_haul", "bowl_maiden", "bowl_econ"],
          "field": ["field"]}


def sr_points(runs, balls):
    """Strike-rate band points; 0 below 10 balls faced."""
    sr = 100 * runs / np.maximum(balls, 1)
    p = np.select([sr > 170, sr > 150, sr >= 130, sr > 70, sr >= 60, sr >= 50], [5, 3, 1, 0, -1, -3], -5)
    return np.where(balls >= 10, p, 0)


def econ_points(runs_conceded, legal_balls):
    """Economy band points; 0 below 12 legal balls (2 overs)."""
    e = 6 * runs_conceded / np.maximum(legal_balls, 1)
    p = np.select([e < 5, e < 6, e <= 7, e < 10, e <= 11, e <= 12], [5, 3, 1, 0, -1, -3], -5)
    return np.where(legal_balls >= 12, p, 0)


def phi(df, six_also_boundary=False, duck_scope="all"):
    r = df["runs"].to_numpy()
    w = df["wickets"].to_numpy()
    c = df["catches"].to_numpy()
    out = pd.DataFrame(index=df.index)
    out["participation"] = 3
    out["bat_runs"] = r
    out["bat_boundary"] = df["fours"] + (3 if six_also_boundary else 2) * df["sixes"]
    out["bat_milestone"] = np.select([r >= 100, r >= 75, r >= 50, r >= 25], [20, 15, 10, 5], 0)
    duck = df["dismissed"].to_numpy() & (r == 0)
    if duck_scope == "faced_ball":
        duck &= df["balls_faced"].to_numpy() > 0
    out["bat_duck"] = np.where(duck, -2, 0)
    out["bat_sr"] = sr_points(r, df["balls_faced"].to_numpy())
    out["bowl_dots"] = df["dots"]
    out["bowl_wkts"] = 25 * w
    out["bowl_lbwb"] = 5 * df["lbw_bowled"]
    out["bowl_haul"] = np.select([w >= 5, w >= 4, w >= 3], [15, 10, 5], 0)
    out["bowl_maiden"] = 6 * df["maidens"]
    out["bowl_econ"] = econ_points(df["runs_conceded"].to_numpy(), df["legal_balls_bowled"].to_numpy())
    out["field"] = 6 * c + 4 * (c >= 3) + 10 * df["stumpings"] + 10 * df["runout_direct"] + 5 * df["runout_indirect"]
    out = out.astype("int32")
    for g, cols in GROUPS.items():
        out["pts_" + g] = out[cols].sum(axis=1)
    out["total"] = out[COMPONENTS].sum(axis=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/interim/player_match_raw.parquet")
    ap.add_argument("--out", default="data/processed/player_match.parquet")
    ap.add_argument("--audit-dir", default="data/audit")
    ap.add_argument("--six-also-boundary", action="store_true")
    ap.add_argument("--duck-scope", default="all", choices=["all", "faced_ball"])
    a = ap.parse_args()
    P = pd.read_parquet(a.inp)
    Z = phi(P, a.six_also_boundary, a.duck_scope)
    P = pd.concat([P, Z], axis=1)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    P.to_parquet(a.out, index=False)
    t = P.loc[~P.no_play, "total"]
    audit = {"rows": len(P), "total_stats_played": t.describe(percentiles=[.5, .9, .99]).to_dict(),
             "frac_only_participation": float((t == 3).mean()),
             "sum_components_eq_total": bool((P[COMPONENTS].sum(axis=1) == P.total).all()),
             "config": {"six_also_boundary": a.six_also_boundary, "duck_scope": a.duck_scope}}
    audit["pass"] = audit["sum_components_eq_total"]
    json.dump(audit, open(os.path.join(a.audit_dir, "points_audit.json"), "w"), indent=1)
    print(json.dumps(audit, indent=1))
    sys.exit(0 if audit["pass"] else 1)


if __name__ == "__main__":
    main()
