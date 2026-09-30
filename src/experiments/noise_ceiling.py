"""E2: how much of Φ is predictable at all? ICC by role + a leaky 'true player mean' ceiling on within-match rank.

python src/noise_ceiling.py --start 2019-01-01 --out results/experiments/E2_noise_ceiling
"""
import argparse, json, os
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2019-01-01")
ap.add_argument("--out", default="results/experiments/E2_noise_ceiling")
a = ap.parse_args()
P = pd.read_parquet("data/processed/player_match.parquet", columns=["match_id", "player_id", "team", "date", "total"])
R = pd.read_parquet("data/processed/roles.parquet")
P = P[P.date >= a.start].merge(R, on=["match_id", "team", "player_id"])
res = {}
for role, g in P.groupby("role"):
    n = g.groupby("player_id").total.transform("size")
    g = g[n >= 10]
    mean_p = g.groupby("player_id").total.transform("mean")
    between = mean_p.groupby(g.player_id).first().var()
    total = g.total.var()
    res[role] = {"icc": float(between / total), "cv_within": float((g.total - mean_p).std() / g.total.mean()),
                 "rows": len(g), "sd_total": float(np.sqrt(total))}
# ceiling: rank players inside each match by their own mean over the same year (uses the future; ceiling only)
P["year"] = P.date.str[:4]
P["m_year"] = P.groupby(["player_id", "year"]).total.transform("mean")
sp = [spearmanr(g.m_year, g.total).statistic for _, g in P[P.date >= "2024-01-01"].groupby("match_id") if g.total.nunique() > 1]
res["ceiling_spearman_played_same_year_mean"] = float(np.nanmean(sp))
res["irreducible_mae_if_true_mean_known"] = float((P.total - P.m_year).abs().mean())
os.makedirs(a.out, exist_ok=True)
json.dump({"start": a.start}, open(os.path.join(a.out, "config.json"), "w"))
json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
icc = " ".join(f"icc_{k}={v['icc']:.3f}" for k, v in res.items() if isinstance(v, dict))
line = f"| {date.today()} | E2 noise_ceiling | - | {icc} ceiling_spearman_played={res['ceiling_spearman_played_same_year_mean']:.3f} mae_if_true_year_mean_known={res['irreducible_mae_if_true_mean_known']:.2f} | leaky ceiling, not a model |\n"
open("results/experiments/LOG.md", "a", encoding="utf-8").write(line); print(line, json.dumps(res, indent=1))
