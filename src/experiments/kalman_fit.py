"""E4b: fit the memory length (Kalman local-level λ) by one-step-ahead error instead of choosing it by hand.

Steady-state Kalman filter for a random walk + noise = EWMA with gain K = (−λ + √(λ²+4λ))/2, λ = q/r.
Grid over half-life h (appearances): K = 1 − 0.5^(1/h), λ = K²/(1−K). Score = MSE of predicting the next Φ.
python src/kalman_fit.py --start 2019-01-01 --end 2023-12-31
"""
import argparse, json, os
from datetime import date

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2019-01-01")
ap.add_argument("--end", default="2023-12-31")
ap.add_argument("--out", default="results/experiments/E4b_kalman_fit")
a = ap.parse_args()
P = pd.read_parquet("data/processed/player_match.parquet", columns=["player_id", "date", "match_id", "total", "pts_bat", "pts_bowl", "pts_field"])
P = P[P.date <= a.end].sort_values(["date", "match_id"], kind="stable").reset_index(drop=True)
ev = (P.date >= a.start).to_numpy()
res = {}
for col in ["total", "pts_bat", "pts_bowl", "pts_field"]:
    g = P.groupby("player_id", sort=False)[col]
    prior = g.transform(lambda x: x.expanding().mean().shift(1))
    ok = ev & prior.notna().to_numpy()
    rows = {}
    for h in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100, 1e9]:
        pred = g.transform(lambda x: x.ewm(halflife=h).mean().shift(1)).to_numpy()
        mse = float(np.nanmean((P[col].to_numpy()[ok] - pred[ok]) ** 2))
        K = 1 - 0.5 ** (1 / h)
        rows[str(h)] = {"mse": mse, "K": K, "lambda": K * K / (1 - K)}
    best = min(rows, key=lambda k: rows[k]["mse"])
    res[col] = {"grid": rows, "best_halflife": best, "best_lambda": rows[best]["lambda"], "var": float(P.loc[ok, col].var())}
    print(col, "best h", best, "lambda", rows[best]["lambda"], "mse", rows[best]["mse"], "var", res[col]["var"])
os.makedirs(a.out, exist_ok=True)
json.dump(vars(a), open(os.path.join(a.out, "config.json"), "w"))
json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
s = " ".join(f"{c}:h*={v['best_halflife']},λ={v['best_lambda']:.4f},R2={1 - v['grid'][v['best_halflife']]['mse'] / v['var']:.3f}" for c, v in res.items())
open("results/experiments/LOG.md", "a", encoding="utf-8").write(f"| {date.today()} | E4b kalman_fit | - | {s} | one-step EWMA/Kalman memory fit |\n")
