"""Mock test_fixtures.json from real matches (pseudo-squads as squads, cricsheet names as display strings),
plus the truth file, for an end-to-end dry run of evaluate.py.

python src/mock_fixtures.py --start 2025-10-01 --end 2025-12-31 --n 40 --out data/mock/test_fixtures.json
"""
import argparse, json, os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2025-10-01")
ap.add_argument("--end", default="2025-12-31")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--out", default="data/mock/test_fixtures.json")
a = ap.parse_args()
S = pd.read_parquet("data/processed/squads.parquet")
S = S[S.in_base & (S.date >= a.start) & (S.date <= a.end)]
P = pd.read_parquet("data/processed/player_match.parquet", columns=["match_id", "player_id", "name", "team", "total"])
names = P.drop_duplicates("player_id").set_index("player_id").name
M = pd.read_parquet("data/interim/matches.parquet").set_index("match_id")
rng = np.random.default_rng(0)
mids = rng.choice(np.sort(S.match_id.unique()), min(a.n, S.match_id.nunique()), replace=False)
fx, truth = [], []
for mid in sorted(mids, key=lambda m: (M.loc[m, "date"], m)):
    m = M.loc[mid]
    sq = {t: [names[p] for p in S[(S.match_id == mid) & (S.team == t)].player_id] for t in (m.team1, m.team2)}
    fx.append({"match_date": m.date, "team1": m.team1, "team2": m.team2, "venue": m.venue, "squads": sq})
    t = P[P.match_id == mid]
    truth += [{"match_date": m.date, "team1": m.team1, "team2": m.team2, "player": r.name, "actual_points": int(r.total)} for r in t.itertuples()]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(fx, open(a.out, "w"), indent=1)
pd.DataFrame(truth).to_csv(a.out.replace(".json", "_truth.csv"), index=False)
print(len(fx), "fixtures ->", a.out)
