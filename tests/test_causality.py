"""Gate G6, the time-travel test. Run: python tests/test_causality.py

For 200 random squad rows (spread over 10 random match dates), rebuild features from history truncated to
date < match date and assert equality with the full-pipeline features. Also assert every stored
feat_source_max_date is strictly before its match date.
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from shared.preprocessing import features as F  # noqa: E402


def test_time_travel(n_dates=10, per_date=20, seed=0):
    P = pd.read_parquet("data/processed/player_match.parquet")
    X = pd.read_parquet("data/processed/features.parquet")
    assert (X.feat_source_max_date.isna() | (X.feat_source_max_date < X.date)).all()
    rng = np.random.default_rng(seed)
    dates = rng.choice(np.sort(X.loc[X.date >= "2015-01-01", "date"].unique()), n_dates, replace=False)
    cols = None
    checked, bad = 0, []
    for d in dates:
        rows = X[X.date == d]
        Q = rows[["match_id", "team", "player_id", "date"]]
        Xt = F.build(P[P.date < d], Q)  # history truncated: the future does not exist
        cols = cols or [c for c in F.feature_cols(Xt) if Xt[c].dtype != object]
        pick = rng.choice(len(Q), min(per_date, len(Q)), replace=False)
        a = Xt.iloc[pick][cols].to_numpy(float)
        # full pipeline (stored). Q holds whole (match, team) squads, so squad-relative ranks are comparable.
        b = X.set_index(["match_id", "team", "player_id"]).loc[pd.MultiIndex.from_frame(Q.iloc[pick][["match_id", "team", "player_id"]]), cols].to_numpy(float)
        same = np.isclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-9)
        if not same.all():
            bad.append((d, [cols[j] for j in np.where(~same.all(0))[0]]))
        checked += len(pick)
    json.dump({"rows_checked": checked, "dates": list(map(str, dates)), "mismatches": bad, "pass": not bad},
              open("data/audit/causality_audit.json", "w"), indent=1)
    assert not bad, bad
    print("time-travel ok on", checked, "rows")



def test_context_time_travel(n_dates=4, seed=1):
    """E14 x_ features (squad/opponent means, venue running mean) rebuilt from truncated history must match."""
    from shared.preprocessing import context
    P = pd.read_parquet("data/processed/player_match.parquet", columns=["match_id", "date", "total", "pts_bat", "pts_bowl"])
    M = pd.read_parquet("data/interim/matches.parquet", columns=["match_id", "date", "venue"])
    X = pd.read_parquet("data/processed/features.parquet")
    rng = np.random.default_rng(seed)
    for d in rng.choice(np.sort(X.loc[X.date >= "2020-01-01", "date"].unique()), n_dates, replace=False):
        Xd = X[X.date == d]
        full = context.add_match_context(Xd, P, M)
        trunc = context.add_match_context(Xd, P[P.date < d], M[M.date < d])
        cols = [c for c in full.columns if c.startswith("x_")]
        assert np.allclose(full[cols].to_numpy(float), trunc[cols].to_numpy(float), equal_nan=True), d
    print("context time-travel ok")


if __name__ == "__main__":
    test_time_travel()
    test_context_time_travel()
