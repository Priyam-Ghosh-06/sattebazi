"""Graph model: rank-4 Poisson factorisation of the batter x bowler graph, E[y_ij] = n_ij exp(β + a_i + b_j + u_i·v_j),
one for runs and one for dismissals, refitted monthly on deliveries dated < each snapshot (L2 priors = shrinkage);
graph features = the opponent squad's expected effect on each candidate as of its cutoff c; gate G8."""

import json
import numpy as np
import os
import pandas as pd
import pickle
import sys
import time
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from graph.ingestion import PAIRS, SNAPS  # noqa: E402


K, HALF_LIFE, SD_AB, SD_UV = 4, 730.0, 0.3, 0.15


FIRST, LAST = "2017-10-01", "2026-01-01"


FEATS, STALE = "data/processed/matchup_features.parquet", "data/processed/stale_features.parquet"


KEY = ["match_id", "team", "player_id"]


def days(s):
    return pd.to_datetime(pd.Series(s)).to_numpy().astype("datetime64[D]").astype(np.int64)


def aggregate(D, sday):
    """Pair totals over deliveries dated < the snapshot, recency-weighted."""
    d = D[D.day < sday]
    w = 0.5 ** ((sday - d.day.to_numpy()) / HALF_LIFE)
    g = pd.DataFrame({"bi": d.bi.to_numpy(), "wi": d.wi.to_numpy(), "n": w * d.n.to_numpy(), "r": w * d.r.to_numpy(), "w": w * d.w.to_numpy()})
    return g.groupby(["bi", "wi"], sort=True).sum().reset_index(), int(d.day.max()) if len(d) else None


def fit(bi, wi, n, y, nb, nw, init, iters):
    """MAP Poisson factorisation, full-batch Adam (deterministic on CPU). Returns numpy params."""
    t = lambda x: torch.tensor(x, dtype=torch.float64)
    bi_, wi_, n_, y_ = torch.tensor(bi), torch.tensor(wi), t(n), t(y)
    p = {k: t(v.copy()).requires_grad_(True) for k, v in init.items()}
    opt = torch.optim.Adam(p.values(), lr=0.05)
    logn = torch.log(n_)
    for _ in range(iters):
        eta = p["beta"] + p["a"][bi_] + p["b"][wi_] + (p["u"][bi_] * p["v"][wi_]).sum(1)
        nll = (torch.exp(logn + eta) - y_ * eta).sum()
        pen = ((p["a"] ** 2).sum() + (p["b"] ** 2).sum()) / (2 * SD_AB ** 2) + ((p["u"] ** 2).sum() + (p["v"] ** 2).sum()) / (2 * SD_UV ** 2)
        loss = nll + pen
        opt.zero_grad(); loss.backward(); opt.step()
    return {k: v.detach().numpy().copy() for k, v in p.items()}


def snapshot_dates():
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(FIRST, LAST, freq="MS")]


def load_pairs():
    D = pd.read_parquet(PAIRS)
    bats, bowls = np.sort(D.bat.unique()), np.sort(D.bowl.unique())
    D["bi"] = np.searchsorted(bats, D.bat); D["wi"] = np.searchsorted(bowls, D.bowl); D["day"] = days(D.date)
    return D, bats, bowls


def fit_snapshot(D, sday, prev, nb, nw, rng, iters_first=400, iters_warm=80):
    agg, maxday = aggregate(D, sday)
    out = {"max_source_day": maxday}
    for tgt in ("r", "w"):
        if prev is None:
            init = {"beta": np.array(np.log(agg[tgt].sum() / agg.n.sum())), "a": np.zeros(nb), "b": np.zeros(nw),
                    "u": rng.normal(0, 0.01, (nb, K)), "v": rng.normal(0, 0.01, (nw, K))}
        else:
            init = prev[tgt]
        out[tgt] = fit(agg.bi.to_numpy(), agg.wi.to_numpy(), agg.n.to_numpy(), agg[tgt].to_numpy(), nb, nw, init,
                       iters_first if prev is None else iters_warm)
    seen_b = np.zeros(nb, bool); seen_b[agg.bi.unique()] = True
    seen_w = np.zeros(nw, bool); seen_w[agg.wi.unique()] = True
    out["seen_b"], out["seen_w"] = seen_b, seen_w
    return out


def snapshots():
    torch.set_num_threads(4); torch.manual_seed(0)
    D, bats, bowls = load_pairs()
    rng = np.random.default_rng(0)
    snaps, prev, t0 = [], None, time.time()
    for s in snapshot_dates():
        cur = fit_snapshot(D, int(days([s])[0]), prev, len(bats), len(bowls), rng)
        prev = {"r": cur["r"], "w": cur["w"]}
        snaps.append({"date": s, **cur})
        print(f"snapshot {s}: max source {np.datetime64(cur['max_source_day'], 'D') if cur['max_source_day'] else None} ({time.time() - t0:.0f}s)", flush=True)
    pickle.dump({"bats": bats, "bowls": bowls, "snaps": snaps, "k": K}, open(SNAPS, "wb"))


def lw_mean(w, M):
    """log( Σ_j w_j exp(M_ij) / Σ_j w_j ) per row i; NaN if the weights are all 0."""
    s = w.sum()
    return np.log(np.exp(M) @ w / s) if s > 0 else np.full(M.shape[0], np.nan)


def params(sn, tgt, bi, wi):
    """(a, u) for batter indices, (b, v) for bowler indices at one snapshot; unseen players -> 0 (the prior)."""
    p = sn[tgt]
    ok_b = (bi >= 0) & sn["seen_b"][np.maximum(bi, 0)]
    ok_w = (wi >= 0) & sn["seen_w"][np.maximum(wi, 0)]
    a = np.where(ok_b, p["a"][np.maximum(bi, 0)], 0.0); u = np.where(ok_b[:, None], p["u"][np.maximum(bi, 0)], 0.0)
    b = np.where(ok_w, p["b"][np.maximum(wi, 0)], 0.0); v = np.where(ok_w[:, None], p["v"][np.maximum(wi, 0)], 0.0)
    return a, u, b, v, ok_b, ok_w


def matchup_block(sn, own, opp):
    """x_mu_ features for the candidates in `own` against the opponent squad `opp` (DataFrames with bi, wi, weights)."""
    f = {}
    wbowl = np.nan_to_num(opp.share10.to_numpy() * opp.lb10.to_numpy())
    wbat = np.nan_to_num(opp.share10.to_numpy() * opp.bf10.to_numpy())
    for tgt, name in (("r", "run"), ("w", "out")):
        a, u, _, _, okb, _ = params(sn, tgt, own.bi.to_numpy(), own.wi.to_numpy())
        _, _, bo, vo, _, _ = params(sn, tgt, opp.bi.to_numpy(), opp.wi.to_numpy())
        f[f"x_mu_bat_{name}"] = lw_mean(wbowl, bo[None, :] + u @ vo.T)
        f[f"x_mu_self_bat_{name}"] = np.where(okb, a, np.nan)
        _, _, b, v, _, okw = params(sn, tgt, own.bi.to_numpy(), own.wi.to_numpy())
        ao, uo, _, _, _, _ = params(sn, tgt, opp.bi.to_numpy(), opp.wi.to_numpy())
        f[f"x_mu_bowl_{name}"] = lw_mean(wbat, (ao[:, None] + uo @ v.T).T)
        f[f"x_mu_self_bowl_{name}"] = np.where(okw, b, np.nan)
        if tgt == "r":
            f["x_mu_bat_run_int"] = lw_mean(wbowl, u @ vo.T)
            f["x_mu_bowl_run_int"] = lw_mean(wbat, (uo @ v.T).T)
    return pd.DataFrame(f, index=own.index)


def features():
    S = pickle.load(open(SNAPS, "rb"))
    sdays = days([s["date"] for s in S["snaps"]])
    X = pd.read_parquet(STALE, columns=KEY + ["opp", "c_date", "share10", "lb10", "bf10"])
    X["bi"] = pd.Series(pd.Categorical(X.player_id, categories=S["bats"]).codes, index=X.index)
    X["wi"] = pd.Series(pd.Categorical(X.player_id, categories=S["bowls"]).codes, index=X.index)
    X["si"] = np.searchsorted(sdays, days(X.c_date), side="right") - 1  # latest snapshot <= c
    assert (X.si >= 0).all(), "a cutoff before the first snapshot"
    assert (sdays[X.si.to_numpy()] <= days(X.c_date)).all(), "G8: snapshot after the cutoff"
    parts = []
    for (mid, si), g in X.groupby(["match_id", "si"], sort=False):
        sn = S["snaps"][si]
        teams = g.team.unique()
        for t in teams:
            own, opp = g[g.team == t], g[g.team != t]
            if len(opp):
                parts.append(matchup_block(sn, own, opp))
    Fx = pd.concat(parts).reindex(X.index)
    out = pd.concat([X[KEY + ["c_date"]], Fx], axis=1)
    out["mu_snapshot"] = np.array([s["date"] for s in S["snaps"]])[X.si.to_numpy()]
    out.to_parquet(FEATS, index=False)
    print(json.dumps({"rows": len(out), "null_rate": Fx.isna().mean().round(3).to_dict()}))


def g8():
    """Gate G8 (SPEC §15.1)."""
    S = pickle.load(open(SNAPS, "rb"))
    bad = [s["date"] for s in S["snaps"] if s["max_source_day"] is not None and s["max_source_day"] >= days([s["date"]])[0]]
    F = pd.read_parquet(FEATS, columns=["c_date", "mu_snapshot"])
    rows_ok = bool((F.mu_snapshot <= F.c_date).all())
    # time travel: refit one snapshot from deliveries truncated at its date, warm-started from the stored previous snapshot
    D, bats, bowls = load_pairs()
    k = len(S["snaps"]) // 2
    s = S["snaps"][k]; prev = S["snaps"][k - 1]
    sday = int(days([s["date"]])[0])
    torch.set_num_threads(4); torch.manual_seed(0)
    re = fit_snapshot(D[D.day < sday].reset_index(drop=True), sday, {"r": prev["r"], "w": prev["w"]}, len(bats), len(bowls), np.random.default_rng(0))
    same = all(np.array_equal(re[t][p], s[t][p]) for t in ("r", "w") for p in ("a", "b", "u", "v", "beta"))
    au = {"snapshots_with_future_source": bad, "rows_snapshot_le_cutoff": rows_ok, "time_travel_snapshot": s["date"],
          "time_travel_identical": bool(same), "pass": not bad and rows_ok and same}
    json.dump(au, open("data/audit/g8_audit.json", "w"), indent=1)
    print(json.dumps(au))
    sys.exit(0 if au["pass"] else 1)


def add_graph_features(X, train_end, snaps=SNAPS):
    """Test time: x_mu_ features for the fixture candidates in X (columns match_id, team, player_id, date, share10, lb10, bf10),
    each fixture from the latest snapshot <= its cutoff c = min(fixture date, train_end + 1 day)."""
    S = pickle.load(open(snaps, "rb"))
    sdays = days([s["date"] for s in S["snaps"]])
    end = days([(pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")])[0]
    X = X.copy()
    X["bi"] = pd.Categorical(X.player_id, categories=S["bats"]).codes
    X["wi"] = pd.Categorical(X.player_id, categories=S["bowls"]).codes
    parts = []
    for mid, g in X.groupby("match_id", sort=False):
        c = min(int(days([g.date.iloc[0]])[0]), int(end))
        sn = S["snaps"][int(np.searchsorted(sdays, c, side="right") - 1)]
        for t in g.team.unique():
            own, opp = g[g.team == t], g[g.team != t]
            if len(opp):
                parts.append(matchup_block(sn, own, opp))
    Fx = pd.concat(parts).reindex(X.index)
    return pd.concat([X.drop(columns=["bi", "wi"]), Fx], axis=1)
