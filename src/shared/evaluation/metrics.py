"""Shared evaluation: team-level metrics (captured, oracle, captured/oracle via the exact ILP), within-fixture
rank correlation, the squad-change segment, paired bootstrap, gate collection and the data hash."""

import glob
import hashlib
import json
import numpy as np
import os
import pandas as pd
import sys
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # src/
from shared.optimization.selection import select_xi  # noqa: E402


OUT = "results/experiments/E22_final"


PRED = os.path.join(OUT, "pred")


CB_ARMS, SEEDS, YEARS = ("CB64", "CB128"), (0, 1), ("F24", "F25")


KEY = ["fx_id", "delta", "team", "player_id"]


def _xi(args):
    score, y, role, team, pid = args
    idx = select_xi(score, role, team, pid)[0]
    return float(y[idx].sum()), float((y[idx] > 0).mean())


def team_eval(df, cols, pool):
    """Per (fixture, Δ): captured points and share of picked players who played, for every score column."""
    groups = list(df.groupby(["fx_id", "delta"], sort=True))
    out = pd.DataFrame([k for k, _ in groups], columns=["fx_id", "delta"])
    for c in cols:
        r = pool.map(_xi, [(g[c].to_numpy(float), g.y.to_numpy(float), g.role.to_numpy(), g.team.to_numpy(), g.player_id.to_numpy())
                           for _, g in groups], chunksize=64)
        out[c + "_captured"] = [a for a, _ in r]; out[c + "_played"] = [b for _, b in r]
    out["random"] = [11 * g.y.mean() for _, g in groups]
    return out


def spearman_within(df, col):
    v = [spearmanr(g[col], g.y).statistic for _, g in df.groupby(["fx_id", "delta"]) if g[col].nunique() > 1 and g.y.nunique() > 1]
    return float(np.mean(v))


def squad_change(P, fx_ids):
    """Fixture flag: in either team, a player of the team's previous match is missing from the XI, or the XI has a
    player with no earlier appearance for that team (realised XI; used only to segment results)."""
    tm = P[["match_id", "team", "date"]].drop_duplicates().sort_values(["team", "date", "match_id"], kind="stable")
    tm["prev"] = tm.groupby("team").match_id.shift()
    xi = P.groupby(["match_id", "team"]).player_id.agg(frozenset).to_dict()
    first = P.groupby(["team", "player_id"]).date.min().to_dict()
    d = dict(zip(zip(P.match_id, P.team), P.date))
    flag = {}
    for r in tm[tm.match_id.isin(fx_ids)].itertuples():
        now = xi[(r.match_id, r.team)]
        dropped = isinstance(r.prev, str) and bool(xi[(r.prev, r.team)] - now)
        debut = any(first[(r.team, p)] >= d[(r.match_id, r.team)] for p in now)
        flag[r.match_id] = flag.get(r.match_id, False) or dropped or debut
    return flag


def boot(diff, n=2000):
    bs = np.random.default_rng(0).choice(diff, (n, len(diff))).mean(1)
    return {"mean": float(diff.mean()), "lo": float(np.quantile(bs, .025)), "hi": float(np.quantile(bs, .975)), "n": int(len(diff))}


def data_hash():
    h = hashlib.sha256()
    for f in sorted(glob.glob("data/processed/*.parquet")):
        h.update(os.path.basename(f).encode()); h.update(open(f, "rb").read())
    return h.hexdigest()[:16]


def gates():
    """R1 inputs: every audit's pass flag (pipeline gates are shared by all arms)."""
    g = {}
    for f in sorted(glob.glob("data/audit/*_audit.json")):
        a = json.load(open(f))
        g[os.path.basename(f)[:-11]] = bool(a.get("pass", a.get("pass_", True)))
    return g
