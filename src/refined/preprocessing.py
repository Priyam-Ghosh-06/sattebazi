"""Refined preprocessing: per-fixture tensors (tokens without cutoff-relative inputs, capacities, n_XI, cutoff)
and the batch layout (fixtures, teams, slots)."""

import numpy as np
import os
import pandas as pd
import pickle
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from baseline.config import CATS, N_NUM  # noqa: E402
from baseline.ingestion import days  # noqa: E402
from refined.config import HP, MU_COLS, N_SC, SNAPS  # noqa: E402


class Fixtures(torch.utils.data.Dataset):
    """One item = one fixture. Tokens are truncated at c = day − Δ, or at the fixture day when to_date (A5 pairs)."""

    def __init__(self, store, fx, L, to_date=False, true_xi=True):
        self.s, self.fx, self.L, self.to_date, self.true_xi = store, fx, L, to_date, true_xi
        self.snaps = mu_snaps() if HP["mu"] else None
        self.plan(np.zeros(len(fx), np.int64), np.zeros(len(fx), np.int64))

    def plan(self, deltas, swaps):
        self.deltas, self.swaps = np.asarray(deltas), np.asarray(swaps)

    def __len__(self):
        return len(self.fx)

    def __getitem__(self, k):
        f, s, L = self.fx[k], self.s, self.L
        delta = int(self.deltas[k]); c = f["day"] - delta
        cut = f["day"] if self.to_date else c
        n = len(f["pc"])
        kind = np.zeros((n, L), np.int64); cat = np.zeros((n, L, len(CATS)), np.int64)
        num = np.zeros((n, L, N_NUM), np.float32); sc = np.zeros((n, L, N_SC), np.float32)
        day = np.full((n, L), 1e6, np.float32)  # token day relative to the fixture day; padding far in the future
        caps = np.zeros((n, 5), np.float32); nxi = np.zeros(n, np.int64)
        for j in range(n):
            kd, nm, rows = s.tokens(f["pc"][j], f["tc"][j], cut, L)
            caps[j] = s.capacity(f["pc"][j], c)
            m = len(kd)
            if m:
                dd = s.T_day[rows]
                kind[j, :m] = kd; num[j, :m] = nm; cat[j, :m] = s.T_cat[rows]
                sc[j, :m, :2] = s.T_sc[rows]; sc[j, :m, 2] = s.T_team[rows] == f["tc"][j]
                sc[j, :m, 3] = np.log1p(np.diff(dd, prepend=dd[0])) / 8
                day[j, :m] = dd - f["day"]
        side = f["side"] ^ int(self.swaps[k])
        for sd in (0, 1):  # keeper capacity: direct evidence, or the most keeper-like candidate of the team (roles.py rule)
            t = f["side"] == sd
            if t.any():
                top = caps[t, 1] >= caps[t, 1].max()
                caps[t, 0] = np.maximum(caps[t, 0], (top & (caps[t, 1] >= 0.3)).astype(np.float32))
                nxi[t] = int(f["played"][t].sum()) if self.true_xi else s.xi_size(f["tc"][t][0], c)
        mu = mu_features(self.snaps, s, f, c) if self.snaps is not None else np.zeros((n, 0), np.float32)
        return dict(kind=kind, num=num, cat=cat, sc=sc, day=day, caps=caps[:, [0, 2, 3, 4]], nxi=nxi, y=f["y"], mu=mu,
                    played=f["played"], side=side, fxcat=f["cat"], month=f["month"],
                    fxsc=np.array([f["sched"] / 20, f["bpo"] / 6, np.log1p(delta) / 6], np.float32),
                    cut=np.float32(-delta), delta=np.float32(delta))



_SNAPS = {}


def mu_snaps():
    """Monthly graph snapshots (src/graph/graph_model.py), loaded once per process; each fitted on balls dated < its date."""
    if "s" not in _SNAPS:
        S = pickle.load(open(SNAPS, "rb"))
        S["days"] = days([x["date"] for x in S["snaps"]])
        S["bat_ix"] = {p: i for i, p in enumerate(S["bats"])}
        S["bowl_ix"] = {p: i for i, p in enumerate(S["bowls"])}
        _SNAPS["s"] = S
    return _SNAPS["s"]


def mu_features(S, store, f, c):
    """The 10 graph features for every candidate of fixture f at cutoff c (latest snapshot <= c), each side against the
    other side's squad; bowling / batting weights = mean legal balls bowled / balls faced over the last 10 appearances."""
    from graph.graph_model import matchup_block
    si = int(np.searchsorted(S["days"], c, side="right")) - 1
    n = len(f["pc"])
    if si < 0:
        return np.zeros((n, len(MU_COLS)), np.float32)
    assert S["days"][si] <= c, "snapshot after the cutoff"
    rec = np.array([store.recent(p, c) for p in f["pc"]])
    df = pd.DataFrame({"bi": [S["bat_ix"].get(p, -1) for p in f["player_id"]], "wi": [S["bowl_ix"].get(p, -1) for p in f["player_id"]],
                       "share10": 1.0, "lb10": rec[:, 0], "bf10": rec[:, 1]})
    out = np.zeros((n, len(MU_COLS)), np.float32)
    for sd in (0, 1):
        own, opp = f["side"] == sd, f["side"] != sd
        if own.any() and opp.any():
            out[own] = matchup_block(S["snaps"][si], df[own], df[opp])[MU_COLS].to_numpy(np.float32)
    return np.nan_to_num(out)


def collate(items):
    b = {k: torch.from_numpy(np.concatenate([it[k] for it in items])) for k in ("kind", "num", "cat", "sc", "day", "caps", "nxi", "mu")}
    b["y"] = torch.from_numpy(np.concatenate([it["y"] for it in items]).astype(np.float32))
    b["played"] = torch.from_numpy(np.concatenate([it["played"] for it in items]).astype(np.float32))
    side = np.concatenate([it["side"] for it in items]).astype(np.int64)
    n = [len(it["y"]) for it in items]
    fx = np.repeat(np.arange(len(items)), n)
    team = fx * 2 + side
    order = np.lexsort((np.arange(len(team)), team))
    tslot = np.empty(len(team), np.int64)
    tslot[order] = np.arange(len(team)) - np.searchsorted(team[order], team[order])
    b.update(side=torch.from_numpy(side), fx=torch.from_numpy(fx), team=torch.from_numpy(team), tslot=torch.from_numpy(tslot),
             slot=torch.from_numpy(np.concatenate([np.arange(k) for k in n])),
             w=torch.from_numpy(np.concatenate([np.full(k, 1 / (k * len(items)), np.float32) for k in n])),
             fxcat=torch.from_numpy(np.stack([it["fxcat"] for it in items])), month=torch.tensor([it["month"] for it in items]),
             fxsc=torch.from_numpy(np.stack([it["fxsc"] for it in items])),
             cut=torch.tensor([it["cut"] for it in items]), delta=torch.tensor([it["delta"] for it in items]))
    return b


def loader(ds, batches, workers):
    return torch.utils.data.DataLoader(ds, batch_sampler=batches, collate_fn=collate, num_workers=workers,
                                       pin_memory=torch.cuda.is_available())
