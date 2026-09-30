"""Baseline preprocessing: candidate sets (season pseudo-squads, padded in training), the staleness plan (Δ per
fixture and epoch) and token tensors per fixture; fantasy roles as of c for the ILP only."""

import numpy as np
import os
import pandas as pd
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from shared.preprocessing import roles as rl  # noqa: E402
from baseline.config import CATS, HP, N_NUM, N_SC  # noqa: E402
from baseline.ingestion import days  # noqa: E402


class Fixtures(torch.utils.data.Dataset):
    """One item = one fixture, both candidate sets, at the Δ and side swap of the current plan."""

    def __init__(self, store, fx, L):
        self.s, self.fx, self.L = store, fx, L
        self.plan(np.zeros(len(fx), np.int64), np.zeros(len(fx), np.int64))

    def plan(self, deltas, swaps):
        self.deltas, self.swaps = np.asarray(deltas), np.asarray(swaps)

    def __len__(self):
        return len(self.fx)

    def __getitem__(self, k):
        f, s, L = self.fx[k], self.s, self.L
        delta = int(self.deltas[k]); c = f["day"] - delta
        n = len(f["pc"])
        kind = np.zeros((n, L), np.int64); rank = np.zeros((n, L), np.int64); cat = np.zeros((n, L, len(CATS)), np.int64)
        num = np.zeros((n, L, N_NUM), np.float32); sc = np.zeros((n, L, N_SC), np.float32); age = np.zeros((n, L), np.float32)
        ntok = np.zeros(n, np.float32)
        for j in range(n):
            kd, nm, rows = s.tokens(f["pc"][j], f["tc"][j], c, L)
            m = len(kd)
            if not m:
                continue
            dd = s.T_day[rows]
            a = np.log1p(c - dd)
            kind[j, :m] = kd; num[j, :m] = nm; cat[j, :m] = s.T_cat[rows]; age[j, :m] = a
            sc[j, :m, :2] = s.T_sc[rows]; sc[j, :m, 2] = s.T_team[rows] == f["tc"][j]
            sc[j, :m, 3] = a / 8; sc[j, :m, 4] = np.log1p(np.diff(dd, prepend=dd[0])) / 8
            rank[j, :m] = np.arange(m)[::-1]; ntok[j] = np.log1p(m) / 5
        return dict(kind=kind, num=num, cat=cat, sc=sc, age=age, rank=rank, ntok=ntok, y=f["y"],
                    side=f["side"] ^ int(self.swaps[k]), fxcat=f["cat"], month=f["month"],
                    fxsc=np.array([f["sched"] / 20, f["bpo"] / 6, np.log1p(delta) / 6], np.float32))


def collate(items):
    b = {k: torch.from_numpy(np.concatenate([it[k] for it in items])) for k in ("kind", "num", "cat", "sc", "age", "rank", "ntok")}
    b["y"] = torch.from_numpy(np.concatenate([it["y"] for it in items]).astype(np.float32))
    b["side"] = torch.from_numpy(np.concatenate([it["side"] for it in items]).astype(np.int64))
    n = [len(it["y"]) for it in items]
    b["fx"] = torch.from_numpy(np.repeat(np.arange(len(items)), n))
    b["slot"] = torch.from_numpy(np.concatenate([np.arange(k) for k in n]))
    b["w"] = torch.from_numpy(np.concatenate([np.full(k, 1 / (k * len(items)), np.float32) for k in n]))
    b["fxcat"] = torch.from_numpy(np.stack([it["fxcat"] for it in items]))
    b["month"] = torch.tensor([it["month"] for it in items])
    b["fxsc"] = torch.from_numpy(np.stack([it["fxsc"] for it in items]))
    return b


def fixture_rows(P, S, start, end, pad):
    """Candidate rows of fixtures dated in [start, end): season pseudo-squads (in_base); with pad, each team's set is
    filled up to pad_to with extended-squad candidates (appeared for the team in the prior 365 d), most recent first."""
    s = S[(S.date >= start) & (S.date < end)]
    rows = s[s.in_base]
    if pad:
        ext = s[~s.in_base].copy()
        last = P[["team", "player_id", "date"]].assign(k=lambda d: d.team + "|" + d.player_id).rename(columns={"date": "src"})
        last = last.sort_values("src", kind="stable")
        q = ext.assign(k=ext.team + "|" + ext.player_id, _d=pd.to_datetime(ext.date), _i=np.arange(len(ext))).sort_values("_d", kind="stable")
        q = pd.merge_asof(q, last.assign(_d=pd.to_datetime(last.src))[["k", "_d", "src"]], on="_d", by="k",
                          allow_exact_matches=False, direction="backward").sort_values("_i")
        q = q.sort_values(["match_id", "team", "src", "player_id"], ascending=[True, True, False, True], kind="stable")
        room = (HP["pad_to"] - rows.groupby(["match_id", "team"]).size()).clip(lower=0).rename("room").reset_index()
        q = q.merge(room, on=["match_id", "team"])
        q = q[q.groupby(["match_id", "team"]).cumcount() < q.room]
        rows = pd.concat([rows, q[rows.columns]])
    rows = rows.merge(P[["match_id", "team", "player_id", "total"]], on=["match_id", "team", "player_id"], how="left")
    rows["y"] = np.where(rows.played == 1, rows.total.fillna(0), 0.0)
    two = rows.groupby("match_id").team.nunique()
    return rows[rows.match_id.isin(two[two == 2].index)]


def make_fixtures(store, rows, M):
    """rows: [match_id, team, player_id, y] (+ side via M.team1). Returns a list of fixture dicts in (date, id) order."""
    Mi = M.set_index("match_id")
    rows = rows.assign(side=(rows.team != rows.match_id.map(Mi.team1)).astype(np.int64))
    rows = rows.sort_values(["match_id", "side", "player_id"], kind="stable")
    fcat = store.encode(Mi.reindex(rows.match_id.unique()).reset_index().assign(opp=None, team=None, bat_first=None, won=None))
    out = []
    for (mid, g), fc in zip(rows.groupby("match_id", sort=False), fcat):
        m = Mi.loc[mid]
        out.append(dict(id=mid, date=m.date, day=int(days([m.date])[0]), month=int(m.date[5:7]), sched=float(m.sched_overs),
                        bpo=float(m.balls_per_over), cat=fc[:5], side=g.side.to_numpy(),
                        pc=np.array([store.pid_code.get(p, -1) for p in g.player_id]), tc=np.array([store.team_code.get(t, -1) for t in g.team]),
                        y=g.y.to_numpy(np.float32), team=g.team.to_numpy(), player_id=g.player_id.to_numpy(),
                        played=g.played.to_numpy(np.int64) if "played" in g else np.zeros(len(g), np.int64)))
    return sorted(out, key=lambda f: (f["date"], f["id"]))


def role_table(P, fx, deltas, state=None):
    """Fantasy roles as of each copy's cutoff (for the ILP only), grouped by (fixture copy, team) like features.build."""
    Q = pd.DataFrame([(f"{f['id']}@{d}", t, p, str(np.datetime64(f["day"] - d, "D")))
                      for f in fx for d in deltas for t, p in zip(f["team"], f["player_id"])], columns=["match_id", "team", "player_id", "date"])
    R = rl.infer(Q, state if state is not None else rl.role_state(P, rl.DEFAULTS["window"]), **rl.DEFAULTS)
    return dict(zip(zip(Q.match_id, Q.team, Q.player_id), R.role))


def to(b, dev):
    return {k: v.to(dev, non_blocking=True) for k, v in b.items()}


def loader(ds, idx_batches, workers):
    return torch.utils.data.DataLoader(ds, batch_sampler=idx_batches, collate_fn=collate, num_workers=workers,
                                       pin_memory=torch.cuda.is_available())
