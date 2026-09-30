"""Baseline ingestion: player-match and team-match tables -> numpy token store (appearance and absence tokens).

Vocabularies and standardisation come from the training fold only; every token of a candidate is dated < the cutoff c."""

import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from baseline.config import CATS, HP, NUM, N_NUM  # noqa: E402


def days(s):
    return pd.to_datetime(pd.Series(s)).to_numpy().astype("datetime64[D]").astype(np.int64)


def team_matches(P, M):
    """One row per (match, team) with the team's own view of the match context."""
    tm = P.drop_duplicates(["match_id", "team"])[["match_id", "team", "opp", "date"]].merge(
        M[["match_id", "match_type", "gender", "team_type", "event", "venue", "sched_overs", "balls_per_over",
           "toss_winner", "toss_decision", "winner"]], on="match_id", how="left")
    bat = (tm.toss_winner == tm.team) == (tm.toss_decision == "bat")
    tm["bat_first"] = np.where(tm.toss_winner.isna(), "?", np.where(bat, "1", "0"))
    tm["won"] = np.select([tm.winner == tm.team, tm.winner.notna()], ["1", "0"], "?")
    return tm


class Store:
    """Numpy token tables. Vocabularies and standardisation come from the training fold (dates < fold_end) or are passed in."""

    def __init__(self, P, M, fold_end=None, vocab=None, norm=None, min_freq=HP["min_freq"]):
        tm = team_matches(P, M)
        if vocab is None:
            f = tm[tm.date < fold_end]
            vocab = {}
            for c in CATS:
                vc = f[c].fillna("?").astype(str).value_counts()
                vocab[c] = {v: i + 1 for i, v in enumerate(sorted(vc[vc >= min_freq].index))}  # 0 = UNK / padding
        self.vocab = vocab
        order = M.sort_values(["date", "match_id"], kind="stable").match_id
        self.mord = dict(zip(order, range(len(order))))
        self.team_code = {t: i for i, t in enumerate(sorted(tm.team.unique()))}
        tm = tm.assign(tcode=tm.team.map(self.team_code), day=days(tm.date), mo=tm.match_id.map(self.mord))
        tm = tm.sort_values(["tcode", "mo"], kind="stable").reset_index(drop=True)
        self.T_day, self.T_mo, self.T_team, self.T_mid = tm.day.to_numpy(), tm.mo.to_numpy(), tm.tcode.to_numpy(), tm.match_id.to_numpy()
        self.T_cat = self.encode(tm)
        self.T_sc = np.c_[tm.sched_overs / 20, tm.balls_per_over / 6].astype(np.float32)
        self.t_off = self.offsets(self.T_team)
        tpos = dict(zip(zip(tm.match_id, tm.team), range(len(tm))))
        A = P[["player_id", "match_id", "team", "date", "total", *NUM]].copy()
        A["mo"] = A.match_id.map(self.mord)
        self.pid_code = {p: i for i, p in enumerate(sorted(A.player_id.unique()))}
        A["pc"] = A.player_id.map(self.pid_code)
        A = A.sort_values(["pc", "mo"], kind="stable").reset_index(drop=True)
        pos = np.array([tpos[k] for k in zip(A.match_id, A.team)])
        raw = np.c_[np.log1p(A[NUM].to_numpy(float)), A.total.to_numpy(float)]
        if norm is None:
            fit = raw[(A.date < fold_end).to_numpy()]
            norm = {"mean": fit.mean(0).tolist(), "std": (fit.std(0) + 1e-6).tolist()}
        self.norm = norm
        self.A_num = ((raw - np.array(norm["mean"])) / np.array(norm["std"])).astype(np.float32)
        self.A_day, self.A_mo, self.A_pos = self.T_day[pos], self.T_mo[pos], pos
        self.a_off = self.offsets(A.pc.to_numpy())
        # (player, team) -> sorted team-match positions where the player played (absence tokens exclude these)
        pc, tc = A.pc.to_numpy(), self.T_team[pos]
        o = np.lexsort((pos, tc, pc))
        key = pc[o].astype(np.int64) * 100000 + tc[o]
        cut = np.flatnonzero(np.diff(key)) + 1
        self.played = {int(k[0]): v for k, v in zip(np.split(key, cut), np.split(pos[o], cut))}

    @staticmethod
    def offsets(code):
        cut = np.r_[0, np.flatnonzero(np.diff(code)) + 1, len(code)]
        return {int(code[a]): (a, b) for a, b in zip(cut[:-1], cut[1:])}

    def encode(self, df):
        return np.stack([df[c].fillna("?").astype(str).map(self.vocab[c]).fillna(0).to_numpy(np.int64) for c in CATS], 1)

    def tokens(self, pc, tc, c, L):
        """Token rows of candidate (player code pc, fixture team code tc) at cutoff day c: the L most recent of
        appearances (any team) and absences (team tc matches in [max(first app for tc, c − 365 d), c) he missed)."""
        ai = bi = np.empty(0, np.int64)
        if pc in self.a_off:
            s, e = self.a_off[pc]
            hi = s + np.searchsorted(self.A_day[s:e], c)  # strictly before c
            ai = np.arange(max(s, hi - L), hi)
        pos = self.played.get(pc * 100000 + tc) if pc >= 0 else None
        if pos is not None:
            s, e = self.t_off[tc]
            d = self.T_day[s:e]
            lo = s + np.searchsorted(d, max(self.T_day[pos[0]], c - HP["absence_days"]))
            hi = s + np.searchsorted(d, c)
            if hi > lo:
                bi = np.setdiff1d(np.arange(lo, hi), pos, assume_unique=True)
        rows = np.r_[self.A_pos[ai], bi]
        o = np.argsort(self.T_mo[rows], kind="stable")[-L:]
        kind = np.r_[np.ones(len(ai), np.int64), np.full(len(bi), 2, np.int64)][o]
        num = np.r_[self.A_num[ai], np.zeros((len(bi), N_NUM), np.float32)][o]
        return kind, num, rows[o]


def load_tables(train_end="2025-12-31"):
    P = pd.read_parquet("data/processed/player_match.parquet")
    M = pd.read_parquet("data/interim/matches.parquet")
    S = pd.read_parquet("data/processed/squads.parquet")
    assert (P.date <= train_end).all() and (M.date <= train_end).all(), "leakage guard: post-cutoff rows"
    return P, M, S
