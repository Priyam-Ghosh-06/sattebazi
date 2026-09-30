"""Refined ingestion: the baseline token store + A0a bench tokens (season rule, cutoff-independent), capacities
(keeper, bowling option, batting depth) from appearances dated < c, and the XI-size rule."""

import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from baseline.config import N_NUM  # noqa: E402
from baseline.ingestion import Store as BaseStore  # noqa: E402


class Store(BaseStore):
    """CB's token tables + A0a bench tokens (season rule), raw stats for capacities, XI sizes."""

    def __init__(self, P, M, fold_end=None, vocab=None, norm=None):
        super().__init__(P, M, fold_end, vocab, norm)
        season = pd.Series(self.T_mid).map(M.set_index("match_id").season).fillna("?").to_numpy()
        df = pd.DataFrame({"t": self.T_team, "s": season, "d": self.T_day})
        first = df.groupby(["t", "s"]).d.transform("min")
        self.T_srank = df.assign(f=first).groupby("t").f.rank(method="dense").to_numpy(np.int64)
        inv = {v: k for k, v in self.team_code.items()}
        xi = P.groupby(["match_id", "team"]).size()
        self.T_xi = np.array([xi.get((m, inv[t]), 11) for m, t in zip(self.T_mid, self.T_team)], np.int64)
        A = P[["player_id", "match_id", "legal_balls_bowled", "bat_position", "stumpings", "catches"]].copy()
        A["mo"] = A.match_id.map(self.mord); A["pc"] = A.player_id.map(self.pid_code)
        A = A.sort_values(["pc", "mo"], kind="stable")  # the same order as the parent's appearance arrays
        self.A_lb = A.legal_balls_bowled.to_numpy(); self.A_bp = A.bat_position.replace(0, 11).to_numpy()
        self.A_st = A.stumpings.to_numpy(); self.A_ct = A.catches.to_numpy(float)
        self.A_cst = A.groupby("pc").stumpings.cumsum().to_numpy()
        self.A_bf = P[["player_id", "match_id", "balls_faced"]].assign(mo=lambda d: d.match_id.map(self.mord), pc=lambda d: d.player_id.map(self.pid_code)) \
            .sort_values(["pc", "mo"], kind="stable").balls_faced.to_numpy(float)

    def recent(self, pc, c):
        """(mean legal balls bowled, mean balls faced) over the last 10 appearances before c: who is likely to bowl / bat."""
        if pc not in self.a_off:
            return 0.0, 0.0
        s, e = self.a_off[pc]
        hi = s + np.searchsorted(self.A_day[s:e], c)
        w = slice(max(s, hi - 10), hi)
        return (float(self.A_lb[w].mean()), float(self.A_bf[w].mean())) if hi > s else (0.0, 0.0)

    def tokens(self, pc, tc, c, L):
        """The L most recent of: appearances (any team) and bench tokens (team tc matches he missed while he was a
        current squad member: an appearance for tc earlier in the same or the previous season), all dated < c."""
        ai = bi = np.empty(0, np.int64)
        if pc in self.a_off:
            s, e = self.a_off[pc]
            hi = s + np.searchsorted(self.A_day[s:e], c)
            ai = np.arange(max(s, hi - L), hi)
        pos = self.played.get(pc * 100000 + tc) if pc >= 0 else None
        if pos is not None:
            s, e = self.t_off[tc]
            d = self.T_day[s:e]
            lo, hi = s + np.searchsorted(d, self.T_day[pos[0]]), s + np.searchsorted(d, c)
            if hi > lo:
                cand = np.setdiff1d(np.arange(lo, hi), pos, assume_unique=True)
                k = np.searchsorted(pos, cand)
                ok = k > 0
                cand, prev = cand[ok], pos[k[ok] - 1]
                bi = cand[self.T_srank[prev] >= self.T_srank[cand] - 1]
        rows = np.r_[self.A_pos[ai], bi]
        o = np.argsort(self.T_mo[rows], kind="stable")[-L:]
        kind = np.r_[np.ones(len(ai), np.int64), np.full(len(bi), 2, np.int64)][o]
        num = np.r_[self.A_num[ai], np.zeros((len(bi), N_NUM), np.float32)][o]
        return kind, num, rows[o]

    def capacity(self, pc, c):
        """(keeper-direct, wk score, bowling option, batting depth, unk) from appearances dated < c."""
        if pc not in self.a_off:
            return 0.0, 0.0, 0.0, 0.0, 1.0
        s, e = self.a_off[pc]
        hi = s + np.searchsorted(self.A_day[s:e], c)
        if hi == s:
            return 0.0, 0.0, 0.0, 0.0, 1.0
        w30, w10 = slice(max(s, hi - 30), hi), slice(max(s, hi - 10), hi)
        st, ct = self.A_st[w30].sum(), self.A_ct[w30].mean()
        score = float(self.A_cst[hi - 1] > 0) + ct + st
        return float(st >= 1 or ct >= 1.2), score, float((self.A_lb[w10] >= 12).mean() >= 0.5), float(np.median(self.A_bp[w30]) <= 7), 0.0

    def xi_size(self, tc, c):
        """n_XI rule at test time: the XI size of the team's last match before c (11 or 12, impact-sub competitions)."""
        if tc not in self.t_off:
            return 11
        s, e = self.t_off[tc]
        k = s + np.searchsorted(self.T_day[s:e], c)
        return int(np.clip(self.T_xi[k - 1], 11, 12)) if k > s else 11
