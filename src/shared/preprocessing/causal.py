"""Strict causality in one place.

state_table(): per-player rolling stats *including* each appearance (the state after it).
asof():        joins any query (player, date) to the last state with state.date < query.date (strict),
               so played rows, benched squad rows and test-time rows all see only earlier matches,
               and same-day doubleheaders never see each other.
"""
import zlib

import numpy as np
import pandas as pd


def state_table(P, key, windows, ewms):
    """P sorted by date. windows: {name: (col, n|None, how)}; ewms: {name: (col, halflife)}."""
    P = P.sort_values(["date", "match_id"], kind="stable")
    S = P[[key, "date"]].copy()
    g = P.groupby(key, sort=False)
    S["n_apps"] = g.cumcount() + 1
    for name, (col, n, how) in windows.items():
        r = g[col].expanding() if n is None else g[col].rolling(n, min_periods=1)
        S[name] = getattr(r, how)().reset_index(level=0, drop=True)
    for name, (col, hl) in ewms.items():
        S[name] = g[col].transform(lambda x: x.ewm(halflife=hl).mean())
    S["src_date"] = S["date"]
    return S


def asof(Q, S, key):
    """Q needs [key, date]. Returns Q with state columns from the last state strictly before Q.date."""
    Q = Q.copy()
    Q["_d"] = pd.to_datetime(Q["date"]); S = S.copy(); S["_d"] = pd.to_datetime(S["date"])
    Q["_i"] = np.arange(len(Q))
    # stable sorts: among same-date state rows the chronologically last (cumulative) one must win
    out = pd.merge_asof(Q.sort_values("_d", kind="stable"), S.drop(columns="date").sort_values("_d", kind="stable"), on="_d", by=key,
                        allow_exact_matches=False, direction="backward")
    return out.sort_values("_i").drop(columns=["_d", "_i"]).reset_index(drop=True)


# ---------------------------------------------------------------- staleness: history ends at a cutoff c = date − Δ
DELTAS = (0, 90, 180, 270, 365)  # validation / test grid; headline = equal-weight mean over it
SPLITS = {  # name: (train from, validation from, test from, test to); test from = end of the training window
    "F24": ("2019-01-01", "2023-10-01", "2024-01-01", "2024-12-31"),
    "F25": ("2019-01-01", "2024-10-01", "2025-01-01", "2025-12-31"),
    "FINAL": ("2019-01-01", "2025-10-01", "2026-01-01", None),
}


def draw_delta(key):
    """Training staleness in days: Δ ~ 0.10·δ0 + 0.90·U[1, 450], seeded by `key` (fixture id, or 'fixture|epoch')."""
    r = np.random.default_rng(zlib.crc32(str(key).encode()))
    return 0 if r.random() < 0.10 else int(r.integers(1, 451))


def cutoff(fx_date, train_end):
    """Test-time cutoff: history ends at train_end, so c = min(fixture date, train_end + 1 day); Δ = fixture − c."""
    c = min(pd.Timestamp(fx_date), pd.Timestamp(train_end) + pd.Timedelta(days=1))
    return c.strftime("%Y-%m-%d"), (pd.Timestamp(fx_date) - c).days
