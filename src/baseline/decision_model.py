"""Baseline decision model: Ŷ -> the exact fantasy-XI ILP (shared/optimization); captured/oracle on a set of fixtures."""

import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from shared.optimization.selection import select_xi  # noqa: E402


def capture(pred, roles, col="yhat"):
    """Per (fixture, Δ): captured points of the ILP XI and the hindsight oracle."""
    rows = []
    for (fx, d), g in pred.groupby(["fx_id", "delta"], sort=False):
        r = np.array([roles[(f"{fx}@{d}", t, p)] for t, p in zip(g.team, g.player_id)])
        y, t, p = g.y.to_numpy(), g.team.to_numpy(), g.player_id.to_numpy()
        cap = y[select_xi(g[col].to_numpy(), r, t, p)[0]].sum()
        orc = y[select_xi(y, r, t, p)[0]].sum()
        rows.append((fx, d, cap, orc))
    return pd.DataFrame(rows, columns=["fx_id", "delta", "captured", "oracle"])


def headline(cap):
    """Equal-weight mean over the Δ grid of the per-fixture mean captured/oracle."""
    c = cap[cap.oracle > 0]
    return float((c.captured / c.oracle).groupby(c.delta).mean().mean())
