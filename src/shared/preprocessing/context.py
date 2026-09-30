"""Experimental match-context features (prefix x_), all causal:
- opponent / own squad strength: squad means of each player's pre-match state (the squad is a pre-match input)
- venue: running mean of per-player Φ at the venue in strictly earlier matches (shrunk to the global mean)
"""
import os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # src/
from shared.preprocessing.causal import asof, state_table  # noqa: E402

SQUAD_STATS = ["ewm10_pts_bat", "ewm10_pts_bowl", "ewm10_total"]


def add_match_context(X, P, M, k=5):
    X = X.copy()
    b = X[X.in_base]
    g = b.groupby(["match_id", "team"])[SQUAD_STATS].mean()
    own = g.add_prefix("x_own_").reset_index()
    opp = X[["match_id", "team", "opp"]].drop_duplicates().merge(
        g.add_prefix("x_opp_").reset_index().rename(columns={"team": "opp"}), on=["match_id", "opp"], how="left").drop(columns="opp")
    X = X.merge(own, on=["match_id", "team"], how="left").merge(opp, on=["match_id", "team"], how="left")
    # venue: match-level mean Φ per player and batting share, strictly earlier matches only
    V = P.groupby("match_id").agg(v_total=("total", "mean"), v_bat=("pts_bat", "mean"), v_bowl=("pts_bowl", "mean")).reset_index()
    V = V.merge(M[["match_id", "date", "venue"]], on="match_id").dropna(subset=["venue"])
    S = state_table(V.reset_index(drop=True), "venue", {"x_venue_total": ("v_total", None, "mean"),
                                                        "x_venue_bat": ("v_bat", None, "mean"),
                                                        "x_venue_bowl": ("v_bowl", None, "mean")}, {})
    S = S.rename(columns={"n_apps": "x_venue_n"}).drop(columns="src_date")
    q = asof(X[["venue", "date"]].fillna({"venue": "?"}), S, "venue")
    # shrinkage target: the global running mean, also strictly before the match
    G = state_table(V.assign(k="all").reset_index(drop=True), "k", {c: (c, None, "mean") for c in ["v_total", "v_bat", "v_bowl"]}, {})
    gq = asof(X[["date"]].assign(k="all"), G.drop(columns=["src_date", "n_apps"]), "k")
    n = q.x_venue_n.fillna(0).to_numpy()
    for c, gc in [("x_venue_total", "v_total"), ("x_venue_bat", "v_bat"), ("x_venue_bowl", "v_bowl")]:
        X[c] = (n * q[c].fillna(0).to_numpy() + k * gq[gc].to_numpy()) / (n + k)
    X["x_venue_n"] = n
    return X
