"""Exact fantasy-XI selection: maximise Σ x·score s.t. 11 players, role bounds, ≥1 per team (scipy milp)."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

BOUNDS = {"WK": (1, 4), "BAT": (3, 6), "AR": (1, 4), "BOWL": (3, 6)}


def select_xi(score, role, team, player_id=None):
    """Returns indices of the chosen 11. UNK counts as BAT. Deterministic tie-break by player_id rank.
    If role bounds are infeasible, falls back to team constraint only (flagged via second return)."""
    score = np.asarray(score, float)
    role = np.where(np.asarray(role) == "UNK", "BAT", np.asarray(role))
    team = np.asarray(team)
    n = len(score)
    tb = np.argsort(np.argsort(player_id if player_id is not None else np.arange(n))) * 1e-7
    c = -(score + tb)
    A, lo, hi = [np.ones(n)], [11], [11]
    for r, (a, b) in BOUNDS.items():
        A.append((role == r).astype(float)); lo.append(a); hi.append(b)
    for t in np.unique(team):
        A.append((team == t).astype(float)); lo.append(1); hi.append(np.inf)
    res = milp(c, constraints=LinearConstraint(np.array(A), lo, hi), integrality=np.ones(n), bounds=Bounds(0, 1))
    if res.status == 0:
        return np.where(res.x > 0.5)[0], True
    res = milp(c, constraints=LinearConstraint(np.array([A[0]] + A[5:]), [11] + lo[5:], [11] + hi[5:]),
               integrality=np.ones(n), bounds=Bounds(0, 1))
    return np.where(res.x > 0.5)[0], False
