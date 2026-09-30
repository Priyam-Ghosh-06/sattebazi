"""Refined decision model, A3: the exact constrained choice layer over each squad,
P(X) ∝ exp(Σ_{i∈X} u_i + φ_keep(n_keep) + φ_bowl(n_bowl)) 1[|X| = n_XI];
log Z by dynamic programming, marginals π = ∂log Z/∂u (the gradient of log Z). sum_to_n is A0b's independent approximation."""

import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from refined.config import BOWL_MAX, HP, KEEP_MAX, NEG, REF  # noqa: E402


def _sat_shift(x, dim, top):
    """Saturating +1 along dim: new[j] = old[j−1] for j ≥ 1, new[top] also keeps old[top]."""
    lo = x.narrow(dim, 0, top - 1)
    last = torch.logaddexp(x.narrow(dim, top - 1, 1), x.narrow(dim, top, 1))
    pad = torch.full_like(x.narrow(dim, 0, 1), NEG)
    return torch.cat([pad, lo, last], dim)


def log_z(u, mask, keep, bowl, nxi, phik, phib, M):
    """Exact log normaliser of P(X) ∝ exp(Σ_{i∈X} u_i + φk(n_keep) + φb(n_bowl)) 1[|X| = n_XI] for T teams at once.
    u, mask, keep, bowl [T, C]; nxi [T]; forward push over candidates with saturating counters."""
    T, C = u.shape
    a = u.new_full((T, M + 1, KEEP_MAX + 1, BOWL_MAX + 1), NEG)
    a = a.index_put((torch.arange(T, device=u.device), torch.zeros(T, dtype=torch.long, device=u.device),
                     torch.zeros(T, dtype=torch.long, device=u.device), torch.zeros(T, dtype=torch.long, device=u.device)),
                    u.new_zeros(T))
    for j in range(C):
        add = a + u[:, j, None, None, None]
        add = torch.cat([torch.full_like(add[:, :1], NEG), add[:, :-1]], 1)  # one more selected (overflow dropped)
        add = torch.where(keep[:, j, None, None, None] > 0, _sat_shift(add, 2, KEEP_MAX), add)
        add = torch.where(bowl[:, j, None, None, None] > 0, _sat_shift(add, 3, BOWL_MAX), add)
        a = torch.where(mask[:, j, None, None, None], torch.logaddexp(a, add), a)
    sel = a[torch.arange(T, device=u.device), nxi]
    return torch.logsumexp((sel + phik[:, None] + phib[None, :]).flatten(1), 1)


class Choice(nn.Module):
    """Count potentials with the gauge fixed at the reference counts (φ = 0 there)."""

    def __init__(self):
        super().__init__()
        self.pk, self.pb = nn.Parameter(torch.zeros(KEEP_MAX + 1)), nn.Parameter(torch.zeros(BOWL_MAX + 1))
        mk, mb = torch.ones(KEEP_MAX + 1), torch.ones(BOWL_MAX + 1)
        mk[list(REF["keep"])] = 0; mb[list(REF["bowl"])] = 0
        self.register_buffer("mk", mk); self.register_buffer("mb", mb)

    def phis(self):
        return self.pk * self.mk, self.pb * self.mb

    def nll(self, u, mask, keep, bowl, nxi, x):
        """Choice loss per team: log Z − energy of the observed XI x (0/1)."""
        phik, phib = self.phis()
        lz = log_z(u, mask, keep, bowl, nxi, phik, phib, HP["max_xi"])
        nk = (keep * x).sum(1).long().clamp(max=KEEP_MAX); nb = (bowl * x).sum(1).long().clamp(max=BOWL_MAX)
        return lz - (u * x * mask).sum(1) - phik[nk] - phib[nb]

    def marginals(self, u, mask, keep, bowl, nxi):
        """π_i = ∂ log Z / ∂u_i (the gradient of log Z)."""
        with torch.enable_grad():
            u = u.detach().requires_grad_(True)
            phik, phib = self.phis()
            lz = log_z(u, mask, keep, bowl, nxi, phik.detach(), phib.detach(), HP["max_xi"])
            (g,) = torch.autograd.grad(lz.sum(), u)
        return g * mask


def sum_to_n(logit, team, nxi, iters=60):
    """Per team shift τ with Σ σ(s − τ) = n_XI (A0b's post-hoc correction)."""
    g = pd.factorize(team)[0]
    size = np.bincount(g).astype(float)
    target = np.minimum(np.bincount(g, weights=nxi) / size, size - 1e-6)
    lo, hi = np.full(g.max() + 1, -30.0), np.full(g.max() + 1, 30.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        big = np.bincount(g, weights=1 / (1 + np.exp(-(logit - mid[g])))) > target
        lo, hi = np.where(big, mid, lo), np.where(big, hi, mid)
    return (1 / (1 + np.exp(-(logit - ((lo + hi) / 2)[g])))).astype(np.float32)
