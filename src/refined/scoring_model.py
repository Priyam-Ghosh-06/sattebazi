"""Refined scoring model: A0a CT-WKV encoder, A0b factorised utility/μ heads with capacity binding, A2 FiLM,
A1 squad attention (μ path), A5 monthly latent transition, A6 student; E[Y] = π·μ."""

import copy
import math
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from baseline.config import CATS, N_NUM  # noqa: E402
from refined.config import HP, NEG, N_SC, TAU  # noqa: E402
from refined.decision_model import Choice, sum_to_n  # noqa: E402


def inv_softplus(x):
    return x + np.log(-np.expm1(-x))


def wkv_scan(k, v, la_app, la_day, day, mask):
    """RWKV-v4-style scan in log space. k, v [N,L,d]; la_app, la_day [d] (>0); day [N,L]; mask [N,L].
    Returns per-token (num, den, m) states: value estimate = num/den, log D = log den + m."""
    N, L, d = k.shape
    num = k.new_zeros(N, d); den = k.new_zeros(N, d); m = k.new_full((N, d), NEG)
    prev = day[:, 0]
    nums, dens, ms = [], [], []
    for n in range(L):
        ok = mask[:, n, None]
        gap = (day[:, n] - prev).clamp(min=0)[:, None]
        la = -(la_app + la_day * gap)
        m2 = torch.maximum(m + la, k[:, n])
        e1, e2 = torch.exp(m + la - m2), torch.exp(k[:, n] - m2)
        num = torch.where(ok, e1 * num + e2 * v[:, n], num)
        den = torch.where(ok, e1 * den + e2, den)
        m = torch.where(ok, m2, m)
        prev = torch.where(mask[:, n], day[:, n], prev)
        nums.append(num); dens.append(den); ms.append(m)
    return torch.stack(nums, 1), torch.stack(dens, 1), torch.stack(ms, 1)


class TimeMix(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.k, self.v, self.r, self.o = (nn.Linear(d, d) for _ in range(4))
        g = np.random.default_rng(0)
        self.w_app = nn.Parameter(torch.tensor(inv_softplus(np.log(2) / g.permutation(np.geomspace(2, 64, d))), dtype=torch.float32))
        self.w_day = nn.Parameter(torch.tensor(inv_softplus(np.log(2) / g.permutation(np.geomspace(30, 1000, d))), dtype=torch.float32))

    def rates(self):
        return Fn.softplus(self.w_app), Fn.softplus(self.w_day)

    def forward(self, x, day, mask):
        la_app, la_day = self.rates()
        num, den, m = wkv_scan(self.k(x), self.v(x), la_app, la_day, day, mask)
        r = torch.sigmoid(self.r(x))
        return self.o(r * num / den.clamp_min(1e-30)), (num, den, m, r)


class Encoder(nn.Module):
    """Token embedding (no cutoff-relative inputs, so one pass gives the state at every cutoff) + CT-WKV blocks."""

    def __init__(self, vocab_sizes, d, layers, edrop):
        super().__init__()
        self.cat = nn.ModuleList([nn.Embedding(n, d) for n in vocab_sizes])
        self.kind = nn.Embedding(3, d)
        self.num = nn.Linear(N_NUM + N_SC, d)
        self.edrop = nn.Dropout(edrop)
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.tm = nn.ModuleList([TimeMix(d) for _ in range(layers)])
        self.cm = nn.ModuleList([nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(layers)])
        self.cold = nn.Parameter(torch.zeros(2 * d))

    def states(self, b):
        """Per-token state of the last layer (num, den, m, r) and the token days/mask."""
        mask = b["kind"] > 0
        x = self.num(torch.cat([b["num"], b["sc"]], -1)) + self.kind(b["kind"])
        for k, e in enumerate(self.cat):
            x = x + e(b["cat"][..., k])
        x = self.edrop(x)
        for ln1, ln2, tm, cm in zip(self.ln1, self.ln2, self.tm, self.cm):
            h, st = tm(ln1(x), b["day"], mask)
            x = x + h
            x = x + cm(ln2(x))
        return st, b["day"], mask

    def readout(self, st, day, mask, cut):
        """z at cutoff cut [N] (days relative to the fixture): [N/D of the last token before cut, (log D − λ_day·gap)/10].
        Cold (no token before cut): the learned cold vector. Returns z [N, 2d] and cold flag [N]."""
        num, den, m, r = st
        n_before = ((day < cut[:, None]) & mask).sum(1)
        cold = n_before == 0
        idx = (n_before - 1).clamp(min=0)
        g = lambda t: t[torch.arange(len(idx), device=t.device), idx]
        la_day = self.tm[-1].rates()[1]
        gap = (cut - g(day)).clamp(min=0)[:, None]
        o = g(r) * g(num) / g(den).clamp_min(1e-30)
        logd = (torch.log(g(den).clamp_min(1e-30)) + g(m) - la_day * gap) / 10
        z = torch.cat([o, logd], -1)
        z = torch.where(cold[:, None], self.cold.expand_as(z), z)
        return z, cold, (cut - g(day)).clamp(min=0) * (~cold)

    def forward(self, b, cut):
        return self.readout(*self.states(b), cut)


class Decoder(nn.Module):
    def __init__(self, vocab_sizes, d, heads, drop, ybar, ysd):
        super().__init__()
        self.fcat = nn.ModuleList([nn.Embedding(n, d) for n in vocab_sizes[:5]])
        self.month, self.fnum = nn.Embedding(13, d), nn.Linear(3, d)
        self.inp = nn.Sequential(nn.Linear(2 * d + 3 + d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.film = nn.Linear(d, 2 * d)  # A2
        self.R = nn.Parameter(torch.zeros(d, 3))  # capacity binding (keep, bowl, bat)
        self.p = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.own = nn.ModuleList([nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True) for _ in range(2)])  # A1
        self.opp = nn.ModuleList([nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True) for _ in range(2)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(2)])
        self.mu = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.choice = Choice()
        self.register_buffer("ybar", torch.tensor(float(ybar))); self.register_buffer("ysd", torch.tensor(float(ysd)))

    def fixture(self, b):
        g = self.month(b["month"]) + self.fnum(b["fxsc"])
        for k in range(5):
            g = g + self.fcat[k](b["fxcat"][:, k])
        return g

    def forward(self, z, gapd, b):
        """Utilities s and μ for every candidate."""
        g = self.fixture(b)[b["fx"]]
        unk = b["caps"][:, 3:4]
        u = self.inp(torch.cat([z, torch.log1p(gapd)[:, None] / 8, unk, b["side"][:, None].float(), g], -1))
        if HP["a2"]:
            ga, be = self.film(g).chunk(2, -1)
            u = u * (1 + ga) + be
        bnd = u * (1 + b["caps"][:, :3] @ self.R.t())
        s = self.p(bnd).squeeze(-1)
        h = bnd
        if HP["a1"]:
            B, C = b["fxcat"].shape[0], int(b["slot"].max()) + 1
            H = h.new_zeros(B, C, h.shape[-1]).index_put((b["fx"], b["slot"]), h)
            valid = torch.zeros(B, C, dtype=torch.bool, device=h.device).index_put((b["fx"], b["slot"]), torch.ones_like(b["fx"], dtype=torch.bool))
            sd = torch.full((B, C), -1, dtype=torch.long, device=h.device).index_put((b["fx"], b["slot"]), b["side"])
            same = (sd[:, :, None] == sd[:, None, :]) & valid[:, None, :]
            same = same | ~same.any(-1, keepdim=True)  # padding rows: attend somewhere finite; their output is unused
            other = (sd[:, :, None] != sd[:, None, :]) & valid[:, None, :] & valid[:, :, None]
            other = other | ~other.any(-1, keepdim=True)  # rows with no opponent (padding) attend to all: output unused
            heads = self.own[0].num_heads
            for k in range(2):
                x = self.ln[2 * k](H)
                H = H + self.own[k](x, x, x, attn_mask=(~same).repeat_interleave(heads, 0), need_weights=False)[0] \
                      + self.opp[k](x, x, x, attn_mask=(~other).repeat_interleave(heads, 0), need_weights=False)[0]
                H = H + self.ff[k](self.ln[2 * k + 1](H))
            h = H[b["fx"], b["slot"]]
        mu = self.mu(h).squeeze(-1) * self.ysd + self.ybar
        return s, mu

    def teams(self, b, *vals):
        """Scatter candidate vectors into [T, C] team tensors (T = 2 per fixture)."""
        T, C = int(b["team"].max()) + 1, int(b["tslot"].max()) + 1
        idx = (b["team"], b["tslot"])
        mask = torch.zeros(T, C, dtype=torch.bool, device=b["team"].device).index_put(idx, torch.ones_like(b["team"], dtype=torch.bool))
        out = [v.new_zeros(T, C).index_put(idx, v) for v in vals]
        nxi = torch.full((T,), 11, dtype=torch.long, device=b["team"].device).index_put((b["team"],), b["nxi"])
        return mask, nxi, out

    def expected(self, s, mu, b, how="a3"):
        """E[Y] = π μ with π from the choice layer (A3) or independent logits + sum-to-XI shift τ (A0b)."""
        if how == "a3":
            mask, nxi, (u, keep, bowl) = self.teams(b, s, b["caps"][:, 0], b["caps"][:, 1])
            pi = self.choice.marginals(u, mask, keep, bowl, nxi)[b["team"], b["tslot"]]
        else:
            pi = torch.from_numpy(sum_to_n(s.detach().cpu().numpy(), b["team"].cpu().numpy(), b["nxi"].cpu().numpy())).to(s.device)
        return pi * mu, pi

    def loss(self, s, mu, b):
        """P1 loss: choice NLL per selected player (A3; BCE if off) + L2 on μ over played rows."""
        pl = b["played"] > 0
        l_mu = (((mu - b["y"]) / self.ysd) ** 2 * pl).sum() / pl.sum().clamp(min=1)
        if HP["a3"]:
            mask, nxi, (u, keep, bowl, x) = self.teams(b, s, b["caps"][:, 0], b["caps"][:, 1], b["played"])
            l_p = (self.choice.nll(u, mask, keep, bowl, nxi, x) / nxi.clamp(min=1)).mean()
        else:
            l_p = Fn.binary_cross_entropy_with_logits(s, b["played"])
        return l_p + l_mu


class Transition(nn.Module):
    """ẑ_{k+1} = z_k + MLP([z_k, season phase, gender, team type]) (one shared g: Stat-LuPTS)."""

    def __init__(self, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * d + 4, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)

    def forward(self, z, phase, gender, ttype):
        return z + self.f(torch.cat([z, torch.sin(phase)[:, None], torch.cos(phase)[:, None], gender[:, None], ttype[:, None]], -1))


class Student(nn.Module):
    def __init__(self, d_in, d):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d_in, 2 * d), nn.GELU(), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


class ADV(nn.Module):
    def __init__(self, vocab_sizes, ybar, ysd):
        super().__init__()
        d = HP["d"]
        self.enc = Encoder(vocab_sizes, d, HP["layers"], HP["edrop"])
        self.dec = Decoder(vocab_sizes, d, HP["heads"], HP["drop"], ybar, ysd)
        self.dec_roll = copy.deepcopy(self.dec)  # P3 copy (A5 path)
        self.g = Transition(d)
        self.student = Student(2 * d + 1 + 4 + 3, d)

    def roll(self, z, cold, b, steps):
        """z* = g^steps(z(c)); steps per candidate; the phase advances one month per step. Cold stays cold."""
        fx = b["fx"]
        month0 = (b["month"][fx].float() - 1 - (b["delta"][fx] / 30.44)) % 12
        gender, ttype = (b["fxcat"][fx, 1] == 1).float(), (b["fxcat"][fx, 2] == 1).float()
        out = z
        for k in range(int(steps.max().item()) if len(steps) else 0):
            nxt = self.g(out, 2 * math.pi * ((month0 + k) % 12) / 12, gender, ttype)
            out = torch.where(((steps > k) & ~cold)[:, None], nxt, out)
        return out

    def predict_many(self, b, paths):
        """{'<path>_<how>': E[Y]} for several predictors from one encoder pass (and at most one roll-out)."""
        z, cold, gapd = self.enc(b, b["cut"][b["fx"]])
        out, cache = {}, {}
        for path, how in paths:
            if path == "direct":
                if "d" not in cache:
                    cache["d"] = self.dec(z, gapd, b)
                out[f"{path}_{how}"] = self.dec.expected(*cache["d"], b, how)[0]
            elif path == "student":
                out[f"{path}_{how}"] = self.student(student_x(z, gapd, b)) * self.dec.ysd + self.dec.ybar
            else:
                if "r" not in cache:
                    cache["r"] = self.dec_roll(self.roll(z, cold, b, torch.ceil(b["delta"][b["fx"]] / TAU).long()), gapd, b)
                out[f"{path}_{how}"] = self.dec_roll.expected(*cache["r"], b, how)[0]
        return out

    def predict(self, b, path="direct", how="a3"):
        """E[Y] per candidate for one of the predictors: direct (P1), rollout (A5, P3), student (A6)."""
        z, cold, gapd = self.enc(b, b["cut"][b["fx"]])
        if path == "direct":
            s, mu = self.dec(z, gapd, b)
            return self.dec.expected(s, mu, b, how)[0]
        if path == "student":
            return self.student(student_x(z, gapd, b)) * self.dec.ysd + self.dec.ybar
        zr = self.roll(z, cold, b, torch.ceil(b["delta"][b["fx"]] / TAU).long())
        s, mu = self.dec_roll(zr, gapd, b)
        return self.dec_roll.expected(s, mu, b, how)[0]


def student_x(z, gapd, b):
    fx = b["fx"]
    return torch.cat([z, torch.log1p(gapd)[:, None] / 8, b["caps"], b["fxsc"][fx]], -1)


def build(store, fx_train):
    """μ is points *if he plays*: its output scale comes from played rows."""
    y = np.concatenate([f["y"][f["played"] > 0] for f in fx_train])
    return ADV([len(store.vocab[c]) + 1 for c in CATS], y.mean(), y.std() + 1e-6)
