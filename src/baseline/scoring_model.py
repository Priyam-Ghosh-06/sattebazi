"""Baseline scoring model CB: Transformer history encoder per candidate -> set decoder over both squads and the
fixture token -> Ŷ = E[Y] per candidate, trained by squared error (its minimiser is E[Y | inputs] = p·μ)."""

import copy
import math
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from baseline.config import CATS, HP, N_NUM, N_SC  # noqa: E402
from baseline.preprocessing import loader, to  # noqa: E402


class CB(nn.Module):
    def __init__(self, vocab_sizes, L, ybar, ysd, d=64, heads=4, layers=2, drop=0.1, edrop=0.05, **_):
        super().__init__()
        self.cat = nn.ModuleList([nn.Embedding(n, d) for n in vocab_sizes])  # shared by tokens and the fixture token
        self.kind, self.rank = nn.Embedding(3, d), nn.Embedding(L, d)
        self.num = nn.Linear(N_NUM + N_SC + 16, d)
        self.register_buffer("omega", torch.tensor(np.geomspace(0.05, 2.0, 8), dtype=torch.float32))
        self.sum = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, "gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.proj = nn.Linear(d + 1, d)
        self.month, self.fnum, self.side = nn.Embedding(13, d), nn.Linear(3, d), nn.Embedding(2, d)
        self.dec = nn.TransformerEncoder(copy.deepcopy(layer), layers, norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.edrop = nn.Dropout(edrop)
        self.register_buffer("ybar", torch.tensor(float(ybar))); self.register_buffer("ysd", torch.tensor(float(ysd)))

    def forward(self, b):
        kind = b["kind"]
        ang = 2 * math.pi * b["age"][..., None] * self.omega
        x = self.num(torch.cat([b["num"], b["sc"], ang.sin(), ang.cos()], -1)) + self.kind(kind) + self.rank(b["rank"])
        for k, e in enumerate(self.cat):
            x = x + e(b["cat"][..., k])
        N, d = x.shape[0], x.shape[-1]
        x = torch.cat([self.sum.expand(N, 1, d), self.edrop(x)], 1)
        pad = torch.cat([torch.zeros(N, 1, dtype=torch.bool, device=x.device), kind == 0], 1)
        h = self.proj(torch.cat([self.enc(x, src_key_padding_mask=pad)[:, 0], b["ntok"][:, None]], -1))
        B, C = b["fxcat"].shape[0], int(b["slot"].max()) + 1
        g = self.month(b["month"]) + self.fnum(b["fxsc"])
        for k in range(5):
            g = g + self.cat[k](b["fxcat"][:, k])
        z = torch.cat([g[:, None], x.new_zeros(B, C, d)], 1).index_put((b["fx"], b["slot"] + 1), h + self.side(b["side"]))
        zpad = torch.ones(B, C + 1, dtype=torch.bool, device=x.device)
        zpad[:, 0] = False; zpad[b["fx"], b["slot"] + 1] = False
        out = self.dec(self.edrop(z), src_key_padding_mask=zpad)[b["fx"], b["slot"] + 1]
        return self.head(out).squeeze(-1) * self.ysd + self.ybar


@torch.no_grad()
def predict(model, ds, deltas, dev, workers=0, bs=64):
    """Ŷ for every (fixture, candidate) at each Δ, sides as in the fixture (no swap). Returns a DataFrame."""
    model.eval()
    out = []
    for d in deltas:
        ds.plan(np.full(len(ds), d), np.zeros(len(ds), np.int64))
        batches = [list(range(i, min(i + bs, len(ds)))) for i in range(0, len(ds), bs)]
        yh = np.concatenate([model(to(b, dev)).float().cpu().numpy() for b in loader(ds, batches, workers)])
        f = ds.fx
        out.append(pd.DataFrame({"fx_id": np.concatenate([[x["id"]] * len(x["y"]) for x in f]), "delta": d,
                                 "team": np.concatenate([x["team"] for x in f]), "player_id": np.concatenate([x["player_id"] for x in f]),
                                 "y": np.concatenate([x["y"] for x in f]), "yhat": yh}))
    return pd.concat(out, ignore_index=True)


def build_model(store, L, fx_train):
    y = np.concatenate([f["y"] for f in fx_train])
    return CB([len(store.vocab[c]) + 1 for c in CATS], L, y.mean(), y.std() + 1e-6, **HP)
