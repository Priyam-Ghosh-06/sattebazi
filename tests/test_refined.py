"""Refined approach (ADV) gates: T1 causality/time travel of bench tokens and capacities, T2 EWMA
containment, T6 staleness moves confidence only, T11 privileged isolation, T14 exact DP marginals vs brute force,
T16 gauge, T4 permutation equivariance, T7 cold start. Run: python tests/test_refined.py (writes data/audit/adv_tests_audit.json)
"""
import itertools, json, os, sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from baseline.ingestion import load_tables as base_load_tables  # noqa: E402
from baseline.preprocessing import fixture_rows as base_fixture_rows, make_fixtures as base_make_fixtures  # noqa: E402
from refined.config import BOWL_MAX, KEEP_MAX, REF  # noqa: E402
from refined.decision_model import Choice, log_z  # noqa: E402
from refined.ingestion import Store  # noqa: E402
from refined.preprocessing import Fixtures, collate  # noqa: E402
from refined.scoring_model import build, wkv_scan  # noqa: E402
from shared.preprocessing.causal import SPLITS, draw_delta  # noqa: E402

D = SPLITS["F24"]


def setup():
    P, M, S = base_load_tables()
    P, M = P[P.date < D[2]], M[M.date < D[2]]
    st = Store(P, M, fold_end=D[1])
    fx = base_make_fixtures(st, base_fixture_rows(P, S, D[0], D[1], pad=True), M)
    return P, M, st, fx


def t1(P, M, st, fx, n_fx=8, seed=0):
    """Tokens and capacities dated < c; identical when rebuilt from history truncated at c; bench tokens obey the rule."""
    rng = np.random.default_rng(seed)
    xi = set(zip(P.match_id, P.team, P.player_id))
    n = 0
    for k in rng.choice(len(fx), n_fx, replace=False):
        f = fx[k]; c = f["day"] - draw_delta(f"{f['id']}|0"); cd = str(np.datetime64(c, "D"))
        tr = Store(P[P.date < cd], M[M.date < cd], vocab=st.vocab, norm=st.norm)
        for pid, team in zip(f["player_id"], f["team"]):
            a = st.tokens(st.pid_code.get(pid, -1), st.team_code.get(team, -1), c, 128)
            b = tr.tokens(tr.pid_code.get(pid, -1), tr.team_code.get(team, -1), c, 128)
            assert (st.T_day[a[2]] < c).all(), "T1: token >= c"
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(st.T_day[a[2]], tr.T_day[b[2]]) \
                and np.array_equal(st.T_cat[a[2]], tr.T_cat[b[2]]), "T1: truncated rebuild differs"
            assert st.capacity(st.pid_code.get(pid, -1), c) == tr.capacity(tr.pid_code.get(pid, -1), c), "T1: capacity differs"
            bench = a[2][a[0] == 2]
            assert not any((m, team, pid) in xi for m in st.T_mid[bench]), "bench token for a match he played"
            n += 1
    return n


def t2():
    """EWMA containment: with k constant, λ_day = 0, v = y the scan is pandas' EWMA with halflife ln2/λ_app."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=40)
    hl = np.array([3.0, 10.0, 30.0])
    num, den, m = wkv_scan(torch.zeros(1, 40, 3, dtype=torch.float64), torch.tensor(np.repeat(y[:, None], 3, 1)[None]),
                               torch.tensor(np.log(2) / hl), torch.zeros(3, dtype=torch.float64),
                               torch.arange(40, dtype=torch.float64)[None] * 7, torch.ones(1, 40, dtype=torch.bool))
    est = (num / den)[0].numpy()
    ref = np.stack([pd.Series(y).ewm(halflife=h).mean().to_numpy() for h in hl], 1)
    err = float(np.abs(est - ref).max())
    assert err < 1e-5, err
    return err


def t6_t11(st, fx):
    """T6: reading out at two cutoffs with no token between leaves N/D unchanged and lowers log D by λ_day·gap.
    T11: z(c) from the sequence truncated at the fixture date equals z(c) from the sequence truncated at c
    (players with < L tokens, so truncation does not differ): the gap tokens never reach the X1 path."""
    torch.manual_seed(0)
    model = build(st, fx[:50]).eval()
    f = [x for x in fx if x["date"] >= "2023-01-01"][0]
    delta = 200
    dc, dd = Fixtures(st, [f], 128), Fixtures(st, [f], 128, to_date=True)
    dc.plan([delta], [0]); dd.plan([delta], [0])
    bc, bd = collate([dc[0]]), collate([dd[0]])
    with torch.no_grad():
        stt, day, mask = model.enc.states(bc)
        last = torch.where(mask, day, torch.full_like(day, -1e9)).max(1).values
        c1 = last + 1; c2 = last + 101
        z1, cold, _ = model.enc.readout(stt, day, mask, c1); z2, _, _ = model.enc.readout(stt, day, mask, c2)
        d = z1.shape[1] // 2
        la = model.enc.tm[-1].rates()[1]
        warm = ~cold
        e6 = float((z1[warm, :d] - z2[warm, :d]).abs().max())
        e6b = float(((z1[warm, d:] - z2[warm, d:]) * 10 - la * 100).abs().max())
        assert e6 < 1e-5 and e6b < 1e-3, (e6, e6b)
        za, _, _ = model.enc(bc, bc["cut"][bc["fx"]])
        zb, _, _ = model.enc(bd, bd["cut"][bd["fx"]])
        short = (bd["kind"] > 0).sum(1) < 128
        e11 = float((za[short] - zb[short]).abs().max())
        assert e11 < 1e-4, e11
    return e6, e11


def t14_t16():
    """Exact DP marginals and log Z vs brute force over all C(14, 11) = 364 subsets, with random potentials; gauge."""
    rng = np.random.default_rng(0)
    n, nxi = 14, 11
    u = torch.tensor(rng.normal(size=(1, n)), dtype=torch.float64)
    keep = torch.tensor((rng.random((1, n)) < 0.2).astype(float)); bowl = torch.tensor((rng.random((1, n)) < 0.5).astype(float))
    phik = torch.tensor(rng.normal(size=KEEP_MAX + 1)); phib = torch.tensor(rng.normal(size=BOWL_MAX + 1))
    mask = torch.ones(1, n, dtype=torch.bool); nx = torch.tensor([nxi])
    uu = u.clone().requires_grad_(True)
    lz = log_z(uu, mask, keep, bowl, nx, phik, phib, 13)
    (pi,) = torch.autograd.grad(lz.sum(), uu)
    E, inc = [], np.zeros(n)
    for X in itertools.combinations(range(n), nxi):
        X = list(X)
        k = min(int(keep[0, X].sum()), KEEP_MAX); b = min(int(bowl[0, X].sum()), BOWL_MAX)
        E.append(float(u[0, X].sum() + phik[k] + phib[b]))
    E = np.array(E); lz_bf = np.logaddexp.reduce(E); w = np.exp(E - lz_bf)
    for w_, X in zip(w, itertools.combinations(range(n), nxi)):
        inc[list(X)] += w_
    e14 = max(abs(float(lz) - lz_bf), float(np.abs(pi[0].numpy() - inc).max()))
    assert e14 < 1e-6, e14
    ch = Choice()
    ch.pk.data.normal_(); ch.pb.data.normal_()
    pk, pb = ch.phis()
    assert all(float(pk[r]) == 0 for r in REF["keep"]) and all(float(pb[r]) == 0 for r in REF["bowl"]), "T16 gauge"
    # affine re-gauge φ_q(k) + a + b·k with u − b·c_q leaves π unchanged
    a_, b_ = 0.7, -0.3
    u2 = (u - b_ * keep).clone().requires_grad_(True)
    lz2 = log_z(u2, mask, keep, bowl, nx, phik + a_ + b_ * torch.arange(KEEP_MAX + 1), phib, 13)
    (pi2,) = torch.autograd.grad(lz2.sum(), u2)
    # saturation (k counted up to 3) keeps the re-gauge exact only when no XI has > 3 keepers
    if int(keep.sum()) <= KEEP_MAX:
        assert float((pi2 - pi).abs().max()) < 1e-8, "T16 re-gauge"
    return e14


def t4_t7(st, fx):
    torch.manual_seed(0)
    model = build(st, fx[:50]).eval()
    ds = Fixtures(st, [fx[i] for i in (0, 1, 2)], 128, true_xi=False)
    ds.plan([90] * 3, [0] * 3)
    items = [ds[i] for i in range(3)]
    with torch.no_grad():
        out = model.predict(collate(items), "direct", "a3").numpy()
    n0 = len(items[0]["y"]); perm = np.random.default_rng(0).permutation(n0)
    it = dict(items[0])
    for k in ("kind", "num", "cat", "sc", "day", "caps", "nxi", "y", "played", "side"):
        it[k] = it[k][perm]
    with torch.no_grad():
        out2 = model.predict(collate([it] + items[1:]), "direct", "a3").numpy()
    e4 = float(np.abs(out2[:n0] - out[:n0][perm]).max())
    assert e4 < 1e-3, e4
    f = dict(fx[0]); f["pc"] = f["pc"].copy(); f["pc"][0] = -1
    d7 = Fixtures(st, [f], 128, true_xi=False); d7.plan([365], [0])
    with torch.no_grad():
        o7 = model.predict(collate([d7[0]]), "direct", "a3")
    assert torch.isfinite(o7).all()
    return e4


if __name__ == "__main__":
    res = {"t2_max_err": t2(), "t14_max_err": t14_t16()}
    P, M, st, fx = setup()
    res["t1_candidates"] = t1(P, M, st, fx)
    res["t6_err"], res["t11_err"] = t6_t11(st, fx)
    res["t4_max_err"] = t4_t7(st, fx); res["t7"] = "ok"
    res["pass"] = True
    json.dump(res, open("data/audit/adv_tests_audit.json", "w"), indent=1)
    print(json.dumps(res))
