"""Baseline (CB) gates (docs/BASELINE.md §9): T1 causality + time travel, T1c absence-token rule, T4 permutation equivariance, T7 cold start.
Run: python tests/test_baseline.py   (writes data/audit/cb_tests_audit.json; T3 = `baseline/pipeline.py overfit`, T5 = two evaluate.py runs)
"""
import json, os, sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from baseline.config import HP  # noqa: E402
from baseline.ingestion import Store, days, load_tables  # noqa: E402
from baseline.preprocessing import Fixtures, collate, fixture_rows, make_fixtures  # noqa: E402
from baseline.scoring_model import build_model  # noqa: E402
from shared.preprocessing.causal import SPLITS, draw_delta  # noqa: E402

D = SPLITS["F24"]


def setup():
    P, M, S = load_tables()
    P, M = P[P.date < D[2]], M[M.date < D[2]]
    st = Store(P, M, fold_end=D[1])
    fx = make_fixtures(st, fixture_rows(P, S, D[0], D[1], pad=True), M)
    return P, M, st, fx


def tok(st, pid, team, c, L=64):
    kind, num, rows = st.tokens(st.pid_code.get(pid, -1), st.team_code.get(team, -1), c, L)
    return kind, num, st.T_cat[rows], st.T_day[rows], st.T_team[rows], rows


def test_t1_t1c(P, M, st, fx, n_fx=10, seed=0):
    """Every token < c; tokens rebuilt from history truncated at c are identical; absence tokens obey the rule."""
    rng = np.random.default_rng(seed)
    xi = set(zip(P.match_id, P.team, P.player_id))
    first = P.groupby(["player_id", "team"]).date.min().to_dict()
    checked = 0
    for k in rng.choice(len(fx), n_fx, replace=False):
        f = fx[k]; delta = draw_delta(f"{f['id']}|0"); c = f["day"] - delta
        cd = str(np.datetime64(c, "D"))
        tr = Store(P[P.date < cd], M[M.date < cd], vocab=st.vocab, norm=st.norm)
        for pid, team in zip(f["player_id"], f["team"]):
            a, b = tok(st, pid, team, c), tok(tr, pid, team, c)
            assert (a[3] < c).all(), "T1: token dated >= c"
            for x, y in zip(a[:4], b[:4]):
                assert np.array_equal(x, y), f"T1: truncated rebuild differs for {pid} in {f['id']}"
            ab = a[0] == 2
            if ab.any():  # T1c
                assert (a[4][ab] == st.team_code[team]).all(), "T1c: absence from another team"
                lo = max(days([first[(pid, team)]])[0], c - HP["absence_days"])
                assert ((a[3][ab] >= lo) & (a[3][ab] < c)).all(), "T1c: absence outside [max(first app, c-365), c)"
                assert not any((m, team, pid) in xi for m in st.T_mid[a[5][ab]]), "T1c: absence token for a match he played"
            checked += 1
    return checked


def batch(st, fx, idx, L=64):
    ds = Fixtures(st, [fx[i] for i in idx], L)
    ds.plan([90] * len(idx), [0] * len(idx))
    return [ds[i] for i in range(len(idx))]


def test_t4(st, fx):
    """Relabelling the candidates permutes the outputs identically (Proposition CB-2)."""
    torch.manual_seed(0)
    items = batch(st, fx, [0, 1, 2])
    m = build_model(st, 64, fx[:50]).eval()
    out = m(collate(items)).detach().numpy()
    n0 = len(items[0]["y"]); perm = np.random.default_rng(0).permutation(n0)
    it = dict(items[0])
    for k in ("kind", "num", "cat", "sc", "age", "rank", "ntok", "y", "side"):
        it[k] = it[k][perm]
    out2 = m(collate([it] + items[1:])).detach().numpy()
    err = float(np.abs(out2[:n0] - out[:n0][perm]).max())
    assert err < 1e-4 * float(m.ysd), err
    assert np.allclose(out2[n0:], out[n0:], atol=1e-4 * float(m.ysd))
    return err


def test_t7(st, fx):
    """A candidate with no tokens (unknown player) runs and gets a finite output."""
    f = dict(fx[0]); f["pc"] = f["pc"].copy(); f["pc"][0] = -1
    ds = Fixtures(st, [f], 64); ds.plan([365], [0])
    it = ds[0]
    assert it["kind"][0].sum() == 0
    out = build_model(st, 64, fx[:50]).eval()(collate([it]))
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    P, M, st, fx = setup()
    res = {"t1_candidates_checked": test_t1_t1c(P, M, st, fx), "t4_max_err": test_t4(st, fx)}
    test_t7(st, fx); res["t7"] = "ok"; res["pass"] = True
    json.dump(res, open("data/audit/cb_tests_audit.json", "w"), indent=1)
    print(json.dumps(res))
