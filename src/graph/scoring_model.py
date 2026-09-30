"""Graph approach, scoring model (3a = B0s, the submitted model): LightGBM p = P(plays | squad) (logistic,
then the sum-to-XI shift τ) and μ = E[points | plays] (L2), on staleness-aware features; score = p·μ."""

import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from shared.preprocessing import features as F  # noqa: E402


CONTEXT_PREFIX = ("x_opp_", "x_own_", "x_venue")  # E14: opponent/venue context, replicated +2.6 pts/match


BASE_PARAMS = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8, lambda_l2=1.0,
                   seed=0, deterministic=True, force_row_wise=True, num_threads=4, verbose=-1)


CATS = {"role": ["WK", "BAT", "AR", "BOWL", "UNK"], "gender": ["male", "female"],
        "team_type": ["club", "international"], "match_type": ["T20", "IT20"]}


def prep(X):
    """Numeric model matrix: feature_cols + integer-coded categoricals."""
    X = X.copy()
    for c, levels in CATS.items():
        X[c + "_c"] = pd.Categorical(X[c], categories=levels).codes.astype("int8")
    return X


def cols_of(X):
    return [c for c in F.feature_cols(X) if not c.endswith("_c") and not c.startswith("x_")] + [c + "_c" for c in CATS]


def split_valid(D, frac=0.1):
    """Time-ordered split: last `frac` of dates is the early-stopping set."""
    cut = np.sort(D.date.unique())[int(len(D.date.unique()) * (1 - frac))]
    return D[D.date < cut], D[D.date >= cut]


def valid_start(train_end, months=3):
    """First day of the early-stopping window: 2025-12-31 -> 2025-10-01."""
    return (pd.Timestamp(train_end) + pd.Timedelta(days=1) - pd.DateOffset(months=months)).strftime("%Y-%m-%d")


def fit(D, target, cols, objective="regression", params=None, weight=None, rounds=2000, feval_metric=None, valid_from=None, **kw):
    p = {**BASE_PARAMS, "objective": objective, **(params or {})}
    tr, va = split_valid(D) if valid_from is None else (D[D.date < valid_from], D[D.date >= valid_from])
    w_tr = None if weight is None else weight.loc[tr.index]
    w_va = None if weight is None else weight.loc[va.index]
    dtr = lgb.Dataset(tr[cols], tr[target], weight=w_tr, categorical_feature=[c for c in cols if c.endswith("_c")], free_raw_data=False, **kw)
    dva = lgb.Dataset(va[cols], va[target], weight=w_va, reference=dtr)
    if feval_metric:
        p["metric"] = feval_metric
    m = lgb.train(p, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m


def sum_to_n(logit, groups, n=11.0, iters=60):
    """Per-group shift τ so that Σ σ(logit − τ) = min(n, group size): minimum-KL correction to the XI size."""
    g = pd.factorize(groups)[0]
    size = np.bincount(g).astype(float)
    target = np.minimum(n, size - 1e-6)
    lo, hi = np.full(g.max() + 1, -30.0), np.full(g.max() + 1, 30.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        s = np.bincount(g, weights=1 / (1 + np.exp(-(logit - mid[g]))))
        big = s > target
        lo = np.where(big, mid, lo); hi = np.where(big, hi, mid)
    tau = (lo + hi) / 2
    return 1 / (1 + np.exp(-(logit - tau[g])))


def fit_pipeline(Xtr, mu_obj="regression", mu_params=None, use_context=True, valid_from=None, extra=()):
    """μ on played rows (target Φ total), p on base-squad rows (target played). extra: more x_ prefixes (E24: 'x_mu_')."""
    cols = cols_of(Xtr) + ([c for c in Xtr.columns if c.startswith(CONTEXT_PREFIX + tuple(extra))] if use_context else [])
    mu = fit(Xtr[Xtr.played == 1], "y", cols, mu_obj, mu_params, valid_from=valid_from)
    pm = fit(Xtr[Xtr.in_base], "played", cols, "binary", valid_from=valid_from)
    return {"mu": mu, "p": pm, "cols": cols}


def predict(model, X):
    Xp = prep(X) if "role_c" not in X else X
    mu = model["mu"].predict(Xp[model["cols"]], num_iteration=model["mu"].best_iteration)
    logit = model["p"].predict(Xp[model["cols"]], num_iteration=model["p"].best_iteration, raw_score=True)
    p = sum_to_n(logit, Xp.match_id.astype(str) + "|" + Xp.team)
    return mu, p
