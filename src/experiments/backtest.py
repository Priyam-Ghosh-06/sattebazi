"""Walk-forward experiments on smoke match sets. Each experiment writes results/experiments/E<k>_<name>/ and one LOG line.

python src/backtest.py --exp E3 B0 E5 ... [--test-years 2024 2025] [--n-matches 400] [--train-start 2019-01-01]
Reference pipeline B0: μ = LightGBM L2 on played rows, p = logistic LightGBM on pseudo-squads + sum-to-11,
score = p·μ, XI = exact ILP. Every variant is compared to B0 on the same matches (paired bootstrap).
"""
import argparse, hashlib, json, os, random, sys, time
from datetime import date

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

random.seed(0); np.random.seed(0)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from graph import scoring_model as T  # noqa: E402
from shared.optimization.selection import select_xi  # noqa: E402

EXP_DIR = "results/experiments"
REF = ["B0_B0_L2_logit_sum11"]
GROUPS = {
    "form": ["n_apps", "last5_total", "car_total", "car_bat", "car_bowl", "car_field", "sd20_total", "big20", "r25_20",
             "w2_20", "runs20", "wk20", "ct20", "days_since"] + [f"ewm{h}_{c}" for h in (3, 10, 30) for c in ("total", "pts_bat", "pts_bowl", "pts_field")],
    "opportunity": ["pos10", "bf10", "lb10", "bowl12_10", "role_bshare_w", "role_pos_w", "role_catch_w", "role_stump_w", "role_c"],
    "appearance": ["team_apps", "team_matches_before", "team_days_since", "team_gap_matches", "share3", "share5", "share10"],
    "squad": ["squad_n", "rank_share10", "rank_ewm10", "role_rank_share10", "role_n_in_squad"],
    "context": ["gender_c", "team_type_c", "match_type_c", "sched_overs", "balls_per_over"],
}


# ---------------------------------------------------------------- data / folds
def load(args):
    X = pd.read_parquet(args.features)
    X = T.prep(X[X.date >= args.train_start])
    return X.reset_index(drop=True)


def folds(X, args):
    rng = np.random.default_rng(args.seed)
    for y in args.test_years:
        tr = X[X.date < f"{y}-01-01"]
        te = X[(X.date >= f"{y}-01-01") & (X.date <= f"{y}-12-31") & X.in_base]
        ms = np.sort(te.match_id.unique())
        pick = rng.choice(ms, min(args.n_matches, len(ms)), replace=False)
        te = te[te.match_id.isin(pick)].reset_index(drop=True)
        yield y, tr, te


# ---------------------------------------------------------------- evaluation
def spearman_within(df, col):
    vals = []
    for _, g in df.groupby("match_id", sort=False):
        if g[col].nunique() > 1 and g.y.nunique() > 1:
            vals.append(spearmanr(g[col], g.y).statistic)
    return float(np.mean(vals))


def evaluate(te, pred, oracle):
    d = te[["match_id", "team", "player_id", "role", "y", "played"]].copy()
    d["score"] = pred["score"]
    out = {"mae_squad": float((d.score - d.y).abs().mean()), "rmse_squad": float(np.sqrt(((d.score - d.y) ** 2).mean())),
           "spearman_squad": spearman_within(d, "score")}
    if "mu" in pred:
        d["mu"] = pred["mu"]
        pl = d[d.played == 1]
        out.update(mae_played=float((pl.mu - pl.y).abs().mean()), rmse_played=float(np.sqrt(((pl.mu - pl.y) ** 2).mean())),
                   spearman_played=spearman_within(pl, "mu"))
    if "p" in pred:
        p = np.clip(pred["p"], 1e-6, 1 - 1e-6)
        out.update(logloss_p=float(-(d.played * np.log(p) + (1 - d.played) * np.log(1 - p)).mean()),
                   brier_p=float(((p - d.played) ** 2).mean()))
    rows, fallback = [], 0
    for mid, g in d.groupby("match_id", sort=False):
        idx, ok = select_xi(g.score.to_numpy(), g.role.to_numpy(), g.team.to_numpy(), g.player_id.to_numpy())
        fallback += not ok
        r = {"match_id": mid, "captured": g.y.to_numpy()[idx].sum(), "oracle": oracle[mid],
             "random": 11 * g.y.mean()}
        if "mu" in pred:  # hindsight XI + model μ: removes the appearance error
            h = g[g.played == 1]
            hi, _ = select_xi(h.mu.to_numpy(), h.role.to_numpy(), h.team.to_numpy(), h.player_id.to_numpy())
            r["hindsight_captured"] = h.y.to_numpy()[hi].sum()
            oi, _ = select_xi(h.y.to_numpy(), h.role.to_numpy(), h.team.to_numpy(), h.player_id.to_numpy())
            r["hindsight_oracle"] = h.y.to_numpy()[oi].sum()
        rows.append(r)
    pm = pd.DataFrame(rows)
    out.update(captured=float(pm.captured.mean()), oracle=float(pm.oracle.mean()),
               captured_over_oracle=float((pm.captured / pm.oracle).mean()), captured_minus_random=float((pm.captured - pm.random).mean()),
               selection_fallbacks=fallback)
    if "hindsight_captured" in pm:
        out.update(hindsight_over_oracle=float((pm.hindsight_captured / pm.hindsight_oracle).mean()))
    return out, pm


def oracle_of(te):
    return {mid: g.y.to_numpy()[select_xi(g.y.to_numpy(), g.role.to_numpy(), g.team.to_numpy(), g.player_id.to_numpy())[0]].sum()
            for mid, g in te.groupby("match_id", sort=False)}


def paired(pm_a, pm_b, col="captured", n=2000):
    m = pm_a[["match_id", col]].merge(pm_b[["match_id", col]], on="match_id", suffixes=("_a", "_b"))
    diff = (m[col + "_a"] - m[col + "_b"]).to_numpy()
    rng = np.random.default_rng(0)
    bs = rng.choice(diff, (n, len(diff))).mean(1)
    return float(diff.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


# ---------------------------------------------------------------- variants
def cols_all(X):
    return T.cols_of(X)


def b0(tr, te, mu_obj="regression", mu_params=None, cols=None, weight=None, p_train=None, p_cols=None):
    cols = cols or cols_all(tr)
    pl = tr[tr.played == 1]
    mu = T.fit(pl, "y", cols, mu_obj, mu_params, weight=None if weight is None else weight.loc[pl.index])
    pt = p_train if p_train is not None else tr[tr.in_base]
    pm = T.fit(pt, "played", p_cols or cols, "binary", weight=None if weight is None else weight.reindex(pt.index))
    m = mu.predict(te[cols], num_iteration=mu.best_iteration)
    logit = pm.predict(te[p_cols or cols], num_iteration=pm.best_iteration, raw_score=True)
    p = T.sum_to_n(logit, te.match_id + "|" + te.team)
    return {"score": p * m, "mu": m, "p": p, "_models": (mu, pm), "_logit": logit}


CACHE = {}


def base(tr, te, y):
    """B0 predictions per fold, cached (used as the reference and as stage-1 inputs)."""
    if y not in CACHE:
        CACHE[y] = b0(tr, te)
    return CACHE[y]


def lgb_rank(tr, cols, label, group_col="match_id", objective="lambdarank", params=None, fobj=None, valid_label=None):
    """Grouped (query = match) booster. Early stopping on NDCG@11 with integer relevance labels."""
    tr = tr.sort_values([group_col, "team"], kind="stable")
    a, v = T.split_valid(tr)
    def ds(d, lab, ref=None):
        return lgb.Dataset(d[cols], lab(d), group=d.groupby(group_col, sort=False).size().to_numpy(), reference=ref,
                           categorical_feature=[c for c in cols if c.endswith("_c")] if ref is None else "auto", free_raw_data=False)
    da = ds(a, label); dv = ds(v, valid_label or label, da)
    p = {**T.BASE_PARAMS, "objective": objective, "metric": "ndcg", "eval_at": [11], **(params or {})}
    if fobj:
        p["objective"] = fobj
    return lgb.train(p, da, 2000, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])


def rel_label(d):
    return np.clip(np.round(d.y / 10), 0, 30).astype(int)


def listnet_obj(temp=20.0):
    """Listwise softmax cross-entropy within each match: the selection objective hardcoded into boosting."""
    def f(preds, ds):
        y = ds.get_label(); grp = ds.get_group()
        g = np.empty_like(preds); h = np.empty_like(preds); s = 0
        for n in grp:
            sl = slice(s, s + n); s += n
            z = preds[sl] - preds[sl].max(); pz = np.exp(z) / np.exp(z).sum()
            t = np.exp((y[sl] - y[sl].max()) / temp); t /= t.sum()
            g[sl] = pz - t; h[sl] = np.maximum(pz * (1 - pz), 1e-6)
        return g, h
    return f


def topk_mask(v, gid, k=11):
    """1 for the top-k values of v inside each group (vectorised; groups are integer codes)."""
    order = np.lexsort((-v, gid))
    sg = gid[order]
    start = np.r_[0, np.flatnonzero(np.diff(sg)) + 1]
    first = np.repeat(start, np.diff(np.r_[start, len(sg)]))
    m = np.zeros(len(v)); m[order] = (np.arange(len(v)) - first) < k
    return m


def spo_plus_obj(gid, hess=0.2):
    """SPO+ (Elmachtoub & Grigas) for 'pick the top 11 by points' inside each match:
    grad = 2·(x*(2ŝ − y) − x*(y)). The selection decision itself is the training signal.
    Oracle relaxation: top-11 per match (role/team bounds ignored inside training)."""
    def f(preds, ds):
        y = ds.get_label()
        g = 2 * (topk_mask(2 * preds - y, gid) - topk_mask(y, gid))
        return g, np.full_like(preds, hess)
    return f


def captured11_feval(gid):
    def f(preds, ds):
        y = ds.get_label()
        cap = np.bincount(gid, weights=y * topk_mask(preds, gid)); orc = np.bincount(gid, weights=y * topk_mask(y, gid))
        ok = orc > 0
        return "captured11", float((cap[ok] / orc[ok]).mean()), True
    return f


def ew_weights(tr, halflife_days, ref_date):
    age = (pd.Timestamp(ref_date) - pd.to_datetime(tr.date)).dt.days
    return pd.Series(0.5 ** (age / halflife_days), index=tr.index)


def backfit(pl, te, target, cycles=3):
    """Orthogonal-experts via backfitting: one GBDT per feature group, each fit to the residual of the others."""
    groups = [g for g in GROUPS if g != "context"]
    ctx = GROUPS["context"]
    f_tr = {g: np.zeros(len(pl)) for g in groups}; f_te = {g: np.zeros(len(te)) for g in groups}
    mean = pl[target].mean()
    models = {g: [] for g in groups}
    for c in range(cycles):
        for g in groups:
            others = sum(f_tr[o] for o in groups if o != g)
            d = pl.assign(_r=pl[target] - mean - others)
            m = T.fit(d, "_r", GROUPS[g] + ctx, "regression", {"learning_rate": 0.1})
            new_tr = m.predict(pl[GROUPS[g] + ctx], num_iteration=m.best_iteration)
            new_te = m.predict(te[GROUPS[g] + ctx], num_iteration=m.best_iteration)
            f_tr[g] = new_tr - new_tr.mean(); f_te[g] = new_te - new_tr.mean()
    corr = np.corrcoef(np.vstack([f_tr[g] for g in groups]))
    var = {g: float(np.var(f_tr[g])) for g in groups}
    return mean + sum(f_te.values()), {"expert_corr": pd.DataFrame(corr, groups, groups).round(3).to_dict(), "expert_var": var}


def interaction_constrained(pl, te, cols):
    """Each tree may only use one feature group (+ context): an additive model whose groups are the experts."""
    idx = {c: i for i, c in enumerate(cols)}
    ctx = [idx[c] for c in GROUPS["context"]]
    ic = [[idx[c] for c in GROUPS[g] if c in idx] + ctx for g in GROUPS if g != "context"]
    m = T.fit(pl, "y", cols, "regression", {"interaction_constraints": ic})
    return m.predict(te[cols], num_iteration=m.best_iteration), m


def expert_corr_from_constrained(m, X, cols):
    """Per-group contributions from SHAP-style pred_contrib, and their correlation."""
    contrib = m.predict(X[cols], num_iteration=m.best_iteration, pred_contrib=True)[:, :-1]
    out = {}
    for g in GROUPS:
        ix = [cols.index(c) for c in GROUPS[g] if c in cols]
        out[g] = contrib[:, ix].sum(1)
    D = pd.DataFrame(out)
    return D.corr().round(3).to_dict(), D.var().round(2).to_dict()


def run_variant(name, fold_data, fn):
    """fn(tr, te, y) -> pred dict. Returns aggregated metrics and per-match frame."""
    mets, pms, extra = [], [], {}
    for y, tr, te, orc in fold_data:
        t0 = time.time()
        pred = fn(tr, te, y)
        m, pm = evaluate(te, pred, orc)
        m["fold"] = y; m["secs"] = round(time.time() - t0, 1)
        if "_extra" in pred:
            extra[str(y)] = pred["_extra"]
        mets.append(m); pms.append(pm.assign(fold=y))
    agg = {k: float(np.mean([m[k] for m in mets])) for k in mets[0] if k != "fold"}
    return {"per_fold": mets, "mean": agg, "extra": extra}, pd.concat(pms)


# ---------------------------------------------------------------- experiment registry
def experiments(X):
    """Returns {exp_id: {variant_name: fn}}. Variant fns: (tr, te, y) -> pred."""
    C = cols_all(X)
    missing = set(C) ^ set(sum(GROUPS.values(), []))
    assert not missing, f"feature/group mismatch: {missing}"
    E = {}
    E["B0"] = {"B0_L2_logit_sum11": lambda tr, te, y: base(tr, te, y)}

    def baseline(stat):
        def f(tr, te, y):
            pl = tr[tr.played == 1]
            rm = pl.groupby("role").y.mean()
            prior = te.role.map(rm).fillna(pl.y.mean()).to_numpy()
            if stat == "role_mean":
                mu = prior
            elif stat == "eb_career":
                n = te.n_apps.fillna(0).to_numpy(); k = 10
                mu = (n * te.car_total.fillna(0).to_numpy() + k * prior) / (n + k)
            else:
                mu = te[stat].fillna(pd.Series(prior, index=te.index)).to_numpy()
            p = te.share10.fillna(0).to_numpy()
            return {"score": mu * p, "mu": mu}
        return f
    E["E3"] = {f"E3_{s}_x_share10": baseline(s) for s in ["role_mean", "eb_career", "last5_total", "ewm10_total", "ewm3_total", "ewm30_total"]}

    E["E5"] = {
        "E5_mu_L1": lambda tr, te, y: b0(tr, te, "regression_l1"),
        "E5_mu_huber10": lambda tr, te, y: b0(tr, te, "huber", {"alpha": 10.0}),
        "E5_mu_huber25": lambda tr, te, y: b0(tr, te, "huber", {"alpha": 25.0}),
        "E5_direct_squad_L2": lambda tr, te, y: _direct(tr, te, C, "regression"),
        "E5_direct_squad_L1": lambda tr, te, y: _direct(tr, te, C, "regression_l1"),
    }

    def p_variant(kind):
        def f(tr, te, y):
            b = base(tr, te, y)
            if kind == "proxy_share10":
                p = te.share10.fillna(0).to_numpy()
            elif kind == "logit_uncalibrated":
                p = 1 / (1 + np.exp(-b["_logit"]))
            elif kind == "ext_squad_train":
                return b0(tr, te, p_train=tr)
            elif kind == "no_appearance_feats":
                pc = [c for c in C if c not in GROUPS["appearance"] + GROUPS["squad"]]
                return b0(tr, te, p_cols=pc)
            return {"score": p * b["mu"], "mu": b["mu"], "p": p}
        return f
    E["E6"] = {f"E6_p_{k}": p_variant(k) for k in ["proxy_share10", "logit_uncalibrated", "ext_squad_train", "no_appearance_feats"]}

    def components(tr, te, y):
        b = base(tr, te, y); pl = tr[tr.played == 1]
        mu = 3.0
        for t in ["pts_bat", "pts_bowl", "pts_field"]:
            m = T.fit(pl, t, C, "regression")
            mu = mu + m.predict(te[C], num_iteration=m.best_iteration)
        return {"score": b["p"] * mu, "mu": mu, "p": b["p"]}
    E["E7"] = {"E7_components_sum": components}

    def pooled_split(key):
        def f(tr, te, y):
            out = {k: np.zeros(len(te)) for k in ("score", "mu", "p")}
            for v in te[key].unique():
                r = b0(tr[tr[key] == v], te[te[key] == v])
                msk = (te[key] == v).to_numpy()
                for k in out:
                    out[k][msk] = r[k]
            return out
        return f
    E["E8"] = {"E8_split_by_gender": pooled_split("gender"), "E8_split_by_team_type": pooled_split("team_type")}

    E["E9"] = {f"E9_drop_{g}": (lambda g: lambda tr, te, y: b0(tr, te, cols=[c for c in C if c not in GROUPS[g]]))(g) for g in GROUPS}

    # --- reranker family (user idea: GBDT reranker instead of an LLM reranker)
    def rank_direct(tr, te, y):
        m = lgb_rank(tr[tr.in_base], C, rel_label, params={"label_gain": list(range(31))})
        s = m.predict(te.sort_index()[C], num_iteration=m.best_iteration)
        return {"score": s}

    def rerank_stage2(tr, te, y, k=18):
        # stage 1 trained on years < y-1 gives honest out-of-sample scores for year y-1 (stage-2 training data)
        cut = f"{y - 1}-01-01"
        tr1, tr2 = tr[tr.date < cut], tr[(tr.date >= cut) & tr.in_base]
        s1 = b0(tr1, tr2)
        tr2 = tr2.assign(s1_mu=s1["mu"], s1_p=s1["p"], s1_score=s1["score"])
        s1te = b0(tr1, te)
        te2 = te.assign(s1_mu=s1te["mu"], s1_p=s1te["p"], s1_score=s1te["score"])
        cols2 = C + ["s1_mu", "s1_p", "s1_score"]
        tr2["_r"] = tr2.groupby("match_id").s1_score.rank(ascending=False)
        top = tr2[tr2._r <= k]
        m = lgb_rank(top, cols2, rel_label, params={"label_gain": list(range(31)), "learning_rate": 0.03})
        te2["_r"] = te2.groupby("match_id").s1_score.rank(ascending=False)
        s2 = m.predict(te2[cols2], num_iteration=m.best_iteration)
        # keep stage-1 order outside the top-k; rerank inside it
        score = np.where(te2._r <= k, 1e3 + s2, te2.s1_score.to_numpy() / 1e3)
        return {"score": score}

    def listnet(tr, te, y):
        d = tr[tr.in_base].sort_values(["match_id", "team"], kind="stable")
        a, v = T.split_valid(d)
        mk = lambda z, ref=None: lgb.Dataset(z[C], z.y.to_numpy(float), group=z.groupby("match_id", sort=False).size().to_numpy(),
                                             reference=ref, free_raw_data=False)
        da = mk(a); dv = mk(v, da)
        gv = pd.factorize(v.match_id)[0]
        prm = {**T.BASE_PARAMS, "objective": listnet_obj(20.0), "metric": "None"}
        m = lgb.train(prm, da, 2000, valid_sets=[dv], feval=captured11_feval(gv), callbacks=[lgb.early_stopping(50, verbose=False)])
        return {"score": m.predict(te[C], num_iteration=m.best_iteration)}

    E["E10"] = {"E10_lambdarank_direct": rank_direct, "E10_rerank_stage2_top18": rerank_stage2}

    # --- orthogonal experts (user idea #5)
    def ic(tr, te, y):
        b = base(tr, te, y); pl = tr[tr.played == 1]
        mu, m = interaction_constrained(pl, te, C)
        te_pl = te[te.played == 1]
        c_ic, v_ic = expert_corr_from_constrained(m, te_pl, C)
        c_b0, v_b0 = expert_corr_from_constrained(b["_models"][0], te_pl, C)
        return {"score": b["p"] * mu, "mu": mu, "p": b["p"],
                "_extra": {"group_contrib_corr_constrained": c_ic, "group_contrib_var_constrained": v_ic,
                           "group_contrib_corr_unconstrained_B0": c_b0, "group_contrib_var_unconstrained_B0": v_b0}}

    def bf(tr, te, y):
        b = base(tr, te, y); pl = tr[tr.played == 1]
        mu, info = backfit(pl, te, "y")
        return {"score": b["p"] * mu, "mu": mu, "p": b["p"], "_extra": info}

    def role_gated(tr, te, y):
        b = base(tr, te, y)
        mu = np.zeros(len(te))
        for r in ["WK", "BAT", "AR", "BOWL", "UNK"]:
            msk = (te.role == r).to_numpy()
            if not msk.any():
                continue
            pl = tr[(tr.played == 1) & (tr.role == r)]
            m = T.fit(pl, "y", C, "regression")
            mu[msk] = m.predict(te.loc[msk, C], num_iteration=m.best_iteration)
        return {"score": b["p"] * mu, "mu": mu, "p": b["p"]}
    E["E12"] = {"E12_interaction_constrained_groups": ic, "E12_backfit_orthogonal_experts": bf, "E12_role_gated_experts": role_gated}

    # --- structure hardcoded into the training algorithm (user idea)
    def recency(hl):
        def f(tr, te, y):
            return b0(tr, te, weight=ew_weights(tr, hl, f"{y}-01-01"))
        return f

    def monotone(tr, te, y):
        inc = set(["last5_total", "car_total", "share3", "share5", "share10"] + [f"ewm{h}_total" for h in (3, 10, 30)])
        mc = [1 if c in inc else 0 for c in C]
        return b0(tr, te, mu_params={"monotone_constraints": mc, "monotone_constraints_method": "advanced"})

    def boundary_weight(tr, te, y):
        # weight μ-training rows by closeness to the XI cut (rank 11 of a causal proxy score within the match)
        proxy = tr.share10.fillna(0) * tr.ewm10_total.fillna(tr.y.mean())
        r = proxy.groupby(tr.match_id).rank(ascending=False)
        w = 1 + 2 * np.exp(-np.abs(r - 11) / 3)
        return b0(tr, te, weight=w)

    E["E13"] = {"E13_recency_hl1y": recency(365), "E13_recency_hl2y": recency(730), "E13_recency_hl4y": recency(1460),
                "E13_monotone_form_share": monotone, "E13_selection_boundary_weights": boundary_weight,
                "E13_listnet_custom_objective": listnet}

    # --- memory length (EWMA half-life subsets)
    def memory(keep):
        drop = [f"ewm{h}_{c}" for h in (3, 10, 30) if h not in keep for c in ("total", "pts_bat", "pts_bowl", "pts_field")]
        return lambda tr, te, y: b0(tr, te, cols=[c for c in C if c not in drop])
    E["E4"] = {"E4_ewm_hl3_only": memory([3]), "E4_ewm_hl10_only": memory([10]), "E4_ewm_hl30_only": memory([30]), "E4_no_ewm": memory([])}

    def clean(tr, te, y):
        return b0(tr[tr.n_violations == 0] if "n_violations" in tr else tr, te)
    E["E11"] = {"E11_drop_violating_matches_train": clean}

    # --- decision-focused learning: SPO+ fine-tuning of a cross-fitted stage 1 (selection regret as the loss)
    def stage1(tr, te, y):
        key = ("s1", y)
        if key not in CACHE:
            cut = f"{y - 1}-01-01"
            tr1, tr2 = tr[tr.date < cut], tr[(tr.date >= cut) & tr.in_base].sort_values(["match_id", "team"], kind="stable")
            CACHE[key] = (tr2, b0(tr1, tr2)["score"], b0(tr1, te))
        return CACHE[key]

    def s1_only(tr, te, y):
        return stage1(tr, te, y)[2]

    def spo(tr, te, y, hess=0.2, lr=0.05):
        tr2, s_tr2, s_te = stage1(tr, te, y)
        a_mask = (tr2.date < np.sort(tr2.date.unique())[int(tr2.date.nunique() * 0.8)]).to_numpy()
        gid = pd.factorize(tr2.match_id)[0]
        ga, gv = pd.factorize(tr2.match_id[a_mask])[0], pd.factorize(tr2.match_id[~a_mask])[0]
        da = lgb.Dataset(tr2.loc[a_mask, C], tr2.y[a_mask], init_score=s_tr2[a_mask], free_raw_data=False)
        dv = lgb.Dataset(tr2.loc[~a_mask, C], tr2.y[~a_mask], init_score=s_tr2[~a_mask], reference=da)
        prm = {**T.BASE_PARAMS, "objective": spo_plus_obj(ga, hess), "learning_rate": lr, "metric": "None", "min_data_in_leaf": 100}
        m = lgb.train(prm, da, 1000, valid_sets=[dv], feval=captured11_feval(gv), callbacks=[lgb.early_stopping(50, verbose=False)])
        score = s_te["score"] + m.predict(te[C], num_iteration=m.best_iteration)
        return {"score": score, "_extra": {"spo_best_iter": m.best_iteration}}

    E["E15"] = {"E15_stage1_crossfit_control": s1_only, "E15_spo_plus_finetune": spo,
                "E15_spo_plus_finetune_h1": lambda tr, te, y: spo(tr, te, y, hess=1.0)}

    # --- opponent / venue / own-squad context (user point 2: opponents and match setting)
    XC = [c for c in X.columns if c.startswith("x_")]
    opp_c = [c for c in XC if c.startswith(("x_opp_", "x_own_"))]
    ven_c = [c for c in XC if c.startswith("x_venue")]
    E["E14"] = {"E14_plus_opp_own_squad": lambda tr, te, y: b0(tr, te, cols=C + opp_c),
                "E14_plus_venue": lambda tr, te, y: b0(tr, te, cols=C + ven_c),
                "E14_plus_opp_venue": lambda tr, te, y: b0(tr, te, cols=C + opp_c + ven_c)}
    return E


def _direct(tr, te, C, obj):
    m = T.fit(tr[tr.in_base], "y", C, obj)
    s = m.predict(te[C], num_iteration=m.best_iteration)
    return {"score": s}


# ---------------------------------------------------------------- logging
def log(eid, name, cfg, res, ref_pm, pm):
    d = os.path.join(EXP_DIR, f"{eid}_{name}")
    os.makedirs(d, exist_ok=True)
    verdict, cmp = "reference", {}
    if ref_pm is not None and f"{eid}_{name}" != REF[0]:
        for col in ["captured"]:
            cmp[col] = paired(pm, ref_pm, col)
        lo, hi = cmp["captured"][1], cmp["captured"][2]
        verdict = "better than B0" if lo > 0 else "worse than B0" if hi < 0 else "no significant difference vs B0"
    res["vs_B0_paired_captured_diff_mean_ci95"] = cmp.get("captured")
    res["verdict"] = verdict
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(d, "metrics.json"), "w"), indent=1, default=str)
    pm.to_csv(os.path.join(d, "per_match.csv"), index=False)
    h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    m = res["mean"]
    keys = ["captured_over_oracle", "captured", "spearman_squad", "mae_squad", "mae_played", "spearman_played", "rmse_played", "hindsight_over_oracle", "logloss_p"]
    kv = " ".join(f"{k}={m[k]:.4f}" for k in keys if k in m)
    ci = cmp.get("captured")
    ci_s = f" dCaptured={ci[0]:+.2f} [{ci[1]:+.2f},{ci[2]:+.2f}]" if ci else ""
    with open(os.path.join(EXP_DIR, "LOG.txt"), "a", encoding="utf-8") as f:
        f.write(f"| {date.today()} | {eid} {name} | {h} | {kv}{ci_s} | {verdict} |\n")
    print(f"[{eid}] {name}: {kv}{ci_s} -> {verdict}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.parquet")
    ap.add_argument("--exp", nargs="+", default=["B0", "E3"])
    ap.add_argument("--only", nargs="*", default=None, help="variant names to run within the chosen experiments")
    ap.add_argument("--test-years", nargs="+", type=int, default=[2024, 2025])
    ap.add_argument("--n-matches", type=int, default=400)
    ap.add_argument("--train-start", default="2019-01-01")
    ap.add_argument("--seed", type=int, default=0, help="seed of the smoke test-match sample")
    ap.add_argument("--ref", default="B0_B0_L2_logit_sum11", help="experiment dir used as the paired reference")
    ap.add_argument("--tag", default="", help="suffix for variant names (e.g. when changing --train-start)")
    args = ap.parse_args()
    os.makedirs(EXP_DIR, exist_ok=True)
    if not os.path.exists(os.path.join(EXP_DIR, "LOG.txt")):
        open(os.path.join(EXP_DIR, "LOG.txt"), "w").write("| date | experiment | config hash | key metrics (mean over folds) | verdict |\n|---|---|---|---|---|\n")
    X = load(args)
    M = pd.read_parquet("data/interim/matches.parquet", columns=["match_id", "n_violations"])
    X = X.merge(M, on="match_id", how="left")
    if "E14" in args.exp:
        from shared.preprocessing import context
        P = pd.read_parquet("data/processed/player_match.parquet", columns=["match_id", "total", "pts_bat", "pts_bowl"])
        Mv = pd.read_parquet("data/interim/matches.parquet", columns=["match_id", "date", "venue"])
        X = context.add_match_context(X, P, Mv)
    fold_data = [(y, tr, te, oracle_of(te)) for y, tr, te in folds(X, args)]
    for y, tr, te, _ in fold_data:
        print(f"fold {y}: train rows {len(tr)} (played {int(tr.played.sum())}), test matches {te.match_id.nunique()} rows {len(te)}", flush=True)
    E = experiments(X)
    REF[0] = args.ref
    ref_path = os.path.join(EXP_DIR, args.ref, "per_match.csv")
    for eid in args.exp:
        for name, fn in E[eid].items():
            if args.only and name not in args.only:
                continue
            name = name + args.tag
            cfg = {"exp": eid, "variant": name, "test_years": args.test_years, "n_matches": args.n_matches, "seed": args.seed,
                   "train_start": args.train_start, "lgb": T.BASE_PARAMS}
            t0 = time.time()
            res, pm = run_variant(name, fold_data, fn)
            res["secs_total"] = round(time.time() - t0, 1)
            ref_pm = pd.read_csv(ref_path, dtype={"match_id": str}) if os.path.exists(ref_path) else None
            pm["match_id"] = pm.match_id.astype(str)
            log(eid, name, cfg, res, ref_pm, pm)


if __name__ == "__main__":
    main()
