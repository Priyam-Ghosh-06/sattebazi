"""Figures for the README and the three approach pages (all from files written by the runs; nothing is re-selected).

python src/shared/evaluation/figures.py [--refit-lgbm]
Validation curves: baseline (CB) and refined (ADV) per epoch from their checkpoints' done.json; the graph approach (LightGBM)
per boosting iteration, refitted here with the same data and settings and cached in results/experiments/lgbm_curves.json.
Common validation quantities for all three: captured/oracle (headline over the Δ grid) and the MSE of expected points.
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # src/
from graph import scoring_model as T  # noqa: E402
from shared.evaluation.metrics import _xi  # noqa: E402
from shared.preprocessing.causal import DELTAS, SPLITS  # noqa: E402

CK = "model_artifacts/checkpoints"
EXP = "results/experiments"
C = {"baseline": "#d1495b", "refined": "#2e86ab", "graph": "#3b8b5a", "graph+": "#8fbf6f", "B0s": "#3b8b5a"}
LAB = {"baseline": "1 Baseline (CB128)", "refined": "2 Refined (ADV)", "graph": "3a Graph base (submitted)",
       "graph+": "3b Graph + graph features"}
YEARS = ("F24", "F25")
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def done(run):
    p = f"{CK}/{run}/done.json"
    return json.load(open(p)) if os.path.exists(p) else None


def cb_ysd(run):
    import torch
    return float(torch.load(f"{CK}/{run}/best.pt", map_location="cpu", weights_only=False)["ysd"])


# ---------------------------------------------------------------- LightGBM curves (graph approach)
def _headline_mse(args):
    d, score = args
    rows = [_xi((g[score].to_numpy(float), g.y.to_numpy(float), g.role.to_numpy(), g.team.to_numpy(), g.player_id.to_numpy()))[0]
            for _, g in d.groupby(["fx_id", "delta"], sort=True)]
    orc = [_xi((g.y.to_numpy(float), g.y.to_numpy(float), g.role.to_numpy(), g.team.to_numpy(), g.player_id.to_numpy()))[0]
           for _, g in d.groupby(["fx_id", "delta"], sort=True)]
    keys = [k for k, _ in d.groupby(["fx_id", "delta"], sort=True)]
    r = pd.DataFrame(keys, columns=["fx_id", "delta"]).assign(ratio=np.array(rows) / np.maximum(np.array(orc), 1e-9))
    return float(r.groupby("delta").ratio.mean().mean()), float(((d[score] - d.y) ** 2).mean())


def lgbm_curves(step=10):
    X = pd.read_parquet("data/processed/stale_features.parquet")
    G = pd.read_parquet("data/processed/matchup_features.parquet").drop(columns=["c_date", "mu_snapshot"])
    X = T.prep(X.merge(G, on=["match_id", "team", "player_id"], how="left"))
    out = {}
    for arm, extra in (("graph", ()), ("graph+", ("x_mu_",))):
        for split in YEARS:
            tr_from, va_from, te_from, _ = SPLITS[split]
            D = X[(X.part == "train") & (X.date >= tr_from) & (X.date < te_from)]
            cols = T.cols_of(D) + [c for c in D.columns if c.startswith(T.CONTEXT_PREFIX + extra)]
            rec, models = {}, {}
            for part, target, obj, rows in (("mu", "y", "regression", D[D.played == 1]), ("p", "played", "binary", D[D.in_base])):
                tr, va = rows[rows.date < va_from], rows[rows.date >= va_from]
                dtr = lgb.Dataset(tr[cols], tr[target], categorical_feature=[c for c in cols if c.endswith("_c")], free_raw_data=False)
                dva = lgb.Dataset(va[cols], va[target], reference=dtr)
                ev = {}
                m = lgb.train({**T.BASE_PARAMS, "objective": obj}, dtr, 2000, valid_sets=[dtr, dva], valid_names=["train", "valid"],
                              callbacks=[lgb.early_stopping(50, verbose=False), lgb.record_evaluation(ev)])
                rec[part] = {"train": ev["train"]["l2" if obj == "regression" else "binary_logloss"],
                             "valid": ev["valid"]["l2" if obj == "regression" else "binary_logloss"], "best": m.best_iteration}
                models[part] = m
            g = X[(X.part != "train") & (X.date >= va_from) & (X.date < te_from)].reset_index(drop=True)
            n_max = max(models["mu"].num_trees(), models["p"].num_trees())
            its = sorted(set(list(range(step, n_max + 1, step)) + [models["mu"].best_iteration, models["p"].best_iteration]))
            jobs = []
            for k in its:
                mu = models["mu"].predict(g[cols], num_iteration=min(k, models["mu"].num_trees()))
                lo = models["p"].predict(g[cols], num_iteration=min(k, models["p"].num_trees()), raw_score=True)
                p = T.sum_to_n(lo, g.match_id + "|" + g.team)
                jobs.append((g[["fx_id", "delta", "team", "player_id", "y", "role"]].assign(s=mu * p), "s"))
            with Pool(4) as pool:
                res = pool.map(_headline_mse, jobs)
            rec["iters"] = its
            rec["valid_headline"] = [r[0] for r in res]
            rec["valid_mse"] = [r[1] for r in res]
            out[f"{arm}_{split}"] = rec
            print(arm, split, "best mu/p", rec["mu"]["best"], rec["p"]["best"], flush=True)
    json.dump(out, open(f"{EXP}/lgbm_curves.json", "w"), indent=1)
    return out


# ---------------------------------------------------------------- figures
def fig_validation(curves):
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5))
    for j, split in enumerate(YEARS):
        a_h, a_m = ax[0, j], ax[1, j]
        for seed, ls in ((0, "-"), (1, "--")):
            h = (done(f"CB128_{split}_s{seed}") or {}).get("history", [])
            if h:
                ep = [x["epoch"] + 1 for x in h]
                a_h.plot(ep, [x["valid_headline"] for x in h], ls, color=C["baseline"], label=LAB["baseline"] if seed == 0 else None)
                ysd = cb_ysd(f"CB128_{split}_s{seed}")
                a_m.plot(ep, [x["valid_loss"] * ysd ** 2 for x in h], ls, color=C["baseline"], label=LAB["baseline"] if seed == 0 else None)
            h = (done(f"ADV_{split}_s{seed}") or {}).get("history", [])
            p1 = [x for x in h if x["phase"] == "P1"]
            p3 = [x for x in h if x["phase"] == "P3"]
            if p1:
                ep = [x["epoch"] + 1 for x in p1]
                a_h.plot(ep + [len(p1) + x["epoch"] + 1 for x in p3], [x["valid_headline"] for x in p1 + p3], ls, color=C["refined"],
                         label=LAB["refined"] + " (P1 then P3)" if seed == 0 else None)
                a_m.plot(ep, [x["valid_mse"] for x in p1], ls, color=C["refined"], label=LAB["refined"] + " (P1)" if seed == 0 else None)
        for arm in ("graph", "graph+"):
            r = curves.get(f"{arm}_{split}")
            if r:
                x = np.array(r["iters"]) / max(r["iters"]) * 20  # boosting iterations rescaled onto the 20-epoch axis
                a_h.plot(x, r["valid_headline"], ":", color=C[arm], label=LAB[arm] + " (iterations, rescaled)")
                a_m.plot(x, r["valid_mse"], ":", color=C[arm], label=LAB[arm] + " (iterations, rescaled)")
        yr = "2023 Q4" if split == "F24" else "2024 Q4"
        a_h.set_title(f"{split}: validation captured/oracle ({yr}, mean over Δ grid)")
        a_m.set_title(f"{split}: validation MSE of expected points (points²)")
        a_m.set_xlabel("epoch (CB, ADV)  |  LightGBM iterations rescaled to 0-20")
    ax[0, 0].legend(fontsize=7, loc="lower right"); ax[1, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig("results/overall/validation_curves.png"); plt.close(fig)


ARMS = ("baseline", "refined", "graph", "graph+")
GROUPS = ("IPL", "other men's leagues", "men's internationals", "women's cricket")


def _pick(args):
    """Chosen XI and best XI (hindsight) for one fixture copy: (captured, oracle, overlap)."""
    score, y, role, team, pid = args
    from shared.optimization.selection import select_xi
    a, b = select_xi(score, role, team, pid)[0], select_xi(y, role, team, pid)[0]
    return float(y[a].sum()), float(y[b].sum()), len(set(a) & set(b))


def team_table():
    """Per (approach, fold, fixture, Δ) on the test years: captured, oracle, overlap with the best XI.
    Scores come from the stored predictions (baseline and refined: mean of the two seeds); 3b is refitted (its fold
    predictions were not stored). Cached in results/experiments/team_table.parquet."""
    cache = f"{EXP}/team_table.parquet"
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    key = ["fx_id", "delta", "team", "player_id"]
    X = None
    rows = []
    for split in YEARS:
        b = pd.read_parquet(f"{EXP}/E22_final/pred/B0s_{split}.parquet")
        b = b[b.subset == "test"]
        d = b[key + ["y", "role"]].assign(graph=b.score.to_numpy())
        for arm, pat in (("baseline", "E22_final/pred/CB128_{}_s{}"), ("refined", "E23_adv/pred/ADV_{}_s{}")):
            e = pd.concat([pd.read_parquet(f"{EXP}/{pat.format(split, s)}.parquet")[key + ["yhat"]] for s in (0, 1)])
            d = d.merge(e.groupby(key).yhat.mean().rename(arm).reset_index(), on=key, how="left")
        if X is None:
            G = pd.read_parquet("data/processed/matchup_features.parquet").drop(columns=["c_date", "mu_snapshot"])
            X = T.prep(pd.read_parquet("data/processed/stale_features.parquet").merge(G, on=["match_id", "team", "player_id"], how="left"))
        tr_from, va_from, te_from, te_to = SPLITS[split]
        m = T.fit_pipeline(X[(X.part == "train") & (X.date >= tr_from) & (X.date < te_from)], valid_from=va_from, extra=("x_mu_",))
        g = X[(X.part != "train") & (X.date >= te_from) & (X.date <= te_to)].reset_index(drop=True)
        mu, pp = T.predict(m, g)
        d = d.merge(g[key].assign(**{"graph+": mu * pp}), on=key, how="left")
        groups = list(d.groupby(["fx_id", "delta"], sort=True))
        with Pool(4) as pool:
            for arm in ARMS:
                res = pool.map(_pick, [(x[arm].to_numpy(float), x.y.to_numpy(float), x.role.to_numpy(), x.team.to_numpy(), x.player_id.to_numpy())
                                       for _, x in groups], chunksize=64)
                rows.append(pd.DataFrame([(arm, split, f, dl, *r) for ((f, dl), _), r in zip(groups, res)],
                                         columns=["arm", "split", "fx_id", "delta", "captured", "oracle", "overlap"]))
        print("team table", split, flush=True)
    t = pd.concat(rows, ignore_index=True)
    M = pd.read_parquet("data/interim/matches.parquet", columns=["match_id", "event", "gender", "team_type"]).set_index("match_id")
    ev, gen, tt = t.fx_id.map(M.event), t.fx_id.map(M.gender), t.fx_id.map(M.team_type)
    t["group"] = np.select([gen == "female", ev == "Indian Premier League", tt == "club"], [GROUPS[3], GROUPS[0], GROUPS[1]], GROUPS[2])
    t["pct"] = 100 * t.captured / t.oracle.where(t.oracle > 0)
    t.to_parquet(cache, index=False)
    return t


def fig_pct_best(t):
    """Per match: our XI's points as % of the best possible XI (the usual 'share of the dream team' view)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for j, split in enumerate(YEARS):
        f = t[t.split == split].groupby(["arm", "fx_id"]).pct.mean()  # each fixture averaged over the Δ grid
        data = [f.loc[a].dropna().to_numpy() for a in ARMS]
        bp = ax[j].boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops={"color": "k"})
        for patch, a in zip(bp["boxes"], ARMS):
            patch.set_facecolor(C[a]); patch.set_alpha(0.75)
        for i, v in enumerate(data, 1):
            ax[j].plot(i, v.mean(), "D", color="white", markeredgecolor="k", ms=5)
            ax[j].text(i, v.mean() + 1.5, f"{v.mean():.1f}%", ha="center", fontsize=8)
        ax[j].set_xticks(range(1, 5), ["1 Baseline", "2 Refined", "3a Graph base\n(submitted)", "3b + graph"], fontsize=8)
        ax[j].set_title(f"{2024 + j} test season: {len(f.loc[ARMS[0]])} matches")
    ax[0].set_ylabel("our XI's points as % of the best possible XI")
    fig.suptitle("How much of the best possible team's score each approach captures, match by match (diamond = mean)", fontsize=10)
    fig.tight_layout(); fig.savefig("results/overall/pct_of_best_team.png"); plt.close(fig)


def fig_competitions(t):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    w = 0.2
    for j, split in enumerate(YEARS):
        f = t[t.split == split]
        for i, a in enumerate(ARMS):
            v = [f[(f.arm == a) & (f.group == g)].groupby("fx_id").pct.mean().mean() for g in GROUPS]
            ax[j].bar(np.arange(4) + (i - 1.5) * w, v, w, color=C[a], label=LAB[a])
        n = [f[(f.arm == ARMS[0]) & (f.group == g)].fx_id.nunique() for g in GROUPS]
        ax[j].set_xticks(range(4), [f"{g}\n({k} matches)" for g, k in zip(GROUPS, n)], fontsize=8)
        ax[j].set_ylim(50, 72); ax[j].set_title(f"{2024 + j} test season, by competition")
    ax[0].set_ylabel("% of the best possible XI (mean)"); ax[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig("results/overall/by_competition.png"); plt.close(fig)


def fig_staleness(t):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for j, split in enumerate(YEARS):
        f = t[t.split == split]
        for a in ARMS:
            v = f[f.arm == a].groupby("delta").pct.mean()
            ax[j].plot(v.index, v.values, "o-", color=C[a], label=LAB[a])
        ax[j].set_xticks(DELTAS); ax[j].set_xlabel("Δ = days between the last known match and the fixture")
        ax[j].set_title(f"{2024 + j} test season: the older the history, the less we capture")
    ax[0].set_ylabel("% of the best possible XI (mean)"); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("results/overall/by_staleness.png"); plt.close(fig)


def fig_overlap(t):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    w = 0.2
    for j, split in enumerate(YEARS):
        f = t[t.split == split]
        for i, a in enumerate(ARMS):
            o = f[f.arm == a].overlap
            share = np.bincount(o, minlength=12)[:12] / len(o) * 100
            ax[j].bar(np.arange(12) + (i - 1.5) * w, share, w, color=C[a], label=f"{LAB[a]} (mean {o.mean():.2f})")
        ax[j].set_xticks(range(12)); ax[j].set_xlim(0.5, 11.5)
        ax[j].set_xlabel("how many of our 11 were in the best possible XI"); ax[j].set_title(f"{2024 + j} test season")
        ax[j].legend(fontsize=6.5, loc="upper right")
    ax[0].set_ylabel("% of matches")
    fig.tight_layout(); fig.savefig("results/overall/best_team_overlap.png"); plt.close(fig)


def fig_baseline():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    for arm, color in (("CB64", "#f4a259"), ("CB128", C["baseline"])):
        for split, ls in (("F24", "-"), ("F25", "--")):
            h = (done(f"{arm}_{split}_s0") or {}).get("history", [])
            ep = [x["epoch"] + 1 for x in h]
            ax[0].plot(ep, [x["train_loss"] for x in h], ls, color=color, label=f"{arm} {split}")
            ax[1].plot(ep, [x["valid_loss"] for x in h], ls, color=color)
            ax[2].plot(ep, [x["valid_headline"] for x in h], ls, color=color)
    for a, t in zip(ax, ("training loss (normalised MSE)", "validation loss (normalised MSE)", "validation captured/oracle")):
        a.set_title(t); a.set_xlabel("epoch")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig("results/baseline/training_curves.png"); plt.close(fig)


def fig_refined():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    for split, ls in (("F24", "-"), ("F25", "--")):
        for seed, color in ((0, C["refined"]), (1, "#7fb7d4")):
            h = (done(f"ADV_{split}_s{seed}") or {}).get("history", [])
            p1 = [x for x in h if x["phase"] == "P1"]; p2 = [x for x in h if x["phase"] == "P2"]; p3 = [x for x in h if x["phase"] == "P3"]
            ax[0].plot([x["epoch"] + 1 for x in p1], [x["valid_headline"] for x in p1], ls, color=color, label=f"{split} s{seed}")
            ax[1].plot([x["epoch"] + 1 for x in p2], [x["latent_mse"] for x in p2], ls, color=color)
            ax[2].plot([x["epoch"] + 1 for x in p3], [x["valid_headline"] for x in p3], ls, color=color)
    for a, t in zip(ax, ("P1 validation captured/oracle", "P2 (A5) one-step latent MSE", "P3 validation captured/oracle (roll-out)")):
        a.set_title(t); a.set_xlabel("epoch")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig("results/refined/training_phases.png"); plt.close(fig)
    m = json.load(open(f"{EXP}/E23_adv/metrics.json"))
    vs = [("ADV_direct_tau", "direct, τ shift\n(no choice layer)"), ("ADV_direct_a3", "direct +\nchoice layer (A3)"),
          ("ADV_rollout", "+ roll-forward\n(A5)"), ("ADV_student", "distilled student\n(A6)"), ("ADV", "final (chosen on\nvalidation)")]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    for j, split in enumerate(YEARS):
        v = [100 * m[k][split]["test"]["headline"]["captured_over_oracle"] for k, _ in vs]
        ax.bar(np.arange(len(vs)) + (j - 0.5) * 0.35, v, 0.35, color=["#7fb7d4", C["refined"]][j], label=f"{2024 + j} test")
        for i, x in enumerate(v):
            ax.text(i + (j - 0.5) * 0.35, x + 0.1, f"{x:.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(vs)), [l for _, l in vs], fontsize=8); ax.set_ylim(58, 64.5)
    ax.set_ylabel("% of the best possible XI"); ax.legend(fontsize=7)
    ax.set_title("each piece of the refined model, same trained weights")
    fig.tight_layout(); fig.savefig("results/refined/component_ladder.png"); plt.close(fig)


def fig_graph(curves):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
    for arm in ("graph", "graph+"):
        for split, ls in (("F24", "-"), ("F25", "--")):
            r = curves.get(f"{arm}_{split}")
            if not r:
                continue
            ax[0].plot(r["mu"]["valid"], ls, color=C[arm], label=f"{LAB[arm][:2]} {split}")
            ax[1].plot(r["p"]["valid"], ls, color=C[arm])
    ax[0].set_title("μ model: validation L2 (played rows)"); ax[1].set_title("p model: validation log-loss (squad rows)")
    for a in ax[:2]:
        a.set_xlabel("boosting iteration")
    ax[0].legend(fontsize=7)
    m = json.load(open(f"{EXP}/E24_matchups/metrics.json"))
    for arm, key in (("graph", "B0s"), ("graph+", "B0sM")):
        for split, ls in (("F24", "-"), ("F25", "--")):
            bd = m[key][split]["test"]["by_delta"]
            ax[2].plot([int(k) for k in bd], [100 * v for v in bd.values()], "o" + ls, color=C[arm], label=f"{LAB[arm][:2]} {2024 + (split == 'F25')}")
    ax[2].set_xticks(DELTAS); ax[2].set_title("% of the best XI by staleness Δ"); ax[2].set_xlabel("Δ (days)"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig("results/graph/curves_and_gain.png"); plt.close(fig)
    imp = json.load(open(f"{EXP}/E24_matchups/bootstrap.json"))["importance"]
    names = list(imp["F25_mu"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for j, key in enumerate(("F24_mu", "F25_mu", "F24_p", "F25_p")):
        ax.barh(np.arange(len(names)) + (j - 1.5) * 0.2, [imp[key][n] * 100 for n in names], 0.2, label=key.replace("_", " model ").replace("F", "fold F"))
    ax.set_yticks(range(len(names)), names); ax.invert_yaxis(); ax.set_xlabel("% of total split gain"); ax.legend(fontsize=6.5)
    ax.set_title("graph features: share of LightGBM gain"); fig.tight_layout(); fig.savefig("results/graph/graph_feature_gain.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-lgbm", action="store_true", help="recompute the LightGBM validation curves (≈ 5 min)")
    a = ap.parse_args()
    for d in ("results/overall", "results/baseline", "results/refined", "results/graph"):
        os.makedirs(d, exist_ok=True)
    cache = f"{EXP}/lgbm_curves.json"
    curves = lgbm_curves() if a.refit_lgbm or not os.path.exists(cache) else json.load(open(cache))
    t = team_table()
    fig_validation(curves); fig_pct_best(t); fig_competitions(t); fig_staleness(t); fig_overlap(t)
    fig_baseline(); fig_refined(); fig_graph(curves)
    print("figures written")


if __name__ == "__main__":
    main()
