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
LAB = {"baseline": "1 Baseline (CB128)", "refined": "2 Refined (ADV)", "graph": "3a Graph base (B0s, submitted)",
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


def load_test():
    """Per-fixture captured (averaged over Δ) and headline by Δ for every arm, from E22/E23/E24 per_fixture tables."""
    e23 = pd.read_parquet(f"{EXP}/E23_adv/per_fixture.parquet")
    e24 = pd.read_parquet(f"{EXP}/E24_matchups/per_fixture.parquet")
    arms = {"baseline": (e23, "CB128"), "refined": (e23, "ADV"), "graph": (e23, "B0s"), "graph+": (e24, "B0sM")}
    return arms


def by_delta(df, col, split, subset="test"):
    d = df[(df.split == split) & (df.subset == subset)]
    return d.assign(r=d[col + "_captured"] / d.oracle.where(d.oracle > 0)).groupby("delta").r.mean()


def fig_test(arms):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.3, 1.3]})
    w = 0.2
    for i, (arm, (df, col)) in enumerate(arms.items()):
        vals = [by_delta(df, col, s).mean() for s in YEARS]
        ax[0].bar(np.arange(2) + (i - 1.5) * w, vals, w, color=C[arm], label=LAB[arm])
        for j, s in enumerate(YEARS):
            bd = by_delta(df, col, s)
            ax[j + 1].plot(bd.index, bd.values, "o-", color=C[arm], label=LAB[arm])
    ax[0].set_xticks([0, 1], ["2024 test", "2025 test"]); ax[0].set_ylim(0.58, 0.645)
    ax[0].set_title("headline captured/oracle (mean over Δ)"); ax[0].legend(fontsize=6.5, loc="upper left")
    for j, s in enumerate(YEARS):
        ax[j + 1].set_title(f"{2024 + j} test: captured/oracle by staleness Δ")
        ax[j + 1].set_xlabel("Δ = days between history cutoff and fixture"); ax[j + 1].set_xticks(DELTAS)
    fig.tight_layout(); fig.savefig("results/overall/test_results.png"); plt.close(fig)


def fig_forest():
    b23 = json.load(open(f"{EXP}/E23_adv/bootstrap.json"))
    b22 = json.load(open(f"{EXP}/E22_final/bootstrap.json"))
    b24 = json.load(open(f"{EXP}/E24_matchups/bootstrap.json"))["per_year"]
    rows = [("1 Baseline CB64 − 3a", b22, "CB64-B0s {}"), ("1 Baseline CB128 − 3a", b22, "CB128-B0s {}"),
            ("2 Refined ADV − 3a", b23, "ADV-B0s {}"), ("3b Graph features − 3a", {k: v for k, v in b24.items()}, "{}"),
            ("2 Refined − 1 Baseline", b23, "ADV-CB128 {}")]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for i, (lab, b, key) in enumerate(rows):
        for j, s in enumerate(YEARS):
            v = b[key.format(s)]
            y = i + (j - 0.5) * 0.3
            ax.errorbar(v["mean"], y, xerr=[[v["mean"] - v["lo"]], [v["hi"] - v["mean"]]], fmt="o", color=["#555", "#e08e0b"][j],
                        capsize=3, label=["2024 test", "2025 test"][j] if i == 0 else None)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("difference in captured points per fixture (mean over Δ), 95% paired bootstrap")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("results/overall/bootstrap_differences.png"); plt.close(fig)


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
    b = json.load(open(f"{EXP}/E23_adv/bootstrap.json"))
    rows = [("A3 choice layer − τ shift", "ADV_direct_a3-ADV_direct_tau"), ("A5 roll-out − direct", "ADV_rollout-ADV_direct_a3"),
            ("A6 student − direct", "ADV_student-ADV_direct_a3"), ("ADV − baseline CB128", "ADV-CB128"), ("ADV − 3a B0s", "ADV-B0s")]
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, (lab, key) in enumerate(rows):
        for j, s in enumerate(YEARS):
            v = b[f"{key} {s}"]
            ax.errorbar(v["mean"], i + (j - 0.5) * 0.3, xerr=[[v["mean"] - v["lo"]], [v["hi"] - v["mean"]]], fmt="o",
                        color=["#555", "#e08e0b"][j], capsize=3, label=["2024 test", "2025 test"][j] if i == 0 else None)
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel("captured points per fixture, 95% paired bootstrap"); ax.legend(fontsize=7)
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
    b = json.load(open(f"{EXP}/E24_matchups/bootstrap.json"))["per_delta"]
    for j, (s, color) in enumerate((("F24", "#555"), ("F25", "#e08e0b"))):
        xs = [int(k.split("d")[1]) for k in b if k.startswith(s)]
        v = [b[f"{s} d{x}"] for x in xs]
        ax[2].errorbar(np.array(xs) + (j - 0.5) * 8, [x["mean"] for x in v], yerr=[[x["mean"] - x["lo"] for x in v], [x["hi"] - x["mean"] for x in v]],
                       fmt="o-", color=color, capsize=3, label=f"{2024 + j} test")
    ax[2].axhline(0, color="k", lw=0.8); ax[2].set_xticks(DELTAS)
    ax[2].set_title("3b − 3a captured per fixture by Δ"); ax[2].set_xlabel("Δ (days)"); ax[2].legend(fontsize=7)
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
    fig_validation(curves); fig_test(load_test()); fig_forest(); fig_baseline(); fig_refined(); fig_graph(curves)
    print("figures written")


if __name__ == "__main__":
    main()
