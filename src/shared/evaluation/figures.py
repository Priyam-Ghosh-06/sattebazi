"""Every figure in the README, drawn with matplotlib from the files the runs wrote (nothing is re-selected or re-tuned here).

python src/shared/evaluation/figures.py [--refit-lgbm]

- explainers: how the pipeline works, the staleness timeline, why we predict p and μ separately, one real match worked through;
- results: the headline, validation curves, results by staleness and by competition, one training figure per approach.
The neural nets' curves come from their checkpoints' done.json. The trees' (LightGBM) curves are refitted here with the same
data and settings and cached in results/experiments/lgbm_curves.json. Team-level results per test match are cached in
results/experiments/team_table.parquet.
"""
import argparse, json, os, sys
from multiprocessing import Pool

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # src/
from graph import scoring_model as T  # noqa: E402
from shared.evaluation.metrics import _xi  # noqa: E402
from shared.optimization.selection import select_xi  # noqa: E402
from shared.preprocessing.causal import DELTAS, SPLITS  # noqa: E402

CK = "model_artifacts/checkpoints"
EXP = "results/experiments"
ARMS = ("baseline", "refined", "graph", "graph+", "advm")
C = {"baseline": "#d1495b", "refined": "#2e86ab", "graph": "#3b8b5a", "graph+": "#8fbf6f", "advm": "#7b5ea7", "random": "#b0b0b0"}
LAB = {"baseline": "1 Baseline: neural net learns everything itself", "refined": "2 Refined: neural net + cricket structure",
       "graph": "3a Trees (submitted)", "graph+": "3b Trees + batter-vs-bowler graph", "advm": "3c Refined net + batter-vs-bowler graph"}
SHORT = {"baseline": "1 Baseline\n(net learns alone)", "refined": "2 Refined\n(net + structure)", "graph": "3a Trees\n(submitted)",
         "graph+": "3b Trees\n+ graph", "advm": "3c Refined net\n+ graph", "random": "a random\nvalid team"}
YEARS = ("F24", "F25")
YEAR = {"F24": "2024", "F25": "2025"}
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def done(run):
    p = f"{CK}/{run}/done.json"
    return json.load(open(p)) if os.path.exists(p) else None


def cb_ysd(run):
    import torch
    return float(torch.load(f"{CK}/{run}/best.pt", map_location="cpu", weights_only=False)["ysd"])


def save(fig, path):
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


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


# ---------------------------------------------------------------- team-level results on the test seasons
GROUPS = ("IPL", "other men's leagues", "men's internationals", "women's cricket")


def _pick(args):
    """Our XI and the best XI in hindsight for one fixture copy: (points of our XI, points of the best XI, overlap)."""
    score, y, role, team, pid = args
    a, b = select_xi(score, role, team, pid)[0], select_xi(y, role, team, pid)[0]
    return float(y[a].sum()), float(y[b].sum()), len(set(a) & set(b))


def team_table():
    """One row per (approach, test season, match, Δ): points of our XI, of the best possible XI, and their overlap.
    Scores come from the stored predictions (neural nets: mean of the two seeds); 3b is refitted (its fold predictions were
    not stored). Cached in results/experiments/team_table.parquet."""
    cache = f"{EXP}/team_table.parquet"
    if os.path.exists(cache):
        t = pd.read_parquet(cache)
        if set(ARMS) <= set(t.arm):
            return t
    key = ["fx_id", "delta", "team", "player_id"]
    X = None
    rows = []
    for split in YEARS:
        b = pd.read_parquet(f"{EXP}/E22_final/pred/B0s_{split}.parquet")
        b = b[b.subset == "test"]
        d = b[key + ["y", "role"]].assign(graph=b.score.to_numpy())
        for arm, pat in (("baseline", "E22_final/pred/CB128_{}_s{}"), ("refined", "E23_adv/pred/ADV_{}_s{}"), ("advm", "E25_advm/pred/ADVM_{}_s{}")):
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


def random_pct(split):
    """A random valid XI, as % of the best XI (ratio of the season means; from the E22 metrics)."""
    h = json.load(open(f"{EXP}/E22_final/metrics.json"))["B0s"][split]["test"]["headline"]
    return 100 * (h["captured"] - h["captured_minus_random"]) / h["oracle"]


# ---------------------------------------------------------------- explainers
def _box(ax, x, y, w, h, text, color, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal", wrap=True)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.1))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")
    _box(ax, 0.1, 1.6, 1.8, 1.4, "Cricsheet\nball-by-ball data\n\nT20 matches\nup to 31 Dec 2025", "#eeeeee")
    _box(ax, 2.3, 2.5, 2.1, 1.2, "clean tables +\nfantasy points of every\nplayer in every match", "#eeeeee")
    _box(ax, 2.3, 0.9, 2.1, 1.2, "rebuild the squads\n(who was available,\nnot only who played)", "#eeeeee")
    ax.text(6.55, 4.35, "three ways to predict each squad player's expected points", ha="center", fontsize=9, weight="bold")
    lanes = [("baseline", "1 Baseline: a neural net reads every player's\nraw match history and both squads, and\noutputs expected points directly"),
             ("refined", "2 Refined: a neural net with cricket structure:\nchance of playing (a choice among the squad)\n× points if he plays"),
             ("graph", "3 Trees: hand-made form features,\nchance of playing × points if he plays\n(3b, 3c: + batter-vs-bowler match-ups)")]
    for i, (arm, text) in enumerate(lanes):
        _box(ax, 4.8, 3.05 - i * 1.25, 3.5, 1.05, text, C[arm] + "40", fs=7.6)
        _arrow(ax, 4.4, 2.3, 4.8, 3.55 - i * 1.25)
    _box(ax, 8.7, 1.6, 1.7, 1.4, "expected fantasy\npoints for every\nsquad player", "#fff3c4")
    for i in range(3):
        _arrow(ax, 8.3, 3.55 - i * 1.25, 8.7, 2.3)
    _box(ax, 10.8, 2.5, 2.1, 1.2, "exact optimiser picks\nthe best legal 11\n(role rules, both teams)", "#fff3c4", bold=True)
    _box(ax, 10.8, 0.9, 2.1, 1.2, "CSV: 11 players per\nmatch + an optional\nLLM-written summary", "#fff3c4")
    _arrow(ax, 1.9, 2.3, 2.3, 3.1); _arrow(ax, 1.9, 2.3, 2.3, 1.5); _arrow(ax, 10.4, 2.3, 10.8, 3.1); _arrow(ax, 11.85, 2.5, 11.85, 2.1)
    ax.text(3.35, 0.35, "everything a model sees is dated strictly before the match (and before 1 Jan 2026)", ha="center", fontsize=8, style="italic")
    save(fig, "results/overall/how_it_works.png")


def fig_timeline():
    d = lambda s: pd.Timestamp(s)  # noqa: E731
    fig, ax = plt.subplots(2, 1, figsize=(11, 3.9), sharex=True)
    a = ax[0]
    a.barh(0, (d("2025-12-31") - d("2024-01-01")).days, left=d("2024-01-01"), height=0.5, color="#3b8b5a", alpha=0.8)
    a.barh(0, (d("2026-09-15") - d("2026-01-01")).days, left=d("2026-01-01"), height=0.5, color="#dddddd", hatch="//", ec="#999")
    a.plot(d("2026-09-15"), 0, "r*", ms=16)
    a.annotate("", xy=(d("2026-09-15"), 0.45), xytext=(d("2026-01-01"), 0.45), arrowprops=dict(arrowstyle="<->", color="k"))
    a.text(d("2026-05-08"), 0.6, "Δ: no data about these months", ha="center", fontsize=8.5)
    a.text(d("2025-01-01"), 0, "all the data we are allowed to use", ha="center", va="center", color="white", fontsize=9, weight="bold")
    a.text(d("2026-09-15"), -0.55, "a hidden\ntest match", ha="center", va="top", fontsize=8, color="r")
    a.set_title("The real situation: the history stops on 31 Dec 2025, the matches we must predict come months later", fontsize=9.5, loc="left")
    a = ax[1]
    m = d("2025-06-15")
    a.barh(0, (m - pd.Timedelta(days=180) - d("2024-01-01")).days, left=d("2024-01-01"), height=0.5, color="#3b8b5a", alpha=0.8)
    a.barh(0, 180, left=m - pd.Timedelta(days=180), height=0.5, color="#dddddd", hatch="//", ec="#999")
    a.plot(m, 0, "r*", ms=16)
    a.text(d("2024-08-01"), 0, "history the model may see", ha="center", va="center", color="white", fontsize=9, weight="bold")
    a.text(m - pd.Timedelta(days=90), 0.6, "hidden on purpose (here Δ = 180 days)", ha="center", fontsize=8.5)
    a.text(m, -0.55, "a past match\n(true points known)", ha="center", va="top", fontsize=8, color="r")
    a.set_title("How I train and test for it: for every past match, cut the history Δ days earlier (Δ = 0, 90, 180, 270, 365)",
                fontsize=9.5, loc="left")
    for a in ax:
        a.set_yticks([]); a.set_ylim(-1.2, 1.0); a.spines["left"].set_visible(False)
    save(fig, "results/overall/staleness_timeline.png")


def fig_points():
    b = pd.concat([pd.read_parquet(f"{EXP}/E22_final/pred/B0s_{s}.parquet") for s in YEARS])
    b = b[(b.subset == "test") & (b.delta == 0)]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    share = b.played.mean() * 100
    ax[0].bar(["played", "sat out\n(0 points)"], [share, 100 - share], color=["#3b8b5a", "#bbbbbb"])
    for i, v in enumerate([share, 100 - share]):
        ax[0].text(i, v + 1, f"{v:.0f}%", ha="center")
    ax[0].set_ylabel("% of squad members"); ax[0].set_title("1. Many squad members don't play at all")
    y = b[b.played == 1].y.clip(upper=200)
    ax[1].hist(y, bins=np.arange(-5, 205, 5), color="#2e86ab")
    ax[1].axvline(y.mean(), color="k", ls="--"); ax[1].text(y.mean() + 3, ax[1].get_ylim()[1] * 0.85, f"mean {y.mean():.0f}", fontsize=8)
    ax[1].set_xlabel("fantasy points (players who played)"); ax[1].set_ylabel("player-matches")
    ax[1].set_title("2. Those who play: most score little, a few score big")
    bins = np.linspace(0, 1, 11)
    k = np.digitize(b.p, bins[1:-1])
    pr, ac = b.groupby(k).p.mean(), b.groupby(k).played.mean()
    ax[2].plot([0, 1], [0, 1], ":", color="#999", label="perfect")
    ax[2].plot(pr, ac, "o-", color="#3b8b5a", label="3a (submitted)")
    ax[2].set_xlabel("predicted chance of playing"); ax[2].set_ylabel("how often they actually played")
    ax[2].set_title("3. The chance-of-playing model is honest"); ax[2].legend(fontsize=7)
    fig.suptitle("Why I predict  expected points = chance of playing × points if he plays  (2024–25 test matches)", fontsize=10)
    save(fig, "results/overall/why_p_times_mu.png")


def fig_example():
    """One real match worked through by a model trained on data up to 30 Sep 2025 (the mock matches are Oct-Dec 2025).
    Not hand-picked: the match whose share of the best XI is the median of the 40 mock matches."""
    import pickle
    from evaluate import predict_fixtures
    from shared.ingestion.fixtures import load_fixtures
    path = "model_artifacts/model_2025-09-30.pkl"
    if not os.path.exists(path):
        print(f"skip worked example: train it with  python src/train.py --train-end 2025-09-30 --out {path}"); return
    model = pickle.load(open(path, "rb"))
    P = pd.read_parquet("data/processed/player_match.parquet"); M = pd.read_parquet("data/interim/matches.parquet")
    fxs = load_fixtures("data/mock/test_fixtures.json")
    _, XA = predict_fixtures(model, fxs, P, M[M.date <= model["train_end"]], "data/raw/registry", log=lambda *_: None)
    tr = pd.read_csv("data/mock/test_fixtures_truth.csv")
    res = []
    for i, f in enumerate(fxs):
        X = XA[XA.match_id == f"fx{i}"].reset_index(drop=True)
        t = tr[(tr.match_date == f["date"]) & (tr.team1 == f["team1"]) & (tr.team2 == f["team2"])].set_index("player").actual_points
        X = X.assign(actual=X.name.map(t).fillna(0).to_numpy(float))
        ours = set(select_xi(X.score.to_numpy(), X.role.to_numpy(), X.team.to_numpy(), X.player_id.to_numpy())[0])
        best = set(select_xi(X.actual.to_numpy(), X.role.to_numpy(), X.team.to_numpy(), X.player_id.to_numpy())[0])
        res.append((X[X.index.isin(ours)].actual.sum() / max(X[X.index.isin(best)].actual.sum(), 1), i, X, ours, best))
    res.sort(key=lambda r: r[:2])
    _, k, X, ours, best = res[(len(res) - 1) // 2]
    fx = [fxs[k]]
    print(f"worked example: mock match {k}; mean over the {len(res)} mock matches = {100 * np.mean([r[0] for r in res]):.1f}%")
    X = X.assign(ours=X.index.isin(ours), best=X.index.isin(best)).sort_values("score")
    teams = list(dict.fromkeys(X.team_display)); tc = {teams[0]: "#e0a100", teams[1]: "#2b2b2b"}
    fig, ax = plt.subplots(figsize=(10, 0.27 * len(X) + 1.6))
    yy = np.arange(len(X))
    ax.barh(yy, X.mu, color="#e8e8e8", label="points if he plays (μ)")
    ax.barh(yy, X.score, color=[tc[t] if o else "#bdbdbd" for t, o in zip(X.team_display, X.ours)], label="expected points = p × μ")
    ax.plot(X.actual, yy, "D", color="#d1495b", ms=5, label="what he actually scored")
    for i, r in enumerate(X.itertuples()):
        ax.text(max(r.mu, r.score) + 2, i, f"p = {r.p:.2f}" + ("   ★ in the best possible XI" if r.best else ""), va="center", fontsize=7)
    ax.set_yticks(yy, [f"{n} ({r})" + ("  ✔" if o else "") for n, r, o in zip(X.name, X.role, X.ours)], fontsize=7.5)
    ours_pts, best_pts = X[X.ours].actual.sum(), X[X.best].actual.sum()
    ax.set_title(f"A typical match (the median of the 40 mock matches): {fx[0]['team1']} vs {fx[0]['team2']}, {fx[0]['date']}\n"
                 f"✔ = our 11 (coloured by team).  Our XI scored "
                 f"{ours_pts:.0f} points, the best possible XI {best_pts:.0f} → {100 * ours_pts / best_pts:.0f}%.  "
                 f"{len(ours & best)} of our 11 were in it.", fontsize=8.5, loc="left")
    from matplotlib.patches import Patch
    h, l = ax.get_legend_handles_labels()
    h += [Patch(color=tc[t]) for t in teams]; l += [f"picked, {t}" for t in teams]
    ax.set_xlabel("fantasy points"); ax.legend(h, l, fontsize=7, loc="lower right")
    save(fig, "results/overall/worked_example.png")


# ---------------------------------------------------------------- results
def fig_headline(t):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    rows = ["random"] + list(ARMS)
    for j, split in enumerate(YEARS):
        f = t[t.split == split].groupby(["arm", "fx_id"]).pct.mean()
        v = [random_pct(split)] + [f.loc[a].mean() for a in ARMS]
        yy = np.arange(len(rows)) + (j - 0.5) * 0.38
        ax.barh(yy, v, 0.38, color=[C[r] for r in rows], alpha=[0.55, 1.0][j], label=f"{YEAR[split]} test season")
        for y_, x in zip(yy, v):
            ax.text(x + 0.5, y_, f"{x:.1f}%  ({YEAR[split]})", va="center", fontsize=7.5)
    ax.set_yticks(range(len(rows)), [SHORT[r].replace("\n", " ") for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.axvline(100, color="k", lw=0.8)
    ax.set_xlabel("points our 11 scored, as % of the best possible 11 (picked in hindsight), average over test matches")
    ax.set_title("All approaches end up close to each other and far above a random team; the limit is match-day luck, not the model",
                 fontsize=9.5, loc="left")
    save(fig, "results/overall/headline.png")


def fig_staleness(t):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for j, split in enumerate(YEARS):
        f = t[t.split == split]
        for a in ARMS:
            v = f[f.arm == a].groupby("delta").pct.mean()
            ax[j].plot(v.index, v.values, "o-", color=C[a], label=LAB[a], lw=2.2 if a == "graph" else 1.4)
        ax[j].set_xticks(DELTAS); ax[j].set_xlabel("Δ = days between the last match in our data and the match to predict")
        ax[j].set_title(f"{YEAR[split]} test season")
    ax[0].set_ylabel("% of the best possible XI"); ax[1].legend(fontsize=7)
    fig.suptitle("The older the history, the less anyone captures; the trees win with fresh history, the nets once it is old", fontsize=10)
    save(fig, "results/overall/by_staleness.png")


def fig_competitions(t):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    w = 0.16
    for j, split in enumerate(YEARS):
        f = t[t.split == split]
        for i, a in enumerate(ARMS):
            v = [f[(f.arm == a) & (f.group == g)].groupby("fx_id").pct.mean().mean() for g in GROUPS]
            ax[j].bar(np.arange(4) + (i - 2) * w, v, w, color=C[a], label=LAB[a])
        n = [f[(f.arm == ARMS[0]) & (f.group == g)].fx_id.nunique() for g in GROUPS]
        ax[j].set_xticks(range(4), [f"{g}\n({k} matches)" for g, k in zip(GROUPS, n)], fontsize=8)
        ax[j].set_ylim(50, 72); ax[j].set_title(f"{YEAR[split]} test season")
    ax[0].set_ylabel("% of the best possible XI"); ax[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Women's cricket and internationals are far more predictable than the men's leagues", fontsize=10)
    save(fig, "results/overall/by_competition.png")


def fig_validation(curves):
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    for j, split in enumerate(YEARS):
        a_h, a_m = ax[0, j], ax[1, j]
        for seed, ls in ((0, "-"), (1, "--")):
            h = (done(f"CB128_{split}_s{seed}") or {}).get("history", [])
            if h:
                ep = [x["epoch"] + 1 for x in h]
                a_h.plot(ep, [100 * x["valid_headline"] for x in h], ls, color=C["baseline"], label=LAB["baseline"] if seed == 0 else None)
                ysd = cb_ysd(f"CB128_{split}_s{seed}")
                a_m.plot(ep, [x["valid_loss"] * ysd ** 2 for x in h], ls, color=C["baseline"], label=LAB["baseline"] if seed == 0 else None)
            h = (done(f"ADV_{split}_s{seed}") or {}).get("history", [])
            run = [x for x in h if x["phase"] in ("P1", "P3")]
            p1 = [x for x in h if x["phase"] == "P1"]
            if run:
                a_h.plot(range(1, len(run) + 1), [100 * x["valid_headline"] for x in run], ls, color=C["refined"], label=LAB["refined"] if seed == 0 else None)
                a_m.plot(range(1, len(p1) + 1), [x["valid_mse"] for x in p1], ls, color=C["refined"], label=LAB["refined"] if seed == 0 else None)
            hm = [x for x in (done(f"ADVM_{split}_s{seed}") or {}).get("history", []) if x["phase"] in ("P1", "P3")]
            if hm and run:  # warm-started from the refined model: drawn as a continuation of its curve
                a_h.plot(range(len(run), len(run) + len(hm) + 1), [100 * run[-1]["valid_headline"]] + [100 * x["valid_headline"] for x in hm], ls,
                         color=C["advm"], label=LAB["advm"] + " (continues from 2)" if seed == 0 else None)
        for arm in ("graph", "graph+"):
            r = curves.get(f"{arm}_{split}")
            if r:
                x = np.array(r["iters"]) / max(r["iters"]) * 20  # tree-building rounds stretched onto the 20-epoch axis
                a_h.plot(x, 100 * np.array(r["valid_headline"]), ":", color=C[arm], lw=2, label=LAB[arm])
                a_m.plot(x, r["valid_mse"], ":", color=C[arm], lw=2, label=LAB[arm])
        yr = "Oct–Dec 2023" if split == "F24" else "Oct–Dec 2024"
        a_h.set_title(f"{YEAR[split]} fold: validation matches {yr}\nscore: % of the best possible XI (higher = better)", fontsize=9)
        a_m.set_title("error of the predicted points: mean squared error (lower = better)", fontsize=9)
        a_m.set_xlabel("training progress: epochs for the nets,\ntree-building rounds for the trees (stretched to the same width)")
    ax[0, 0].legend(fontsize=6.5, loc="lower right"); ax[1, 0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("Validation during training (solid/dashed = two random seeds of the same net)", fontsize=10)
    save(fig, "results/overall/validation_curves.png")


def fig_baseline():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for arm, color, n in (("CB64", "#f4a259", 64), ("CB128", C["baseline"], 128)):
        for split, ls in (("F24", "-"), ("F25", "--")):
            h = (done(f"{arm}_{split}_s0") or {}).get("history", [])
            ep = [x["epoch"] + 1 for x in h]
            ax[0].plot(ep, [x["train_loss"] for x in h], ls, color=color, label=f"last {n} matches of history, {YEAR[split]} fold")
            ax[1].plot(ep, [x["valid_loss"] for x in h], ls, color=color)
            ax[2].plot(ep, [100 * x["valid_headline"] for x in h], ls, color=color)
    titles = ("training error\n(keeps falling: the net is learning)", "validation error\n(flattens: stop there, or it memorises)",
              "validation score\n(% of the best possible XI)")
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=9); a.set_xlabel("epoch (one pass over the training matches)")
    ax[0].set_ylabel("squared error (scaled)"); ax[0].legend(fontsize=6.5)
    save(fig, "results/baseline/training_curves.png")


def fig_refined():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for split, ls in (("F24", "-"), ("F25", "--")):
        for seed, color in ((0, C["refined"]), (1, "#7fb7d4")):
            h = (done(f"ADV_{split}_s{seed}") or {}).get("history", [])
            p1 = [x for x in h if x["phase"] == "P1"]; p2 = [x for x in h if x["phase"] == "P2"]; p3 = [x for x in h if x["phase"] == "P3"]
            lab = f"{YEAR[split]} fold, seed {seed}"
            ax[0].plot([x["epoch"] + 1 for x in p1], [100 * x["valid_headline"] for x in p1], ls, color=color, label=lab)
            ax[1].plot([x["epoch"] + 1 for x in p2], [x["latent_mse"] for x in p2], ls, color=color)
            ax[2].plot([x["epoch"] + 1 for x in p3], [100 * x["valid_headline"] for x in p3], ls, color=color)
    titles = ("phase 1: learn players + who plays\nvalidation score (% of best XI)",
              "phase 2: learn how form drifts month to month\nerror of the one-month-ahead guess (lower = better)",
              "phase 3: re-tune on form rolled forward\nto the match date: validation score")
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=9); a.set_xlabel("epoch")
    ax[0].legend(fontsize=6.5)
    save(fig, "results/refined/training_phases.png")
    m = json.load(open(f"{EXP}/E23_adv/metrics.json")); mm = json.load(open(f"{EXP}/E25_advm/metrics.json"))
    vs = [(m, "ADV_direct_tau", "no choice model\n(simple 'sum to 11'\nfix)"), (m, "ADV_direct_a3", "+ choice model\n(who plays = a\nchoice among the\nsquad)"),
          (m, "ADV_rollout", "+ form rolled\nforward to the\nmatch date"), (m, "ADV_student", "one-pass student\n(failed, dropped)"),
          (m, "ADV", "2 Refined final\n(each seed picks\non validation)"), (mm, "ADVM", "3c: + batter-vs-\nbowler graph\nfeatures")]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for j, split in enumerate(YEARS):
        v = [100 * src[k][split]["test"]["headline"]["captured_over_oracle"] for src, k, _ in vs]
        ax.bar(np.arange(len(vs)) + (j - 0.5) * 0.36, v, 0.36, color=[C["advm"] if k == "ADVM" else ["#7fb7d4", C["refined"]][j] for _, k, _ in vs],
               alpha=[0.6, 1][j], label=f"{YEAR[split]} test season")
        for i, x in enumerate(v):
            ax.text(i + (j - 0.5) * 0.36, x + 0.08, f"{x:.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(vs)), [l for *_, l in vs], fontsize=7.5); ax.set_ylim(58, 64.5)
    ax.set_ylabel("% of the best possible XI"); ax.legend(fontsize=7, loc="lower left")
    ax.set_title("What each piece of the refined model is worth on the test seasons (the first five share the same trained weights)", fontsize=9)
    save(fig, "results/refined/component_ladder.png")


FEATURE_NAMES = {
    "x_mu_self_bat_run": "batting: his own run-scoring quality", "x_mu_bat_run": "batting: how the opposing bowlers affect his runs",
    "x_mu_bat_run_int": "batting: his personal match-ups only (runs)", "x_mu_self_bat_out": "batting: his own risk of getting out",
    "x_mu_bat_out": "batting: how the opposing bowlers affect his getting out", "x_mu_self_bowl_run": "bowling: his own economy quality",
    "x_mu_bowl_run": "bowling: how the opposing batters affect runs he concedes", "x_mu_bowl_run_int": "bowling: his personal match-ups only (runs)",
    "x_mu_self_bowl_out": "bowling: his own wicket-taking quality", "x_mu_bowl_out": "bowling: how the opposing batters affect his wickets"}


def fig_graph(curves):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for arm in ("graph", "graph+"):
        for split, ls in (("F24", "-"), ("F25", "--")):
            r = curves.get(f"{arm}_{split}")
            if r:
                ax[0].plot(r["mu"]["valid"], ls, color=C[arm], label=f"{LAB[arm]}, {YEAR[split]} fold")
                ax[1].plot(r["p"]["valid"], ls, color=C[arm])
    ax[0].set_title("'points if he plays' model: validation error\n(mean squared error, players who played)", fontsize=9)
    ax[1].set_title("'chance of playing' model: validation error\n(log-loss, every squad member)", fontsize=9)
    for a in ax:
        a.set_xlabel("tree-building round (each new tree corrects the previous ones)")
    ax[0].legend(fontsize=6.5)
    save(fig, "results/graph/curves_and_gain.png")
    imp = json.load(open(f"{EXP}/E24_matchups/bootstrap.json"))["importance"]
    names = sorted(FEATURE_NAMES, key=lambda n: -(imp["F24_mu"][n] + imp["F25_mu"][n]))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for j, (part, lab) in enumerate((("mu", "'points if he plays' model"), ("p", "'chance of playing' model"))):
        ax.barh(np.arange(len(names)) + (j - 0.5) * 0.38, [50 * (imp[f"F24_{part}"][n] + imp[f"F25_{part}"][n]) for n in names], 0.38,
                color=["#3b8b5a", "#8fbf6f"][j], label=lab)
    ax.set_yticks(range(len(names)), [FEATURE_NAMES[n] for n in names], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("% of the trees' total improvement that came from this feature (mean of 2024 and 2025 folds)"); ax.legend(fontsize=7)
    ax.set_title("Which graph features the trees used most", fontsize=9.5)
    save(fig, "results/graph/graph_feature_gain.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-lgbm", action="store_true", help="recompute the LightGBM validation curves (≈ 5 min)")
    a = ap.parse_args()
    for d in ("results/overall", "results/baseline", "results/refined", "results/graph"):
        os.makedirs(d, exist_ok=True)
    cache = f"{EXP}/lgbm_curves.json"
    curves = lgbm_curves() if a.refit_lgbm or not os.path.exists(cache) else json.load(open(cache))
    fig_pipeline(); fig_timeline(); fig_points(); fig_example()
    t = team_table()
    fig_headline(t); fig_staleness(t); fig_competitions(t); fig_validation(curves)
    fig_baseline(); fig_refined(); fig_graph(curves)
    print("figures written")


if __name__ == "__main__":
    main()
