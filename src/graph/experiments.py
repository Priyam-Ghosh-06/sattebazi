"""Graph approach experiments: B0s folds (E22 reference) and E24 (B0s + graph features vs B0s, pre-registered two-year rule).

python src/graph/experiments.py ingest | snapshots | features | g8 | folds | e24"""

import argparse
import hashlib
import json
import numpy as np
import os
import pandas as pd
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from graph import scoring_model as T  # noqa: E402
from shared.preprocessing.causal import SPLITS  # noqa: E402
from graph.graph_model import FEATS, FIRST, HALF_LIFE, K, KEY, LAST, SD_AB, SD_UV, STALE, features, g8, snapshots  # noqa: E402
from graph.ingestion import ingest  # noqa: E402


OUT = "results/experiments/E24_matchups"


def backtest():
    from shared.evaluation import metrics as FR
    X = pd.read_parquet(STALE).merge(pd.read_parquet(FEATS).drop(columns=["c_date", "mu_snapshot"]), on=KEY, how="left", validate="one_to_one")
    X = T.prep(X)
    mu_cols = [c for c in X.columns if c.startswith("x_mu_")]
    os.makedirs(OUT, exist_ok=True)
    frames, imp = [], {}
    for split in FR.YEARS:
        tr_from, va_from, te_from, te_to = SPLITS[split]
        tr = X[(X.part == "train") & (X.date >= tr_from) & (X.date < te_from)]
        g = X[(X.part != "train") & (X.date >= va_from) & (X.date <= te_to)].reset_index(drop=True)
        out = g[FR.KEY + ["y", "played", "role"]].assign(subset=np.where(g.date < te_from, "valid", "test"), split=split)
        for arm, extra in (("B0s", ()), ("B0sM", ("x_mu_",))):
            m = T.fit_pipeline(tr, valid_from=va_from, extra=extra)
            mu, p = T.predict(m, g)
            out[arm] = mu * p
            if arm == "B0sM":
                for part in ("mu", "p"):
                    gain = dict(zip(m["cols"], m[part].feature_importance("gain")))
                    tot = sum(gain.values())
                    imp[f"{split}_{part}"] = {c: round(gain[c] / tot, 4) for c in mu_cols}
            print(split, arm, "mu iters", m["mu"].best_iteration, "p iters", m["p"].best_iteration, flush=True)
        frames.append(out)
    D = pd.concat(frames, ignore_index=True)
    metrics, per_fx = {}, []
    from multiprocessing import Pool as P_
    with P_(4) as pool:
        for (split, subset), d in D.groupby(["split", "subset"]):
            te = FR.team_eval(d, ["B0s", "B0sM"], pool)
            te["oracle"] = FR.team_eval(d.assign(oracle=d.y), ["oracle"], pool).oracle_captured.to_numpy()
            per_fx.append(te.assign(split=split, subset=subset))
            for a in ("B0s", "B0sM"):
                r = te.assign(ratio=te[a + "_captured"] / te.oracle.where(te.oracle > 0))
                by_d = {int(dl): float(x.ratio.mean()) for dl, x in r.groupby("delta")}
                metrics.setdefault(a, {}).setdefault(split, {})[subset] = {"captured_over_oracle": float(np.mean(list(by_d.values()))),
                                                                         "by_delta": by_d, "captured": float(r[a + "_captured"].mean())}
    PF = pd.concat(per_fx, ignore_index=True)
    PF.to_parquet(os.path.join(OUT, "per_fixture.parquet"), index=False)
    t = PF[PF.subset == "test"]
    bs, bs_d = {}, {}
    for split in FR.YEARS:
        f = t[t.split == split].groupby("fx_id")[["B0s_captured", "B0sM_captured"]].mean()
        bs[split] = FR.boot((f.B0sM_captured - f.B0s_captured).to_numpy())
        for dl, x in t[t.split == split].groupby("delta"):
            bs_d[f"{split} d{int(dl)}"] = FR.boot((x.B0sM_captured - x.B0s_captured).to_numpy())
    g8ok = json.load(open("data/audit/g8_audit.json")).get("pass", False) if os.path.exists("data/audit/g8_audit.json") else False
    adopt = g8ok and all(bs[s]["lo"] > 0 for s in FR.YEARS)
    lines = [f"G8: {'pass' if g8ok else 'FAIL/missing'}",
             "B0s+M - B0s captured per fixture (test, mean over Δ): " + "; ".join(f"{s} {v['mean']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]" for s, v in bs.items()),
             f"-> {'ADOPT B0s+M' if adopt else 'keep B0s'} (two-year rule)",
             "by Δ: " + "; ".join(f"{k} {v['mean']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]" for k, v in bs_d.items()),
             "headline captured/oracle (test): " + " ".join(f"{a}:{s}={metrics[a][s]['test']['captured_over_oracle']:.4f}" for a in ("B0s", "B0sM") for s in FR.YEARS),
             "x_mu_ gain share: " + json.dumps(imp), "R4: written once from these numbers."]
    open(os.path.join(OUT, "decision.txt"), "w", encoding="utf-8").write("E24 graph features: decision\n\n" + "\n".join(f"- {l}" for l in lines) + "\n")
    json.dump({"adopt": adopt, "gain_vs_b0s": {s: v["mean"] for s, v in bs.items()}}, open(os.path.join(OUT, "decision.json"), "w"), indent=1)
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    json.dump({"per_year": bs, "per_delta": bs_d, "importance": imp}, open(os.path.join(OUT, "bootstrap.json"), "w"), indent=1)
    cfg = {"k": K, "half_life": HALF_LIFE, "sd_ab": SD_AB, "sd_uv": SD_UV, "snapshots": [FIRST, LAST], "data_hash": FR.data_hash()}
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)
    heads = " ".join(f"{a}:{s}={metrics[a][s]['test']['captured_over_oracle']:.4f}" for a in ("B0s", "B0sM") for s in FR.YEARS)
    with open("results/experiments/LOG.txt", "a", encoding="utf-8") as f:
        f.write(f"| {date.today()} | E24 matchups | {hashlib.sha256(json.dumps(cfg).encode()).hexdigest()[:8]} | {heads} "
                f"| dCaptured F24 {bs['F24']['mean']:+.2f} [{bs['F24']['lo']:+.2f},{bs['F24']['hi']:+.2f}] F25 {bs['F25']['mean']:+.2f} "
                f"[{bs['F25']['lo']:+.2f},{bs['F25']['hi']:+.2f}] | {'adopt' if adopt else 'keep B0s'} |\n")
    print("\n".join(lines))


def folds(features="data/processed/stale_features.parquet"):
    """B0s on F24 and F25: fit on the 'train' copy, predict the Δ grid of validation + test (E22 reference)."""
    from shared.evaluation.metrics import KEY as MKEY, PRED as MPRED, YEARS
    X = T.prep(pd.read_parquet(features))
    os.makedirs(MPRED, exist_ok=True)
    for split in YEARS:
        tr_from, va_from, te_from, te_to = SPLITS[split]
        model = T.fit_pipeline(X[(X.part == "train") & (X.date >= tr_from) & (X.date < te_from)], valid_from=va_from)
        g = X[(X.part != "train") & (X.date >= va_from) & (X.date <= te_to)].reset_index(drop=True)
        mu, p = T.predict(model, g)
        out = g[MKEY + ["y", "played", "role"]].assign(mu=mu, p=p, score=mu * p, subset=np.where(g.date < te_from, "valid", "test"))
        out.to_parquet(os.path.join(MPRED, f"B0s_{split}.parquet"), index=False)
        print(split, "mu iters", model["mu"].best_iteration, "p iters", model["p"].best_iteration, len(out), "rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ingest", "snapshots", "features", "g8", "folds", "e24"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"ingest": lambda: ingest(a.workers), "snapshots": snapshots, "features": features, "g8": g8, "folds": folds, "e24": backtest}[a.cmd]()


if __name__ == "__main__":
    main()
