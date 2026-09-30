"""Shared evaluation reports, each written once from its pre-registered rule (R4):
E22 (SPEC §13: B0s vs baseline CB64/CB128) and E23 (SPEC §14: refined vs B0s and CB128).

python src/shared/evaluation/report.py report | adv"""

import argparse
import hashlib
import json
import numpy as np
import os
import pandas as pd
import subprocess
import sys
from datetime import date
from multiprocessing import Pool

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # src/
from graph import scoring_model as T  # noqa: E402
from shared.preprocessing.causal import DELTAS, SPLITS  # noqa: E402
from shared.evaluation.metrics import CB_ARMS, KEY, OUT, PRED, SEEDS, YEARS, boot, data_hash, gates, spearman_within, squad_change, team_eval  # noqa: E402


def report():
    from baseline import config as cb  # HP for the config
    P = pd.read_parquet("data/processed/player_match.parquet")
    M = pd.read_parquet("data/interim/matches.parquet").set_index("match_id")
    G = gates()
    shared_ok = all(v for k, v in G.items() if not k.startswith(("cb_", "g7")))
    arms, notes, frames = {}, [], []
    for split in YEARS:
        b = pd.read_parquet(os.path.join(PRED, f"B0s_{split}.parquet")).rename(columns={"score": "B0s"})
        for arm in CB_ARMS:
            fs = [os.path.join(PRED, f"{arm}_{split}_s{s}.parquet") for s in SEEDS]
            if not all(os.path.exists(f) for f in fs):
                notes.append(f"{arm} {split}: missing seed predictions -> arm excluded (R1)"); arms[arm] = False; continue
            e = pd.concat([pd.read_parquet(f)[KEY + ["yhat"]] for f in fs]).groupby(KEY, sort=False).yhat.agg(["mean", "std"])
            e = e.rename(columns={"mean": arm, "std": arm + "_seed_sd"}).reset_index()
            n0 = len(b); b = b.merge(e, on=KEY, how="left")
            miss = int(b[arm].isna().sum())
            if miss or len(b) != n0:  # T1b: identical candidate sets for every arm
                notes.append(f"{arm} {split}: candidate sets differ from B0s ({miss} rows) -> arm excluded (R1)"); arms[arm] = False
            arms.setdefault(arm, True)
        frames.append(b.assign(split=split))
    D = pd.concat(frames, ignore_index=True)
    live = ["B0s"] + [a for a in CB_ARMS if arms.get(a)]
    arm_ok = {"B0s": shared_ok and G.get("g7", False),
              **{a: shared_ok and arms.get(a, False) and G.get("cb_smoke", False) and G.get("cb_overfit", False) and G.get("cb_tests", False) for a in CB_ARMS}}
    seg = M.assign(ipl=M.event.eq("Indian Premier League"))[["gender", "team_type", "ipl"]]
    chg = squad_change(P, set(D.fx_id))
    metrics, per_fx = {}, []
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        for (split, subset), d in D.groupby(["split", "subset"]):
            te = team_eval(d, live, pool)
            orc = team_eval(d.assign(oracle=d.y), ["oracle"], pool)
            te["oracle"] = orc.oracle_captured.to_numpy()
            te = te.assign(split=split, subset=subset)
            per_fx.append(te)
            for a in live:
                r = te.assign(ratio=te[a + "_captured"] / te.oracle.where(te.oracle > 0))
                by_d = {}
                for dl, x in d.groupby("delta"):
                    rx = r[r.delta == dl]
                    m = {"captured": float(rx[a + "_captured"].mean()), "oracle": float(rx.oracle.mean()),
                         "captured_over_oracle": float(rx.ratio.mean()), "captured_minus_random": float((rx[a + "_captured"] - rx.random).mean()),
                         "picked_played": float(rx[a + "_played"].mean()),
                         "mae_squad": float((x[a] - x.y).abs().mean()), "rmse_squad": float(np.sqrt(((x[a] - x.y) ** 2).mean())),
                         "spearman_squad": spearman_within(x, a)}
                    pl = x[x.played == 1]
                    pp = pl.mu if a == "B0s" else pl[a]  # B0s: μ on played rows; CB: its Ŷ
                    m.update(mae_played=float((pp - pl.y).abs().mean()), rmse_played=float(np.sqrt(((pp - pl.y) ** 2).mean())))
                    if a == "B0s":
                        p = np.clip(x.p, 1e-6, 1 - 1e-6)
                        m.update(logloss_p=float(-(x.played * np.log(p) + (1 - x.played) * np.log(1 - p)).mean()), brier_p=float(((p - x.played) ** 2).mean()))
                    else:
                        tt = x.groupby(["fx_id", "team"])[[a, "y"]].sum()
                        m.update(team_total_bias_mean=float((tt[a] - tt.y).mean()), team_total_bias_sd=float((tt[a] - tt.y).std()),
                                 team_total_bias_rel=float((tt[a].sum() - tt.y.sum()) / tt.y.sum()), seed_sd_mean=float(x[a + "_seed_sd"].mean()))
                    by_d[int(dl)] = m
                head = {k: float(np.mean([by_d[dl][k] for dl in by_d])) for k in by_d[DELTAS[0]]}
                segs = {}
                if subset == "test":
                    r = r.assign(**{c: r.fx_id.map(seg[c]) for c in seg}, change=r.fx_id.map(chg))
                    for name, mask in {"male": r.gender == "male", "female": r.gender == "female", "club": r.team_type == "club",
                                       "international": r.team_type == "international", "ipl": r.ipl == True, "squad_change": r.change == True,
                                       "no_squad_change": r.change == False}.items():
                        segs[name] = {"fixtures": int(r[mask].fx_id.nunique()), "captured_over_oracle": float(r[mask].ratio.groupby(r[mask].delta).mean().mean())}
                metrics.setdefault(a, {}).setdefault(split, {})[subset] = {"headline": head, "by_delta": by_d, "segments": segs}
    PF = pd.concat(per_fx, ignore_index=True)
    PF.to_parquet(os.path.join(OUT, "per_fixture.parquet"), index=False)
    # paired bootstrap on per-fixture captured, each fixture averaged over the five Δ (test years)
    bs = {}
    t = PF[PF.subset == "test"]
    pairs = [(a, "B0s") for a in CB_ARMS if a in live] + ([("CB128", "CB64")] if {"CB64", "CB128"} <= set(live) else [])
    for split in YEARS:
        f = t[t.split == split].groupby("fx_id")[[a + "_captured" for a in live]].mean()
        for x, y in pairs:
            bs[f"{x}-{y} {split}"] = boot((f[x + "_captured"] - f[y + "_captured"]).to_numpy())
    # §13.7 decision
    lines = []
    for a in ["B0s", *CB_ARMS]:
        lines.append(f"R1 {a}: {'passes' if arm_ok[a] else 'EXCLUDED'} (gates: {json.dumps({k: v for k, v in G.items()})})")
    cands = [a for a in CB_ARMS if arm_ok[a]]
    choice = None
    if cands:
        val = {a: sum(metrics[a][s]["valid"]["headline"]["captured_over_oracle"] for s in YEARS) for a in cands}
        choice = cands[0] if len(cands) == 1 else ("CB64" if abs(val["CB128"] - val["CB64"]) < 0.005 else max(val, key=val.get))
        lines.append(f"R2 choice of L: validation headline sums {json.dumps({k: round(v, 4) for k, v in val.items()})} -> {choice}")
    else:
        lines.append("R2 choice of L: no CB arm passed R1")
    submit = "B0s"
    if choice:
        wins = [bs[f"{choice}-B0s {s}"]["lo"] > 0 for s in YEARS]
        ci = "; ".join(f"{s} {bs[f'{choice}-B0s {s}']['mean']:+.2f} [{bs[f'{choice}-B0s {s}']['lo']:+.2f}, {bs[f'{choice}-B0s {s}']['hi']:+.2f}]" for s in YEARS)
        submit = choice if all(wins) else "B0s"
        lines.append(f"R3 submission: {choice} - B0s captured per fixture: {ci} -> submit {submit}")
    else:
        lines.append("R3 submission: B0s (no CB arm eligible)")
    lines.append("R4 no second look: this file is written once, from the numbers above.")
    lines += [f"note: {n}" for n in notes]
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "decision.md"), "w", encoding="utf-8").write("# E22 final run: decision (SPEC §13.7)\n\n" + "\n".join(f"- {l}" for l in lines) + "\n")
    json.dump({"submit": submit, "cb_choice": choice, "arms_ok": arm_ok}, open(os.path.join(OUT, "decision.json"), "w"), indent=1)
    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
    json.dump(bs, open(os.path.join(OUT, "bootstrap.json"), "w"), indent=1)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "no-git"
    except OSError:
        commit = "no-git"
    cfg = {"commit": commit, "data_hash": data_hash(), "splits": SPLITS, "deltas": DELTAS, "cb_hp": {k: v for k, v in cb.HP.items()},
           "cb_arms": cb.ARMS, "seeds": SEEDS, "b0s_params": T.BASE_PARAMS, "bootstrap": 2000}
    json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1, default=str)
    heads = " ".join(f"{a}:{s}={metrics[a][s]['test']['headline']['captured_over_oracle']:.4f}" for a in live for s in YEARS)
    with open("results/experiments/LOG.md", "a", encoding="utf-8") as f:
        f.write(f"| {date.today()} | E22 final | {hashlib.sha256(json.dumps(cfg, default=str).encode()).hexdigest()[:8]} | {heads} | submit {submit} |\n")
    print("\n".join(lines))


ADV_OUT = "results/experiments/E23_adv"


ADV_COLS = {"ADV": "yhat", "ADV_direct_a3": "direct_a3", "ADV_direct_tau": "direct_tau", "ADV_rollout": "rollout_a3", "ADV_student": "student_a3"}


def adv_report():
    """SPEC §14: ADV (two-seed ensemble of each seed's validation-chosen predictor) vs B0s and CB128, variants from the
    same runs (A3 vs τ, roll-out, student), paired bootstrap, and the pre-registered submission rule."""
    G = gates()
    ok_gates = all(v for k, v in G.items() if not k.startswith(("cb_", "g7", "adv_", "t5"))) and all(G.get(k, False) for k in ("adv_tests", "adv_smoke", "adv_overfit"))
    notes, frames = [], []
    for split in YEARS:
        b = pd.read_parquet(os.path.join(PRED, f"B0s_{split}.parquet")).rename(columns={"score": "B0s"})
        fs = [os.path.join(PRED, f"CB128_{split}_s{s}.parquet") for s in SEEDS]
        if all(os.path.exists(f) for f in fs):
            e = pd.concat([pd.read_parquet(f)[KEY + ["yhat"]] for f in fs]).groupby(KEY, sort=False).yhat.mean().rename("CB128").reset_index()
            b = b.merge(e, on=KEY, how="left")
        fa = [os.path.join(ADV_OUT, "pred", f"ADV_{split}_s{s}.parquet") for s in SEEDS]
        if not all(os.path.exists(f) for f in fa):
            notes.append(f"ADV {split}: missing seed predictions"); ok_gates = False; continue
        e = pd.concat([pd.read_parquet(f)[KEY + list(ADV_COLS.values())] for f in fa]).groupby(KEY, sort=False).mean()
        e = e.rename(columns={v: k for k, v in ADV_COLS.items()}).reset_index()
        n0 = len(b); b = b.merge(e, on=KEY, how="left")
        if b.ADV.isna().any() or len(b) != n0:
            notes.append(f"ADV {split}: candidate sets differ from B0s"); ok_gates = False
        frames.append(b.assign(split=split))
    if not frames:
        raise SystemExit("no ADV predictions")
    D = pd.concat(frames, ignore_index=True)
    arms = ["B0s"] + (["CB128"] if "CB128" in D else []) + list(ADV_COLS)
    metrics, per_fx = {}, []
    with Pool(min(4, os.cpu_count() or 1)) as pool:
        for (split, subset), d in D.groupby(["split", "subset"]):
            te = team_eval(d, arms, pool)
            te["oracle"] = team_eval(d.assign(oracle=d.y), ["oracle"], pool).oracle_captured.to_numpy()
            per_fx.append(te.assign(split=split, subset=subset))
            for a in arms:
                r = te.assign(ratio=te[a + "_captured"] / te.oracle.where(te.oracle > 0))
                by_d = {int(dl): {"captured_over_oracle": float(r[r.delta == dl].ratio.mean()), "captured": float(r[r.delta == dl][a + "_captured"].mean()),
                                  "mae_squad": float((x[a] - x.y).abs().mean()), "rmse_squad": float(np.sqrt(((x[a] - x.y) ** 2).mean())),
                                  "spearman_squad": spearman_within(x, a)} for dl, x in d.groupby("delta")}
                head = {k: float(np.mean([v[k] for v in by_d.values()])) for k in by_d[DELTAS[0]]}
                metrics.setdefault(a, {}).setdefault(split, {})[subset] = {"headline": head, "by_delta": by_d}
    PF = pd.concat(per_fx, ignore_index=True)
    os.makedirs(ADV_OUT, exist_ok=True)
    PF.to_parquet(os.path.join(ADV_OUT, "per_fixture.parquet"), index=False)
    t = PF[PF.subset == "test"]
    pairs = [("ADV", "B0s"), ("ADV", "CB128"), ("ADV_direct_a3", "ADV_direct_tau"), ("ADV_rollout", "ADV_direct_a3"), ("ADV_student", "ADV_direct_a3")]
    bs = {}
    for split in YEARS:
        f = t[t.split == split].groupby("fx_id")[[a + "_captured" for a in arms]].mean()
        for x, y in pairs:
            if x in arms and y in arms:
                bs[f"{x}-{y} {split}"] = boot((f[x + "_captured"] - f[y + "_captured"]).to_numpy())
    wins = ok_gates and all(bs[f"ADV-B0s {s}"]["lo"] > 0 for s in YEARS)
    submit = "ADV" if wins else "B0s"
    ci = "; ".join(f"{s} {bs[f'ADV-B0s {s}']['mean']:+.2f} [{bs[f'ADV-B0s {s}']['lo']:+.2f}, {bs[f'ADV-B0s {s}']['hi']:+.2f}]" for s in YEARS)
    lines = [f"gates: {'pass' if ok_gates else 'FAIL'} ({json.dumps(G)})", f"ADV - B0s captured per fixture: {ci} -> submit {submit}",
             "variants (same runs, test, per fixture): " + "; ".join(f"{k} {v['mean']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]" for k, v in bs.items() if not k.startswith("ADV-B0s")),
             "headline captured/oracle (test): " + " ".join(f"{a}:{s}={metrics[a][s]['test']['headline']['captured_over_oracle']:.4f}" for a in arms for s in YEARS),
             "R4: written once from these numbers."] + [f"note: {n}" for n in notes]
    open(os.path.join(ADV_OUT, "decision.md"), "w", encoding="utf-8").write("# E23 ADV: decision (SPEC §14)\n\n" + "\n".join(f"- {l}" for l in lines) + "\n")
    json.dump({"submit": submit, "gates_ok": ok_gates}, open(os.path.join(ADV_OUT, "decision.json"), "w"), indent=1)
    json.dump(metrics, open(os.path.join(ADV_OUT, "metrics.json"), "w"), indent=1)
    json.dump(bs, open(os.path.join(ADV_OUT, "bootstrap.json"), "w"), indent=1)
    from refined import config as adv
    cfg = {"data_hash": data_hash(), "splits": SPLITS, "deltas": DELTAS, "adv_hp": adv.HP, "seeds": SEEDS, "bootstrap": 2000}
    json.dump(cfg, open(os.path.join(ADV_OUT, "config.json"), "w"), indent=1, default=str)
    heads = " ".join(f"{a}:{s}={metrics[a][s]['test']['headline']['captured_over_oracle']:.4f}" for a in ("B0s", "ADV") for s in YEARS)
    with open("results/experiments/LOG.md", "a", encoding="utf-8") as f:
        f.write(f"| {date.today()} | E23 ADV | {hashlib.sha256(json.dumps(cfg, default=str).encode()).hexdigest()[:8]} | {heads} | submit {submit} |\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["report", "adv"], help="report: E22 (B0s vs CB64/CB128); adv: E23 (refined vs B0s, CB128)")
    a = ap.parse_args()
    {"report": report, "adv": adv_report}[a.cmd]()


if __name__ == "__main__":
    main()
