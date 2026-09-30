"""Refined pipeline: P1 (encoder + decoder) -> P2 (A5 transition on privileged gap states) -> P3 (decoder on
rolled-forward states) -> P4 (A6 students) -> the predictor chosen on validation; checkpoints every epoch.

python src/refined/pipeline.py train --split F24 --seed 0 --device cuda:0
python src/refined/pipeline.py smoke | overfit"""

import argparse
import copy
import json
import math
import numpy as np
import os
import pandas as pd
import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # src/
from shared.preprocessing.causal import DELTAS, draw_delta, SPLITS  # noqa: E402
from baseline.preprocessing import fixture_rows, make_fixtures, role_table, to  # noqa: E402
from baseline.decision_model import capture, headline  # noqa: E402
from baseline.pipeline import lr_at, run_id as base_run_id  # noqa: E402
from baseline.config import CATS, resolve_device, seed_all  # noqa: E402
from baseline.ingestion import days, load_tables  # noqa: E402
from refined.config import CKPT, HP, PRED, TAU  # noqa: E402
from refined.ingestion import Store  # noqa: E402
from refined.preprocessing import Fixtures, collate, loader  # noqa: E402
from refined.scoring_model import ADV, Student, build, student_x  # noqa: E402


def params_opt(mods, lr):
    ps = [p for m in mods for p in m.parameters()]
    emb = {id(p) for m in mods for mm in m.modules() if isinstance(mm, nn.Embedding) for p in mm.parameters()}
    dec = [p for p in ps if p.ndim == 2 and id(p) not in emb]
    rest = [p for p in ps if not (p.ndim == 2 and id(p) not in emb)]
    return torch.optim.AdamW([{"params": dec, "weight_decay": HP["wd"]}, {"params": rest, "weight_decay": 0.0}], lr=lr, betas=HP["betas"])


@torch.no_grad()
def predict(model, ds, deltas, dev, workers, paths=(("direct", "a3"),), bs=32):
    """Predictions for every (fixture, candidate, Δ) and every requested (path, how). Columns '<path>_<how>'."""
    model.eval()
    out = []
    for d in deltas:
        ds.plan(np.full(len(ds), d), np.zeros(len(ds), np.int64))
        batches = [list(range(i, min(i + bs, len(ds)))) for i in range(0, len(ds), bs)]
        cols = {f"{p}_{h}": [] for p, h in paths}
        for b in loader(ds, batches, workers):
            for k, v in model.predict_many(to(b, dev), paths).items():
                cols[k].append(v.float().cpu().numpy())
        f = ds.fx
        out.append(pd.DataFrame({"fx_id": np.concatenate([[x["id"]] * len(x["y"]) for x in f]), "delta": d,
                                 "team": np.concatenate([x["team"] for x in f]), "player_id": np.concatenate([x["player_id"] for x in f]),
                                 "y": np.concatenate([x["y"] for x in f]), **{k: np.concatenate(v) for k, v in cols.items()}}))
    return pd.concat(out, ignore_index=True)


def val_score(model, ds, roles, dev, workers, path, how="a3"):
    pv = predict(model, ds, DELTAS, dev, workers, ((path, how),))
    col = f"{path}_{how}"
    return headline(capture(pv.rename(columns={col: "yhat"}), roles)), float(((pv[col] - pv.y) ** 2).mean())


def save(path, **kw):
    tmp = path + ".tmp"
    torch.save(kw, tmp)
    os.replace(tmp, path)  # atomic


def run_id(dates, seed):
    return base_run_id("ADV", dates, seed)


def train_run(dates, seed, device="auto", workers=None, max_train=None, epochs=None, log=print, predict_test=True):
    """P1 (encoder + decoder, choice + L2) -> P2 (g on privileged gap states, encoder frozen) -> P3 (decoder copy on
    rolled-forward states) -> P4 (students, λ grid) -> choose the predictor on validation. One checkpoint per epoch."""
    rid = run_id(dates, seed)
    rdir = os.path.join(CKPT, rid)
    if os.path.exists(os.path.join(rdir, "done.json")):
        log(f"[{rid}] already done"); return rid
    os.makedirs(rdir, exist_ok=True)
    dev = torch.device(resolve_device(device))
    workers = (1 if dev.type == "cuda" else 0) if workers is None else workers
    seed_all(seed)
    L = HP["L"]
    tr_from, va_from, te_from, te_to = dates
    t0 = time.time()
    P, M, S = load_tables()
    P, M = P[P.date < te_from], M[M.date < te_from]
    store = Store(P, M, fold_end=va_from)
    fx_tr = make_fixtures(store, fixture_rows(P, S, tr_from, va_from, pad=True), M)
    if max_train:
        fx_tr = fx_tr[:max_train]
    fx_va = make_fixtures(store, fixture_rows(P, S, va_from, te_from, pad=False), M)
    roles_va = role_table(P, fx_va, DELTAS)
    ds_tr, ds_va = Fixtures(store, fx_tr, L), Fixtures(store, fx_va, L, true_xi=False)
    model = build(store, fx_tr).to(dev)
    ck = os.path.join(rdir, "last.pt")
    st = dict(phase="P1", epoch=-1, step=0, best=-1.0, best_loss=float("inf"), bad=0, history=[], best_state=None)
    E = epochs or HP["epochs"]
    spe = math.ceil(len(fx_tr) / HP["batch"])
    ema = copy.deepcopy(model)
    opt = params_opt([model.enc, model.dec], HP["lr"])
    if os.path.exists(ck):
        c = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(c["model"]); ema.load_state_dict(c["ema"]); st = c["state"]
        torch.set_rng_state(c["rng"])
        if dev.type == "cuda" and c.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(c["cuda_rng"], dev)
        if st["phase"] == "P1":
            opt.load_state_dict(c["opt"])
        log(f"[{rid}] resumed: phase {st['phase']} after epoch {st['epoch']}")
    log(f"[{rid}] setup {time.time() - t0:.0f}s: {len(fx_tr)} train / {len(fx_va)} valid fixtures, device={dev}")

    def checkpoint(o):
        save(ck, model=model.state_dict(), ema=ema.state_dict(), opt=o.state_dict() if o is not None else None, state=st,
             rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None)

    def epoch_batches(ep, subset=None):
        rng = np.random.default_rng([seed, ep, {"P1": 1, "P2": 2, "P3": 3}[st["phase"]]])
        ds_tr.plan([draw_delta(f"{f['id']}|{ep}") for f in fx_tr], rng.integers(0, 2, len(fx_tr)))
        idx = rng.permutation(len(fx_tr) if subset is None else subset)
        return [idx[i:i + HP["batch"]].tolist() for i in range(0, len(idx), HP["batch"])]

    def ema_step(src, dst):
        with torch.no_grad():
            for pe, pm in zip(dst.parameters(), src.parameters()):
                pe.mul_(HP["ema"]).add_(pm.detach(), alpha=1 - HP["ema"])

    # ---- P1: encoder + direct decoder (A0a, A0b, A2, A1, A3)
    while st["phase"] == "P1":
        ep = st["epoch"] + 1
        if ep >= E or st["bad"] >= HP["patience"]:
            model.load_state_dict(st["best_state"]); model.to(dev)
            model.dec_roll.load_state_dict(model.dec.state_dict())
            st.update(phase="P2", epoch=-1, p1_best=st["best"], best=-1.0, bad=0, best_state=None)
            checkpoint(None); break
        te = time.time(); model.train(); tl = []
        for b in loader(ds_tr, epoch_batches(ep), workers):
            b = to(b, dev)
            for g in opt.param_groups:
                g["lr"] = HP["lr"] * lr_at(st["step"], E * spe)
            z, cold, gapd = model.enc(b, b["cut"][b["fx"]])
            s, mu = model.dec(z, gapd, b)
            loss = model.dec.loss(s, mu, b)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_([p for m in (model.enc, model.dec) for p in m.parameters()], HP["clip"]); opt.step()
            st["step"] += 1; ema_step(model, ema); tl.append(loss.item())
        hv, vl = val_score(ema, ds_va, roles_va, dev, workers, "direct")
        better = hv > st["best"] + 1e-12
        if better:
            st.update(best=hv, best_loss=vl, bad=0, best_state={k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}, best_epoch=ep)
        else:
            st["bad"] += 1
        st["epoch"] = ep
        st["history"].append(dict(phase="P1", epoch=ep, train_loss=float(np.mean(tl)), first_loss=float(np.mean(tl[:2])),
                                  last_loss=float(np.mean(tl[-2:])), valid_headline=hv, valid_mse=vl, seconds=round(time.time() - te, 1)))
        checkpoint(opt)
        log(f"[{rid}] P1 epoch {ep}: loss {np.mean(tl):.4f} valid {hv:.4f}{' *' if better else ''} ({time.time() - te:.0f}s)")

    for p in model.enc.parameters():
        p.requires_grad_(False)
    # ---- P2: A5 transition g on privileged monthly state pairs (encoder frozen, stop-gradient targets)
    if st["phase"] == "P2":
        if not HP["a5"]:
            st.update(phase="P4", epoch=-1)
        opt2 = params_opt([model.g], HP["p2_lr"])
        ds_full = Fixtures(store, fx_tr, L, to_date=True)
        while st["phase"] == "P2":
            ep = st["epoch"] + 1
            if ep >= HP["p2_epochs"]:
                st.update(phase="P3", epoch=-1, best=-1.0, bad=0); checkpoint(None); break
            te = time.time(); model.g.train(); model.enc.eval(); tl = []
            batches = epoch_batches(ep, len(fx_tr))
            ds_full.plan(ds_tr.deltas, ds_tr.swaps)
            for b in loader(ds_full, batches, workers):
                b = to(b, dev)
                with torch.no_grad():
                    stt, day, mask = model.enc.states(b)
                fx = b["fx"]
                delta = b["delta"][fx]
                K = torch.ceil(delta / TAU).long()
                month0 = (b["month"][fx].float() - 1 - delta / 30.44) % 12
                gender, ttype = (b["fxcat"][fx, 1] == 1).float(), (b["fxcat"][fx, 2] == 1).float()
                loss, cnt = 0.0, 0
                with torch.no_grad():
                    zs = [model.enc.readout(stt, day, mask, torch.minimum(-delta + k * TAU, torch.zeros_like(delta)))[:2]
                          for k in range(int(K.max().item()) + 1)]
                for k in range(len(zs) - 1):
                    m = (K > k) & ~zs[k][1]  # a cold state stays cold in the roll-out: no pair
                    if m.any():
                        pred = model.g(zs[k][0][m], 2 * math.pi * ((month0[m] + k) % 12) / 12, gender[m], ttype[m])
                        loss = loss + ((pred - zs[k + 1][0][m]) ** 2).sum(); cnt += int(m.sum()) * zs[k][0].shape[1]
                if cnt:
                    loss = loss / cnt
                    opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step(); tl.append(loss.item())
            st["epoch"] = ep
            st["history"].append(dict(phase="P2", epoch=ep, latent_mse=float(np.mean(tl)) if tl else None, seconds=round(time.time() - te, 1)))
            checkpoint(None)
            log(f"[{rid}] P2 epoch {ep}: latent mse {np.mean(tl) if tl else float('nan'):.5f} ({time.time() - te:.0f}s)")

    # ---- P3: decoder copy fine-tuned on rolled-forward states (encoder and g frozen)
    for p in model.g.parameters():
        p.requires_grad_(False)
    if st["phase"] == "P3":
        opt3 = params_opt([model.dec_roll], HP["p3_lr"])
        ema3 = copy.deepcopy(model)
        while st["phase"] == "P3":
            ep = st["epoch"] + 1
            if ep >= HP["p3_epochs"] or st["bad"] >= HP["p3_patience"]:
                if st["best_state"] is not None:
                    model.load_state_dict(st["best_state"]); model.to(dev)
                st.update(phase="P4", epoch=-1); checkpoint(None); break
            te = time.time(); model.dec_roll.train(); model.enc.eval(); tl = []
            for b in loader(ds_tr, epoch_batches(ep), workers):
                b = to(b, dev)
                with torch.no_grad():
                    z, cold, gapd = model.enc(b, b["cut"][b["fx"]])
                    zr = model.roll(z, cold, b, torch.ceil(b["delta"][b["fx"]] / TAU).long())
                s, mu = model.dec_roll(zr, gapd, b)
                loss = model.dec_roll.loss(s, mu, b)
                opt3.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.dec_roll.parameters(), HP["clip"]); opt3.step()
                ema_step(model, ema3); tl.append(loss.item())
            hv, vl = val_score(ema3, ds_va, roles_va, dev, workers, "rollout")
            better = hv > st["best"] + 1e-12
            if better:
                st.update(best=hv, bad=0, best_state={k: v.detach().cpu().clone() for k, v in ema3.state_dict().items()})
            else:
                st["bad"] += 1
            st["epoch"] = ep
            st["history"].append(dict(phase="P3", epoch=ep, train_loss=float(np.mean(tl)), valid_headline=hv, seconds=round(time.time() - te, 1)))
            checkpoint(None)
            log(f"[{rid}] P3 epoch {ep}: loss {np.mean(tl):.4f} valid {hv:.4f}{' *' if better else ''} ({time.time() - te:.0f}s)")

    # ---- P4: A6 students on a cached feature set (one Δ draw), λ grid; choose the final predictor on validation
    if st["phase"] == "P4":
        te = time.time()
        model.eval()
        feats = {}
        for name, ds, ds_deltas in (("train", ds_tr, None), ("valid", ds_va, DELTAS)):
            X, Yl, Yt, keys = [], [], [], []
            for d in (ds_deltas or [None]):
                if d is None:
                    ds.plan([draw_delta(f"{f['id']}|999") for f in ds.fx], np.zeros(len(ds), np.int64))
                else:
                    ds.plan(np.full(len(ds), d), np.zeros(len(ds), np.int64))
                bs = [list(range(i, min(i + 32, len(ds)))) for i in range(0, len(ds), 32)]
                with torch.no_grad():
                    for b in loader(ds, bs, workers):
                        b = to(b, dev)
                        z, cold, gapd = model.enc(b, b["cut"][b["fx"]])
                        X.append(student_x(z, gapd, b).cpu()); Yl.append(b["y"].cpu())
                        Yt.append(model.predict(b, "rollout" if HP["a5"] else "direct").cpu())
            feats[name] = (torch.cat(X), torch.cat(Yl), torch.cat(Yt))
        ysd, ybar = float(model.dec.ysd), float(model.dec.ybar)
        Xt, Yl, Yt = (t.to(dev) for t in feats["train"])
        Xv = feats["valid"][0].to(dev)
        res = {}
        pv_base = predict(model, ds_va, DELTAS, dev, workers, (("direct", "a3"), ("rollout", "a3")) if HP["a5"] else (("direct", "a3"),))
        for name in [c for c in pv_base.columns if c.endswith("_a3")]:
            res[name] = headline(capture(pv_base.rename(columns={name: "yhat"}), roles_va))
        best_lam, best_state = None, None
        if HP["a6"]:
            for lam in HP["lambdas"]:
                torch.manual_seed(seed)
                stu = Student(Xt.shape[1], HP["d"]).to(dev)
                o = torch.optim.AdamW(stu.parameters(), lr=HP["p4_lr"], weight_decay=HP["wd"])
                tgt = (lam * Yl + (1 - lam) * Yt - ybar) / ysd
                g = torch.Generator(device="cpu").manual_seed(seed)
                for _ in range(HP["p4_epochs"]):
                    perm = torch.randperm(len(Xt), generator=g).to(dev)
                    for i in range(0, len(perm), 4096):
                        j = perm[i:i + 4096]
                        l = ((stu(Xt[j]) - tgt[j]) ** 2).mean()
                        o.zero_grad(); l.backward(); o.step()
                with torch.no_grad():
                    yv = (stu(Xv) * ysd + ybar).cpu().numpy()
                h = headline(capture(pv_base.assign(yhat=yv), roles_va))
                res[f"student_{lam}"] = h
                if best_lam is None or h > res[f"student_{best_lam}"]:
                    best_lam, best_state = lam, copy.deepcopy(stu.state_dict())
            model.student.load_state_dict(best_state)
        final = max(res, key=res.get)
        st.update(phase="DONE", p4=res, lam=best_lam, final=final.replace("_a3", "").split("_")[0])
        st["history"].append(dict(phase="P4", valid=res, seconds=round(time.time() - te, 1)))
        checkpoint(None)
        log(f"[{rid}] P4: validation headlines {json.dumps({k: round(v, 4) for k, v in res.items()})} -> final predictor {st['final']}")

    # ---- predictions of every predictor on validation (+ test), artifacts
    os.makedirs(PRED, exist_ok=True)
    paths = [("direct", "a3"), ("direct", "tau")] + ([("rollout", "a3")] if HP["a5"] else []) + ([("student", "a3")] if HP["a6"] else [])
    parts = [predict(model, ds_va, DELTAS, dev, workers, paths).assign(subset="valid")]
    if te_to and predict_test:
        P2_, M2, _ = load_tables()
        st2 = Store(P2_, M2, vocab=store.vocab, norm=store.norm)
        end = (pd.Timestamp(te_to) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        fx_te = make_fixtures(st2, fixture_rows(P2_, S, te_from, end, pad=False), M2)
        parts.append(predict(model, Fixtures(st2, fx_te, L, true_xi=False), DELTAS, dev, workers, paths).assign(subset="test"))
    pr = pd.concat(parts, ignore_index=True)
    pr["yhat"] = pr[f"{st['final']}_a3"]
    pr.to_parquet(os.path.join(PRED, f"{rid}.parquet"), index=False)
    save(os.path.join(rdir, "best.pt"), state={k: v.cpu() for k, v in model.state_dict().items()}, vocab=store.vocab, norm=store.norm,
         hp=HP, ybar=float(model.dec.ybar), ysd=float(model.dec.ysd), final=st["final"], dates=list(dates), seed=seed)
    info = dict(run=rid, p1_best=st.get("p1_best"), final=st["final"], lam=st.get("lam"), p4=st.get("p4"), history=st["history"],
                train_fixtures=len(fx_tr), valid_fixtures=len(fx_va), minutes=round((time.time() - t0) / 60, 1))
    json.dump(info, open(os.path.join(rdir, "done.json"), "w"), indent=1, default=float)
    log(f"[{rid}] done in {info['minutes']} min: final predictor {st['final']}")
    return rid


def package(runs, train_start, train_end):
    return dict(arch="adv", runs=runs, members=[torch.load(os.path.join(CKPT, r, "best.pt"), map_location="cpu", weights_only=False) for r in runs],
                train_start=train_start, train_end=train_end)


@torch.no_grad()
def predict_rows(model, Q, info, P, M, bs=32):
    """Seed-mean E[Y] for test candidates (same contract as base_predict_rows), each member's chosen predictor."""
    seed_all(0)
    preds = []
    for mb in model["members"]:
        HP.update({k: v for k, v in mb["hp"].items() if k in HP})
        net = ADV([len(mb["vocab"][c]) + 1 for c in CATS], mb["ybar"], mb["ysd"])
        net.load_state_dict(mb["state"]); net.eval()
        st = Store(P, M, vocab=mb["vocab"], norm=mb["norm"])
        fx = []
        for mid, g in Q.groupby("match_id", sort=False):
            i = info[mid]
            cat = st.encode(pd.DataFrame([{c: i.get(c) for c in CATS}]))[0][:5]
            fx.append(dict(id=mid, day=int(days([i["date"]])[0]), month=int(str(i["date"])[5:7]), sched=float(i["sched_overs"]),
                           bpo=float(i["balls_per_over"]), cat=cat, side=g.side.to_numpy(np.int64),
                           pc=np.array([st.pid_code.get(p, -1) for p in g.player_id]), tc=np.array([st.team_code.get(t, -1) for t in g.team]),
                           y=np.zeros(len(g), np.float32), played=np.zeros(len(g), np.int64), team=g.team.to_numpy(), player_id=g.player_id.to_numpy()))
        ds = Fixtures(st, fx, mb["hp"]["L"], true_xi=False)
        ds.plan([info[f["id"]]["delta"] for f in fx], np.zeros(len(fx), np.int64))
        batches = [list(range(i, min(i + bs, len(ds)))) for i in range(0, len(ds), bs)]
        preds.append(np.concatenate([net.predict(b, mb["final"], "a3").numpy() for b in loader(ds, batches, 0)]))
    return np.mean(preds, 0)


def smoke(device="auto", log=print):
    """200 F24 fixtures, 1 epoch of every phase: finite losses, P1 loss decreasing, predictions for every path."""
    import shutil
    d = SPLITS["F24"]
    rid = run_id(d, 99)
    shutil.rmtree(os.path.join(CKPT, rid), ignore_errors=True)
    keep = {k: HP[k] for k in ("p2_epochs", "p3_epochs", "p4_epochs")}
    HP.update(p2_epochs=1, p3_epochs=1, p4_epochs=2)
    try:
        train_run(d, 99, device, max_train=200, epochs=1, log=log, predict_test=False)
    finally:
        HP.update(keep)
    info = json.load(open(os.path.join(CKPT, rid, "done.json")))
    h = [x for x in info["history"] if x["phase"] == "P1"][0]
    pr = pd.read_parquet(os.path.join(PRED, f"{rid}.parquet"))
    os.remove(os.path.join(PRED, f"{rid}.parquet"))  # no metric from the smoke run is read or recorded
    cols = [c for c in pr.columns if c.endswith(("_a3", "_tau"))]
    res = dict(finite=bool(np.isfinite(pr[cols].to_numpy()).all() and np.isfinite(h["train_loss"])), first=h["first_loss"], last=h["last_loss"],
               phases=sorted({x["phase"] for x in info["history"]}))
    import pickle
    os.makedirs("model_artifacts", exist_ok=True)
    pickle.dump(package([rid], d[0], "2023-12-31"), open("model_artifacts/adv_smoke.pkl", "wb"))
    return res


def overfit(device="auto", n=50, steps=600, log=print):
    """T3: P1 on 50 fixtures at Δ = 0 without dropout: the loss must fall to a small fraction of its start."""
    seed_all(0)
    dev = torch.device(resolve_device(device))
    P, M, S = load_tables()
    d = SPLITS["F24"]
    P, M = P[P.date < d[2]], M[M.date < d[2]]
    store = Store(P, M, fold_end=d[1])
    fx = make_fixtures(store, fixture_rows(P, S, "2023-01-01", d[1], pad=False), M)[:n]
    ds = Fixtures(store, fx, HP["L"])
    model = build(store, fx)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        if isinstance(m, nn.MultiheadAttention):
            m.dropout = 0.0
    model = model.to(dev)
    opt = torch.optim.AdamW(list(model.enc.parameters()) + list(model.dec.parameters()), lr=1e-3, weight_decay=0.0)
    b = to(collate([ds[i] for i in range(len(ds))]), dev)
    losses = []
    for i in range(steps):
        model.train()
        z, cold, gapd = model.enc(b, b["cut"][b["fx"]])
        s, mu = model.dec(z, gapd, b)
        loss = model.dec.loss(s, mu, b)
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    res = dict(first=losses[0], last=losses[-1], ratio=losses[-1] / losses[0], pass_=losses[-1] < 0.1 * losses[0])
    log(f"[T3 adv] overfit {n} fixtures: loss {losses[0]:.3f} -> {losses[-1]:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "smoke", "overfit"])
    ap.add_argument("--split", default="F24", choices=list(SPLITS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "train":
        train_run(SPLITS[a.split], a.seed, a.device, a.workers)
        return
    res = smoke(a.device) if a.cmd == "smoke" else overfit(a.device)
    ok = (res["finite"] and res["last"] < res["first"] and res["phases"] == ["P1", "P2", "P3", "P4"]) if a.cmd == "smoke" else res["pass_"]
    res["pass"] = bool(ok)
    os.makedirs("data/audit", exist_ok=True)
    json.dump(res, open(f"data/audit/adv_{a.cmd}_audit.json", "w"), indent=1, default=float)
    print(json.dumps(res, default=float))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
