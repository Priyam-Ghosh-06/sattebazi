"""Baseline pipeline: training with per-epoch checkpoints and resume, prediction, packaging, smoke and overfit checks.

python src/baseline/pipeline.py train --arm CB128 --split F24 --seed 0 --device cuda:0
python src/baseline/pipeline.py smoke | overfit"""

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
from baseline.config import ARMS, CATS, CKPT, HP, PRED, resolve_device, seed_all  # noqa: E402
from baseline.decision_model import capture, headline  # noqa: E402
from baseline.ingestion import Store, days, load_tables  # noqa: E402
from baseline.preprocessing import Fixtures, collate, fixture_rows, loader, make_fixtures, role_table, to  # noqa: E402
from baseline.scoring_model import CB, build_model, predict  # noqa: E402


def optimizer(model):
    emb = {id(p) for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters()}
    decay = [p for p in model.parameters() if p.ndim == 2 and id(p) not in emb]
    rest = [p for p in model.parameters() if not (p.ndim == 2 and id(p) not in emb)]
    return torch.optim.AdamW([{"params": decay, "weight_decay": HP["wd"]}, {"params": rest, "weight_decay": 0.0}],
                             lr=HP["lr"], betas=HP["betas"])


def lr_at(step, total):
    w = max(1, int(HP["warmup"] * total))
    if step < w:
        return (step + 1) / w
    t = (step - w) / max(1, total - w)
    return HP["floor"] + (1 - HP["floor"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


def run_id(arm, dates, seed):
    name = next((k for k, v in SPLITS.items() if v[:3] == tuple(dates[:3])), f"{dates[0]}_{dates[2]}")
    return f"{arm}_{name}_s{seed}"


def train_run(arm, dates, seed, device="auto", workers=None, max_train=None, epochs=None, log=print, predict_test=True):
    """Train one (arm, split, seed). Checkpoint every epoch to model_artifacts/checkpoints/<run>/last.pt; resume from it.
    Writes validation (+ test) predictions of the best EMA weights and a done.json. Returns the run id."""
    rid = run_id(arm, dates, seed)
    rdir = os.path.join(CKPT, rid)
    if os.path.exists(os.path.join(rdir, "done.json")):
        log(f"[{rid}] already done"); return rid
    os.makedirs(rdir, exist_ok=True)
    dev = torch.device(resolve_device(device))
    workers = (1 if dev.type == "cuda" else 0) if workers is None else workers  # 2 GPU jobs share 4 vCPU
    seed_all(seed)
    L = ARMS[arm]
    tr_from, va_from, te_from, te_to = dates
    t0 = time.time()
    P, M, S = load_tables()
    P = P[P.date < te_from]; M = M[M.date < te_from]  # history the fold may see
    store = Store(P, M, fold_end=va_from)
    fx_tr = make_fixtures(store, fixture_rows(P, S, tr_from, va_from, pad=True), M)
    if max_train:
        fx_tr = fx_tr[:max_train]
    fx_va = make_fixtures(store, fixture_rows(P, S, va_from, te_from, pad=False), M)
    roles_va = role_table(P, fx_va, DELTAS)
    ds_tr, ds_va = Fixtures(store, fx_tr, L), Fixtures(store, fx_va, L)
    model = build_model(store, L, fx_tr).to(dev)
    ema = copy.deepcopy(model)
    opt = optimizer(model)
    E = epochs or HP["epochs"]
    steps_per = math.ceil(len(fx_tr) / HP["batch"]); total = E * steps_per
    st = dict(epoch=-1, step=0, best=-1.0, best_loss=float("inf"), bad=0, history=[], best_state=None)
    ck = os.path.join(rdir, "last.pt")
    if os.path.exists(ck):
        c = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(c["model"]); ema.load_state_dict(c["ema"]); opt.load_state_dict(c["opt"])
        st = c["state"]; torch.set_rng_state(c["rng"])
        if dev.type == "cuda" and c.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(c["cuda_rng"], dev)
        log(f"[{rid}] resumed after epoch {st['epoch']} (best {st['best']:.4f})")
    log(f"[{rid}] setup {time.time() - t0:.0f}s: {len(fx_tr)} train / {len(fx_va)} valid fixtures, L={L}, device={dev}")
    for ep in range(st["epoch"] + 1, E):
        if st["bad"] >= HP["patience"]:
            break
        te = time.time()
        rng = np.random.default_rng([seed, ep])
        ds_tr.plan([draw_delta(f"{f['id']}|{ep}") for f in fx_tr], rng.integers(0, 2, len(fx_tr)))
        perm = rng.permutation(len(fx_tr))
        batches = [perm[i:i + HP["batch"]].tolist() for i in range(0, len(perm), HP["batch"])]
        model.train(); tl = []
        for b in loader(ds_tr, batches, workers):
            b = to(b, dev)
            for g in opt.param_groups:
                g["lr"] = HP["lr"] * lr_at(st["step"], total)
            loss = (((model(b) - b["y"]) / model.ysd) ** 2 * b["w"]).sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), HP["clip"]); opt.step(); st["step"] += 1
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(HP["ema"]).add_(pm.detach(), alpha=1 - HP["ema"])
            tl.append(loss.item())
        pv = predict(ema, ds_va, DELTAS, dev, workers)
        vl = float((((pv.yhat - pv.y) / float(model.ysd)) ** 2).mean())
        hv = headline(capture(pv, roles_va))
        better = hv > st["best"] + 1e-12 or (abs(hv - st["best"]) <= 1e-12 and vl < st["best_loss"])
        if better:
            st.update(best=hv, best_loss=vl, bad=0, best_state={k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}, best_epoch=ep)
        else:
            st["bad"] += 1
        st["epoch"] = ep
        st["history"].append(dict(epoch=ep, train_loss=float(np.mean(tl)), first_loss=float(np.mean(tl[:2])), last_loss=float(np.mean(tl[-2:])), valid_loss=vl,
                                  valid_headline=hv, seconds=round(time.time() - te, 1)))
        tmp = ck + ".tmp"
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), state=st, rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(dev) if dev.type == "cuda" else None), tmp)
        os.replace(tmp, ck)  # atomic: a crash mid-save never corrupts the last good checkpoint
        log(f"[{rid}] epoch {ep}: train {np.mean(tl):.4f} valid {vl:.4f} headline {hv:.4f}{' *' if better else ''} ({time.time() - te:.0f}s)")
    ema.load_state_dict({k: v.to(dev) for k, v in st["best_state"].items()})
    os.makedirs(PRED, exist_ok=True)
    pv = predict(ema, ds_va, DELTAS, dev, workers).assign(subset="valid")
    parts = [pv]
    if te_to and predict_test:  # test fixtures see all history dated < their own cutoff, encoded with this fold's vocabulary and scaling
        P2, M2, _ = load_tables()
        st2 = Store(P2, M2, vocab=store.vocab, norm=store.norm)
        end = (pd.Timestamp(te_to) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        fx_te = make_fixtures(st2, fixture_rows(P2, S, te_from, end, pad=False), M2)
        parts.append(predict(ema, Fixtures(st2, fx_te, L), DELTAS, dev, workers).assign(subset="test"))
    pd.concat(parts, ignore_index=True).to_parquet(os.path.join(PRED, f"{rid}.parquet"), index=False)
    torch.save(dict(state=st["best_state"], vocab=store.vocab, norm=store.norm, L=L, hp=HP, ybar=float(model.ybar), ysd=float(model.ysd),
                    dates=list(dates), seed=seed, arm=arm), os.path.join(rdir, "best.pt"))
    info = dict(run=rid, best_epoch=st.get("best_epoch"), best_valid_headline=st["best"], epochs_run=st["epoch"] + 1,
                history=st["history"], train_fixtures=len(fx_tr), valid_fixtures=len(fx_va), minutes=round((time.time() - t0) / 60, 1))
    json.dump(info, open(os.path.join(rdir, "done.json"), "w"), indent=1)
    log(f"[{rid}] done: best epoch {info['best_epoch']} headline {st['best']:.4f}")
    return rid


def package(runs, train_start, train_end):
    """Model artifact for evaluate.py: the seeds' best EMA weights (the ensemble = mean of their Ŷ)."""
    bs = [torch.load(os.path.join(CKPT, r, "best.pt"), map_location="cpu", weights_only=False) for r in runs]
    return dict(arch="cb", runs=runs, members=bs, train_start=train_start, train_end=train_end)


def load_members(model):
    out = []
    for b in model["members"]:
        m = CB([len(b["vocab"][c]) + 1 for c in CATS], b["L"], b["ybar"], b["ysd"], **b["hp"])
        m.load_state_dict(b["state"]); m.eval(); out.append(m)
    return out


@torch.no_grad()
def predict_rows(model, Q, info, P, M, bs=64):
    """Test time. Q: candidate rows [match_id, team, player_id, side], contiguous per fixture; info[match_id]: date, delta,
    match_type, gender, team_type, event, venue, sched_overs, balls_per_over. Returns the seed-mean Ŷ aligned with Q."""
    seed_all(0)
    preds = []
    for b, net in zip(model["members"], load_members(model)):
        st = Store(P, M, vocab=b["vocab"], norm=b["norm"])
        fx = []
        for mid, g in Q.groupby("match_id", sort=False):
            i = info[mid]
            cat = st.encode(pd.DataFrame([{c: i.get(c) for c in CATS}]))[0][:5]
            fx.append(dict(id=mid, day=int(days([i["date"]])[0]), month=int(str(i["date"])[5:7]), sched=float(i["sched_overs"]),
                           bpo=float(i["balls_per_over"]), cat=cat, side=g.side.to_numpy(np.int64),
                           pc=np.array([st.pid_code.get(p, -1) for p in g.player_id]), tc=np.array([st.team_code.get(t, -1) for t in g.team]),
                           y=np.zeros(len(g), np.float32), team=g.team.to_numpy(), player_id=g.player_id.to_numpy()))
        ds = Fixtures(st, fx, b["L"])
        ds.plan([info[f["id"]]["delta"] for f in fx], np.zeros(len(fx), np.int64))
        batches = [list(range(i, min(i + bs, len(ds)))) for i in range(0, len(ds), bs)]
        preds.append(np.concatenate([net(x).numpy() for x in loader(ds, batches, 0)]))
    return np.mean(preds, 0)


def smoke(device="auto", log=print):
    """200 F24 training fixtures, 1 epoch per arm: finite loss that decreases. Writes a smoke artifact for evaluate.py."""
    res = {}
    for arm in ARMS:
        d = SPLITS["F24"]
        rid = run_id(arm, d, 99)
        import shutil
        shutil.rmtree(os.path.join(CKPT, rid), ignore_errors=True)
        train_run(arm, d, 99, device, max_train=200, epochs=1, log=log, predict_test=False)
        h = json.load(open(os.path.join(CKPT, rid, "done.json")))["history"][0]
        res[arm] = dict(finite=bool(np.isfinite([h["train_loss"], h["valid_loss"]]).all()), first=h["first_loss"], last=h["last_loss"])
        os.remove(os.path.join(PRED, f"{rid}.parquet"))  # no metric from the smoke run is read or recorded
    model = package([run_id("CB64", SPLITS["F24"], 99)], SPLITS["F24"][0], "2023-12-31")
    import pickle
    os.makedirs("model_artifacts", exist_ok=True)
    pickle.dump(model, open("model_artifacts/cb_smoke.pkl", "wb"))
    return res


def overfit(device="auto", n=50, steps=1000, log=print):
    """T3: 50 fixtures at Δ = 0, no dropout: training loss must go to ≈ 0."""
    seed_all(0)
    dev = torch.device(resolve_device(device))
    P, M, S = load_tables()
    d = SPLITS["F24"]
    P = P[P.date < d[2]]; M = M[M.date < d[2]]
    store = Store(P, M, fold_end=d[1])
    fx = make_fixtures(store, fixture_rows(P, S, "2023-01-01", d[1], pad=False), M)[:n]
    ds = Fixtures(store, fx, 64)
    model = build_model(store, 64, fx)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0
        if isinstance(m, nn.MultiheadAttention):
            m.dropout = 0.0
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    b = to(collate([ds[i] for i in range(len(ds))]), dev)
    losses = []
    for i in range(steps):
        model.train()
        loss = (((model(b) - b["y"]) / model.ysd) ** 2 * b["w"]).sum()
        opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    res = dict(first=losses[0], last=losses[-1], ratio=losses[-1] / losses[0], pass_=losses[-1] < 0.05 * losses[0])
    log(f"[T3] overfit {n} fixtures: loss {losses[0]:.3f} -> {losses[-1]:.4f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "smoke", "overfit"])
    ap.add_argument("--arm", default="CB64", choices=list(ARMS))
    ap.add_argument("--split", default="F24", choices=list(SPLITS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "train":
        train_run(a.arm, SPLITS[a.split], a.seed, a.device, a.workers)
    else:
        res = smoke(a.device) if a.cmd == "smoke" else overfit(a.device)
        ok = all(r["finite"] and r["last"] < r["first"] for r in res.values()) if a.cmd == "smoke" else res["pass_"]
        res["pass"] = bool(ok)
        os.makedirs("data/audit", exist_ok=True)
        json.dump(res, open(f"data/audit/cb_{a.cmd}_audit.json", "w"), indent=1, default=float)
        print(json.dumps(res, default=float))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
