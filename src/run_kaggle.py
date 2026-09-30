"""All approaches end to end on a Kaggle GPU box (2x T4, 4 vCPU), resumable.

Run from the repo root (on Kaggle: copy the repo to /kaggle/working/sattebazi and attach the raw data). Raw data is read in place from $RAW_DIR
(default data/raw), never copied, so the saved notebook output stays small.
Every stage writes results/stages/<stage>.json when it finishes and is skipped next time; CB runs checkpoint every
epoch (model_artifacts/checkpoints/<run>/last.pt) and resume, so a crashed or timed-out session restarts where it stopped.
python src/kaggle_run.py [--stage all|<one of STAGES>] [--test-file PATH]
"""
import argparse, filecmp, json, os, queue, shutil, subprocess, sys, threading, time

PY = sys.executable
STAGES = ["pipeline", "stale", "b0s", "cb_checks", "cb_runs", "report", "adv_checks", "adv_runs", "adv_report", "adv_final",
          "advm_runs", "advm_report", "advm_final", "predict", "bundle"]
MARK = "results/stages"
LOGS = "results/logs"
TRAIN_START, TRAIN_END = "2019-01-01", "2025-12-31"
RAW = os.environ.get("RAW_DIR", "data/raw")


def sh(*args, log=None):
    """Run one pipeline command; stream its output (and tee to a log file). Raises on a non-zero exit (a failed gate)."""
    cmd = [PY, "-u", "-W", "ignore", *args]
    print("$", " ".join(args), flush=True)
    t = time.time()
    with open(log or os.devnull, "a") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="", flush=True); lf.write(line)
        p.wait()
    print(f"  ({time.time() - t:.0f}s, exit {p.returncode})", flush=True)
    if p.returncode:
        raise RuntimeError(f"failed: {' '.join(args)}")


def done(stage):
    return os.path.exists(os.path.join(MARK, stage + ".json"))


def mark(stage, t0, **info):
    os.makedirs(MARK, exist_ok=True)
    json.dump({"minutes": round((time.time() - t0) / 60, 1), **info}, open(os.path.join(MARK, stage + ".json"), "w"), indent=1)


# ---------------------------------------------------------------- stages
def pipeline():
    for d in ("data/interim", "data/processed", "data/audit", "model_artifacts", "results", "results/experiments"):
        os.makedirs(d, exist_ok=True)
    sh("src/shared/ingestion/ingest.py", "--raw-dir", os.path.join(RAW, "cricsheet"), "--train-end", TRAIN_END, "--workers", str(os.cpu_count() or 4))
    sh("src/shared/preprocessing/points.py"); sh("tests/test_points.py")
    sh("src/shared/preprocessing/roles.py"); sh("src/shared/preprocessing/squads.py")
    sh("src/shared/preprocessing/features.py", "--train-end", TRAIN_END)
    sh("tests/test_causality.py"); sh("tests/test_select.py")


def stale():
    sh("src/graph/preprocessing.py", "--train-start", TRAIN_START, "--train-end", TRAIN_END)
    sh("src/graph/preprocessing.py", "--g7")


def b0s():
    sh("src/graph/experiments.py", "folds")
    sh("src/train.py", "--approach", "graph", "--train-start", TRAIN_START, "--train-end", TRAIN_END,
       "--out", f"model_artifacts/model_b0s_{TRAIN_END}.pkl")


def cb_checks():
    sh("tests/test_baseline.py")
    sh("src/baseline/pipeline.py", "smoke", "--device", "auto")
    sh("src/baseline/pipeline.py", "overfit", "--device", "auto")
    out = "results/cb_smoke_mock.csv"
    sh("src/evaluate.py", "--model-path", "model_artifacts/cb_smoke.pkl", "--test-file", "data/mock/test_fixtures.json", "--output", out, "--registry-dir", os.path.join(RAW, "registry"))
    import pandas as pd
    d = pd.read_csv(out)
    need = {"match_date", "team1", "team2", "rank", "player", "role", "predicted_points"}
    assert need <= set(d.columns) and (d.groupby(["match_date", "team1", "team2"]).size() == 11).all(), "smoke CSV schema"


def gpus():
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def gpu_queue(jobs, retries=1, status_file="cb_status.json"):
    """jobs: [(run id, [script args])]. One worker thread per GPU; a failed run is retried (it resumes from its checkpoint)."""
    q = queue.Queue()
    for j in jobs:
        q.put(j)
    status, lock = {}, threading.Lock()
    os.makedirs(LOGS, exist_ok=True)

    def worker(dev):
        while True:
            try:
                rid, args = q.get_nowait()
            except queue.Empty:
                return
            ok = False
            for attempt in range(retries + 1):
                try:
                    sh(*args, "--device", dev, log=os.path.join(LOGS, rid + ".log")); ok = True; break
                except RuntimeError:
                    pass
            with lock:
                status[rid] = "done" if ok else "FAILED"
                json.dump(status, open(os.path.join(LOGS, status_file), "w"), indent=1)

    n = gpus()
    devs = [f"cuda:{i}" for i in range(n)] or ["cpu"]
    print(f"runs on {devs}", flush=True)
    th = [threading.Thread(target=worker, args=(d,)) for d in devs]
    [t.start() for t in th]; [t.join() for t in th]
    return status


def cb_runs():
    """12 runs (2 arms x 3 splits x 2 seeds). Folds first (they decide R2/R3), FINAL last."""
    return gpu_queue([(f"{arm}_{split}_s{seed}", ["src/baseline/pipeline.py", "train", "--arm", arm, "--split", split, "--seed", str(seed)])
                      for split in ("F24", "F25", "FINAL") for arm in ("CB64", "CB128") for seed in (0, 1)])


def adv_checks():
    sh("tests/test_refined.py")
    sh("src/refined/pipeline.py", "smoke", "--device", "auto")
    sh("src/refined/pipeline.py", "overfit", "--device", "auto")
    out = "results/adv_smoke_mock.csv"
    sh("src/evaluate.py", "--model-path", "model_artifacts/adv_smoke.pkl", "--test-file", "data/mock/test_fixtures.json", "--output", out,
       "--registry-dir", os.path.join(RAW, "registry"))
    import pandas as pd
    d = pd.read_csv(out)
    assert (d.groupby(["match_date", "team1", "team2"]).size() == 11).all(), "smoke CSV schema"


def adv_runs():
    """Refined runs: F25 then F24 (longer runs first, so the two GPUs finish together), seeds 0 and 1."""
    return gpu_queue([(f"ADV_{split}_s{seed}", ["src/refined/pipeline.py", "train", "--split", split, "--seed", str(seed)])
                      for split in ("F25", "F24") for seed in (0, 1)], status_file="adv_status.json")


def adv_report():
    sh("src/shared/evaluation/report.py", "adv")


def adv_final():
    """Only if ADV won (pre-registered two-year rule): the FINAL two-seed ADV runs, in parallel on the two GPUs."""
    dec = json.load(open("results/experiments/E23_adv/decision.json"))
    if dec["rule_choice"] != "ADV":
        print("ADV did not beat B0s: no FINAL ADV runs", flush=True)
        return {}
    return gpu_queue([(f"ADV_FINAL_s{seed}", ["src/refined/pipeline.py", "train", "--split", "FINAL", "--seed", str(seed)]) for seed in (0, 1)],
                     status_file="adv_status.json")


def advm_runs():
    """Refined + graph features (E25), warm-started from the refined runs: F25 then F24, seeds 0 and 1.
    Needs data/processed/matchup_snapshots.pkl (python src/graph/experiments.py ingest && ... snapshots)."""
    return gpu_queue([(f"ADVM_{split}_s{seed}", ["src/refined/pipeline.py", "train", "--graph-features", "--split", split, "--seed", str(seed)])
                      for split in ("F25", "F24") for seed in (0, 1)], status_file="advm_status.json")


def advm_report():
    sh("src/shared/evaluation/report.py", "advm")


def advm_final():
    """The final model, 3c: the FINAL two-seed runs (warm-started from the refined F25 runs), in parallel on the two GPUs."""
    return gpu_queue([(f"ADVM_FINAL_s{seed}", ["src/refined/pipeline.py", "train", "--graph-features", "--split", "FINAL", "--seed", str(seed)])
                      for seed in (0, 1)], status_file="advm_status.json")


def report():
    sh("src/shared/evaluation/report.py", "report")


def find_test_file(explicit):
    if explicit:
        return explicit
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f in ("test_fixtures.json", "test.json") and "mock" not in root:
                return os.path.join(root, f)
    return None


def predict(test_file):
    """The final model, 3c (refined + graph): package the FINAL runs (trained in advm_final; resumed if unfinished), then the
    mock set twice (determinism check) and the hidden set if it is attached."""
    final = f"model_artifacts/model_{TRAIN_END}.pkl"
    sh("src/train.py", "--train-start", TRAIN_START, "--train-end", TRAIN_END, "--out", final)
    print("final model: 3c (refined + graph)", flush=True)
    a, b = "results/mock_predictions.csv", "results/mock_predictions_rerun.csv"
    for o in (a, b):  # T5: two runs, byte-identical CSVs
        sh("src/evaluate.py", "--model-path", final, "--test-file", "data/mock/test_fixtures.json", "--output", o, "--registry-dir", os.path.join(RAW, "registry"))
    same = filecmp.cmp(a, b, shallow=False)
    json.dump({"pass": same, "model": "3c"}, open("data/audit/t5_determinism_audit.json", "w"), indent=1)
    assert same, "T5: evaluate.py is not deterministic"
    tf = find_test_file(test_file)
    if tf:
        sh("src/evaluate.py", "--model-path", final, "--test-file", tf, "--output", "results/hidden_predictions.csv", "--registry-dir", os.path.join(RAW, "registry"))
    else:
        print("no hidden test file found: attach it and rerun --stage predict (or pass --test-file)", flush=True)
    return {"model": "3c", "hidden_test_file": tf}


def bundle():
    """One zip with everything worth downloading (checkpoints' best weights included, last.pt excluded)."""
    dst = "/kaggle/working/results_bundle" if os.path.isdir("/kaggle/working") else "results_bundle"
    tmp = dst + "_dir"
    shutil.rmtree(tmp, ignore_errors=True)
    for d in ("results", "data/audit"):
        if os.path.isdir(d):
            shutil.copytree(d, os.path.join(tmp, d), ignore=shutil.ignore_patterns("pred"))
    for f in os.listdir("model_artifacts"):
        if f.endswith(".pkl"):
            os.makedirs(os.path.join(tmp, "model_artifacts"), exist_ok=True); shutil.copy(os.path.join("model_artifacts", f), os.path.join(tmp, "model_artifacts", f))
    ck = "model_artifacts/checkpoints"
    for r in os.listdir(ck) if os.path.isdir(ck) else []:
        for f in ("best.pt", "done.json"):
            p = os.path.join(ck, r, f)
            if os.path.exists(p):
                os.makedirs(os.path.join(tmp, os.path.dirname(p)), exist_ok=True); shutil.copy(p, os.path.join(tmp, p))
    shutil.make_archive(dst, "zip", tmp)
    shutil.rmtree(tmp)
    print("wrote", dst + ".zip", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES)
    ap.add_argument("--test-file", default=os.environ.get("TEST_FILE"))
    ap.add_argument("--force", action="store_true", help="rerun the stage even if it is marked done")
    a = ap.parse_args()
    todo = STAGES if a.stage == "all" else [a.stage]
    for s in todo:
        if done(s) and not a.force and s not in ("predict", "bundle"):
            print(f"== {s}: done earlier, skipped", flush=True); continue
        print(f"\n== {s}", flush=True)
        t0 = time.time()
        if s == "cb_runs":
            if not done("cb_checks"):
                print("CB checks did not pass: skipping CB runs (R1 excludes CB)", flush=True); continue
            st = cb_runs()
            mark(s, t0, runs=st)
            if any(v != "done" for v in st.values()):
                print("some CB runs failed: they are excluded by R1; rerun this stage to resume them", flush=True)
        elif s in ("cb_checks", "adv_checks"):
            try:
                globals()[s](); mark(s, t0)
            except (RuntimeError, AssertionError) as e:  # the arm fails R1 and is excluded
                print(f"{s} FAILED ({e}): the arm is excluded, continuing", flush=True)
        elif s == "adv_runs":
            if not done("adv_checks"):
                print("ADV checks did not pass: skipping ADV runs", flush=True); continue
            mark(s, t0, runs=adv_runs())
        elif s in ("adv_report", "adv_final") and not done("adv_runs"):
            print(f"{s}: ADV runs missing, skipped", flush=True); continue
        elif s == "adv_final":
            mark(s, t0, runs=adv_final())
        elif s == "advm_runs":
            if not done("adv_runs"):
                print("advm_runs: it warm-starts from the refined runs, which are missing: skipped", flush=True); continue
            mark(s, t0, runs=advm_runs())
        elif s in ("advm_report", "advm_final") and not done("advm_runs"):
            print(f"{s}: ADVM runs missing, skipped", flush=True); continue
        elif s == "advm_final":
            mark(s, t0, runs=advm_final())
        elif s == "predict":
            mark(s, t0, **predict(a.test_file))
        else:
            globals()[s](); mark(s, t0)


if __name__ == "__main__":
    main()
