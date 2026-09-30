"""Baseline (CB) settings: history lengths, token fields, the fixed training budget (SPEC §13.4), paths, seeding."""

import numpy as np
import os
import random
import torch


ARMS = {"CB64": 64, "CB128": 128}


NUM = ["runs", "balls_faced", "fours", "sixes", "dismissed", "bat_position", "legal_balls_bowled", "runs_conceded", "dots",
       "wickets", "lbw_bowled", "maidens", "catches", "stumpings", "runout_direct", "runout_indirect"]


CATS = ["match_type", "gender", "team_type", "event", "venue", "opp", "team", "bat_first", "won"]  # first 5 = fixture token


N_NUM, N_SC = len(NUM) + 1, 5  # counts + realised points; sched overs, balls/over, is fixture team, log age, log gap


HP = dict(d=64, heads=4, layers=2, drop=0.1, edrop=0.05, lr=1e-3, betas=(0.9, 0.98), wd=0.05, warmup=0.05, floor=0.02,
          clip=1.0, ema=0.999, batch=32, epochs=20, patience=4, min_freq=20, pad_to=20, absence_days=365)


CKPT = "model_artifacts/checkpoints"


PRED = "results/experiments/E22_final/pred"


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
