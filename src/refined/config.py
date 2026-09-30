"""Refined approach (ADV) settings: the fixed training budget, the saturating counters of the choice layer, paths."""


NEG = -1e9


TAU = 30  # A5: one transition step per month


HP = dict(d=64, L=128, layers=2, heads=4, drop=0.1, edrop=0.05, lr=2e-3, betas=(0.9, 0.98), wd=0.05, warmup=0.05, floor=0.02,
          clip=1.0, ema=0.999, batch=32, epochs=15, patience=3, p2_epochs=6, p2_lr=1e-3, p3_epochs=6, p3_patience=2, p3_lr=3e-4,
          p4_epochs=30, p4_lr=1e-3, lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), max_xi=13,
          a1=True, a2=True, a3=True, a5=True, a6=True, mu=False, name="ADV")


# refined + graph (ADVM): the 10 graph features of approach 3b appended to every candidate's decoder input
MU_COLS = ["x_mu_bat_run", "x_mu_self_bat_run", "x_mu_bowl_run", "x_mu_self_bowl_run", "x_mu_bat_run_int", "x_mu_bowl_run_int",
           "x_mu_bat_out", "x_mu_self_bat_out", "x_mu_bowl_out", "x_mu_self_bowl_out"]


SNAPS = "data/processed/matchup_snapshots.pkl"


# warm-started from the refined run of the same fold and seed, so a short fine-tune; the student is dropped (it failed in E23)
ADVM = dict(mu=True, name="ADVM", a6=False, epochs=4, patience=2, lr=5e-4, p2_epochs=2, p3_epochs=2)


KEEP_MAX, BOWL_MAX = 3, 8  # counters saturate: '3 or more' keepers, '8 or more' bowling options


REF = {"keep": (1, 2), "bowl": (5, 6)}  # gauge fix: φ = 0 at these counts, so the tables are identified and comparable


CKPT = "model_artifacts/checkpoints"


PRED = "results/experiments/E23_adv/pred"


def pred_dir():
    return "results/experiments/E25_advm/pred" if HP["mu"] else PRED


N_SC = 4  # sched overs, balls/over, is fixture team, log gap to previous token
