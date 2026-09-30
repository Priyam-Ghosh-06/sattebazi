# Decisions

Format: choice | alternatives | reason | config flag.

## Data / cleaning
- **Scope** = `info.match_type ∈ {T20, IT20}`, `dates[0] ≤ --train-end`. | README competition codes | JSON field is authoritative | `ingest.py --train-end`
- **The Hundred is kept** (cricsheet labels it `T20`, 5-ball overs, 100 balls/innings). Over checks and maidens use `info.balls_per_over` (a "maiden" is a full 5-ball set there); economy and SR use the same per-ball formulas and minimums (12 legal balls, 10 balls faced). | drop it | PS says "all T20 matches available on cricsheet", and cricsheet calls it T20; `balls_per_over` is a feature | `balls_per_over` column
- **Move gate G1**: the pre-move per-directory count failed (an unquoted path with a space), so G1 was verified after the move instead. Each year's file set equals the ID list in its README manifest, and totals equal the pre-move `ls` totals (male 16908, female 4039). See `data/audit/move_audit.json`.
- **G2 threshold**: an innings counts as violating if any invariant fails. Gate: < 2% of innings. Observed 0.95% (100 overs with more than bpo legal balls, 66 short overs mid-innings, 61 bowler-quota excesses: real umpire miscounts). The first run failed at 3.4% only because the check assumed 6-ball overs (The Hundred); the check was fixed, not the threshold.
- **Retired hurt / retired not out** are not dismissals (no duck, no wicket count). Retired out is a dismissal.
- **Replacement players** (`replacements.match[].in`) not already in `info.players` are appended to that team's XI list (participation +3, spec §2.4).
- Duplicate check: 0 duplicate IDs; context duplicates (same date, teams, venue) are logged in `ingest_audit.json`.

## Φ (user-confirmed 2026-09-29)
- **Six = +2 only**, not +1 boundary + 2. | +3 | User decision | `--six-also-boundary`
- **Duck applies to every dismissed player on 0**, including 0-ball run-outs. | only if faced a ball | User decision | `--duck-scope`
- Golden-match check: Φ equals an independent, deliberately naive per-delivery re-scorer (`tests/test_points.py::ref_score`) on 4 matches (latest IPL, latest men's T20I, latest women's match, a super-over match). Band edges are tested separately. This replaces "scored by hand" (time), and is noted as a deviation.

## Roles
- Rolling window 30 appearances. WK if ≥1 stumping in the window, **or** catch rate ≥ 1.2/match, **or** the most keeper-like player of their team group (score = any career stumping + catch rate + window stumpings ≥ 0.3). BOWL: bowling share of quota ≥ 0.25 and median bat position ≥ 8. AR: bowling share ≥ 0.25 and position < 8. BAT otherwise. UNK = no history. | spec thresholds (BOWL needs share ≥ 0.5, which left low-share tail-enders as BAT) | tuned on G4 only: 98.3% feasible → 99.33% (spec §4 allows this) | `roles.py --window --wk-catch-rate --wk-top-min --bowl-share --bowl-pos`
- The "top keeper per group" rule groups the XI in history and the squad at test time. That is legitimate: the group is an input at test time.
- Spot-check (2025 modal role): Kohli BAT, Dhoni WK, Bumrah BOWL, Hardik AR, Pant WK, Rashid BOWL, Buttler WK, SKY BAT, Jadeja AR, Siraj BOWL, Markram AR, de Kock WK, Boult BOWL, Maxwell AR, Cummins BOWL, Babar BAT, Rizwan WK, Perry AR, Healy WK.
- **Selection with UNK**: UNK players count as BAT for the composition constraints. | exclude | same for every model, rare in recent years

## Causality
- One mechanism for everything: a per-player "state after appearance" table joined with a strict `merge_asof` (`src/causal.py`). The time-travel test found a real bug on the first run: unstable sorts let same-date ties (doubleheaders) resolve arbitrarily. Fixed with stable sorts and a unique (team, date) timeline.

## Squads
- Ŝ = players who played for the team in the same (event, season); ±30 days when the event is missing (1.3% of international matches). Extended = Ŝ ∪ played for the team in the prior 365 days.
- **src/selection.py** instead of spec's `select.py`: `select` shadows the stdlib module that subprocess/multiprocessing import.

## Model (from experiments, docs/EXPERIMENTS.md)
- **Final model** = B0 + E14 context (opponent/own-squad pre-match strength + venue running mean). | B0 only | replicated +2.6 pts/match [+0.5, +4.7] | `train.py --no-context` to drop
- **μ loss L2** (selection needs the mean). L1/Huber: same captured, lower MAE, reported alongside. | Huber | no replicated gain
- **Training window starts 2019-01-01** in `run.sh` (features still use all history). The CLI accepts any `--train-start`, including the PS example 2000-01-01. | 2011, 2015 | no gain, slower
- **Kept out** (no replicated gain): rankers, ListNet, SPO+, monotone/recency/boundary weights, gated or orthogonal experts, component models, gender split.
- **Predicted points in the CSV** = expected points p·μ (what the XI maximises). μ and p are also written as separate columns.
- Backtest "random XI" reference = 11 × squad mean Φ (not an ILP over random scores): simpler, and only used as a sanity baseline.

## Layout and reproducibility (2026-09-29 evening)
- **Folders** = `src/` (code + `src/tests/`), `docs/`, `references/`, `output/` (`model_artifacts/`, `results/`, `experiments/`), `data/`. | PS example paths `model_artifacts/`, `results/` at the root | user asked for a minimal layout; `data/` stays because the PS requires raw and processed data in their own location | argparse defaults, `run.sh`
- **Deleted**: `_archive/` and `_claude_analysis/` (early exploration with a placeholder points table), `__pycache__/`. Duplicates of kept material only.
- **CSV line endings** fixed to `\n` (`evaluate.py`), so the output is byte-identical on Windows and Linux. Checked: the mock run from the new paths reproduces `mock_predictions.csv` exactly.
- **Compute**: sweeps may run on Kaggle; the final model is trained by `src/train.py` on CPU, deterministically. `config.json` records the git commit and a hash of `data/processed/`. | train the final model on GPU | viva re-run must be byte-identical
- **torch (CPU)** allowed for the NB/ADV rungs only.

## p-model (2026-09-29 evening; Scrutiny doc S1–S11)
- **Candidate sets are season-level pseudo-squads**; history bench tokens use the causal rule. Never build candidate sets from the causal rule (it makes every zero-history candidate a positive).
- **NB keeps independent logits + τ**; the constrained-choice layer is an ADV component (and an E17 objective for rung B), so its gain is measurable.
- **Capacities** (keeper, bowling option, batting depth, overseas) describe real selection; fantasy roles only constrain our output ILP.

## Baseline (2026-09-29, night)
- **Baseline = caveman CB** (`BASELINE.md` rev. 3): Transformer history encoder + set decoder over both squads, one MSE head on Y. | NB (CT-WKV + capacity binding + BCE/L2 + τ) | user decision: the baseline learns every mapping itself; NB's components become ADV rungs A0a/A0b so their gain is measured | `train.py --arch cb`

## Final run implementation (2026-09-30, SPEC §13; format: choice | alternative | reason | flag)
- **BASELINE §12 items closed at the proposed defaults** (SPEC §13 declares them closed): absence tokens = 365-day window after the first appearance for the fixture team; decoder over both squads; realised 3+Φ inside each token; A/B side swap augmentation; training sets padded with extended-squad candidates; D_Δ = 0.10·δ0 + 0.90·U[1, 450]. | the listed alternatives | pre-registered in §13 | `cb.HP`, `causal.draw_delta`
- **Training padding rule**: each team's candidate set is filled up to 20 with extended-squad candidates (appeared for the team in the prior 365 days, not in this season's pseudo-squad), most recent team appearance first. Validation and test never pad. | all extended candidates (median 27 per team) | 15+ test squads; bounded set size | `cb.HP["pad_to"]`
- **Test-time cutoff** c = min(fixture date, train_end + 1 day), Δ = fixture − c: history dated ≤ 2025-12-31 is all used, and a fixture inside the history window (the mock) is evaluated at Δ = 0. | c = train_end (drops matches on 31 Dec) | tokens and features are strictly before c | `causal.cutoff`
- **B0s** = B0 features computed as of c (the as-of key shifted), day gaps measured to the fixture (+Δ), Δ as an input; one Δ per training fixture (seed = fixture id); validation/test on the Δ grid. The fixture's opponent comes from the matches table (a fixture fact), not from history: gate G7 caught the history-derived version going empty under truncation. | — | G7 | `src/stale.py`
- **Early stopping on the explicit last 3 months** of the training window (SPEC §13.3), for B0s and CB. | last 10% of dates (old `split_valid`) | spec | `train.valid_start`
- **CB is trained on GPU for every split, including FINAL** (Kaggle, 2 × T4). Reproducibility for the viva = `evaluate.py` on CPU from the saved weights, byte-identical across runs (T5). | FINAL on CPU (D13) | a CPU step takes 9–15 s: ~2 h per epoch | `train.py --arch cb --device`
- **CB validation metric** = headline captured/oracle (mean over the Δ grid of per-fixture captured/oracle), ILP with fantasy roles as of each copy's c, EMA weights; ties by validation MSE. | — | SPEC §13.4 | `cb.headline`
- **Squad-change segment**: a team's previous-match player is missing from the XI, or the XI has a player with no earlier appearance *for that team*. | debut = first T20 overall | new signings are also squad changes | `final_run.squad_change`
- **T4 tolerance** 1e-4·σ_Y (float32 attention sums in a different order), not 1e-6. **T4b (A/B swap gap) and T4c (padding diagnostic) are not implemented**: reported as a gap.
- **R1 in practice**: an arm is excluded if a shared pipeline gate, its own gate (B0s: G7; CB: T1/T4/T7, smoke, T3) fails, a seed's predictions are missing, or its candidate sets differ from B0s's (T1b).
- **Kaggle driver** `src/kaggle_run.py`: stage markers (`output/stages/`), per-epoch CB checkpoints with atomic replace and resume (model, EMA, optimiser, RNG state), one job queue per GPU, one retry per run (it resumes). Raw data is read in place from `$RAW_DIR`. | — | the user asked for crash-safe runs

## ADV (2026-09-30, SPEC §14; user asked to build it)
- **A4 innings simulator not built**: it needs over-by-over bowler sequences, batting order and balls per batter, which ingest does not extract, plus seven likelihood heads. | a rushed version | a wrong simulator is worse than none; reported as the designed rung | —
- **Overseas potential not built** (needs nationality inference and gate G-p1). Potentials = keeper (0..3+) and bowling options (0..8+), one global table, gauge φ = 0 at keep {1, 2}, bowl {5, 6}. | per-competition tables | time; D2 | `adv.REF`
- **Bench tokens (A0a)** follow ADVANCED A0.1's season rule (appeared for the team earlier in the same or the previous season), not CB's 365-day window: the rule is cutoff-independent, so one encoder pass gives the state at every cutoff (A5 needs this). | CB's rule | design | `adv.Store.tokens`
- **Token embedding has no cutoff-relative inputs** (no age, no recency rank): time enters only through the CT-WKV decays and the readout at c (Prop. 2). | CB's age/rank features | required for A5 | `adv.Encoder`
- **Capacities** from appearances dated < c: keeper = ≥ 1 stumping in the last 30 or ≥ 1.2 catches per match, or the team's most keeper-like candidate (score ≥ 0.3); bowling option = ≥ 12 legal balls in ≥ half of the last 10; batting depth = median position ≤ 7 over the last 30. | — | D1 | `adv.Store.capacity`
- **n_XI**: observed XI size in training; at validation/test the XI size of the team's last match before c, clipped to 11–12. | always 11 | impact-sub competitions field 12 | `adv.Store.xi_size`
- **Loss P1** = choice NLL per selected player (A3) + L2 on μ over played rows, equal weights; μ head scaled by played-row mean/sd. | BCE (A0b alone) | A3 is the design's p model | `adv.Decoder.loss`
- **A5 pairs** come from the last 128 tokens before the fixture date; z(c) inside the gap is read from that sequence, so for players with > 128 tokens it can differ slightly from the z(c) the direct path sees. Pairs starting from a cold state are skipped (a cold state is never rolled). | exact states from full history | memory/time | `adv.train_run` P2
- **Budget smaller than CB's** (P1 ≤ 15 epochs, patience 3; 2 folds before FINAL): deadline. FINAL ADV runs only if ADV wins (SPEC §14.3).

## E24 matchups (2026-09-30, SPEC §15; user idea, THOUGHT_LOG Entry 10)
- **Graph = the observed batter × bowler delivery graph**, not a dense graph pruned afterwards: the same object, built directly. | dense all-pairs graph | ~10⁸ empty pairs | `matchups.ingest`
- **Generalisation by a rank-4 Poisson factorisation** (a_i + b_j + u_i·v_j, log link, exposure = balls faced), one for runs and one for dismissals, with Gaussian priors as shrinkage (sd 0.3 main effects, 0.15 interaction). Bowler "types" are learned, not labelled (Cricsheet has no bowling style). | raw head-to-head averages; a GNN | raw pairs are a few balls each; the factorisation is the minimal graph model | `matchups.K, SD_AB, SD_UV`
- **Monthly causal snapshots** (2017-10 → 2026-01), each fitted on deliveries dated < the snapshot, warm-started from the previous one (400 then 80 Adam steps); a row uses the latest snapshot ≤ its cutoff c (up to one month older than c). | refit per cutoff | cost | `matchups.snapshot_dates`
- **Recency weight** 0.5^(age / 730 days) on pair totals. | unweighted | form drifts | `matchups.HALF_LIFE`
- **Pre-toss expectation over the opponent squad** with weights share10 × lb10 (bowlers) and share10 × bf10 (batters), as of c: who is likely to play and to bowl/bat. | B0s's own p (circular inside one fit) | available as of c in the feature table | `matchups.matchup_block`
- **Wides are not balls faced**; no-balls are (as in ingest); a dismissal counts for the bowler only if bowler-credited and the striker is out.
