# SPEC v1: Fantasy XI pipeline, data cleaning, Φ, loss experiments

Status: design agreed at the idea level on 2026-09-29. Claude Code implements it, **stage by stage, with a gate between stages**.
Nothing downstream runs if a gate fails. Every "DECISION" below is a default that can be switched by a flag and is also logged in `docs/DECISIONS.md` with its reason.
Every "ASK" is a question for the organisers' group. Until they answer, the default stands.

Deadline: 30 Sep 2026, 23:59 IST. Priorities are marked **[MUST] / [SHOULD] / [COULD]**. Do all MUSTs end-to-end before any SHOULD.

---

## 0. The system being modelled (read this first)

For match *m*, team *T* has a squad *S* (15+, given at test time, never given in training). It also has a playing XI *X ⊂ S* (given in training, unknown at test time).

    Y_im = 1[i ∈ X_m] · (3 + Φ(E_im))        E_im = ball-level events involving i
    E[Y_im] = p_im · μ_im
    p_im = P(i ∈ X_m | S_m, F_<m)   (appearance)
    μ_im = E[3 + Φ | i ∈ X_m, F_<m]  (performance given playing)

The decomposition is exact: it is the law of total expectation, with no independence assumption.
F_<m is everything strictly before match m's date. Selection maximises Σ p·μ subject to composition rules (linearity of expectation, so teammate correlations don't matter for the expected total).

Φ is **known** (the PS scoring table). It is nonlinear: highest-only milestones, threshold-gated SR/economy bands, and +25-sized wicket jumps. So E[Φ] ≠ Φ(E[·]). For example, the batting milestone term satisfies
E[bonus] = 5·[P(R≥25)+P(R≥50)+P(R≥75)+P(R≥100)], which depends on the tails.

---

## 1. Folder layout  [MUST]  (PS: raw / processed / artifacts / training / evaluation must be separate)

Reorganise with `mv` only. **Never delete, never edit raw JSON contents.**

```
D:\traditional ML PS\          (revised 2026-09-29 evening: minimal layout)
  CLAUDE.md  README.md  requirements.txt  run.sh
  src/          ingest, points, roles, squads, causal, features, context, train, evaluate, selection, backtest
    tests/      test_points.py, test_causality.py, test_select.py
  docs/         PIPELINE.md, BASELINE.md, ADVANCED.md, SPEC.md, DECISIONS.md, EXPERIMENTS.md, THOUGHT_LOG.md
  references/   problem statement, LuPTS paper
  output/       model_artifacts/, results/, experiments/ (E<k>_<name>/{config.json, metrics.json}, LOG.md)
  data/
    raw/cricsheet/{male,female}/<year>/<match_id>.json (+ README.txt per year)   read-only
    raw/registry/people.csv  names.csv  SOURCE.txt
    interim/      matches.parquet, innings_audit.parquet, player_match_raw.parquet
    processed/    player_match.parquet (with Φ components), roles.parquet, squads.parquet, features.parquet
    audit/        <stage>_audit.json  (gate reports)
    mock/         mock test fixtures + truth
```

Gate G1: file count under `data/raw/cricsheet` equals the count before the move. Compare JSON-file counts per gender per year before and after, and write them to `data/audit/move_audit.json`.

run.sh: if `data/raw/cricsheet` is empty, download from cricsheet.org (verify the correct zip URLs on the site first) and re-split by gender/year. Otherwise skip. Then run the stages in order and stop on the first gate failure (`set -e`, each stage exits non-zero on gate fail).

---

## 2. Ingest and cleaning  [MUST]  → `src/ingest.py`

Parallelise across files (multiprocessing, deterministic output order: sort by `date, match_id`). Cache to parquet. Re-runs read the cache unless `--rebuild`.

### 2.1 Scope filter
- Keep `info.match_type ∈ {T20, IT20}` (PS: "all T20 and T20I matches"). Drop ODI/ODM/Test/MDM. Log counts per dropped type.
- Keep `info.dates[0] ≤ --train-end` (default 2025-12-31). This is **the first leakage guard**. Assert it again at feature time.
- Keep gender, `team_type` (club/international), `event.name`, `season`, `venue`, `city`, `overs` (scheduled), toss and outcome as columns. Nothing is dropped silently.

### 2.2 Identity
- Use **registry IDs** (`info.registry.people`) as the player key everywhere. Names are display-only.
- Gate: every name appearing in `players` or in deliveries (batter, bowler, non_striker, fielders, player_out) resolves to an ID in that match's registry. Report unresolved counts, which should be 0.
- Build `name_variants` from `data/raw/registry/names.csv` for test-time resolution (§8).
- Team names: output the list of distinct team strings per `event.name` with first and last dates to `DATA_AUDIT.md`. **DECISION**: do not auto-merge franchise renames. Apply a hand-written `team_aliases.json` for renames only after verifying in data that it is the same franchise (e.g. Delhi Daredevils→Delhi Capitals, Kings XI Punjab→Punjab Kings, Royal Challengers Bangalore→Royal Challengers Bengaluru). Only team-level features use the alias.

### 2.3 Match-level cleaning (each becomes a flag column, then a decision)
| Case | Detect | DECISION (default) |
|---|---|---|
| Duplicate match | same `match_id` in 2 files, or same (date, teams, venue) | keep one, log both paths |
| Abandoned, zero deliveries | `innings` empty | exclude from μ-training (no performance info); keep for p (XI was named). Flag `no_play` |
| No result with play | `outcome.result == "no result"` and deliveries > 0 | keep; flag |
| Reduced overs (DLS/rain) | `info.overs` < 20 or `target.overs` present | keep; feature `scheduled_overs`, `balls_available` |
| Super over | innings with `super_over: true` | **exclude those deliveries from Φ** (PS rule) |
| Team list ≠ 11 | `len(players[team])` | 12 = impact/concussion sub (see 2.4); other sizes → flag and inspect |
| Parse error | exception | log path; must be 0 or explained |

### 2.4 Replacements (impact player / concussion sub)
From `deliveries[].replacements.match[]` (`in`, `out`, `reason`).
- **DECISION**: every player in `info.players[team]` gets participation +3. That includes a 12th listed player who is an impact or concussion substitute, and also the player they replaced. Both accrue their own events. Flag `is_sub_in`, `is_sub_out`. Config switch: `subs_get_participation`.
- **ASK**: does the organisers' "official Playing XI" include impact subs?

### 2.5 Delivery-level invariants (gate G2; report violations, don't auto-fix)
1. Per innings: Σ(runs.total) equals Σ(batter) + Σ(extras). Check each delivery: `total == batter + extras`.
2. Legal balls per over ≤ 6 (`> 6` → flag; umpire miscount exists in real cricket). Count overs with < 6 legal balls not at innings end.
3. Wickets per innings ≤ 10.
4. Every batter and bowler belongs to the correct team's `players` list (batting team vs fielding team).
5. Balls per bowler ≤ ceil(scheduled_overs/5)·6 legal balls (4 overs in a 20-over game); flag excess.
6. A fielder in a dismissal must be in the fielding XI **or** have `substitute: true`.
7. `dates[0]` monotone vs `match_id` is **not** assumed.

G2 passes if violation rates are below a small threshold (report exact numbers). Rows attached to violating innings are flagged, not dropped. An experiment later checks whether dropping them matters.

### 2.6 Per-player-match raw counts → `interim/player_match_raw.parquet`
One row per (match, team, player in XI list):
`runs, balls_faced (legal: not wides; no-balls count), fours, sixes, dismissed, dismissal_kind, bat_position (order of first appearance as striker or non-striker), legal_balls_bowled, runs_conceded (batter runs + wides + noball runs; byes/legbyes excluded), dots, wickets_bowler_credited, lbw_bowled, maidens, catches, stumpings, runout_direct, runout_indirect, is_sub_in, is_sub_out` + match context columns.

Substitute fielders (`substitute: true`) are not in the XI, so they get **no** fielding points (PS: non-players score 0). Count how many dismissals this affects.

---

## 3. Φ, the points function  [MUST]  → `src/points.py`

Pure function: raw counts → **components** `{participation, bat_runs, bat_boundary, bat_milestone, bat_duck, bat_sr, bowl_dots, bowl_wkts, bowl_lbwb, bowl_haul, bowl_maiden, bowl_econ, field}` and `total`. Store the components, not just the total. The component models and the error decomposition need them.

| Item | Rule (PS) | Interpretation / DECISION | ASK? |
|---|---|---|---|
| Participation | +3 if in playing XI | §2.4 | yes (subs) |
| Runs | +1 per run | runs off the bat only (no extras) | |
| Boundary | +1 per four | four = `runs.batter==4` and not `non_boundary` | |
| Six | +2 per six | **DECISION**: six gets +2 only (not +1 +2). Flag `six_also_boundary` | yes |
| Milestone | 25:+5, 50:+10, 75:+15, 100:+20, highest only | thresholds are ≥ | |
| Duck | −2 | dismissed with 0 runs, any player, any balls (incl. 0-ball run-out). Flag `duck_scope` | yes |
| Strike rate | min 10 balls faced | SR=100·runs/balls_faced. Bands (half-open to close PS gaps): >170:+5; (150,170]:+3; [130,150]:+1; (70,130):0; [60,70]:−1; [50,60):−3; <50:−5 | |
| Dot ball | +1 | legal ball (no wide/no-ball) with 0 runs charged to bowler (leg-byes/byes count as dot). Flag `dot_def` | yes |
| Wicket | +25 excl. run out | credited kinds: bowled, caught, caught and bowled, lbw, stumped, hit wicket. Not: run out, retired hurt/out, obstructing the field, etc. | |
| LBW/Bowled | +5 each | | |
| Haul | 3w:+5, 4w:+10, 5w+:+15, highest only | | |
| Maiden | +6 | one bowler, 6 legal balls, 0 runs charged to bowler; wides/no-balls break it; byes/leg-byes don't. Overs split between two bowlers → no maiden | |
| Economy | min 2 overs (12 legal balls) | econ=6·runs_conceded/legal_balls. Bands: <5:+5; [5,6):+3; [6,7]:+1; (7,10):0; [10,11]:−1; (11,12]:−3; >12:−5 | |
| Catch | +6; 3-catch bonus +4 (once, at ≥3) | caught-and-bowled → bowler gets the catch | |
| Stumping | +10 | | |
| Run out | direct +10, indirect +5 | **heuristic**: exactly 1 non-substitute fielder listed → direct (+10 to them); ≥2 → +5 to each listed. Cricsheet has no direct-hit field; state this in the report | |

Tests (`src/tests/test_points.py`), gate G3:
- One test per band **boundary** (SR 150.00, 150.01, 170.00, 170.01, 130, 70, 60, 50; econ 5, 6, 7, 7.01, 10, 11, 11.01, 12, 12.01; 9 balls vs 10 balls; 11 vs 12 legal balls bowled).
- Milestone highest-only: 99→+15, 100→+20, not +50.
- 3 **golden matches** scored by hand from the scorecard in the JSON (one IPL, one T20I, one women's). Φ must match exactly. Write the hand computation in `src/tests/golden/*.md`.
- Invariant: Σ components == total, for every row.
- Distribution sanity (into DATA_AUDIT.md): mean/median/p90/p99/min/max of total by inferred role; fraction of zero-contribution rows beyond participation.

---

## 4. Roles  [MUST]  → `src/roles.py`

Roles are latent and must be inferred **causally**, using only matches before t. Cricsheet has no wicket-keeper field.

- WK evidence: stumpings (only a keeper stumps). Per (player, team, window): `stump_rate`. **DECISION**: WK if the player has ≥1 stumping in the last N=20 appearances **or** a career stumping share ≥ θ. Tune θ with the gate below.
- Batting and bowling share: `bowl_share` = mean legal balls bowled per appearance / 24; `bat_share` from median batting position and balls faced.
  - BOWL: bowl_share ≥ 0.5 and median bat_pos ≥ 8
  - AR: bowl_share ≥ 0.25 and median bat_pos ≤ 7
  - BAT: otherwise (and not WK)
  - Thresholds live in config; tune only against gate G4.
- Cold start (no history): role unknown. **DECISION**: at test time use the provided role if the test file has one (**ASK**). Otherwise use a label marked "unknown" and exclude it from selection unless needed for feasibility.

Gate G4:
1. In ≥ 99% of historical matches, the actual 22 players admit at least one feasible fantasy XI under the composition rules (WK 1–4, BAT 3–6, AR 1–4, BOWL 3–6, ≥1 per team). Infeasible matches point to broken role inference.
2. Per-XI role histogram looks sane (≈1 WK per team, etc.). Report it.
3. Spot-check 20 well-known players' roles by hand in DATA_AUDIT.md.

---

## 5. Pseudo-squads (fixes the missing negatives for p)  [MUST]  → `src/squads.py`

Problem: training has only positives (the XI). Benched squad members never appear. Workaround: reconstruct Ŝ.

- Club matches: Ŝ(team, event, season) = everyone who appeared for that team in that event and season.
- International: Ŝ = everyone who appeared for that team in the same `event.name` (series/tournament). If there is no event name, use a ±30-day window.
- Rows: every (match, i ∈ Ŝ) with `played ∈ {0,1}`.
- Costs, to be stated in the report:
  1. Ŝ ⊆ S. Squad members who never played are missing, and those are exactly the p≈0 players. Test squads will contain them, so the model extrapolates. Mitigation: a feature `apps_for_team_in_window == 0`.
  2. Ŝ uses appearances later in the same season or series. This is defensible **only** because squads are announced before the season or series. It is still membership, not performance, and no performance data from the future enters. Write this in the leakage audit.
- Variant [SHOULD]: "extended squad" = Ŝ ∪ {players who appeared for the team in the prior 365 days}. This moves closer to the 15+ test squads. Compare in E6.

Gate G5: distribution of |Ŝ| (report median, p10, p90; expect ~13–20 for leagues), mean played share, and the fraction of matches where |Ŝ| < 11 (must be ≈0; otherwise reconstruction is broken).

---

## 6. Features  [MUST core / SHOULD rest]  → `src/features.py`

**Causality rule**: a feature for match m uses only matches with `date < date(m)`. Strict inequality: same-day doubleheaders are excluded from each other.

Core [MUST]:
- Performance history per player (all T20+IT20, split by gender naturally): count n, EWMA of `total` and of each **component group** (bat, bowl, field) at half-lives {3, 10, 30} appearances, EB-shrunk career mean `(n·ȳ + κ·μ_role)/(n+κ)`, days since last appearance.
- Opportunity: median bat position (last 10), mean balls faced, mean legal balls bowled, fraction of appearances bowling ≥ 12 balls.
- Appearance (for p): share of the team's last {5, 10} matches played, played in the team's previous match, streak length, days since last appearance for this team, role rank within Ŝ by recent appearance share.
- Context: gender, team_type, match_type, scheduled_overs.

SHOULD:
- Venue mean total points per role (causal, shrunk to the global mean).
- Opponent: points conceded per role by the opposition (causal, shrunk).
- Competition level: event-level mean, and later (COULD) league fixed effects identified through players who play in several leagues.
- Kalman local-level filter: fit λ = q/r per stat by walk-forward CV, gain K = (−λ + √(λ²+4λ))/2. Output the filtered mean and posterior variance as features. The EWMA half-life sweep (E4) tests whether this adds anything.

Gate G6 (`src/tests/test_causality.py`), the **time-travel test**:
for 200 random (player, match) rows, recompute features from a dataset truncated to `date < date(m)` and assert equality with the full-pipeline features. Also assert max(source_date) < date(m) for every row via a stored `feat_source_max_date` column. This is the leakage audit's main evidence.

---

## 7. Models and loss  [MUST]  → `src/train.py`

Model family: LightGBM (traditional ML, fast, deterministic with `seed`, `deterministic=True`, `force_row_wise=True`, fixed `num_threads`). Fix all seeds (numpy, random, lightgbm) at the top of each script.

- **μ-model** (rows: played==1). Target = Φ total (incl. participation).
- **p-model** (rows: pseudo-squad). Binary logistic on `played`. Then **sum-to-XI calibration** per (match, team): find τ with Σ_i σ(s_i − τ) = n_XI (11, or 12 if subs are enabled), solved by bisection. This is the minimum-KL correction of independent Bernoullis to the correct expected XI size.
- Score for selection = p·μ.

Loss is the central experiment (E5). Theory for the report:
- L2 → conditional mean, which is what the selection objective Σ E[Y] needs.
- L1 → conditional median, which is MAE-optimal. For the unconditional squad-level target, the median is 0 whenever p < 0.5.
- The target is zero-inflated, heavy-tailed and quantised (wicket jumps of 25–30), so L2 is outlier-sensitive. Huber is a compromise.
- Therefore: **select on the L2/mean model; report MAE both for the mean model and for an L1 model**, and state which metric suits fantasy (§9).

---

## 8. Selection and test-time adapter  [MUST]  → `src/select.py`, `src/evaluate.py`

- ILP (scipy `milp`): maximise Σ x_i·score_i, Σx=11, role bounds, ≥1 per team, x binary. Deterministic tie-break: add ε·rank(player_id). Output the players sorted by predicted points, with role and predicted points (PS CSV: match date, team1, team2, 11 players, roles, predicted points).
- Infeasible (e.g. unknown roles) → relax by assigning unknowns their most likely role, log it.
- `test_fixtures.json` schema is **unknown** (ASK). Build an adapter with one function `load_fixtures(path) -> list[{date, team1, team2, squad1: [names], squad2: [names], roles?: {}}]`. Build a **mock fixtures file** from late-2025 matches (pseudo-squads as squads, names as display strings) and run evaluate.py end-to-end on it.
- Name resolution: exact match on registry name → match in `names.csv` variants → `difflib.get_close_matches` restricted to players who appeared for that team → unresolved (cold start: role prior μ, low p). Log the resolution rate per fixture. It is a known failure mode, so print it loudly.

Tests (`src/tests/test_select.py`): output always satisfies every constraint on 1,000 random squads; determinism (two runs → identical CSV).

---

## 9. Evaluation protocol  [MUST]  → `src/backtest.py`

Walk-forward by year: for Y ∈ {2022, 2023, 2024, 2025}: train on matches ≤ Y−1, test on Y. Feature computation is causal anyway. The model is retrained per fold. Use 2025 as the headline fold.

Metrics, reported on **two populations**: (a) played rows, (b) squad rows with Y=0 for benched.
- Player level: MAE, RMSE, **within-match Spearman** (rank correlation among the match's candidates, averaged over matches).
- Team level: **points captured** = actual points of the chosen XI; **oracle** = best feasible XI from the squad in hindsight; report captured/oracle, and captured − (random valid XI).
- Error decomposition: (i) hindsight XI + model μ (removes p-error), (ii) squad + model p·μ. The gap between them measures the cost of not knowing the XI.
- Segments: gender, club/international, top events, role.

Report justification (write in the report): points captured / within-match rank correlation match the task (choose 11 of ~30). MAE is dominated by per-player noise and by the zero mass of benched players.

---

## 10. Experiments (pre-registered; each writes output/experiments/E<k>_*/)

| ID | Priority | Question | Measure | Expectation / decision rule |
|---|---|---|---|---|
| E0 | MUST | Data audit | counts from §2 gates | all gates pass |
| E1 | MUST | Is Φ correct? | golden matches, boundary tests | exact match |
| E2 | MUST | Noise ceiling | CV of Φ per player; ICC = between-player var / total var of Φ (played rows, per role) | Sets what MAE is achievable. Low ICC means rank metrics matter more |
| E3 | MUST | Baseline ladder | role mean; EB career mean; last-5 mean; EWMA; each × (p=appearance-share proxy) | model must beat the best baseline on captured and Spearman |
| E4 | SHOULD | Memory length | EWMA half-life sweep vs Kalman-fitted λ | pick by walk-forward Spearman |
| E5 | MUST | Loss | L2 vs L1 vs Huber (δ tuned) for μ | select-model = best captured; report MAE of each |
| E6 | MUST | p-model | proxy vs logistic on Ŝ vs logistic on extended Ŝ; ± sum-to-11 calibration | log-loss, Brier, captured |
| E7 | SHOULD | Components vs direct | sum of component-group models vs direct Φ model | captured, Spearman |
| E8 | SHOULD | Pooling | one model vs split by gender / club-intl | captured per segment |
| E9 | SHOULD | Feature-group ablation | drop each group (form, opportunity, appearance, context, venue, opponent) | for the report's explainability section |
| E10 | COULD | Direct ranking | LightGBM lambdarank, query = match, NDCG@11 | only if time remains |
| E11 | SHOULD | Cleaning sensitivity | train with vs without G2-violating innings | shows whether the cleaning decisions matter |

`output/experiments/LOG.md`: one line per run → `date | ID | config hash | key metrics | verdict`.

---

## 11. Open questions for organisers (ASK list)
1. Are impact/concussion subs in the "official Playing XI" (+3)?
2. Does a six also earn the +1 boundary bonus?
3. Does the duck apply to all players? Does a 0-ball run-out count?
4. Dot ball: do leg-byes/byes count as dots for the bowler?
5. test_fixtures.json schema: are names or IDs given? Are roles given? Is gender given?
6. Which competitions and genders are in the hidden set?
7. Test release "after 30 Sep" vs submission deadline 30 Sep 23:59: which one holds?
8. Hidden-set scoring: actual points of our XI? Compared to what?

---

## 12. Revision 2026-09-29 (evening): staleness, choice objective, team totals  [supersedes where it conflicts]

Context: `docs/PIPELINE.md`. Every backtest in `docs/EXPERIMENTS.md` used fresh history (Δ = 0) on smoke samples. At test time history stops at $c$ = 2025-12-31 and the fixtures are later. This section adds what rung B needs before its numbers mean anything, and the pre-registered experiments that decide the submission.

### 12.1 Staleness-aware features  [MUST]  → `src/features.py`, `src/causal.py`
- Every feature row gets a cutoff $c$ and staleness $\Delta = \text{date} - c$. The as-of join key becomes $c$ instead of the match date: the existing `causal.asof` with the key shifted. Δ is an input feature.
- Features that only exist at Δ = 0 are redefined as of $c$: "played the team's previous match" becomes "played the team's last match before $c$"; "days since last appearance" is measured from that appearance to the fixture date (so it includes Δ). Team-level features use the team's state at $c$.
- Training: one Δ per training fixture from $\mathcal D_\Delta$ (PIPELINE D3), seeded by fixture ID. Evaluation: fixed grid Δ ∈ {0, 90, 180, 270, 365}.
- **Gate G7** (extends G6): for 200 random (row, Δ) pairs, recompute features from data truncated at $c$ and require exact equality; assert `feat_source_max_date < c`.

### 12.2 Candidate-set guard  [MUST]
Candidate sets stay the season pseudo-squad (`squads.py`). Test: no candidate set is built from the causal bench rule (Scrutiny S8).

### 12.3 Capacities  [MUST before 12.4]  → `src/roles.py`
Add a capacity vector per player-date alongside the fantasy role, using the rules in PIPELINE D1: keeper, bowling option, batting depth, and overseas (the IT20 team the player appeared for before $c$; none → domestic). Gates: fraction of historical XIs with 0 inferred keepers ≈ 0; bowling options per XI concentrated around 5–7 (report the histogram); **G-p1** zero overseas-cap violations per competition, with the cap = maximum observed count.

### 12.4 Constrained-choice objective for p  [SHOULD]  → `src/train.py --p-objective choice`
LightGBM custom objective, one group per (fixture, team). With current raw scores $u$, compute exact marginals $\pi$ by the DP in `ADVANCED.md` §2.2 (numpy, vectorised across groups); gradient $\pi_i - y_i$, Hessian $\pi_i(1-\pi_i)$ (the diagonal of the PSD covariance). The potentials φ (≈ 30 numbers after gauge fixing) are updated by one Newton step on the convex NLL every 10 boosting rounds (block-coordinate descent). Stages, each its own experiment: (a) exact conditional Bernoulli, no potentials; (b) + keeper; (c) + bowling options and overseas. Cost grows with the state space (12 → 48 → 2,160 states per group); stage (c) runs on Kaggle if CPU is too slow.

### 12.5 Team-total diagnostics  [SHOULD]  → `src/backtest.py`
Per (fixture, team) on the validation folds compare $\sum_i p_i\,E[\text{legal balls bowled}_i\mid\text{plays}]$ with the realised team total (≤ 6 × scheduled overs), and $\sum_i p_i\,E[\text{credited wickets}_i\mid\text{plays}]$ with the realised bowler-credited wickets. A bias above 5% (PIPELINE D12) triggers a per-team rescaling of the opportunity components, the same move as sum-to-XI applied to overs and wickets.

### 12.6 Experiments (pre-registered)
| ID | Priority | Question | Measure | Decision rule |
|---|---|---|---|---|
| E16 | MUST | B0 under staleness | captured/oracle, hindsight gap, log-loss at each Δ; full 2024 and 2025 folds | none: this is the honest baseline every rung must beat |
| E17 | SHOULD | choice objective for p (12.4 a/b/c) | log-loss overall and on the squad-change segment (G-p2); captured | adopt a stage if squad-change log-loss improves and captured is not significantly worse |
| E18 | SHOULD | team-total bias (12.5) | relative bias of expected overs and wickets per team | rescale only if bias > 5% |
| E19 | MUST | full-fold re-run of the adopted B (+E14 context) at the test-matching Δ | as E16 | replaces the smoke headline in the report |
| E20 | SHOULD | cold-start prior (Scrutiny S9) | truncated-geometric debut hazard; padding test (add k synthetic zero-history candidates to mock fixtures) | adopt if calibration of p for zero-history candidates improves |
| E21 | COULD | NB and ADV ladders | `BASELINE.md` §9, `ADVANCED.md` §10 | pre-registered rule, PIPELINE §4 |

### 12.7 Additional organiser questions
9. May `evaluate.py` read Cricsheet matches after 2025-12-31 as inputs (not training data)?
10. Will the organisers validate our XI with their own role labels?

---

## 13. The final run (pre-registered, single shot)  [MUST]  — supersedes §12.6 where they conflict

Written 2026-09-30, 02:00 IST, before any CB code exists. **This run is the first and last test of the models.** Everything that decides the submission is fixed here: arms, splits, seeds, budget, metrics and rules. Nothing below may be changed after a test-year number has been seen. Every design question in `BASELINE.md` §12 and `ADVANCED.md` §12 is closed.

### 13.1 Arms
| arm | what | spec |
|---|---|---|
| **B0s** | LightGBM B0 + E14 context, **staleness-aware** (§12.1), p by logistic + τ, μ by L2 | this doc §7, §12.1 |
| **CB64** | caveman baseline, history length L = 64 | `BASELINE.md` rev. 3.1 |
| **CB128** | caveman baseline, history length L = 128 | same, L = 128 |

Not in this run (reported as the designed ablation ladder, not as results): ADV rungs A0a–A6, E17 (choice objective for B), E18 (team-total rescaling), E20 (debut-hazard prior). B0s is required regardless of CB: the submitted pipeline must handle Δ > 0, and every earlier backtest was at Δ = 0.

### 13.2 Staleness
Training: Δ ~ 0.10·δ₀ + 0.90·U[1, 450] days, drawn per fixture (B0s: once per fixture, seed = hash(fixture); CB: afresh every epoch, seed = hash(fixture, epoch)). Inputs are rebuilt as of c = date − Δ, every source dated < c (gate G7 / T1).
Validation and test: every fixture is evaluated at **each** Δ ∈ {0, 90, 180, 270, 365} (deterministic, 5 copies). **Headline = the equal-weight mean over these five Δ** (the evaluation analogue of the flat training mixture; the true test Δ is unknown, O2).

### 13.3 Splits (history tokens/features always use all data dated < c, from 2005)
| split | train fixtures | validation (early stopping, L choice) | test (reported once) |
|---|---|---|---|
| F24 | 2019-01-01 → 2023-09-30 (5,376) | 2023-10-01 → 2023-12-31 (300) | 2024 full year (1,776) |
| F25 | 2019-01-01 → 2024-09-30 (7,028) | 2024-10-01 → 2024-12-31 | 2025 full year (1,801) |
| FINAL | 2019-01-01 → 2025-09-30 (8,849) | 2025-10-01 → 2025-12-31 | none: this is the submitted model |

Candidate sets everywhere: season pseudo-squads (`in_base`). CB training additionally pads with extended-squad candidates (BASELINE §6); validation and test never pad.

### 13.4 Budget (fixed; no tuning, no grid)
- CB: d = 64, 2 encoder + 2 decoder layers, 4 heads, FFN 4d, dropout 0.1 / 0.05 (embeddings), AdamW lr 1e-3, β (0.9, 0.98), wd 0.05 on matrices, 5% warm-up + cosine to 2%, clip 1.0, EMA 0.999, batch 32 fixtures, ≤ 20 epochs, patience 4 on validation headline captured/oracle (EMA weights), ties by validation loss.
- **Seeds {0, 1} per (arm, split); the arm's prediction is the mean of the two seeds' Ŷ.** The two-seed ensemble is the model that is evaluated and submitted. Seed-to-seed spread is reported.
- B0s: existing deterministic LightGBM settings, seed 0 (no bagging, so seed variance is nil).
- Runs: CB 2 arms × 3 splits × 2 seeds = 12 GPU runs (Kaggle, parallel notebooks); B0s 3 CPU runs.

### 13.5 Order of execution
1. **Build and gate (CPU, local).** CB data builder and model; tests T1, T1b, T1c, T4 (permutation), T4b (team swap, exact), T5 (determinism of evaluate.py), T7 (cold start); G7 for B0s. Any failure: fix the code and rerun the *test* (tests are not the experiment).
2. **Smoke (mechanics only).** 200 F24 training fixtures, 1 epoch, each CB arm, then `evaluate.py` on the mock fixtures. Pass = finite, decreasing loss and a CSV with the PS schema. No metric from the smoke run is read or recorded.
3. **T3**: overfit 50 fixtures (training loss → ≈ 0). Correctness only.
4. **The run**: all 12 CB runs and 3 B0s runs, configs frozen and hashed into `config.json` before launch.
5. **Evaluate once** with `src/backtest.py`: every arm, split and Δ; write the tables in §13.6.
6. **Apply §13.7**, then produce the submission CSV (mock now; hidden file when released).

**Time gate.** If the CB FINAL models are not trained and gated by **20:00 IST on 30 Sep**, B0s is submitted and CB is reported with whatever folds finished (clearly marked incomplete).

### 13.6 What is measured (all arms, F24 and F25 test years, per Δ and at the headline)
- Team level: captured, oracle, **captured/oracle**, captured − random XI (11 × squad mean Φ), % of picked players who played.
- Player level: MAE and RMSE on squad rows and on played rows (PS requirement), within-fixture Spearman.
- B0s only: log-loss and Brier of p. CB only: team-total bias Σ_i Ŷ_i − realised team fantasy total (mean and sd); padding diagnostic T4c.
- Segments: gender, club/international, IPL, and **squad-change fixtures** (a player who played the team's previous match is absent, or the XI contains a debutant).
- Paired bootstrap (2,000 resamples of fixtures) of per-fixture captured differences, each fixture's value averaged over the five Δ: CB64 − B0s, CB128 − B0s, CB128 − CB64, per test year.

### 13.7 Decision rules (pre-registered)
- **R1 gates.** An arm that fails any §13.5 gate is excluded.
- **R2 choice of L.** Between CB64 and CB128, pick the higher **validation** headline captured/oracle summed over F24 and F25 validation windows. If they differ by < 0.005, pick CB64 (cheaper, fewer ways to overfit). The test-year comparison CB128 − CB64 is reported, not used to choose.
- **R3 submission.** Submit the chosen CB (FINAL, two-seed ensemble) **only if** the 95% bootstrap interval of (CB − B0s) captured lies entirely above 0 in **both** F24 and F25. Otherwise submit B0s (FINAL). A tie is reported as a tie.
- **R4 no second look.** No rerun, retune or reselection after test numbers are seen. A bug found afterwards is reported; it is fixed and rerun only if it is in the submission path and changes the CSV, and then both results are reported.

### 13.8 Expectations written before running
- CB ties B0s (E5's direct squad-level L2 LightGBM tied B0: −1.8 [−5.8, +2.2]).
- CB128 − CB64 within ±0.005 captured/oracle; if anything CB128 is better at Δ ≥ 270, where recent tokens are gone and long memory is all that is left.
- All arms lose captured/oracle as Δ grows; the Δ = 0 → 365 drop is larger for B0s than for CB (CB is trained with Δ as input over the whole range; B0s sees one Δ per fixture).
- CB's team-total bias is within ±10% (it is never told that about 11 play).

### 13.9 Outputs
`output/experiments/E22_final/`: `config.json` (git commit, data hash, every setting above), `metrics.json` (§13.6 tables), `bootstrap.json`, `per_fixture.parquet` (fixture × arm × Δ × seed-ensemble scores), `decision.md` (R1–R4 applied, one line each). One line in `LOG.md`. The report's results section is built only from these files.

---

## 14. ADV run (pre-registered 2026-09-30, 11:30 IST, before any ADV code ran on real folds)  [user request]

Context: E22 (§13) is finished and its test numbers are known: B0s submitted; CB128 − B0s = −5.3 [−8.3, −2.5] (2024), −1.9 [−4.7, +0.9] (2025). This run is **post-hoc to E22** and is reported as such. ADV's settings below are fixed now, without looking at any ADV number.

### 14.1 What is built (`src/adv.py`, `ADVANCED.md`)
A0a CT-WKV encoder (L = 128, 2 layers, d = 64) · A0b factorised utility / μ heads with capacity binding · A2 FiLM · A1 squad attention on μ only · A3 exact choice layer (keeper and bowling-option potentials, global, gauge-fixed) · A5 monthly transition (P2) + decoder fine-tuned on rolled-forward states (P3) · A6 student, λ ∈ {0, .25, .5, .75, 1}. **Not built: A4 innings simulator** (needs a ball-by-ball ingest stage); overseas potential (nationality inference).
Closed open items (defaults from PIPELINE §7): D1 capacities = keeper by the `roles.py` WK rule, bowling option = ≥ 12 legal balls in ≥ half of the last 10 appearances, batting depth = median position ≤ 7 over the last 30; D2 potentials keeper + bowling options, one global table; D5 A1 on μ only; D8 monthly grid; D9 no encoder fine-tune in P3. n_XI = the observed XI in training; at validation/test, the XI size of the team's last match before c (11 or 12).

### 14.2 Runs and budget (fixed)
F24 and F25 (splits and Δ grid as §13.2–13.3), seeds {0, 1}; ensemble = mean of the two seeds' Ŷ. AdamW lr 2e-3 (P1), 3e-4 (P3), 1e-3 (P2, P4); P1 ≤ 15 epochs patience 3; P2 6 epochs; P3 ≤ 6 epochs patience 2; P4 30 epochs per λ on a cached feature set. Each seed's final predictor (direct / roll-out / student) is the best on its **validation** headline. FINAL runs only if 14.3 says ADV.

### 14.3 Decision rule
Submit ADV (FINAL, two seeds) only if ADV's gates pass (T1, T2, T4, T6, T7, T11, T14, T16, smoke, T3) and the 95% paired-bootstrap interval of ADV − B0s captured per fixture (averaged over the five Δ) lies entirely above 0 in **both** 2024 and 2025. Otherwise B0s stays. No rerun or retune after the numbers are seen (R4).

### 14.4 Expectations written before running (ADVANCED §10.2)
ADV ties B0s; ADV − CB128 ≥ 0. A3 vs τ (same weights, `direct_a3 − direct_tau`): +0 to +2 captured, larger on squad-change fixtures. Roll-out (A5) − direct: ≈ 0 at Δ = 0, positive at Δ ≥ 270 only if the monthly dynamics are learned; could reverse. Student (A6) ≥ the worse of direct and roll-out.

### 14.5 Outputs
`output/experiments/E23_adv/`: `decision.md`, `decision.json`, `metrics.json`, `bootstrap.json`, `per_fixture.parquet`, `pred/ADV_<split>_s<seed>.parquet` (columns per predictor), one line in `LOG.md`.

---

## 15. E24: batter–bowler matchup factorisation (pre-registered 2026-09-30, 15:45 IST, before any matchup number exists)  [user idea, THOUGHT_LOG Entry 10]

Post-hoc to E22 (test years seen); reported as such.

### 15.1 Model
Delivery-level pairs (batter i faced bowler j; super overs excluded; wides are not balls faced) aggregated per match: legal balls faced n, runs off the bat r, bowler-credited dismissals of that batter w. Two Poisson log-linear factorisations with rank k = 4:
  E[r_ij] = n_ij · exp(β + a_i + b_j + u_i·v_j),   E[w_ij] = n_ij · exp(β' + a'_i + b'_j + u'_i·v'_j),
fitted on pair totals with recency weight 0.5^(age/730 days); Gaussian priors (L2): sd 0.3 on a, b; sd 0.15 on u, v (shrinkage of thin pairs). Full-batch Adam, fixed seed, CPU.
**Snapshots**: the first day of every month from 2017-10-01 to 2026-01-01; snapshot s uses only deliveries dated < s. A row with cutoff c uses the latest snapshot ≤ c. **G8**: every snapshot's source dates < s; every row's snapshot ≤ c; one snapshot refitted from truncated data is identical.

### 15.2 Features (prefix x_mu_, added to B0s's context)
For candidate i in a fixture copy, the opponent squad's weights: bowlers w_j = share10_j · lb10_j, batters w_i = share10_i · bf10_i (both as of c).
- batting: log Σ_j w_j e^{b_j + u_i·v_j} / Σ w_j for runs and for dismissals, the same with the interaction only (u_i·v_j), and own a_i, a'_i;
- bowling: log Σ_i w_i e^{a_i + u_i·v_j} / Σ w_i for runs conceded and wickets, the interaction-only runs term, and own b_j, b'_j.

### 15.3 Run and rule
B0s + x_mu_ (same LightGBM settings, same splits F24/F25, same Δ grid, same ILP) vs B0s, paired bootstrap of per-fixture captured averaged over Δ. **Adopt** (B0s+M becomes the B0s-family submission) only if G8 passes and the 95% CI of (B0s+M − B0s) lies entirely above 0 in both 2024 and 2025. If ADV and B0s+M both pass their rules, submit the one with the larger mean gain over B0s summed over 2024 and 2025. Expectation: +0 to +3 captured per fixture; the interaction-only features rank below the opponent-quality ones in gain importance.
Outputs: `output/experiments/E24_matchups/` (decision.md, metrics.json, bootstrap.json), one LOG line.
