# The pipeline

Revision 2026-09-29 (evening). This is the pipeline to present: one spine, shared by every model, and one swappable model stage.
The spine (ingest → Φ → labels → squads → causal features → exact selection → CSV) is **built and gated**.
The model stage has three rungs. Rung B (LightGBM) is built; CB (caveman baseline) and ADV are designed (`BASELINE.md`, `ADVANCED.md`) and not yet implemented.
Section 7 lists everything that still has to be defined before code.

---

## 1. The problem as a mathematical object

A fixture $m$ gives a date, two teams, venue and event, and two squads $S_A, S_B$ (15+ each). History is Cricsheet T20/IT20 dated $\le c = $ 2025-12-31.
For each candidate $i \in S_A \cup S_B$ the realised fantasy score is

$$
Y_i = \mathbb 1[i \in X]\,\big(3 + \Phi(E_i)\big),
$$

where $X$ is the (unknown, pre-toss) set of players who actually play and $E_i$ are $i$'s ball-level events. $\Phi$ is the official points table: known, deterministic, nonlinear (highest-only milestones, threshold bands, haul bonuses).

We choose $x \in \{0,1\}^{|S_A|+|S_B|}$ to maximise the expected team score under the fantasy rules $\mathcal C$ (WK 1–4, BAT 3–6, AR 1–4, BOWL 3–6, at least one per team, $\sum x = 11$). By linearity of expectation and the law of total expectation, exactly:

$$
\max_{x \in \mathcal C}\ E\Big[\sum_i x_i Y_i\Big] = \max_{x\in\mathcal C}\ \sum_i x_i\, p_i\,\mu_i,\qquad
p_i = P(i \in X \mid S, \mathcal F_{\le c}),\quad \mu_i = E[3+\Phi \mid i \in X, \mathcal F_{\le c}].
$$

No independence assumption is used: correlations between teammates cannot change an expectation of a sum. The whole modelling problem is therefore two conditional quantities, $p$ (who plays) and $\mu$ (points if he plays), estimated from information available before the toss and before the cutoff.

**Staleness.** At test time the history stops at $c$ while the fixture is on date $t$. Write $\Delta = t - c$. Every input is computed *as of* $c$, never as of the day before the match. If the hidden fixtures are real matches after the release, $\Delta \ge 273$ days. Every model is trained and evaluated with simulated $\Delta$.

---

## 2. The flow

```
Cricsheet JSON (T20 / IT20, dated <= 2025-12-31)
   |  ingest: scope filter, registry IDs, abandoned / reduced / super-over / subs, delivery invariants      [G1, G2]
   v
player-match table + Phi components (official table, every ambiguity a flag)                                [G3]
   |  labels as of each date: fantasy roles (for our output) and capacities (for real selection)            [G4]
   v
candidate sets = season pseudo-squads (with benched negatives); history tokens obey the causal rule         [G5]
   |  as-of state at cutoff c, staleness Delta = date - c                                                   [G6, G7]
   v
MODEL STAGE   p_i = P(i in XI | squad)      mu_i = E[points | plays]
   |   rung B   LightGBM: B0 -> staleness-aware -> constrained-choice objective
   |   rung CB  caveman: raw tokens -> Transformer history encoder -> set decoder over both squads -> E[Y] (one MSE head)
   |   rung ADV CB + CT-WKV encoder + p/mu capacity decoder + choice layer + squad attention + innings simulator + temporal privileged training
   v
exact ILP over both squads: max sum x_i p_i mu_i  s.t. fantasy composition rules
   v
CSV: date, team1, team2, 11 players in descending p*mu, role, predicted points (= p*mu)
```

---

## 3. Stage by stage

| # | stage | what it does | key decision (DECISIONS.md) | gate | status |
|---|---|---|---|---|---|
| 1 | ingest | JSON → matches, innings, player-match counts | registry IDs; T20/IT20 only; super overs excluded; replacements get +3 | G1 file counts, G2 invariants (0.95% innings flagged) | built |
| 2 | Φ | counts → 13 components + total | six = +2 only; duck for any dismissal on 0; run-out direct/indirect by fielder count | G3 band edges + golden matches | built |
| 3 | labels | fantasy role per player-date (WK/BAT/AR/BOWL); capacities (keeper, bowling option, batting depth, overseas) | causal rolling window; one keeper per team group | G4: 99.33% of real XIs feasible | roles built; capacities to define (§7) |
| 4 | squads | season pseudo-squad Ŝ with `played ∈ {0,1}` | candidates season-level; history tokens causal | G5 | built |
| 5 | features / tokens | as-of state, EWMA bank {3,10,30}, opportunity, appearance, opponent/venue context | one as-of mechanism (`causal.py`) | G6 time-travel test; G7 staleness time-travel (new) | built at Δ = 0; staleness to build |
| 6 | model | $p$, $\mu$ | §4 | per-rung tests | B built; CB/ADV designed |
| 7 | selection | exact ILP (`scipy.milp`), deterministic tie-break | score = $p\mu$ | 1,000 random squads always feasible; two runs identical | built |
| 8 | adapter + CSV | fixture file → name resolution → CSV | resolution rate printed per fixture; CSV newline fixed | mock end-to-end, byte-identical reruns | built (schema unknown) |

---

## 4. The model stage: three rungs

| rung | $p$ (who plays) | $\mu$ (points if plays) | status | doc |
|---|---|---|---|---|
| **B** | LightGBM logistic on pseudo-squads + sum-to-XI shift τ → (next) constrained-choice objective | LightGBM L2 on played rows, EWMA bank + opponent/venue | B0 built; staleness and choice objective specified | SPEC §12 |
| **CB** | not separated: one head learns $E[Y] = p\mu$ by MSE (Prop. CB-1); competition only through set attention | same head | designed (rev. 3) | `BASELINE.md` |
| **ADV** | A0b factorised heads, then the exact constrained-choice layer over the squad | squad attention + innings simulator through the exact Φ; temporal privileged training | designed | `ADVANCED.md` |

**Why the $p$-model is the centre of the design.** Independent per-player logits learn a *propensity* tied to the squad they were measured in: a backup keeper who sits behind a first-choice keeper for three seasons is pushed towards $p \to 0$, and when the first-choice keeper is absent a single calibration shift τ cannot move probability within the role. The constrained choice model

$$
P(X \mid S) \propto \exp\Big(\sum_{i\in X} u_i + \sum_q \phi_q\big(n_q(X)\big)\Big)\,\mathbb 1\big[|X| = n_{XI}\big]
$$

learns a squad-independent pecking order $u$; competition lives in the normaliser and in the count potentials $\phi_q$ over capacities (keepers, bowling options, overseas). A within-role ranker is its zero-temperature limit; the finite temperature gives the calibrated $p$ the optimiser needs. Every counter raised against this is recorded with its answer in the Scrutiny doc (S1–S11).

**Submission rule (pre-registered).** Each rung is compared with the staleness-aware B0 by paired bootstrap of per-fixture captured points, on full 2024 and 2025 folds, at the test-matching Δ. The submitted model is the best rung on 2024, confirmed on 2025. If nothing beats B, B is submitted and the ladder is the ablation study.

---

## 5. Validation

- Walk-forward: train on years < Y, test on Y ∈ {2024, 2025}, full years (the existing numbers are smoke samples of 400–700 matches per year).
- Staleness grid Δ ∈ {0, 90, 180, 270, 365} days, headline at the Δ that matches the test file.
- Metrics: captured points and captured/oracle (the task), within-match Spearman, MAE and RMSE on played and squad rows (PS requirement), log-loss and Brier for $p$, log-loss on matches that follow a squad change (where the choice model must win).
- Leakage audit: date filter at ingest, re-asserted at features; one as-of join for every feature; time-travel test on 200 random rows; staleness time-travel test (features at cutoff $c$ use only matches $< c$).

## 6. Evidence so far (smoke samples, Δ = 0)

| result | value |
|---|---|
| B0 captured/oracle | 0.684 |
| + opponent/venue context (adopted) | 0.691, replicated +2.6 pts/match [+0.5, +4.7] |
| knowing the real XI (hindsight) | 0.743: at most ~5 pp left for $p$ at Δ = 0 |
| learned $p$ vs appearance-share proxy | +40 pts/match |
| every structural change to training (rankers, listwise, experts, SPO+, monotone, recency) | tie or worse |
| ICC of Φ | ≈ 0.13: per-match points are mostly noise |
| mock end-to-end, model ≤ 2025-09-30, 40 fixtures Oct–Dec 2025 | 89% of picked players played |

The only stale evaluation so far is the mock (Δ ≤ 92 days, 40 fixtures). E16 gives the honest numbers.

---

## 7. To be defined

### From the organisers
| id | question | what it changes |
|---|---|---|
| O1 | `test_fixtures.json` schema: names or IDs, roles given, gender given? | the adapter; cold-start rate |
| O2 | fixture dates, and does the release "after 30 Sep" really come after the deadline? | the Δ distribution; whether A5 matters |
| O3 | may `evaluate.py` read Cricsheet matches after 2025-12-31 as *inputs*? | if yes, Δ ≈ 0 and most of the staleness work is moot |
| O4 | are impact / concussion subs in the "official Playing XI" (+3)? | $n_{XI}$ = 11 or 12 per competition |
| O5 | will our XI be validated with the organisers' own role labels? | whether the ILP needs a margin from the role bounds |
| O6 | which competitions and genders are in the hidden set? | segment weighting; per-competition potentials |

### From you (design decisions)
| id | decision | current proposal |
|---|---|---|
| D1 | capacity labels: exact rules for keeper, bowling option, batting depth; overseas from the IT20 team a player appeared for before $c$ | keeper = existing WK rule (stumpings, catch rate, one per team group); bowling option = ≥ 12 legal balls in ≥ half of last 10 appearances; batting depth = median position ≤ 7; overseas as proposed. Byes cannot identify the keeper (team-level in Cricsheet) |
| D2 | which count potentials, and global vs per competition | keeper, bowling options, overseas; per competition where ≥ 500 team-matches, else global |
| D3 | staleness distribution for training | 10% at 0, 40% Uniform[1, 270], 50% Uniform[271, 450] days; replace by the empirical test Δ once O2 is known |
| D4 | cold-start prior for squad members who never played (Scrutiny S9) | truncated-geometric debut hazard; mixture only if the padding test shows bias |
| D5 | squad attention (A1) on the $p$ path, or $\mu$ only | $\mu$ first; $p$ as a separate ladder rung |
| D6 | innings simulator training: teacher forcing only, or scheduled sampling | teacher forcing (it is LuPTS's separate fitting); P3 handles roll-out mismatch |
| D7 | toss inside the simulator: bat-first = 0.5 or learned from team/venue | learned, with 0.5 as the ablation |
| D8 | temporal roll-forward grid: monthly or one step per team match | monthly (match count in the gap is unknown at test) |
| D9 | fine-tune the encoder end-to-end in ADV phase P3 | no (reopens state collapse) |
| D10 | Monte Carlo trajectories per candidate | 4096; check against the distribution of score gaps at the ILP boundary |
| D11 | ILP margin for role robustness (O5) | only if O5 says yes: prefer 2 keepers when scores tie within ε |
| D12 | team-total rescaling (bowling overs, wickets) | adopt only if E18 shows a bias above 5% |
| D13 | final-model reproducibility | sweeps on Kaggle; final `train.py` run on CPU, deterministic |
| D14 | submission rule | as in §4 |
| D15 | folder paths vs the PS's literal example command (`model_artifacts/`, `results/` at root) | keep `output/`; README gives the exact commands. Say if you want root-level copies |

---

## 8. Folder layout

```
run.sh  README.md  requirements.txt  CLAUDE.md
src/          code: ingest, points, roles, squads, causal, features, context, train, evaluate, selection, backtest (+ src/tests/)
docs/         PIPELINE (this), BASELINE, ADVANCED, SPEC, DECISIONS, EXPERIMENTS, THOUGHT_LOG
references/   problem statement, LuPTS paper
output/       model_artifacts/, results/, experiments/ (+ LOG.md)
data/         raw/ (read-only), interim/, processed/, audit/, mock/   <- required by the PS (raw and processed data separated)
```
