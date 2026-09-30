# Thought log

The problem-solver's own reasoning, in order: each idea, why it came up, and where it landed after scrutiny.
Entries are appended, never rewritten. A "Resolution" records what survived discussion; the idea itself is kept even when it was dropped.

---

## Entry 0: Framing (before 2026-09-29)

- The data has no fantasy points and no roles. A points function Φ(deliveries) and role inference have to be built from ball-by-ball data.
- Expected points factorise as E[Y] = P(plays | squad) × E[points | plays].
- Under MAE the optimal point prediction is the median, which is 0 for any player with P(plays) < 0.5. So the number reported and the number used for selection should differ: select on expectation.
- With no credit budget and no captain multiplier, selection is a small 0/1 integer program that can be solved exactly.

---

## Entry 1: Bottlenecks, one by one (2026-09-29, early morning)

Method: identify the bottlenecks myself first, before reading any prior analysis.

### 1. Appearance probability
- **Thought**: only the playing XI is given in training. Modelling "will this player be in the team" is ill-posed without knowing who *could* have played.
- **Thought**: it is also time-dependent. A 2003 player won't be in a 2025 team.
- **Thought**: maybe the problem should be bypassed through preprocessing or through how the model is defined, and maybe my framing itself is wrong.
- **Resolution**: the bypass instinct was right. At test time the squad is an *input*, so the target is P(in XI | in squad), not P(plays for team). That also removes the time issue. The real gap is that training data has no squads. Fix: reconstruct pseudo-squads (everyone who played for the team in that season or series), which turns the problem into supervised learning with negatives. This has two stated costs: it misses squad members who never played, and it uses later-in-season membership, which is defensible because squads are pre-announced. Also: per-team probabilities must sum to 11 (bisection on a shared offset), and same-role players compete for slots.

### 2. What drives performance
- **Thought**: three factors (opponent, match setting, event type/time) plus noise. The setting is a *different kind* of information from the opponent, not just another feature: its direction or class differs.
- **Thought**: time and a player's progression matter, and the trend may differ by match type.
- **Thought**: noise (weather, the player's day) is nearly impossible to model.
- **Thought**: two models in an ensemble. One holds the player's longer-term "semantic" identity, the other the immediate match context (opponents, candidate set, venue).
- **Thought**: time should be baked into how the model is trained or fed, not added as a feature. Feed previous hidden states forward, or use something autoregressive or batch-based.
- **Resolution**:
  - "Different kind of information" formalised: in a log-link model the venue term multiplies *everyone's* rate (it conditions the whole distribution), while opponent is a side-specific term, and player×opponent is a pairwise interaction that is mostly unidentifiable from this data.
  - A fourth factor was missing: opportunity (batting position, overs quota). Points ≈ opportunity × efficiency.
  - "Hidden state fed forward" formalised as a filter. Under a random-walk-plus-noise model, the optimal (Kalman) estimate is an EWMA with gain K = (−λ+√(λ²+4λ))/2, λ = signal/noise variance ratio. So memory length is something to fit from data, not choose by hand. Its outputs (filtered form + uncertainty) feed a GBDT.
  - Noise is quantified rather than modelled: estimate the ceiling (between- vs within-player variance).
  - Φ is known and nonlinear (highest-only milestones, threshold bands), so expected points depend on the tails of runs and wickets, not their means.

### 3. Player priority and synergy
- **Thought**: build implicit features beyond the explicit ones, such as player synergy, perhaps as a knowledge graph.
- **Thought**: a ranker that correctly prioritises elite players; possibly an open-weight LLM reranker.
- **Thought**: ArcFace-style angular-margin loss.
- **Resolution**:
  - Expected team score is a sum of individual expectations, so correlation-type synergy does not change the expected-optimal XI. What remains useful is *conditional-mean* synergy (who else plays changes your batting position), as features.
  - The knowledge-graph idea survives in one rigorous form: the player–league graph's connectivity is what makes league-strength effects identifiable.
  - Elite players are the easy end of the ranking (many samples, P(plays) ≈ 1). The costly errors are around candidates 8–14.
  - LLM reranker dropped: an LLM trained after the data cutoff carries post-cutoff knowledge (leakage), and the rules restrict LLMs to the text overview.
  - ArcFace dropped: optimal-XI membership depends on the other candidates and is not an intrinsic class. The rigorous version of the idea is listwise ranking within a match (LambdaMART), kept as an optional experiment.

### 4. Preprocessing
- **Thought**: traditional ML depends heavily on preprocessing and restructuring, which is what makes the model learn what you want it to learn. It needs a solid plan.
- **Resolution**: taken forward as the first build stage: scope filter to T20/T20I, identity via registry IDs, match-state handling (abandoned, reduced overs, super overs, substitutes), delivery-level invariant checks, Φ with every ambiguity written down, causal role inference.

### 5. Hierarchical mixture of experts
- **Thought**: the mapping is complicated, so use a hierarchical MoE. Each expert captures one aspect of performance, weighted by hierarchy. Force the experts to be orthogonal, e.g. through eigenvectors of each expert's output sensitivity, with a constraint on their parameters.
- **Resolution**: the orthogonality idea has an exact classical form, the functional ANOVA decomposition (components orthogonal in L²(P)). That form is also what makes attribution unambiguous. Orthogonal parameters don't give orthogonal functions, and the real factors are correlated (home teams at home venues). The practical version kept: a GBDT with interaction constraints over feature groups (a structured additive model whose groups act as the experts). Neural MoE was dropped given the data size and noise level.

---

## Entry 2: Move to empirical testing (2026-09-29, ~04:00)

- **Thought**: stop theorising and test how well a pipeline actually performs.
- **Thought**: for now, focus the modelling on the loss function rather than on architecture.
- **Thought**: cleaning the dataset is a first-class part of the work, not a chore before it.
- **Thought**: the dataset folder should be reorganised into a clearly structured layout.
- **Decision**: I define the ideas and the spec; Claude Code (running locally on the data) implements the scripts, reorganises the folders and runs the tests and experiments. Spec: `docs/SPEC.md`.
- **Loss-function reasoning carried into the spec**: select on the conditional mean (L2), report MAE for both the mean model and a median (L1) model, and test Huber because the target is zero-inflated and heavy-tailed with 25-point wicket jumps.

---

## Entry 3: Φ ambiguities settled, ideas to test empirically (2026-09-29)

- **Decision**: a six earns its own +2 only, not the +1 boundary bonus on top. The duck applies to every player dismissed on 0.
- **Thought**: some earlier ideas are no longer allowed under the updated PS, but many still stand. They should be tested on small smoke match sets rather than argued about. Specifically:
  - orthogonal experts (entry 1, point 5),
  - a reranker (entry 1, point 3), done with gradient-boosted trees rather than an LLM,
  - "hardcoding in the training algorithm itself": putting structure into how the model is trained (objective, weights, constraints) rather than into features (entry 1, point 2).
- **Resolution** (tested 2026-09-29 on smoke walk-forward sets; details in `docs/EXPERIMENTS.md`):
  - Orthogonal experts: an additive interaction-constrained GBDT, sequential backfitting and role-gated experts all tie with the plain model. The additive version is interpretable at no cost. Form and opportunity experts stay correlated (0.5–0.75), which is the correlated-factors caveat from Entry 1.
  - Reranker (LambdaRank, stage-2 GBDT reranker): no gain. Listwise ListNet objective: worse.
  - Structure put into the training algorithm (monotone constraints, recency weights, selection-boundary weights, SPO+ decision-focused loss): nothing replicated.
  - What worked was information: the appearance model (+40 pts/match over a proxy), several EWMA memories at once, and opponent + venue context (+2.6, replicated). This backs point 2 of Entry 1, where opponent and setting are different kinds of information.

---

## Entry 4: Two neural architectures and privileged training on time (2026-09-29, ~10:00)

- **Thought**: the smoke-set experiments were not conclusive, because they ran on small samples.
- **Thought**: build a generic solution and try to beat it with two architectures: one containing all my ideas, and one plain baseline.
- **Thought (baseline)**: a history encoder that stores time information (candidates: Recurrent Memory Transformer, Block-Recurrent Transformer, xLSTM, RWKV), a current-info encoder for the pre-toss match data, and a decoder with role–filler embeddings. Normal training with good optimisers.
- **Thought (advanced)**: build significantly on the baseline, implementing my earlier ideas in appropriate ways, and train with *privileged information on time*, following Karlsson et al., "Using Time-Series Privileged Information for Provably Efficient Learning of Prediction Models" (AISTATS 2022, LuPTS).
- **Resolution** (designs in `architecture/BASELINE.md` and `architecture/ADVANCED.md`, open for scrutiny before any code):
  - History encoder: RWKV-style recurrence with decay per unit *time* (continuous-time WKV). With constant keys it reduces exactly to the multi-half-life EWMA bank that worked in B0, so the choice follows from the measured result (E4). RMT and Block-Recurrent solve over-length contexts, which careers are not. The advanced model needs an explicit state to roll forward.
  - Role–filler: Hadamard-compressed tensor-product binding of role embeddings with the player representation.
  - "Privileged training on time" has two meanings here. *Vertical*: the within-match chain toss/XI → opportunity → events → Φ, all post-toss and so privileged. *Horizontal*: the player's matches between the history cutoff and the fixture date, which are observable in training and not at test.
  - Checked against the paper's assumptions: Markov fails as stated and is fixed by carrying the pre-toss state forward. The last step (events → Φ) is noiseless and known, so the gain has to come from the noisy transitions. Φ is nonlinear, so the roll-out has to propagate distributions, not point estimates (Jensen). The existing p·μ model is already a two-step instance of the paper's estimator.
  - Found while mapping the paper: every backtest so far used fresh history (staleness 0), while at test time history stops at 2025-12-31. The top appearance feature ("played the team's previous match") does not exist under staleness. All models must be re-evaluated with simulated staleness (E16). Question for the organisers: may inference read post-cutoff matches as inputs?
  - Idea 5 (hierarchical experts) is realised as the supervised chain of heads (each expert identified by its own target) instead of an orthogonality penalty.

---

## Entry 5: Who plays, and compute (2026-09-29, evening)

- **Thought**: there are two problems: whether a player plays, and his points given that he plays. The second is mostly a compression of the player's history into features (an LSTM or transformer-style history encoder). The first is harder and decides most of the score.
- **Thought**: three phenomena drive the probability of playing: players flowing into the team, players flowing out, and past performance getting a player benched.
- **Thought**: the naive model gives every squad member its own logit. With two keepers, W1 (Dhoni) plays every match, so training pushes W2's logit down. If W1 is absent at test time, the model still won't pick W2.
- **Thought (counter)**: make it relative: a ranker. Even if W2 is rated poorly, he is picked when he tops the list.
- **Thought (counter to my counter)**: a ranker would distort the scores the team optimiser uses.
- **Decision**: keep a Scrutiny doc that records every counter I raise against a live part of the pipeline and its answer, for defending the solution.
- **Decision**: compute is not the constraint. Training can be spread across several Kaggle notebooks.
- **Resolution** (details in the Scrutiny doc, S1-S11):
  - Influx and outflow are observed at test time as changes to the squad, so they are conditioned on, not predicted. Form-benching after the cutoff is unobservable; in-season status is regressed toward long-run status as the gap grows.
  - Independent logits learn a propensity tied to the old squad; a single sum-to-XI shift cannot move probability within a role. A ranker over the whole squad is the same model. A ranker within roles is the zero-temperature limit of a constrained choice model, P(X|S) proportional to exp(sum of utilities + count potentials), where a benched backup keeper's gradient is about 0 because the keeper constraint and his rival explain his absence.
  - My second point holds: the optimiser needs calibrated probabilities, and a 0/1 ranker overvalues rotation players at the selection boundary.
  - Refinement from the all-rounder counter: count potentials over overlapping capacities (keeper, bowling options, batting depth, overseas), not disjoint fantasy roles.
  - At fresh history, knowing the true XI adds about 5 pp of captured/oracle, so the headroom from better p is bounded at staleness 0; under staleness it is unmeasured (E16).

---

## Entry 6: The pipeline to present, and a minimal folder (2026-09-29, night)

- **Thought**: I need one pipeline that I will actually show, not a collection of experiments.
- **Thought**: the BASELINE and ADVANCED docs should say in detail how each architecture is built.
- **Thought**: I need a list of what still has to be defined before code.
- **Decision**: keep the folder minimal: one folder for code, one for docs, one for reference papers, one for output.
- **Resolution**:
  - `docs/PIPELINE.md` is the pipeline to present: one shared, gated spine (ingest → Φ → labels → squads → causal features → exact ILP → CSV) and a swappable model stage with three rungs (B = LightGBM, NB, ADV), plus the pre-registered submission rule. Its §7 lists everything still to define (six organiser questions, fifteen design decisions).
  - BASELINE and ADVANCED rewritten as revision 2: capacity binding; the exact constrained-choice layer with probabilities as gradients of log Z; the innings simulator that keeps overs, balls and wickets consistent; staleness up to 450 days.
  - Folders: `src/`, `docs/`, `references/`, `output/`, plus `data/`, which the PS requires (raw and processed data in their own location).

---

## Entry 7: A caveman baseline (2026-09-29, night)

- **Thought**: the baseline should be a simple caveman solution: preprocessing, then the encoders and a decoder, then the output. The model is asked to learn every mapping internally.
- **Resolution** (`docs/BASELINE.md` rev. 3, open for scrutiny):
  - CB = raw history tokens (appearances and absences) → Transformer history encoder per candidate → set decoder over both squads and the fixture → one number per candidate, trained by squared error on the realised score (0 if benched).
  - Squared error alone recovers the product p·μ, because the minimiser of squared error is the conditional mean and E[Y | info] = P(plays) · E[points | plays]. CB and a factorised model share the population optimum; they differ only in inductive bias and sample efficiency, which is what the advanced ladder then measures.
  - The CT-WKV encoder and the capacity-bound p/μ decoder (the old NB) move to `ADVANCED.md` §A0 as the first structured rungs.
  - Kept in CB because they are data correctness or PS rules, not structure: truncation at the cutoff with staleness as an input, pseudo-squad candidate sets, output-only roles, the exact ILP.

---

## Entry 8: Running the final run on Kaggle (2026-09-30, early morning)

- **Thought**: run the whole architecture end to end on a Kaggle T4 box: upload the code and the raw data as zips, get the outputs and the weights.
- **Thought**: save the weights periodically, so that if the notebook fails the run restarts from that point instead of from scratch.
- **Resolution**: `kaggle_notebook.ipynb` + `src/kaggle_run.py`. Every stage marks itself done; every CB run checkpoints each epoch and resumes; a new session resumes from the previous version's output attached as an input.

---

## Entry 9: Build the advanced model after all (2026-09-30, ~11:00)

- **Thought**: the final run tested only the caveman baseline against B0s; the advanced architecture, where most of my ideas are (the utility / choice model, the time-decay encoder, privileged training on time), was never built.
- **Decision**: build ADV the same way as the baseline (checkpointed, resumable, Kaggle) and run it, with its training and validation timeline.
- **Resolution**: `src/adv.py` with A0a, A0b, A2, A1 (μ), A3, A5, A6; A4 (innings simulator) not built today. Pre-registered as SPEC §14, post-hoc to E22.
- **Result** (E23, Kaggle, 2 seeds × F24/F25): ADV − B0s = 2024 −0.90 [−3.52, +1.63], 2025 +2.82 [−0.06, +5.61] → B0s kept (rule needs both years > 0). ADV − CB128 = +4.45 [+1.77, +7.22] and +4.76 [+2.15, +7.40]: the structure beats the learn-everything baseline in both years. Within ADV: choice layer vs sum-to-XI a tie (+0.49, −0.47); roll-out vs direct a tie leaning positive (+0.91, +0.17); the distilled student failed (−17.5, −14.0). By staleness (descriptive, not a decision): B0s's whole lead is at Δ = 0 (ADV −6.7 / −2.4); at Δ = 365 ADV leads +2.5 [−1.2, +6.3] and +5.5 [+2.0, +9.2].

---

## Entry 10: Is the data handling wrong? And a player-interaction graph (2026-09-30, ~14:00)

- **Thought**: the way we handle the data may be fundamentally wrong: some preprocessing step, or the way the architecture's sequences are built. Check what other people do with Cricsheet data on GitHub.
- **Thought (idea)**: model relationships between players. Start from a dense graph over all players and remove the edges that are empty. Each time a squad or a team is given, learn conditions on the edges: he scores well against this opponent and badly against that one. Then generalise: if a player does well against fast bowlers, the conditional expectation of his score rises when he faces fast bowlers.
- **Reasoning so far** (survey, details in the reply of this date):
  - Independent Cricsheet projects on the same kind of task report the same level we reach: a pre-toss P(plays) × points pipeline with MAE 25.4, Spearman 0.359, oracle capture 60.1%; an XGBoost fantasy model with R² 0.127, the same as our ICC ≈ 0.13. No evidence of a preprocessing error; the ceiling looks like per-match noise.
  - What our pipeline does throw away is the within-match structure: it aggregates each player-match to counts, so who bowled to whom (the batter–bowler graph) never reaches any model. Opponent information is the one context signal that replicated (E14, +2.6 pts/match).
  - The dense-then-prune graph is the observed batter × bowler delivery graph. Most pairs have a handful of balls, so raw head-to-head numbers are noise; the generalisation step ("good against pace") is a low-rank factorisation of the pair effect (batter vector · bowler vector), which learns bowler types without labels and predicts unseen pairs. Pre-toss, the effect enters as an expectation over the opponent squad's likely bowlers and their overs.
- **Resolution** (E24, SPEC §15, pre-registered 15:45): built as a rank-4 Poisson factorisation of the batter × bowler delivery graph (runs and dismissals), refitted monthly on data before each cutoff (gate G8 passed), turned into opponent-squad expectations and added to B0s. B0s + matchups − B0s: 2024 −0.77 [−2.10, +0.62], 2025 +1.59 [+0.23, +2.92]; the 2025 gain grows with staleness (+1.4 at Δ = 0 to +2.0 at Δ = 365). Pair-interaction features carry less gain than opponent-quality ones, as expected. Rule needed both years above 0: B0s kept; the idea is reported as a replicated-in-one-year gain.
