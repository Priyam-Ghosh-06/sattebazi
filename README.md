# sattebazi

**Picking a fantasy cricket XI before the toss**, using only data up to 31 December 2025.
Internal hackathon, Traditional ML track. Data: [Cricsheet](https://cricsheet.org) ball-by-ball files, T20 and T20I only.

![results](results/overall/headline.png)

**In one paragraph.** For each upcoming match I get the two squads. For every player I predict *how many fantasy points he
should score on average*. Then an exact optimiser picks the best 11 that obey the fantasy rules. I built three approaches:
1. a neural network that learns everything by itself;
2. a neural network with my understanding of cricket built into it;
3. gradient-boosted decision trees on hand-made features, plus a batter-vs-bowler graph.

On two full seasons they had never seen (2024 and 2025, about 1,800 matches each), all of them pick teams worth **about 62–63%
of the best team you could have picked in hindsight**. A random legal team gets 45%. The best of them, and therefore the
**final model, is 3c: the refined neural net + the batter-vs-bowler graph** (section 8).

| approach | 2024 | 2025 | of our 11, how many were in the best XI |
|---|---|---|---|
| a random legal team | 45.1% | 44.9% | |
| 1 · Baseline: neural net learns everything itself | 61.6% | 62.5% | 5.6 |
| 2 · Refined: neural net + cricket structure | 62.3% | 63.3% | 5.7 |
| 3a · Trees | 62.4% | 62.8% | 5.7 |
| 3b · Trees + batter-vs-bowler graph | 62.4% | 63.1% | 5.7 |
| **3c · Refined net + batter-vs-bowler graph (final model)** | **62.4%** | **63.3%** | 5.7 |

*% = points our 11 actually scored ÷ points of the best possible 11, averaged over every test match and over five levels of
data staleness (explained in section 1).*

Contents: [1 the problem](#1-the-problem-the-way-i-see-it) · [2 how I judge a model](#2-how-i-judge-a-model) ·
[3 the shared pipeline](#3-what-all-three-approaches-share) · [4 baseline](#4-approach-1-baseline-the-model-learns-everything-itself) ·
[5 refined](#5-approach-2-refined-my-understanding-built-into-the-network) · [6 graph](#6-approach-3-trees-and-the-batter-vs-bowler-graph) ·
[7 overall results](#7-overall-results) · [8 the final model](#8-the-final-model) · [9 assumptions](#9-assumptions) ·
[10 glossary](#10-glossary) · [11 how to run](#11-how-to-run) · [12 layout](#12-repository-layout)

---

## 1. The problem, the way I see it

**Input:** a fixture (date, two teams, maybe a venue) and the two squads, usually 15 or more players each.
**Output:** 11 players who follow the fantasy rules (1–4 wicket-keepers, 3–6 batters, 1–4 all-rounders, 3–6 bowlers, at
least one player from each side). The goal is that those 11 score as many fantasy points as possible in the real match.
One CSV row per chosen player:

```
match_date,team1,team2,rank,player,player_team,role,predicted_points,points_if_plays,p_plays
2025-10-03,Australia,New Zealand,1,MW Short,Australia,AR,53.95,57.2,0.943
```

Here is one real match (not hand-picked: the median of the 40 mock matches), predicted by the trees (3a) trained only on data
up to 30 September 2025. I use the trees here because they output p and μ separately, which makes the picture easy to read:

![one match worked through](results/overall/worked_example.png)

Two things stand out. The ✔ players are the ones with the longest dark bars, because that is what the optimiser maximises.
And the red diamonds are all over the place: one player we didn't pick scored 158. **Most of fantasy cricket is match-day
luck**, and that sets the ceiling for every model.

When I look at a player before a match, I see two kinds of information, and both need to be modelled:

- **the temporal side:** his history. How he has been scoring, how his form moves, how often he has been picked lately, and
  how old all of that is by the time the match is played;
- **the present side:** the squad he is in and the one he is up against. Who else is competing for his place, which opponent he
  faces, where and in which competition.

A player's fantasy score is `Y = 1[he plays] · (3 + Φ(what he does in the match))`: 3 points for being in the XI, plus the
official points table Φ applied to his runs, wickets, catches and so on. The table is exact. What happens in the match is
mostly luck: only about 13% of the match-to-match variation in points can be explained by who the player is.

A team's score is a sum, so the best team *on average* is the one with the largest sum of expected scores. And

    expected points = P(he plays) × expected points if he plays        (E[Y] = p · μ)

![why p times mu](results/overall/why_p_times_mu.png)

So the problem boils down to three things:

1. **p, who plays.** 28% of squad members sit out and score 0, so this matters a lot. It is hard because Cricsheet only
   lists the XI. Benched players never appear, so there are no "didn't play" examples to learn from. A backup keeper looks
   like a player who "never plays" until the first-choice keeper is missing.
2. **μ, points if he plays.** Noisy and lopsided (the middle panel): most players score little and a few score huge. The
   bonuses for milestones, strike rate and economy depend on the rare big days, not the average.
3. **the decision:** pick the 11. That part can be solved exactly (section 3).

The obstacles I had to get past before any of that:

| obstacle | what I did |
|---|---|
| no squads in the data, only the XIs | rebuilt **pseudo-squads**: everyone who played for that team in that competition and season. Members who didn't play become "sat out, 0 points" rows |
| no player roles in the data | inferred each player's role from his history before the match (stumpings, catches, share of overs, batting position): 99.3% of real XIs come out legal |
| history stops on 31 Dec 2025, the matches are later | **simulated staleness**, see the figure below |
| leakage (accidentally using the future) | every feature is looked up with one strict "dated before" join, and tests rebuild features from cut-off history and demand exactly the same numbers |
| noise | judge on the task's own score (points of the chosen 11), measure how sure each comparison is, and write every decision rule down before seeing a test number |

**Staleness**... The hidden matches are months after our last data, so a model tested on fresh
history would look better than it really is. I train and test every model with the history cut Δ days before each match:

![staleness timeline](results/overall/staleness_timeline.png)

## 2. How I judge a model

**Train, validate, test.** Each model learns from matches up to September of one year (training). I use October to December to
decide when to stop training and which settings to keep (validation). Then I measure it once on the whole next year, which
it has never seen (test). I do this twice, testing on 2024 and on 2025. That's called walk-forward: the model never sees
the future, just like in real use.

**The score.** For every test match each approach picks its 11. I add up the points those 11 actually scored and divide by
the points of the *best possible* legal 11 (picked in hindsight, knowing the real points). Then I average over the matches and
over the five staleness levels. The hidden test set is scored by the sum of our 11's points, so this is the number I select
on.

**Why not just MAE?** The problem statement also asks for error metrics, and they are in every results table (MAE, RMSE,
and the rank correlation inside a match). But I don't choose models by MAE. Most squad rows are "sat out = 0", and a model
that minimises MAE learns to predict the *median*, which is 0 for anyone less than 50% likely to play. When I trained that
model, it picked worse teams (−6.9 points a match). Ranking losses (LambdaRank, ListNet, SPO+) tied or lost too.

**Is a difference real or luck?** Two models' teams differ by a few points a match, and single matches swing by hundreds. So
I take the per-match differences and resample the ~1,800 test matches 2,000 times with replacement (a **paired bootstrap**).
The middle 95% of the resampled averages is the interval I report, e.g. `+1.59 [+0.23, +2.92]`. If the whole interval is
above zero, the gain is very unlikely to be luck.

**The two-year rule** (written before any test number): a model counts as *clearly* better than another only if the whole
interval is above zero **in both 2024 and 2025**. Nothing is rerun or retuned after a test number is seen.

## 3. What all three approaches share

![how it works](results/overall/how_it_works.png)

Everything around the model is shared, so any difference in the results comes from the models themselves.

| step | code | what it does | how it is checked |
|---|---|---|---|
| ingestion | `src/shared/ingestion/ingest.py` | Cricsheet JSON → one table of matches and one of player-matches. T20/IT20 dated ≤ 2025-12-31 only (13,085 matches; 7,862 Test/ODI files skipped), registry IDs as the player key, super overs excluded | file counts, delivery sanity checks |
| fantasy points | `src/shared/preprocessing/points.py` | the official points table, as 13 separate components | band edges and a few real matches re-scored by hand (`tests/test_points.py`) |
| roles | `src/shared/preprocessing/roles.py` | keeper / batter / all-rounder / bowler as of each date; used only to apply the team rules | 99.33% of real XIs come out legal |
| pseudo-squads | `src/shared/preprocessing/squads.py` | the candidates for each match, sat-out players included (median 15 per team) | < 0.5% have fewer than 11 |
| features | `src/shared/preprocessing/causal.py`, `features.py`, `context.py` | form, opportunity, how often he has been picked, rank inside his squad, opponent and venue; all through one strict as-of join | time-travel test on 200 rows (`tests/test_causality.py`) |
| **picking the XI** | `src/shared/optimization/selection.py` | maximise the sum of expected points over 11 players under the role rules and ≥ 1 per side, solved exactly as an integer program (`scipy.optimize.milp`), ties broken by player id | 1,000 random squads: every rule holds, same answer twice (`tests/test_select.py`) |
| test fixtures | `src/shared/ingestion/fixtures.py` | reads the hidden fixture file (several layouts), matches names to player IDs, prints how many were matched | |
| evaluation | `src/shared/evaluation/` | best possible XI, the score, error metrics, bootstrap, reports and every figure in this README | |
| LLM overview (bonus) | `src/shared/llm/team_overview.py` | a small local open-weight model (Qwen2.5-0.5B-Instruct, Apache-2.0, runs on a CPU) writes a short paragraph per predicted XI. The code computes the facts (composition, team split, top picks, risky picks) and prints them above each paragraph, and the model only phrases them, so a wrong sentence is easy to spot. Greedy decoding, so the text is the same every time. Example: `results/mock_team_overviews.txt` | |

**Why an exact optimiser, not "take the top 11"?** The expected team score is just a sum, so maximising it under the rules is
the best decision the predictions allow. Taking the top 11 and then patching the roles can lose whenever a rule binds: a
third keeper can be worth less than a fourth bowler, even with a higher personal score.

**Leakage checks.** Matches after `--train-end` are dropped at ingestion, and the check is repeated when features are built.
Every row stores the latest date it used, and the audit asserts it is before the cutoff. "Time-travel" tests rebuild rows
from truncated history and demand bit-identical results:
- the tabular features (200 rows) and the stale features (272 rows);
- the neural nets' inputs (388 and 305 candidates);
- the graph snapshots.

Two of these tests caught real bugs on their first run (a same-day tie order, and an opponent read from rows after the
cutoff). Both were fixed before any result was produced.

## 4. Approach 1: Baseline, the model learns everything itself

**The thought.** Before I put any of my own ideas in, I want to know what a model finds on its own. So the baseline is naive
about cricket on purpose. It is never told "keeper", "form" or "probability of playing". I force it to map everything
intrinsically, and I need these results as the reference every idea has to beat.

```
every squad member's raw history (his last L matches, all dated before the cutoff)
   → a Transformer reads each player's history, one past match at a time           (temporal side)
   → a second Transformer lets all players of both squads look at each other       (present side)
   → one number per player: his expected fantasy points                            → the XI optimiser
```

*A Transformer is a neural network that reads a sequence (here, a player's past matches) and learns which parts of it to pay
attention to. The second one treats the squad as a set, so it can learn things like "if the regular keeper is here, the
backup won't play".*

- Each past match is one **token**: 16 raw counts (runs, balls, fours, sixes, wickets, dots, catches, …), the fantasy points
  he scored, the context (format, competition, venue, opponent, team) and how long ago it was. A second kind of token marks
  every match of his team he *missed* in the last year, so the model can see who is a regular and who was dropped.
- One fixture token says what match it is and how stale the history is (Δ).
- It is trained with plain squared error on Y, sat-out players included. The best answer under squared error is the average,
  and the average of Y is p·μ, so it has to learn "plays" and "points if plays" together without being told.
- Two history lengths (64 and 128 matches), two random seeds each; the two seeds are averaged. 128 won on validation.

![baseline training](results/baseline/training_curves.png)

**Code** (`src/baseline/`):
- `config.py`: the fixed training budget;
- `ingestion.py`: history → tokens;
- `preprocessing.py`: squads, staleness per match and epoch, tensors;
- `scoring_model.py`: the two Transformers;
- `decision_model.py`: scores → XI, the validation score;
- `pipeline.py`: training with a checkpoint every epoch and resume, packaging, prediction.

Tests: `tests/test_baseline.py` (time travel, the missed-match token rule, shuffling the squad changes nothing, unknown
players). Trained on Kaggle (2× T4 GPUs): 12 runs in 3.4 h.

**Results.**

| | 2024 | 2025 |
|---|---|---|
| % of the best possible XI (history 128) | 61.6 | 62.5 |
| vs 3a (the trees), points per match | −5.34 [−8.26, −2.49] | −1.93 [−4.74, +0.90] |
| 128 vs 64 matches of history | +1.83 [−0.11, +3.81] | +1.67 [−0.21, +3.53] |
| MAE / RMSE, squad rows | 24.75 / 33.91 | 24.22 / 33.67 |
| rank correlation inside a match | 0.318 | 0.341 |
| predicted team total vs real | −13% | −16% |

*(A team scores about 440 points a match, so −5.34 is about 1.2% of a team's score.)*

**What I read from it.** It does learn the task (54% → 65% on validation), but it doesn't beat the hand-engineered model.
Nothing tells it that exactly 11 players take the field, and it under-predicts team totals by 13–16%. Two things surprised
me. On IPL matches it is the best of all five in 2024 and a close second in 2025 (a small sample, about 70 matches a year;
section 7). And its whole gap is at
Δ = 0: when everyone's history is a year old, it is level with the others.

## 5. Approach 2: Refined, my understanding built into the network

**The thought.** Same inputs, same training data as the baseline, but now I give the network the structure I believe the
problem has. Every piece can be switched off, so I can measure what each one is worth.

- **Memory should fade with real time, not per row.** Form from two years ago matters less than last month's, and staleness
  should make the model *less sure*, not change its opinion. So the history is read by a recurrent memory whose channels fade
  per match and per day (a continuous-time WKV). A test checks that with fixed settings it reproduces exactly the moving
  averages the trees use (to 1e-16), and that a gap without matches only lowers its confidence.
- **"Plays" and "points if plays" are different questions**, so there are two outputs (heads). Each is tied to what the player
  can do: keep wicket, bowl, bat high.
- **Who plays is a choice among the squad, not eleven separate yes/no questions.** If a squad has two keepers and only one
  plays, the backup isn't a bad player; he is second in line. So the model scores a whole XI at once:
  P(XI) ∝ exp(Σ each player's utility + a term for the number of keepers + a term for the number of bowling options), with
  exactly 11 (or 12 where impact substitutes exist). Every possible XI is summed exactly by dynamic programming, and each
  player's chance of playing drops out of that sum. When the first-choice keeper is missing, the backup's chance goes up
  automatically.
- **Context scales everyone** (a flat pitch lifts every batter), so the match context rescales every player. **Squad
  attention** lets the points head see who else is playing (a missing opener pushes No. 3 up the order).
- **Learn what happens inside the staleness gap.** In training I know the matches between the history cutoff and the match;
  at test time I don't. So the model learns a month-by-month drift of a player's form from those matches, and at test time
  rolls the December form forward to the match date. Then a one-pass "student" is trained to copy it.

Prediction: expected points = chance of playing (from the choice model) × points if he plays. Training has four phases:
1. the memory and the heads;
2. the monthly drift;
3. the heads again, on rolled-forward form;
4. the student.

Each seed picks its final predictor on validation. Not built: a full innings simulator (it needs over-by-over sequences the
ingestion doesn't extract) and an overseas-player term.

### The decision (choice) model, written out

This is the part of the refined approach that answers "who plays". The code is `src/refined/decision_model.py`.

**What goes in.** Take one team in one match. Its squad has candidates $i = 1, \dots, C$ and must field $n$ players (11, or 12
where impact substitutes exist). For every candidate the network gives:
- a **utility** $u_i$: how strongly the team wants him in its XI. It comes from his memory state, rescaled by the match
  context and tied to what he can do (keep wicket, bowl, bat high);
- two flags known before the match: $k_i = 1$ if he can keep wicket, $b_i = 1$ if he is a bowling option.

**The model.** An XI is a set $X$ of exactly $n$ squad members. Its probability is

```math
P(X) \;=\; \frac{1}{Z}\,\exp\!\Big(\sum_{i \in X} u_i \;+\; \varphi_K\big(\min(K_X, 3)\big) \;+\; \varphi_B\big(\min(B_X, 8)\big)\Big)\,\mathbb{1}\big[\,|X| = n\,\big],
\qquad K_X = \sum_{i \in X} k_i,\quad B_X = \sum_{i \in X} b_i
```

- $\sum u_i$ says who the team likes;
- $\varphi_K$ is a small learned table over the number of keepers in the XI (0, 1, 2, 3 or more);
- $\varphi_B$ is a small learned table over the number of bowling options (0 to 8 or more);
- $Z$ adds up the same exponential over **every** possible XI, so the probabilities sum to 1.

Only the *differences* between the table entries matter. To make the tables unique, $\varphi_K(1) = \varphi_K(2) = 0$ and
$\varphi_B(5) = \varphi_B(6) = 0$ are fixed.

**Who plays.** A player's chance of playing is the total probability of all XIs that contain him, and it falls out of $Z$ as a
derivative:

```math
\pi_i \;=\; P(i \in X) \;=\; \frac{\partial \log Z}{\partial u_i}, \qquad \sum_i \pi_i = n
```

So one backward pass through $\log Z$ gives every player's exact chance of playing, and the chances of a team always add up
to exactly $n$. The expected points that go to the XI optimiser are $\text{E}[Y_i] = \pi_i \, \mu_i$, with $\mu_i$ from the
"points if he plays" head.

**Why it handles competition for places.** The players are not scored one at a time. If the first-choice keeper is in the
squad, the XIs that include him take most of the probability, so the backup keeper's $\pi$ stays low even with a decent
utility. Remove the first choice, and the probability the XIs with him used to hold has to go somewhere, because the chances
still add up to $n$. The backup gains the most: the XIs with no keeper at all are pushed down by $\varphi_K(0)$, so the
probability flows to the XIs that contain him. His $\pi$ rises without any rule written for it.

**Computing $Z$ exactly.** A 15-player squad has 1,365 possible XIs, and summing them one by one would be slow inside
training. Instead a dynamic program walks through the candidates once. It keeps a table indexed by (how many players are
picked so far $m$, keepers so far $k$, bowling options so far $b$), with $k$ capped at 3 and $b$ at 8. Adding candidate $j$:

```math
\alpha_j(m, k, b) \;=\; \operatorname{logaddexp}\Big(\alpha_{j-1}(m, k, b),\;\; \alpha_{j-1}(m-1,\, k - k_j,\, b - b_j) + u_j\Big)
```

and at the end

```math
\log Z \;=\; \operatorname{logsumexp}_{k,\, b}\Big(\alpha_C(n, k, b) + \varphi_K(k) + \varphi_B(b)\Big)
```

That is $C \times 14 \times 4 \times 9$ cheap steps per team, done for every team of a batch at once on the GPU.
`tests/test_refined.py` checks it against brute force over all 364 XIs of a 14-player squad (difference $6 \times 10^{-15}$).

**Training.** For each team the real XI $X^\ast$ is known, so the loss is its negative log-probability, divided by $n$:

```math
\mathcal{L}_{\text{plays}} \;=\; \frac{1}{n}\Big(\log Z \;-\; \sum_{i \in X^\ast} u_i \;-\; \varphi_K(K_{X^\ast}) \;-\; \varphi_B(B_{X^\ast})\Big)
```

It is added to the squared error of $\mu$ on the players who played. The utilities, the two tables and the rest of the
network are all learned together.

**What it compares against.** The simpler alternative treats each player on his own: $\pi_i = \sigma(s_i - \tau)$, with one
shift $\tau$ per team chosen so the chances add up to $n$. On the same trained weights the two tie on picked points
(+0.49 and −0.47 points a match in 2024 and 2025, both intervals across zero). The learned tables show why: the
team-shape terms are mild, and most of the work is done by the utilities.

| number in the XI | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---|---|---|---|---|---|---|---|---|
| $\varphi_K$ (keepers) | −0.46 | 0 | 0 | −0.26 (3+) | | | | | |
| $\varphi_B$ (bowling options) | −0.35 | −0.37 | −0.45 | −0.42 | −0.20 | 0 | 0 | −0.45 | −0.96 |

*Learned values, 2025 fold, seed 0 (the 2024 fold learns almost the same). Read them as: an XI with no keeper, or with 7 or
more bowling options, is less likely; five or six bowling options is the normal shape.*

![refined training](results/refined/training_phases.png)
![refined pieces](results/refined/component_ladder.png)

**Code** (`src/refined/`):
- `config.py`;
- `ingestion.py`: the baseline tokens + squad-membership tokens, what each player can do as of the cutoff, XI size;
- `preprocessing.py`;
- `scoring_model.py`: memory, heads, attention, drift, student;
- `decision_model.py`: the exact choice model over the squad;
- `pipeline.py`: the four phases, checkpointed and resumable.

Tests: `tests/test_refined.py`. The memory reproduces moving averages, staleness only lowers confidence, gap matches never
leak, and the choice model equals brute force over all 364 possible XIs of a 14-player squad. Trained on Kaggle: 4 runs in
2.9 h.

**Results.**

| | 2024 | 2025 |
|---|---|---|
| % of the best possible XI | 62.3 | **63.3** |
| vs 3a (the trees) | −0.90 [−3.52, +1.63] | +2.82 [−0.06, +5.61] |
| **vs the baseline** | **+4.45 [+1.77, +7.22]** | **+4.76 [+2.15, +7.40]** |
| choice model vs a simple "sum to 11" fix (same weights) | +0.49 [−0.35, +1.27] | −0.47 [−1.27, +0.37] |
| rolled forward vs not | +0.91 [−0.07, +1.94] | +0.17 [−0.76, +1.08] |
| MAE / RMSE, squad rows | 25.90 / 36.78 | 25.53 / 38.33 |
| rank correlation inside a match | 0.341 | 0.359 |

**What I read from it.**
- **Structure pays.** Putting my understanding into the network gives about +4.5 points a match over the baseline in both
  years, and both intervals are above zero.
- **Against the trees it ties on average, but the average hides a pattern.** The trees win only when history is fresh
  (Δ = 0), where "played the team's last match" is a very strong signal. As history gets older, which is exactly the hidden
  set's situation, the refined model leads.
- **The choice model ties the simple fix on picked points.** The order of who is likely to play barely changes.
- **Rolling forward helps a little**, and 3 of 4 seeds picked it.
- **The student failed.** It sees neither the squad nor the choice model, so it can't learn the competition for places.
- **Its point totals are off in level.** In 2025 its validation MSE is about 2,100 against 1,180 for the others (section 7).
  The optimiser only needs the ranking inside a match, so the picks don't suffer.

## 6. Approach 3: Trees, and the batter-vs-bowler graph

**The thought.** Players don't score in a vacuum. A batter's runs depend on who bowls to him, and a bowler's wickets on whom
he bowls to. That is a graph: every ball is an edge between a batter and a bowler. But first the approach needs a strong base.

### 3a, the base

The p·μ split done the classic way, with **gradient-boosted decision trees** (LightGBM). *A decision tree asks a few yes/no
questions about the features ("scored > 30 on average lately?") and outputs a number. Boosting builds hundreds of small
trees, each one correcting the mistakes of the ones before it.*

- **p:** a classifier on all squad rows (sat out = 0), then one shift per team so each team's chances add up to exactly 11;
- **μ:** a regressor on the rows of players who played;
- **score = p·μ** → the XI optimiser;
- **features,** all computed strictly before the cutoff:
  - form as moving averages over three memory lengths (3, 10 and 30 matches), plus career averages;
  - opportunity (batting position, balls faced and bowled);
  - how often he played for this team recently, and where he ranks inside his own squad;
  - evidence for his role;
  - the strength of his own and the opposing squad, and a shrunk venue average;
- **staleness-aware:** every row is built as of the cutoff, gaps are measured to the match date, and Δ itself is a feature.

Before the final run, about 60 small experiments decided what 3a contains:
- the noise ceiling;
- the memory length;
- the loss;
- the playing model;
- extra structure in the training algorithm (all ties);
- opponent and venue context: the one gain that replicated, +2.6 points a match.

The code is in `src/experiments/`, with one line per run in `results/experiments/LOG.txt`.

### 3b, the batter-vs-bowler graph

1. **Edges.** 645k batter–bowler records (one per pair per match), 2.87M balls. For runs and for dismissals separately, a
   pair's expected count is `balls × exp(base + batter quality + bowler quality + batter vector · bowler vector)`.
2. **Generalising.** The vectors are short (4 numbers). So a batter who handles a *kind* of bowler handles every bowler of that
   kind, including ones he has never faced. The bowler vectors group bowlers by type without any label (Cricsheet has no
   bowling-style field), and pairs with few balls are pulled towards the average.
3. **No leakage.** The graph is refitted every month on balls bowled before that month (100 snapshots, 2017-10 → 2026-01).
   A row with cutoff c uses the latest snapshot on or before c.
4. **Before the toss** nobody knows who will bowl to whom. So each player gets the *expected* effect over the opposing squad,
   weighted by how likely each opponent is to play and to bowl (or bat). That gives 10 features, added to 3a.

![graph features](results/graph/graph_feature_gain.png)

### 3c, the graph inside the refined model

Both the refined model and the graph help most when history is old, so the natural next step was to combine them. The same 10
graph features are appended to every candidate's input in the refined network. Each run **starts from the trained refined
model of the same fold and seed**: the new inputs start with zero weight, so at step 0 it computes exactly what the refined
model computed (checked: max difference 1e-5). Then it is fine-tuned for up to 4 epochs. The student is dropped, because it
failed in section 5. I planned 3c after seeing the earlier test results, so it is a follow-up: same rule, reported as such.

![graph training](results/graph/curves_and_gain.png)

**Code** (`src/graph/`):
- `preprocessing.py`: the stale feature table and its time-travel check;
- `scoring_model.py`: the LightGBM p and μ;
- `ingestion.py`: the ball-level graph;
- `graph_model.py`: the factorisation, monthly snapshots, graph features for training and test, and its check;
- `experiments.py`: `ingest`, `snapshots`, `features`, `g8`, `folds`, `e24`.

The decision is p·μ → the shared optimiser. 3c lives in `src/refined/` (`--graph-features`). 3a trains in about a minute on
a CPU and the graph takes about 15 minutes; 3c took 4 GPU runs of 32–48 minutes.

**Results.**

| | 2024 | 2025 |
|---|---|---|
| 3a, % of the best possible XI | 62.4 | 62.8 |
| 3a: our XI / best possible XI / a random legal XI (points) | 437.0 / 702.4 / 317.1 | 437.9 / 700.4 / 314.1 |
| 3a: share of our picks who actually played | 85.7% | 86.2% |
| 3a MAE / RMSE, squad rows | 24.87 / 33.17 | 24.70 / 33.08 |
| 3a MAE / RMSE, players who played (μ) | 26.44 / 34.44 | 26.44 / 34.47 |
| 3a rank correlation inside a match | 0.346 | 0.361 |
| 3a playing model: log-loss / Brier | 0.487 / 0.161 | 0.485 / 0.161 |
| 3b, % of the best possible XI | 62.4 | 63.1 |
| **3b vs 3a**, points per match | −0.77 [−2.10, +0.62] | **+1.59 [+0.23, +2.92]** |
| **3c, % of the best possible XI (final model)** | **62.4** | **63.3** |
| **3c vs 3a** | −0.03 [−2.64, +2.44] | **+3.53 [+0.76, +6.40]** |
| **3c vs the refined model (2)**: the graph's gain inside the net | **+0.86 [+0.26, +1.45]** | +0.71 [−0.26, +1.69] |
| 3c vs the baseline | **+5.31 [+2.66, +8.03]** | **+5.47 [+2.79, +8.05]** |

**What I read from it.**
- **3a is the strongest when history is fresh** (Δ = 0), and it retrains in a minute on a CPU.
- **The graph adds something real, twice over.** On the trees it adds +1.6 points a match in 2025 (interval above zero) and
  ties in 2024. Inside the refined net it adds +0.9 in 2024 (interval above zero) and +0.7 in 2025.
- **3c is the best model on the averages in both years.** It is clearly better than 3a in 2025, and in 2024 the two are a dead
  tie (−0.03).
- **Most of the graph's value is *whom you face***, the opposing attack's or batting's quality. The pure pair effect ("he is
  good against this particular bowler") carries about a quarter as much.

Where I would take it next:
- edges that depend on the squad context;
- edges split by innings phase (powerplay / death);
- a 3c trained from scratch instead of fine-tuned.

## 7. Overall results

**Validation curves.** The same October–December windows for all approaches, while they train:

![validation curves](results/overall/validation_curves.png)

The trees start good and stay flat. The nets start worse and catch up after about 8 epochs; the structured net (blue) ends
level with the trees, and the baseline (red) stays below. Bottom right: the refined net's squared error is high in 2025 even
though its picks are good. Its point *levels* are off; the *order* inside a match, which is what the optimiser uses, is fine.

**How stale the history is changes who wins:**

![by staleness](results/overall/by_staleness.png)

**By competition:**

![by competition](results/overall/by_competition.png)

What I take away:

1. **All five end up within about one point of each other**, far above a random team. Other Cricsheet projects on this task
   report about 60% of the best XI and R² ≈ 0.13. We are at 62–63%, with the same ≈ 13% of explainable variance: the ceiling is match-day luck, not
   the pipeline.
2. **Structure beats learning everything:** the refined net beats the baseline by about 4.5 points a match in both years.
3. **Staleness decides the ranking.** The trees' lead is entirely at Δ = 0. At the gaps the hidden set will have (roughly
   270 days or more), the refined net and the graph are ahead, and 3c, which has both, leads.
4. **Competitions differ a lot.** Women's cricket (67–69% of the best XI) and men's internationals (64–65%) are far more
   predictable than the men's leagues (55–61%). On the IPL, the baseline that learns everything itself is the best of the
   five in 2024 (58.5% against 57.3% for 3a) and second only to 3c in 2025, though that is only about 70 matches a year.

Every number comes from files in `results/experiments/` (one folder per experiment: metrics, bootstrap, decision). Every figure
is drawn with matplotlib by `src/shared/evaluation/figures.py` from those files.

## 8. The final model

Putting it together, 3c is the best model I have:
- **the highest average in both test seasons:** 62.4% in 2024 (level with 3a) and 63.3% in 2025 (the highest of all five);
- **clearly better than 3a in 2025** (+3.53 [+0.76, +6.40]), **than the refined net in 2024** (+0.86 [+0.26, +1.45]) and **than
  the baseline in both years** (+5.3 and +5.5 points a match);
- **never clearly worse than any other model in either year;**
- **the leader where the hidden matches will be,** with history several months old (section 7, point 3);
- **both of my ideas in one model:** the cricket structure of section 5 and the batter-vs-bowler graph of section 6.

No model, 3c included, is *clearly* better than the trees in both years, because 2024 is a tie. The luck in fantasy points
keeps every gap small (section 7, point 1), and on every other count 3c is ahead.

# The final model is therefore 3c: the refined neural net + the batter-vs-bowler graph

It is `model_artifacts/model_2025-12-31.pkl`, and `src/evaluate.py` uses it for the hidden matches. It holds the two seeds of
the 2025 fold (trained on matches before October 2024, early-stopped on October–December 2024). When it predicts, it reads
every player's history up to 31 December 2025. `NEURAL=1 bash run.sh`, or the Kaggle runner, retrains it on everything up to
the cutoff on a GPU.

## 9. Assumptions

The whole pipeline rests on these assumptions. Each one was checked against the code before it was written here. Some are safe,
and some are real risks for the hidden matches.

**Data and rules**
- **Only T20 and T20I matches dated up to 31 Dec 2025 are used, for training and at prediction time.** `evaluate.py` drops
  every match after the model's `train_end` before building a single feature. The problem statement only restricts
  *training* data. If the organisers allow 2026 matches as inputs at prediction time, most of the staleness problem goes
  away. I take the strict reading; this needs a definite answer from the organisers.
- **The fantasy score is the official points table plus 3 points for being in the XI** (`points.py`), with no captain or
  vice-captain multiplier. The team is judged on the plain sum of its 11 players' points.
- **The team rules are 1–4 keepers, 3–6 batters, 1–4 all-rounders, 3–6 bowlers and at least one player per side**
  (`selection.py`). If a squad can't satisfy them, the optimiser keeps only the one-per-side rule and says so.
- **Roles are inferred from each player's history** unless the fixture file gives them, in which case the given role wins.
- **Names in the hidden file can be matched to Cricsheet's player registry.** A name that can't be matched becomes an unknown
  player with no history, and the models treat him as a cold start.

**Squads and who plays**
- **A training squad is everyone who played for that team in that competition and season (pseudo-squads).** Real squads
  can be larger or smaller, so the "sat out" rows are an approximation.
- **The XI has 11 players, or 12 in impact-substitute competitions.** At prediction time, the refined model takes the size
  of the team's last XI before the cutoff.
- **Rain-hit and no-result matches that had some play are kept** (flagged at ingestion, not dropped). Players then score
  little more than the 3 points for playing, which is part of the luck every model faces.

**Time and staleness**
- **The gap is stationary.** The models learn how history before the cutoff maps to points after it from 2019–2025 gaps that
  start at any time of year. The real gap always starts on 1 January and runs through an IPL season and an auction. The
  trees (3a, 3b) have no month or season feature, so they can't learn that seasonal difference. The neural nets see the
  match month, and the refined model's monthly drift also takes the season phase as an input. A cheap improvement would be
  to give more weight to training copies whose cutoff falls near 1 January.
- **Hidden matches are no more than 450 days after the cutoff.** Training staleness is 0 or 1–450 days
  (`causal.py: draw_delta`), and testing covers 0–365. A fixture after about March 2027 would be outside the range the
  models were trained on.
- **The five staleness levels (0, 90, 180, 270, 365 days) count equally in every score.** The real gap was unknown when the
  scores were defined. The hidden matches will probably sit at the long end, where the neural nets and the graph do best.
- **The final model's weights transfer to 2026.** Its two seeds were trained on matches before October 2024 and read history
  up to 31 Dec 2025 when predicting. A GPU retrain (`NEURAL=1`) would add the 2025 season to its training data.

**Evaluation**
- **Past seasons are a fair stand-in for the hidden set.** The 2024 and 2025 test seasons mix IPL, other leagues,
  internationals and women's cricket in their natural proportions. A hidden set with a different mix will score differently
  (section 7, by competition).
- **Matches are independent for the bootstrap.** Each match's staleness copies are averaged into one number before
  resampling, but matches in the same series or tournament are treated as independent. Real intervals may be a little wider.
- **Maximising expected points is the right goal.** The optimiser is risk-neutral: it doesn't trade expected points for a
  more certain team. That is right for a sum-of-points score over many matches.

## 10. Glossary

| term | meaning here |
|---|---|
| XI | the 11 players picked for a match |
| best possible XI | the legal 11 with the most real points, picked after the match; the upper limit |
| p, μ, p·μ | chance he plays; points if he plays; expected points |
| Δ, staleness | days between the last data we use and the match |
| fold (2024 / 2025) | one train → validate → test split; the test is the named year |
| validation / test | matches used to tune and stop training / matches used once, for the final number |
| epoch | one pass of a neural net over all training matches |
| seed | the random start of a training run; two seeds show how much results move by chance |
| MAE, RMSE, MSE | average error / root mean squared error / mean squared error of predicted points |
| rank correlation | how well the predicted order of players inside a match matches the real order |
| log-loss, Brier | how good predicted probabilities are (lower = better) |
| paired bootstrap, `[lo, hi]` | resample the matches 2,000 times; 95% of the resampled differences fall in `[lo, hi]` |
| leakage | using information that would not exist before the match; tested against, never allowed |
| gradient boosting / LightGBM | many small decision trees, each correcting the previous ones |
| Transformer, attention | a neural net that learns which parts of a sequence or set to focus on |

## 11. How to run

```bash
python -m pip install -r requirements.txt           # Python 3.11+
bash run.sh                                         # download → checks → features + graph → train 3a → hidden-set CSV (final model)
```

`run.sh` downloads Cricsheet if `data/raw/` is empty and stops at the first failed check. It takes about 30 minutes on a CPU
after the download, and predicts with the committed final model. Options:
- `NEURAL=1`: retrain the neural models on a GPU, including the final model 3c on everything up to the cutoff;
- `GRAPH_FEATURES=1`: also train 3b;
- `LLM=1`: write the team overviews.

The two commands from the problem statement:

```bash
python src/train.py --train-start 2019-01-01 --train-end 2025-12-31 --device cuda      # the final model, 3c (GPU)
python src/evaluate.py --model-path model_artifacts/model_2025-12-31.pkl --test-file data/raw/test_fixtures.json --output results/hidden_predictions.csv
```

- The other approaches write their own model files, so they never overwrite the final one:
  - `src/train.py --approach graph` for 3a (CPU, a minute) → `model_trees_2025-12-31.pkl`;
  - `--approach graph --graph-features` for 3b;
  - `--approach baseline --device cuda` and `--approach refined --device cuda` for 1 and 2.

  `evaluate.py` reads which approach made a model from the model file.
- The GPU runs were done on Kaggle with `src/run_kaggle.py`. It runs everything in stages and skips a stage once it is done.
  Every training run saves a checkpoint each epoch and resumes after a crash.
- Seeds are fixed and LightGBM is deterministic. Two runs of `evaluate.py` with the final model give byte-identical CSVs.
- Dry run without the hidden file: `--test-file data/mock/test_fixtures.json` (40 real fixtures from Oct–Dec 2025).
- Figures: `python src/shared/evaluation/figures.py`.

## 12. Repository layout

```
src/
  train.py, evaluate.py         the two entry points (all approaches)
  shared/                       used by every approach
    ingestion/                  Cricsheet JSON → tables; the hidden fixture file
    preprocessing/              fantasy points, roles, pseudo-squads, causal features, staleness helpers
    optimization/               the exact XI optimiser
    evaluation/                 metrics, reports, figures
    llm/                        open-weight LLM team overviews
  baseline/                     1 · config, ingestion, preprocessing, scoring_model, decision_model, pipeline
  refined/                      2 and 3c (the final model) · the same structure
  graph/                        3a, 3b · preprocessing, scoring_model, ingestion (graph), graph_model, experiments
  experiments/                  the experiments that shaped 3a
  run_kaggle.py                 resumable GPU runner
tests/                          points, causality, XI rules, baseline and refined model tests
data/
  raw/                          Cricsheet JSON + player registry (downloaded by run.sh, not committed)
  interim/, processed/          rebuilt by run.sh
  audit/                        one check report per stage
  mock/                         40 mock fixtures + their real points
model_artifacts/                model_2025-12-31.pkl = the final model (3c); model_trees_2025-12-31.pkl (3a);
                                model_2025-09-30.pkl (3a trained to Sep 2025, for the worked example); graph_features_*, baseline_cb128_*;
                                checkpoints/ = best weights and training history of every GPU run
results/
  overall/, baseline/, refined/, graph/     figures and mock predictions
  experiments/                  every experiment's metrics, bootstrap and decision; LOG.txt = one line per run
  logs/                         the GPU training logs
```
