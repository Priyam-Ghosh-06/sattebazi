# Overview: what the problem really is

## In words

Before the toss we get a fixture (date, two teams, maybe a venue) and two squads of 15 or more players. We must pick 11 of them,
respecting the fantasy rules (1–4 keepers, 3–6 batters, 1–4 all-rounders, 3–6 bowlers, at least one player from each side),
so that the 11 score as many fantasy points as possible in the real match.

Two things decide a player's score, and they fail in different ways:

1. **Does he play at all?** A squad member who is left out scores 0. Before the toss nobody knows the XI, and in the
   training data benched players never appear (Cricsheet lists only the XI), so this has to be learned from reconstructed squads.
2. **If he plays, how many points?** Fantasy points are a known formula of what happens in the match (runs, boundaries,
   wickets, catches, strike-rate and economy bands, milestones). The formula is exact; the events are mostly luck.

On top of that, history stops on 31 Dec 2025 while the fixtures come later, so everything we know about a player is stale by
Δ days: a regular in December may be dropped by the time the match is played.

## In a few equations

For candidate *i* in fixture *m*, with *X* the (unknown) XI and Φ the official points table,

    Y_i = 1[i ∈ X] · (3 + Φ(events of i))

The expected total of a team is a sum, so by linearity of expectation (no independence assumption needed)

    maximise over valid teams x:   E[ Σ x_i Y_i ] = Σ x_i · E[Y_i]

and by the law of total expectation

    E[Y_i] = p_i · μ_i,     p_i = P(i plays | squad, history),     μ_i = E[3 + Φ | i plays, history].

So the whole task splits into three pieces:

| piece | what it is | why it is hard |
|---|---|---|
| **probability model** p | who plays, given the squad | no benched players in the data; competition for slots (two keepers, one plays); staleness |
| **score model** μ | points if he plays | per-match points are mostly noise (only ~13% of the variance is between players, ICC ≈ 0.13); Φ is non-linear (milestones, bands), so E[Φ] ≠ Φ(E[events]) |
| **decision** | pick the XI | a small integer program; solved exactly (`src/shared/optimization/selection.py`) |

A model can also learn E[Y] directly: squared error is minimised by the conditional mean, which is exactly p·μ. The baseline
does this; the other two approaches keep p and μ apart.

## Obstacles, and how the shared pipeline answers them

| obstacle | answer |
|---|---|
| training data has only the XI, never the squad | **pseudo-squads**: everyone who played for the team in the same event and season; benched members become rows with Y = 0 |
| roles (keeper, batter, all-rounder, bowler) are not in Cricsheet | inferred from each player's history before the match (stumpings, catches, share of overs, batting position); 99.3% of real XIs are feasible under the inferred roles |
| history ends at 2025-12-31, fixtures are later | every model is trained and evaluated with **simulated staleness**: history cut at c = date − Δ, Δ drawn up to 450 days in training, evaluated at Δ ∈ {0, 90, 180, 270, 365} |
| leakage | one as-of join for every feature (strictly earlier dates), time-travel tests that rebuild features from truncated history and require exact equality (gates G6, G7, G8) |
| noise | evaluate with the task's own metric (points captured by the chosen XI), paired bootstrap over fixtures, decisions pre-registered before any test number was seen |

## How results are measured

For each historical fixture in a test year the model picks its XI; **captured** = the actual points those 11 scored,
**oracle** = the best valid XI in hindsight. We report captured/oracle averaged over fixtures and over the Δ grid, plus the
MAE, RMSE and within-match rank correlation the problem statement asks for. Captured points are what the hidden set is scored
on; MAE is dominated by benched zeros and per-match noise, and the model that minimises it (a median) picks worse teams
(details in [evaluation.md](evaluation.md)).
