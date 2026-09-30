# Optimising the fantasy XI

`src/shared/optimization/selection.py`, used by every approach.

Given a score s_i = E[Y_i] for every candidate in both squads, a role r_i and a team t_i, the XI is the exact solution of a
0/1 integer program:

    maximise    Σ_i x_i · s_i
    subject to  Σ_i x_i = 11
                1 ≤ #WK ≤ 4,  3 ≤ #BAT ≤ 6,  1 ≤ #AR ≤ 4,  3 ≤ #BOWL ≤ 6
                Σ_{i in team A} x_i ≥ 1,  Σ_{i in team B} x_i ≥ 1
                x_i ∈ {0, 1}

solved with `scipy.optimize.milp` (HiGHS branch and bound; a 30–45 player instance takes a few milliseconds).

**Why exact, and why expected points.** The fantasy score of a team is the sum of its players' scores, so the expected team
score is the sum of E[Y_i] whatever the correlations between teammates are. Maximising Σ x_i E[Y_i] under the rules is
therefore the exact best decision given the model; a greedy "top-11 then repair" can be beaten by the ILP whenever the role
bounds bind (e.g. the 3rd-best keeper is worth less than a 4th bowler). No captain multipliers exist in this PS, so nothing
non-linear enters the objective.

**Details.**
- Ties are broken deterministically by a 1e-7 · rank(player id) term, so two runs give the same XI (test: 1,000 random squads,
  every constraint holds and the answer is reproducible, `tests/test_select.py`).
- Players whose role is unknown (no history) count as batters.
- If the inferred roles make the bounds infeasible (rare: 37 of 1,000 random squads, 0.7% of real XIs), the program drops the
  role bounds, keeps "11 players, both teams", and logs it.
- The same program with the true points as scores gives the **oracle** XI used by the evaluation.
