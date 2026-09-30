# Experiments (smoke walk-forward, 2026-09-29)

> Every number below is from smoke samples at staleness Δ = 0. Pre-registered next experiments (E16–E21: staleness, choice objective, team totals, full folds, cold start, neural ladders) are in `SPEC.md` §12.

**Protocol.** Walk-forward: train on 2019-01-01 … Dec 31 of year Y−1 (features use all earlier history), test on year Y.
Smoke set: 400 randomly sampled matches per test year (seed 0) for Y ∈ {2024, 2025}; candidates = the pseudo-squad
(median 15 per team). XI chosen by the exact ILP under the PS composition rules.
**Captured** = actual Φ of the chosen XI (benched = 0). **Oracle** = best feasible XI in hindsight.
Every variant is compared with B0 **on the same matches** (paired bootstrap of the per-match captured difference, 2000 resamples).
Rows ending `_confirm` are a fresh replication: seed 1, 700 matches × {2023, 2024, 2025}, ref = B0 rerun on those matches.
About 40 variants were compared, so one "better" at 95% on the first sample is expected by chance. Only a result that also
holds on the fresh sample counts.

**B0** = μ: LightGBM L2 on played rows; p: LightGBM logistic on pseudo-squads + sum-to-11 shift; score = p·μ.

## Headline
| | captured/oracle | captured pts/match |
|---|---|---|
| best hand baseline (EWMA-3 × appearance share) | 0.633 | 439 |
| B0 | 0.684 | 476 |
| B0 + opponent/own-squad strength + venue (E14, adopted) | 0.691 (confirm: +2.6 [+0.5, +4.7]) | 480 |
| knowing the real XI (hindsight, B0's μ) | 0.734 | — |
| noise ceiling (E2): leaky same-year player mean, within-match Spearman among players | 0.483 (B0: 0.347) | — |

Mock end-to-end (`evaluate.py`, model trained ≤ 2025-09-30, 40 real Oct–Dec 2025 fixtures): XI captured 63% of a
hindsight top-11, and 89% of picked players actually played. Two runs gave byte-identical CSVs; retraining gave an identical model.

## Results table (mean over folds)
| variant | captured/oracle | captured | Δcaptured vs ref [95% CI] | Spearman squad | Spearman played | MAE played | MAE squad | verdict |
|---|---|---|---|---|---|---|---|---|
| B0_L2_logit_sum11 | 0.684 | 476.1 | ref | 0.501 | 0.347 | 25.890 | 22.843 | reference |
| B0_L2_logit_sum11_confirm | 0.678 | 474.2 | ref | 0.495 | 0.339 | 25.887 | 22.832 | reference |
| B0_L2_logit_sum11_train2011 | 0.682 | 475.2 | -0.9 [-4.2, +2.2] | 0.499 | 0.342 | 25.912 | 22.866 | no significant difference |
| B0_L2_logit_sum11_train2015 | 0.681 | 473.7 | -2.4 [-5.3, +0.6] | 0.500 | 0.343 | 25.886 | 22.860 | no significant difference |
| E10_lambdarank_direct | 0.681 | 473.3 | -2.8 [-6.8, +1.1] | 0.489 |  |  | 28.736 | no significant difference |
| E10_rerank_stage2_top18 | 0.681 | 474.0 | -2.1 [-5.8, +1.5] | 0.495 |  |  | n/a (offset scores) | no significant difference |
| E11_drop_violating_matches_train | 0.681 | 474.0 | -2.1 [-4.8, +0.4] | 0.501 | 0.347 | 25.884 | 22.837 | no significant difference |
| E12_backfit_orthogonal_experts | 0.682 | 474.6 | -1.5 [-4.5, +1.7] | 0.497 | 0.337 | 25.955 | 22.869 | no significant difference |
| E12_interaction_constrained_groups | 0.683 | 475.2 | -0.9 [-3.6, +1.7] | 0.499 | 0.343 | 25.972 | 22.880 | no significant difference |
| E12_role_gated_experts | 0.680 | 473.4 | -2.7 [-5.8, +0.6] | 0.498 | 0.342 | 25.932 | 22.873 | no significant difference |
| E13_listnet_custom_objective | 0.677 | 470.8 | -5.3 [-9.4, -1.1] | 0.460 |  |  | 28.288 | worse |
| E13_monotone_form_share | 0.681 | 473.8 | -2.2 [-4.7, +0.1] | 0.500 | 0.346 | 25.894 | 22.845 | no significant difference |
| E13_recency_hl1y | 0.682 | 475.1 | -0.9 [-4.1, +2.3] | 0.498 | 0.345 | 25.933 | 22.898 | no significant difference |
| E13_recency_hl2y | 0.683 | 475.9 | -0.2 [-3.3, +2.9] | 0.500 | 0.344 | 25.907 | 22.863 | no significant difference |
| E13_recency_hl4y | 0.683 | 474.9 | -1.1 [-3.8, +1.6] | 0.501 | 0.346 | 25.904 | 22.860 | no significant difference |
| E13_selection_boundary_weights | 0.682 | 474.7 | -1.4 [-4.0, +1.2] | 0.500 | 0.346 | 25.911 | 22.873 | no significant difference |
| E14_plus_opp_own_squad | 0.686 | 477.7 | +1.6 [-2.0, +5.2] | 0.505 | 0.355 | 25.733 | 22.747 | no significant difference |
| E14_plus_opp_own_squad_confirm | 0.681 | 475.7 | +1.5 [-0.5, +3.6] | 0.499 | 0.347 | 25.755 | 22.729 | no significant difference |
| E14_plus_opp_venue | 0.691 | 480.2 | +4.1 [+0.6, +7.7] | 0.504 | 0.357 | 25.691 | 22.720 | better |
| E14_plus_opp_venue_confirm | 0.682 | 476.8 | +2.6 [+0.5, +4.7] | 0.499 | 0.348 | 25.730 | 22.718 | better |
| E14_plus_venue | 0.681 | 474.5 | -1.5 [-4.5, +1.4] | 0.499 | 0.347 | 25.851 | 22.822 | no significant difference |
| E15_spo_plus_finetune | 0.683 | 475.9 | -0.2 [-3.7, +3.1] | 0.497 |  |  | 23.023 | no significant difference |
| E15_spo_plus_finetune_confirm | 0.678 | 474.0 | -0.2 [-2.1, +1.7] | 0.493 |  |  | 22.981 | no significant difference |
| E15_spo_plus_finetune_h1 | 0.682 | 474.9 | -1.2 [-4.6, +1.9] | 0.498 |  |  | 22.927 | no significant difference |
| E15_stage1_crossfit_control | 0.681 | 474.3 | -1.8 [-5.2, +1.4] | 0.497 | 0.342 | 25.943 | 22.920 | no significant difference |
| E15_stage1_crossfit_control_confirm | 0.677 | 473.6 | -0.6 [-2.4, +1.4] | 0.493 | 0.337 | 25.918 | 22.875 | no significant difference |
| E3_eb_career_x_share10 | 0.612 | 425.0 | -51.0 [-58.3, -43.7] | 0.382 | 0.308 | 26.609 | 23.655 | worse |
| E3_ewm10_total_x_share10 | 0.628 | 436.3 | -39.8 [-46.5, -32.8] | 0.405 | 0.317 | 26.367 | 23.346 | worse |
| E3_ewm30_total_x_share10 | 0.624 | 433.4 | -42.7 [-49.1, -35.7] | 0.400 | 0.315 | 26.337 | 23.430 | worse |
| E3_ewm3_total_x_share10 | 0.633 | 439.3 | -36.8 [-43.4, -29.9] | 0.411 | 0.304 | 26.822 | 23.305 | worse |
| E3_last5_total_x_share10 | 0.625 | 434.7 | -41.3 [-47.9, -34.7] | 0.402 | 0.277 | 27.620 | 23.517 | worse |
| E3_role_mean_x_share10 | 0.583 | 405.3 | -70.7 [-78.5, -62.3] | 0.352 | 0.242 | 27.431 | 23.989 | worse |
| E4_ewm_hl10_only | 0.678 | 472.0 | -4.1 [-7.1, -1.1] | 0.498 | 0.343 | 25.924 | 22.892 | worse |
| E4_ewm_hl30_only | 0.678 | 471.9 | -4.2 [-7.0, -1.0] | 0.496 | 0.343 | 25.925 | 22.902 | worse |
| E4_ewm_hl3_only | 0.680 | 473.4 | -2.7 [-5.6, -0.0] | 0.500 | 0.346 | 25.900 | 22.852 | worse |
| E4_no_ewm | 0.680 | 473.3 | -2.8 [-5.8, +0.5] | 0.498 | 0.344 | 25.927 | 22.895 | no significant difference |
| E5_direct_squad_L1 | 0.675 | 469.2 | -6.9 [-11.2, -2.7] | 0.486 |  |  | 21.265 | worse |
| E5_direct_squad_L2 | 0.682 | 474.3 | -1.8 [-5.8, +2.2] | 0.489 |  |  | 23.199 | no significant difference |
| E5_mu_L1 | 0.683 | 475.3 | -0.8 [-4.4, +2.8] | 0.488 | 0.342 | 24.931 | 21.814 | no significant difference |
| E5_mu_huber10 | 0.686 | 477.7 | +1.6 [-1.5, +4.9] | 0.493 | 0.344 | 24.941 | 21.859 | no significant difference |
| E5_mu_huber25 | 0.686 | 477.6 | +1.5 [-1.3, +4.4] | 0.499 | 0.344 | 25.155 | 22.099 | no significant difference |
| E5_mu_huber25_confirm | 0.679 | 474.6 | +0.4 [-1.4, +2.3] | 0.495 | 0.341 | 25.137 | 22.079 | no significant difference |
| E6_p_ext_squad_train | 0.676 | 470.4 | -5.7 [-9.8, -2.1] | 0.477 | 0.347 | 25.890 | 23.039 | worse |
| E6_p_logit_uncalibrated | 0.683 | 475.3 | -0.8 [-3.7, +2.1] | 0.493 | 0.347 | 25.890 | 23.145 | no significant difference |
| E6_p_no_appearance_feats | 0.667 | 463.7 | -12.4 [-16.5, -8.0] | 0.457 | 0.347 | 25.890 | 23.626 | worse |
| E6_p_proxy_share10 | 0.629 | 436.3 | -39.8 [-46.6, -32.6] | 0.401 | 0.347 | 25.890 | 23.491 | worse |
| E7_components_sum | 0.683 | 475.5 | -0.6 [-3.6, +2.3] | 0.500 | 0.345 | 25.832 | 22.790 | no significant difference |
| E8_split_by_gender | 0.680 | 473.3 | -2.8 [-6.0, +0.3] | 0.497 | 0.344 | 25.950 | 22.910 | no significant difference |
| E8_split_by_team_type | 0.683 | 475.9 | -0.2 [-3.8, +3.2] | 0.500 | 0.343 | 25.911 | 22.840 | no significant difference |
| E9_drop_appearance | 0.669 | 465.2 | -10.9 [-15.1, -6.8] | 0.474 | 0.342 | 25.911 | 23.360 | worse |
| E9_drop_context | 0.680 | 473.2 | -2.9 [-5.9, +0.1] | 0.498 | 0.344 | 25.957 | 22.939 | no significant difference |
| E9_drop_form | 0.671 | 467.2 | -8.9 [-13.1, -4.8] | 0.489 | 0.334 | 26.041 | 23.030 | worse |
| E9_drop_opportunity | 0.678 | 472.1 | -4.0 [-7.0, -0.8] | 0.497 | 0.340 | 25.991 | 22.929 | worse |
| E9_drop_squad | 0.682 | 474.8 | -1.3 [-4.1, +1.4] | 0.498 | 0.345 | 25.926 | 22.905 | no significant difference |

Non-model rows: E2 (ICC of Φ: BAT 0.13, BOWL 0.10, AR 0.13, WK 0.16; leaky Spearman ceiling 0.48; MAE 23.2 even if the
player's same-year mean were known) and E4b (Kalman/EWMA memory fitted by one-step error on 2019–2023: best half-life
20 appearances for total (λ≈0.0012), 30 for batting, 15 for bowling, 50 for fielding; one-step R² 0.04 / 0.19 / 0.25 / 0.01).

## What each idea did

**Loss (E5).** L2, L1 and Huber give the same captured points. L1/Huber lower MAE (25.9 → 24.9) and slightly lower rank
correlation. Direct L1 on squad rows is significantly worse (−6.9): the median of a zero-inflated target is ~0 for fringe
players, which is the theory from Entry 0. → Select on L2; report MAE for both. MAE is the wrong metric for this task.

**Appearance (E6, E9).** The largest controllable loss. The learned p beats the appearance-share proxy by +40 pts/match.
Dropping appearance features costs −11 to −12. Training p on the 365-day extended squad is worse (−5.7): the negatives
are too easy. Sum-to-11 calibration improves log-loss (0.414 → 0.395) with no captured change. Knowing the real XI would
add ~5 points of captured/oracle; that gap is the cost of the pre-toss constraint.

**Memory / "hidden state fed forward" (E4, E4b).** The fitted Kalman memory is long (half-life ≈ 20 apps): signal is small
relative to noise. A single half-life is significantly worse than the {3, 10, 30} set (−2.7 to −4.2), so the GBDT
uses several memories at once. That is the empirical answer to "bake time into the model": give it several filters.
Recency sample weights (E13, half-life 1/2/4 years), i.e. putting time into training instead, did nothing.

**Orthogonal experts / hierarchical MoE (E12).** All ties with B0:
- interaction-constrained GBDT, where each tree uses one feature group, making an additive model whose groups are the experts: −0.9 [−3.6, +1.7];
- backfitting, one expert per group fitted in turn on the others' residuals ("one at a time"): −1.5;
- hard gating by role (one expert per role): −2.7 [−5.8, +0.6].

Expert output correlations (2025 fold): form×opportunity 0.75 when fitted simultaneously, 0.49 under backfitting,
all other pairs about 0. Forcing exact orthogonality only changes how the sum is attributed, not the prediction.
Finding: cross-group interactions carry no predictive value here, so the additive, interpretable version is free.

**Reranker (E10), ListNet (E13), SPO+ (E15).** The direct LambdaRank ties (−2.8). The stage-2 GBDT reranker on the top 18
adds −0.3 over its own cross-fitted stage 1. ListNet listwise softmax as a custom objective is significantly worse (−5.3).
Decision-focused SPO+ fine-tuning (the selection regret as the boosting loss) is +1.6 [−0.5, +3.7] over its control on
the first sample and +0.4 [−1.0, +1.7] on the fresh one: not a result.

**Hardcoded structure in training (E13).** Monotone constraints (form, appearance share): −2.2 [−4.7, +0.1]. Weighting
by closeness to the selection boundary: −1.4. Recency weights: 0. Components (bat/bowl/field) summed (E7): −0.6.
Splitting by gender or club/intl (E8): ties.

**Match setting and opponent (E14; your point 2).** Venue alone: nothing. Opponent + own-squad strength: +1.5 (n.s.).
Both together: +4.1 [+0.6, +7.7], replicated at +2.6 [+0.5, +4.7] on 2,100 fresh matches. **Adopted in the final model.**
They are the 6th and 9th features by gain in μ.

**Cleaning (E11).** Dropping matches with delivery-invariant violations from training: −2.1 (n.s.). The cleaning decisions don't move results.

**Training window.** A 2011 or 2015 start is no better than 2019 (−0.9, −2.4; n.s.). The final model trains from 2019
(features still use all history).

## Overall reading
With ICC ≈ 0.13, per-match Φ is mostly noise. B0 already reaches ~72% of the leaky rank ceiling. Algorithmic structure
(losses, rankers, experts, custom objectives, constraints) moved nothing measurable. Only **information** moved the
result: appearance features (+40 vs proxy), multi-scale memory, and opponent/venue context. The remaining gap to
hindsight is mostly who plays, which the pre-toss rule makes unknowable.

## Feature importance (final model, gain share)
- μ: ewm10_total 0.30, ewm30_total 0.11, ewm3_total 0.08, car_total 0.06, lb10 (balls bowled) 0.05, x_opp_ewm10_total 0.04, n_apps 0.03, ewm3_pts_bowl 0.03, x_opp_ewm10_pts_bat 0.02, pos10 0.02
- p: team_gap_matches (played the team's last match?) 0.44, team_type 0.10, share3 0.09, team_days_since 0.06, rank_share10 0.03
