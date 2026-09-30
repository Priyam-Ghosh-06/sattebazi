# E23 ADV: decision (SPEC §14)

- gates: pass ({"adv_overfit": true, "adv_smoke": true, "adv_tests": true, "causality": true, "cb_overfit": true, "cb_smoke": true, "cb_tests": true, "features": true, "g7": true, "ingest": true, "points": true, "roles": true, "squads": true, "stale": true, "t5_determinism": true})
- ADV - B0s captured per fixture: F24 -0.90 [-3.52, +1.63]; F25 +2.82 [-0.06, +5.61] -> submit B0s
- variants (same runs, test, per fixture): ADV-CB128 F24 +4.45 [+1.77, +7.22]; ADV_direct_a3-ADV_direct_tau F24 +0.49 [-0.35, +1.27]; ADV_rollout-ADV_direct_a3 F24 +0.91 [-0.07, +1.94]; ADV_student-ADV_direct_a3 F24 -17.54 [-20.34, -14.64]; ADV-CB128 F25 +4.76 [+2.15, +7.40]; ADV_direct_a3-ADV_direct_tau F25 -0.47 [-1.27, +0.37]; ADV_rollout-ADV_direct_a3 F25 +0.17 [-0.76, +1.08]; ADV_student-ADV_direct_a3 F25 -14.02 [-16.84, -11.37]
- headline captured/oracle (test): B0s:F24=0.6245 B0s:F25=0.6284 CB128:F24=0.6165 CB128:F25=0.6254 ADV:F24=0.6234 ADV:F25=0.6326 ADV_direct_a3:F24=0.6232 ADV_direct_a3:F25=0.6324 ADV_direct_tau:F24=0.6224 ADV_direct_tau:F25=0.6330 ADV_rollout:F24=0.6245 ADV_rollout:F25=0.6326 ADV_student:F24=0.5975 ADV_student:F25=0.6115
- R4: written once from these numbers.
