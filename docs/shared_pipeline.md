# The shared spine: ingestion, preprocessing, gates

All three approaches start from the same tables. Every stage writes `data/audit/<stage>_audit.json` and stops the run if its
gate fails.

| stage | code | what it does | gate |
|---|---|---|---|
| ingestion | `src/shared/ingestion/ingest.py` | Cricsheet JSON → match table + per-player-per-match counts. Keeps T20/IT20 dated ≤ `--train-end` only (13,085 matches; 7,862 Test/ODI/ODM/MDM files skipped). Registry IDs are the player key. Super-over deliveries are excluded. | G1 file counts, G2 delivery invariants (0.95% of innings flagged, not dropped) |
| fantasy points | `src/shared/preprocessing/points.py` | the PS points table as 13 components + total (six = +2 only; duck for any dismissal on 0; run-out direct/indirect from the fielder count) | G3: band-edge tests and golden matches re-scored by an independent per-delivery scorer (`tests/test_points.py`) |
| roles | `src/shared/preprocessing/roles.py` | WK / BAT / AR / BOWL as of each date from the last 30 appearances; used only by the XI optimiser | G4: 99.33% of historical XIs feasible |
| pseudo-squads | `src/shared/preprocessing/squads.py` | squad = everyone who played for the team in the event and season (±30 days without an event); median 15 players | G5: < 0.5% of squads smaller than 11 |
| causal features | `src/shared/preprocessing/causal.py`, `features.py`, `context.py` | one strict as-of join for every feature: form (EWMA bank, career means), opportunity, appearance shares, squad-relative ranks, opponent and venue context | G6: time-travel test on 200 rows (`tests/test_causality.py`) |
| test fixtures | `src/shared/ingestion/fixtures.py` | reads the hidden fixture file (schema unknown, several layouts accepted), resolves names to registry IDs (exact → registry variants → fuzzy within the team) and prints the resolution rate per fixture | — |

Each approach then adds its own preprocessing (staleness copies, history tokens, capacities, the delivery graph); see the
approach pages.

## Leakage audit

1. Ingestion drops every match dated after `--train-end`; feature building asserts it again.
2. Every feature goes through `causal.asof`: a strict `merge_asof`, so a row dated *d* (or cut at *c*) sees only matches dated
   before it. Same-day doubleheaders never see each other.
3. Time-travel tests rebuild features from history truncated at the cutoff and require exact equality:
   G6 (tabular features, 200 rows), G7 (staleness-aware features, 272 rows at Δ > 0), T1 (neural history tokens, 388 and 305
   candidates), G8 (graph snapshots: no source dated ≥ snapshot; a snapshot refitted from truncated data is bit-identical).
   G6 found a real tie-ordering bug on its first run; G7 found the opponent being read from post-cutoff rows. Both are fixed.
4. Walk-forward folds retrain on years before the test year; the test years were scored once (rule R4 in the design history).
5. Every candidate row stores the latest source date it used (`feat_source_max_date`), and every audit asserts it is before the
   cutoff.
