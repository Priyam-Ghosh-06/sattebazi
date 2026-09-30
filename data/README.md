# data

| folder | contents | in git |
|---|---|---|
| `raw/cricsheet/{male,female}/<year>/<match_id>.json` | Cricsheet JSON (all formats; only T20/IT20 dated ≤ 2025-12-31 are used) | no: `run.sh` downloads it |
| `raw/registry/` | Cricsheet people and name registries (player IDs, name variants) | no: downloaded |
| `raw/test_fixtures.json` | the hidden test file, when released | no |
| `interim/` | matches, per-player-per-match raw counts, batter × bowler pairs | no: rebuilt |
| `processed/` | player-match with Φ, roles, pseudo-squads, features, staleness table, graph snapshots and features | no: rebuilt |
| `audit/` | one gate report per stage (`<stage>_audit.json`) | yes |
| `mock/` | 40 mock fixtures from Oct–Dec 2025 (pseudo-squads as squads) + their truth, for end-to-end dry runs | yes |

Raw data is read-only. No file dated after 2025-12-31 enters any table used for training.
