"""Stage 5: pseudo-squad reconstruction (negatives for the appearance model) + gate G5.

Ŝ(match, team) = everyone who played for the team in the same (event, season); ±30 days if no event.
in_ext marks the SHOULD variant: Ŝ ∪ {played for the team in the prior 365 days}.
python src/squads.py --in data/processed/player_match.parquet --out data/processed/squads.parquet
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd


def build(P, window_days=30, ext_days=365):
    P = P[["match_id", "team", "player_id", "date", "event", "season"]].copy()
    P["d"] = pd.to_datetime(P.date)
    TM = P.drop_duplicates(["match_id", "team"])[["match_id", "team", "d", "event", "season"]]
    has_ev = TM.event.notna()
    # base squad by (team, event, season)
    mem = P[P.event.notna()].drop_duplicates(["team", "event", "season", "player_id"])[["team", "event", "season", "player_id"]]
    A = TM[has_ev].merge(mem, on=["team", "event", "season"])
    # fallback: ±window days
    nb = TM[~has_ev].merge(P[["team", "player_id", "d"]].rename(columns={"d": "pd"}), on="team")
    B = nb[(nb.pd - nb.d).abs() <= pd.Timedelta(days=window_days)].drop(columns="pd")
    S = pd.concat([A, B]).drop_duplicates(["match_id", "team", "player_id"])[["match_id", "team", "player_id"]]
    S["in_base"] = True
    # extended: anyone who played for the team in the prior ext_days (strictly before)
    tp = P[["team", "player_id", "d"]].rename(columns={"d": "pd"})
    E = TM[["match_id", "team", "d"]].merge(tp, on="team")
    E = E[(E.pd < E.d) & (E.pd >= E.d - pd.Timedelta(days=ext_days))].drop_duplicates(["match_id", "team", "player_id"])[["match_id", "team", "player_id"]]
    S = S.merge(E.assign(in_ext_only=True), on=["match_id", "team", "player_id"], how="outer")
    S["in_base"] = S.in_base.fillna(False).astype(bool)
    S = S.drop(columns="in_ext_only")
    played = P[["match_id", "team", "player_id"]].assign(played=1)
    S = S.merge(played, on=["match_id", "team", "player_id"], how="left")
    S["played"] = S.played.fillna(0).astype("int8")
    return S.merge(TM[["match_id", "team", "d"]], on=["match_id", "team"]).assign(
        date=lambda x: x.d.dt.strftime("%Y-%m-%d")).drop(columns="d").sort_values(["date", "match_id", "team", "player_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/processed/player_match.parquet")
    ap.add_argument("--out", default="data/processed/squads.parquet")
    ap.add_argument("--audit-dir", default="data/audit")
    ap.add_argument("--window-days", type=int, default=30)
    ap.add_argument("--ext-days", type=int, default=365)
    a = ap.parse_args()
    P = pd.read_parquet(a.inp)
    S = build(P, a.window_days, a.ext_days)
    S.to_parquet(a.out, index=False)
    base = S[S.in_base]
    sz = base.groupby(["match_id", "team"]).size()
    ext = S.groupby(["match_id", "team"]).size()
    miss = len(P) - int(base.played.sum())
    au = {"rows_base": len(base), "rows_ext": len(S),
          "base_size": {"median": float(sz.median()), "p10": float(sz.quantile(.1)), "p90": float(sz.quantile(.9))},
          "ext_size": {"median": float(ext.median()), "p10": float(ext.quantile(.1)), "p90": float(ext.quantile(.9))},
          "played_share_base": float(base.played.mean()), "played_share_ext": float(S.played.mean()),
          "frac_base_lt_11": float((sz < 11).mean()), "played_rows_missing_from_base": miss,
          "played_rows_in_ext_only_frac": float(S[~S.in_base].played.mean()) if (~S.in_base).any() else 0.0}
    au["pass"] = au["frac_base_lt_11"] < 0.005 and miss == 0
    json.dump(au, open(os.path.join(a.audit_dir, "squads_audit.json"), "w"), indent=1)
    print(json.dumps(au, indent=1))
    sys.exit(0 if au["pass"] else 1)


if __name__ == "__main__":
    main()
