"""Stage 2: raw Cricsheet JSON -> interim tables + cleaning flags + gate G2.

python src/ingest.py --raw-dir data/raw/cricsheet --out-dir data/interim --audit-dir data/audit --train-end 2025-12-31
"""
import argparse, glob, json, math, os, random, sys
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

random.seed(0); np.random.seed(0)

KEEP_TYPES = {"T20", "IT20"}
BOWLER_KINDS = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}
NOT_DISMISSED = {"retired hurt", "retired not out"}


def parse(path, train_end):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    info = d["info"]
    mt = info.get("match_type")
    date = info["dates"][0]
    if mt not in KEEP_TYPES:
        return {"skip": mt}
    if date > train_end:
        return {"skip": "post_cutoff"}
    mid = os.path.basename(path)[:-5]
    reg = info.get("registry", {}).get("people", {})
    teams = info["teams"]
    xi = {t: list(info["players"].get(t, [])) for t in teams}
    sched = info.get("overs", 20)
    bpo = info.get("balls_per_over", 6)  # The Hundred: 5
    v = Counter()  # invariant violations
    unresolved = set()

    def pid(n):
        if n not in reg:
            unresolved.add(n)
        return reg.get(n, "UNRES:" + n)

    stats = defaultdict(Counter)  # (team, name) -> counts
    first_bat = {}  # (team, name) -> order
    dism_kind = {}
    sub_in, sub_out = set(), set()
    sub_fielder_dismissals = 0
    n_deliv = 0
    super_over = False
    innings_rows = []
    target_overs = None
    for ii, inn in enumerate(d.get("innings", [])):
        if inn.get("super_over"):
            super_over = True
            continue  # PS: no points for super-over events
        bt = inn["team"]
        ft = [t for t in teams if t != bt][0] if len(teams) == 2 else None
        if inn.get("target", {}).get("overs") is not None:
            target_overs = inn["target"]["overs"]
        order = 0
        inn_wk = 0
        bowler_balls = Counter()
        overs = inn.get("overs", [])
        for oi, ov in enumerate(overs):
            legal = 0
            bowlers = set()
            charged = 0
            for dl in ov["deliveries"]:
                n_deliv += 1
                r = dl["runs"]; ex = dl.get("extras", {})
                if r["total"] != r["batter"] + r["extras"]:
                    v["total_mismatch"] += 1
                b, ns, bw = dl["batter"], dl["non_striker"], dl["bowler"]
                for rp in dl.get("replacements", {}).get("match", []):
                    t = rp.get("team")
                    if t in xi:
                        sub_in.add((t, rp["in"])); sub_out.add((t, rp["out"]))
                        if rp["in"] not in xi[t]:
                            xi[t].append(rp["in"])
                for nm in (b, ns):
                    if (bt, nm) not in first_bat:
                        order += 1
                        first_bat[(bt, nm)] = order
                if b not in xi.get(bt, []) or ns not in xi.get(bt, []):
                    v["batter_not_in_xi"] += 1
                if ft and bw not in xi.get(ft, []):
                    v["bowler_not_in_xi"] += 1
                wide = "wides" in ex; nb = "noballs" in ex
                is_legal = not wide and not nb
                s = stats[(bt, b)]
                s["runs"] += r["batter"]
                if not wide:
                    s["balls_faced"] += 1
                if not dl.get("non_boundary"):
                    s["fours"] += r["batter"] == 4
                    s["sixes"] += r["batter"] == 6
                bs = stats[(ft, bw)]
                c = r["batter"] + ex.get("wides", 0) + ex.get("noballs", 0)
                bs["runs_conceded"] += c
                charged += c
                bowlers.add(bw)
                if is_legal:
                    legal += 1
                    bs["legal_balls_bowled"] += 1
                    bowler_balls[bw] += 1
                    bs["dots"] += r["batter"] == 0
                for w in dl.get("wickets", []):
                    k = w["kind"]
                    if k not in NOT_DISMISSED:
                        inn_wk += 1
                        dism_kind[(bt, w["player_out"])] = k
                    if k in BOWLER_KINDS:
                        bs["wickets"] += 1
                        bs["lbw_bowled"] += k in ("lbw", "bowled")
                    fl = w.get("fielders", [])
                    for fd in fl:
                        if not fd.get("substitute") and ft and fd.get("name") and fd["name"] not in xi.get(ft, []):
                            v["fielder_not_in_xi"] += 1
                    real = [fd["name"] for fd in fl if not fd.get("substitute") and fd.get("name")]
                    if len(real) < len([fd for fd in fl if fd.get("name")]):
                        sub_fielder_dismissals += 1
                    if k == "caught" and real:
                        stats[(ft, real[0])]["catches"] += 1
                    elif k == "caught and bowled":
                        bs["catches"] += 1
                    elif k == "stumped" and real:
                        stats[(ft, real[0])]["stumpings"] += 1
                    elif k == "run out" and real:
                        key = "runout_direct" if len(real) == 1 else "runout_indirect"
                        for nm in real:
                            stats[(ft, nm)][key] += 1
            if legal > bpo:
                v["over_gt6_legal"] += 1
            if legal < bpo and oi < len(overs) - 1:
                v["over_lt6_midinnings"] += 1
            if len(bowlers) == 1 and legal == bpo and charged == 0:
                stats[(ft, next(iter(bowlers)))]["maidens"] += 1
        if inn_wk > 10:
            v["wickets_gt10"] += 1
        cap = math.ceil(sched / 5) * 6
        v["bowler_over_quota"] += sum(n > cap for n in bowler_balls.values())
        innings_rows.append({"match_id": mid, "innings": ii, "bat_team": bt, "violations": sum(v.values())})

    outcome = info.get("outcome", {})
    m = {
        "match_id": mid, "date": date, "gender": info.get("gender"), "team_type": info.get("team_type"),
        "match_type": mt, "balls_per_over": bpo, "event": info.get("event", {}).get("name"), "season": str(info.get("season")),
        "venue": info.get("venue"), "city": info.get("city"), "sched_overs": sched,
        "team1": teams[0], "team2": teams[1] if len(teams) > 1 else None,
        "toss_winner": info.get("toss", {}).get("winner"), "toss_decision": info.get("toss", {}).get("decision"),
        "result": outcome.get("result"), "winner": outcome.get("winner"),
        "n_deliveries": n_deliv, "no_play": n_deliv == 0,
        "no_result_with_play": outcome.get("result") == "no result" and n_deliv > 0,
        "reduced": sched < 20 or (target_overs is not None and target_overs < sched),
        "super_over": super_over, "xi_size1": len(xi[teams[0]]), "xi_size2": len(xi.get(teams[-1], [])),
        "n_violations": sum(v.values()), "n_unresolved": len(unresolved),
        "sub_fielder_dismissals": sub_fielder_dismissals, "path": path,
    }
    rows = []
    for t in teams:
        for nm in xi[t]:
            s = stats.get((t, nm), Counter())
            k = dism_kind.get((t, nm))
            rows.append({
                "match_id": mid, "team": t, "opp": [x for x in teams if x != t][0] if len(teams) == 2 else None,
                "player_id": pid(nm), "name": nm,
                "runs": s["runs"], "balls_faced": s["balls_faced"], "fours": s["fours"], "sixes": s["sixes"],
                "dismissed": k is not None, "dismissal_kind": k, "bat_position": first_bat.get((t, nm), 0),
                "legal_balls_bowled": s["legal_balls_bowled"], "runs_conceded": s["runs_conceded"], "dots": s["dots"],
                "wickets": s["wickets"], "lbw_bowled": s["lbw_bowled"], "maidens": s["maidens"],
                "catches": s["catches"], "stumpings": s["stumpings"],
                "runout_direct": s["runout_direct"], "runout_indirect": s["runout_indirect"],
                "is_sub_in": (t, nm) in sub_in, "is_sub_out": (t, nm) in sub_out,
            })
    # names that appear in events but not in the XI (non-sub) still need a registry id
    for (t, nm) in stats:
        if t is not None:
            pid(nm)
    return {"match": m, "rows": rows, "innings": innings_rows, "violations": dict(v), "unresolved": sorted(unresolved)}


def _parse(args):
    try:
        return parse(*args)
    except Exception as e:  # noqa: BLE001  gate reports parse errors
        return {"error": f"{args[0]}: {e!r}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw/cricsheet")
    ap.add_argument("--out-dir", default="data/interim")
    ap.add_argument("--audit-dir", default="data/audit")
    ap.add_argument("--train-end", default="2025-12-31")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    out = os.path.join(a.out_dir, "player_match_raw.parquet")
    if os.path.exists(out) and not a.rebuild:
        print("cached:", out); return
    files = sorted(glob.glob(os.path.join(a.raw_dir, "*", "*", "*.json")))
    skipped, errors, matches, rows, inns, viol, unres = Counter(), [], [], [], [], Counter(), Counter()
    with Pool(a.workers) as pool:
        for r in pool.imap(_parse, [(f, a.train_end) for f in files], chunksize=64):
            if "error" in r:
                errors.append(r["error"])
            elif "skip" in r:
                skipped[str(r["skip"])] += 1
            else:
                matches.append(r["match"]); rows += r["rows"]; inns += r["innings"]
                viol.update(r["violations"]); unres.update(r["unresolved"])
    M = pd.DataFrame(matches).sort_values(["date", "match_id"]).reset_index(drop=True)
    # duplicates: same id twice, or same (date, teams, venue)
    M["teams_key"] = M[["team1", "team2"]].astype(str).apply(lambda x: "|".join(sorted(x)), axis=1)
    dup_id = M.duplicated("match_id", keep=False)
    dup_ctx = M.duplicated(["date", "teams_key", "venue"], keep=False)
    dups = M.loc[dup_id | dup_ctx, ["match_id", "date", "teams_key", "venue", "path"]]
    M = M.drop_duplicates("match_id", keep="first")
    P = pd.DataFrame(rows)
    P = P[P.match_id.isin(M.match_id)].drop_duplicates(["match_id", "team", "player_id"])
    ctx = M[["match_id", "date", "gender", "team_type", "match_type", "event", "season", "venue", "sched_overs", "balls_per_over", "no_play", "reduced", "n_violations"]]
    P = P.merge(ctx, on="match_id", how="left").sort_values(["date", "match_id", "team", "player_id"]).reset_index(drop=True)
    os.makedirs(a.out_dir, exist_ok=True); os.makedirs(a.audit_dir, exist_ok=True)
    M.to_parquet(os.path.join(a.out_dir, "matches.parquet"), index=False)
    pd.DataFrame(inns).to_parquet(os.path.join(a.out_dir, "innings_audit.parquet"), index=False)
    P.to_parquet(out, index=False)

    n_del = int(M.n_deliveries.sum())
    xi_sizes = Counter(list(M.xi_size1) + list(M.xi_size2))
    rates = {k: val / max(n_del, 1) for k, val in viol.items()}
    innings_viol_rate = float((pd.DataFrame(inns).violations > 0).mean())
    assert (M.date <= a.train_end).all(), "leakage guard: post-cutoff match"
    gate = {
        "unresolved_names_zero": len(unres) == 0,
        "parse_errors_zero": len(errors) == 0,
        "innings_with_violation_rate_lt_2pct": innings_viol_rate < 0.02,
        "all_dates_le_train_end": bool((M.date <= a.train_end).all()),
        "xi_sizes_sane": sum(c for s, c in xi_sizes.items() if s not in (11, 12)) / sum(xi_sizes.values()) < 0.01,
    }
    audit = {
        "files_scanned": len(files), "kept_matches": len(M), "skipped_by_type": dict(skipped),
        "parse_errors": errors[:50], "n_parse_errors": len(errors), "duplicates": dups.to_dict("records"),
        "n_player_rows": len(P), "n_deliveries": n_del, "violations": dict(viol), "violation_rate_per_delivery": rates,
        "innings_with_any_violation_rate": innings_viol_rate, "unresolved_names": dict(unres.most_common(50)),
        "xi_sizes": {str(k): v for k, v in sorted(xi_sizes.items())},
        "no_play": int(M.no_play.sum()), "no_result_with_play": int(M.no_result_with_play.sum()),
        "reduced": int(M.reduced.sum()), "super_over": int(M.super_over.sum()),
        "sub_fielder_dismissals": int(M.sub_fielder_dismissals.sum()),
        "by_gender": M.gender.value_counts().to_dict(), "by_team_type": M.team_type.value_counts().to_dict(),
        "date_range": [M.date.min(), M.date.max()], "gate": gate, "pass": all(gate.values()),
    }
    with open(os.path.join(a.audit_dir, "ingest_audit.json"), "w") as f:
        json.dump(audit, f, indent=1, default=str)
    print(json.dumps({k: audit[k] for k in ("kept_matches", "n_player_rows", "skipped_by_type", "violations", "innings_with_any_violation_rate", "xi_sizes", "gate")}, indent=1, default=str))
    sys.exit(0 if audit["pass"] else 1)


if __name__ == "__main__":
    main()
