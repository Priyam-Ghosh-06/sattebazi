"""Gate G3. Run: python tests/test_points.py  (also pytest-compatible)."""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from shared.preprocessing.points import econ_points, phi, sr_points  # noqa: E402


def sr(runs, balls):
    return int(sr_points(np.array([runs]), np.array([balls]))[0])


def ec(rc, balls):
    return int(econ_points(np.array([rc]), np.array([balls]))[0])


def test_sr_bands():
    # SR = 100*runs/balls; use 100 balls so runs == SR
    assert sr(171, 100) == 5 and sr(170, 100) == 3
    assert sr(150.01, 100) == 3 and sr(150, 100) == 1
    assert sr(130, 100) == 1 and sr(129.99, 100) == 0
    assert sr(70.01, 100) == 0 and sr(70, 100) == -1
    assert sr(60, 100) == -1 and sr(59.99, 100) == -3
    assert sr(50, 100) == -3 and sr(49.99, 100) == -5
    assert sr(0, 9) == 0 and sr(0, 10) == -5  # 10-ball minimum


def test_econ_bands():
    # econ = 6*rc/balls; use 60 balls so rc/10 == econ
    assert ec(49.9, 60) == 5 and ec(50, 60) == 3
    assert ec(59.9, 60) == 3 and ec(60, 60) == 1
    assert ec(70, 60) == 1 and ec(70.1, 60) == 0
    assert ec(99.9, 60) == 0 and ec(100, 60) == -1
    assert ec(110, 60) == -1 and ec(110.1, 60) == -3
    assert ec(120, 60) == -3 and ec(120.1, 60) == -5
    assert ec(100, 11) == 0 and ec(100, 12) == -5  # 2-over minimum


def row(**kw):
    base = dict(runs=0, balls_faced=0, fours=0, sixes=0, dismissed=False, legal_balls_bowled=0, runs_conceded=0,
                dots=0, wickets=0, lbw_bowled=0, maidens=0, catches=0, stumpings=0, runout_direct=0, runout_indirect=0)
    base.update(kw)
    return phi(pd.DataFrame([base])).iloc[0]


def test_milestone_highest_only():
    assert row(runs=99, balls_faced=200).bat_milestone == 15
    assert row(runs=100, balls_faced=200).bat_milestone == 20
    assert row(runs=24).bat_milestone == 0 and row(runs=25).bat_milestone == 5


def test_haul_duck_field_six():
    assert row(wickets=5).bowl_haul == 15 and row(wickets=2).bowl_haul == 0
    assert row(dismissed=True).bat_duck == -2 and row(dismissed=True, runs=1).bat_duck == 0
    assert row(catches=3).field == 22 and row(stumpings=1, runout_indirect=1).field == 15
    assert row(sixes=1, runs=6).bat_boundary == 2  # DECISION: six = +2 only
    r = row(runs=30, balls_faced=10, fours=3, sixes=1, wickets=3, lbw_bowled=1, dots=8,
            legal_balls_bowled=24, runs_conceded=20, maidens=1)
    # 3 + 30 + (3+2) + 5 + 0 + SR300:+5 | 8 + 75 + 5 + 5 + 6 + econ5.0:+3
    assert r.total == 3 + 30 + 5 + 5 + 5 + 8 + 75 + 5 + 5 + 6 + 3, r.total


def ref_score(path, pid_by_name):
    """Independent, deliberately naive per-delivery re-scorer for golden matches."""
    d = json.load(open(path, encoding="utf-8"))
    from collections import defaultdict
    s = defaultdict(lambda: defaultdict(int))
    for inn in d["innings"]:
        if inn.get("super_over"):
            continue
        for ov in inn["overs"]:
            dl = ov["deliveries"]
            bw = {x["bowler"] for x in dl}
            legal = [x for x in dl if not ({"wides", "noballs"} & set(x.get("extras", {})))]
            if len(bw) == 1 and len(legal) == d["info"].get("balls_per_over", 6) and \
                    sum(x["runs"]["batter"] + x.get("extras", {}).get("wides", 0) + x.get("extras", {}).get("noballs", 0) for x in dl) == 0:
                s[dl[0]["bowler"]]["maid"] += 1
            for x in dl:
                e = x.get("extras", {})
                b = s[x["batter"]]; o = s[x["bowler"]]
                b["r"] += x["runs"]["batter"]
                b["bf"] += "wides" not in e
                b["b4"] += x["runs"]["batter"] == 4 and not x.get("non_boundary")
                b["b6"] += x["runs"]["batter"] == 6 and not x.get("non_boundary")
                o["rc"] += x["runs"]["batter"] + e.get("wides", 0) + e.get("noballs", 0)
                if "wides" not in e and "noballs" not in e:
                    o["lb"] += 1
                    o["dot"] += x["runs"]["batter"] == 0
                for w in x.get("wickets", []):
                    k = w["kind"]
                    if k not in ("retired hurt", "retired not out"):
                        s[w["player_out"]]["out"] = 1
                    if k in ("bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"):
                        o["w"] += 1; o["lbwb"] += k in ("lbw", "bowled")
                    f = [z["name"] for z in w.get("fielders", []) if not z.get("substitute") and "name" in z]
                    if k == "caught" and f: s[f[0]]["ct"] += 1
                    if k == "caught and bowled": o["ct"] += 1
                    if k == "stumped" and f: s[f[0]]["st"] += 1
                    if k == "run out":
                        for z in f: s[z]["ro1" if len(f) == 1 else "ro2"] += 1
    out = {}
    for team, names in d["info"]["players"].items():
        for n in names:
            v = s[n]; r = v["r"]
            t = 3 + r + v["b4"] + 2 * v["b6"]
            t += 20 if r >= 100 else 15 if r >= 75 else 10 if r >= 50 else 5 if r >= 25 else 0
            t -= 2 if v["out"] and r == 0 else 0
            if v["bf"] >= 10:
                q = 100 * r / v["bf"]
                t += 5 if q > 170 else 3 if q > 150 else 1 if q >= 130 else 0 if q > 70 else -1 if q >= 60 else -3 if q >= 50 else -5
            t += v["dot"] + 25 * v["w"] + 5 * v["lbwb"] + (15 if v["w"] >= 5 else 10 if v["w"] >= 4 else 5 if v["w"] >= 3 else 0) + 6 * v["maid"]
            if v["lb"] >= 12:
                q = 6 * v["rc"] / v["lb"]
                t += 5 if q < 5 else 3 if q < 6 else 1 if q <= 7 else 0 if q < 10 else -1 if q <= 11 else -3 if q <= 12 else -5
            t += 6 * v["ct"] + (4 if v["ct"] >= 3 else 0) + 10 * v["st"] + 10 * v["ro1"] + 5 * v["ro2"]
            out[pid_by_name[n]] = t
    return out


def test_golden_matches():
    pm = "data/processed/player_match.parquet"
    if not os.path.exists(pm):
        print("skip golden: run points.py first"); return
    P = pd.read_parquet(pm)
    M = pd.read_parquet("data/interim/matches.parquet")
    picks = [M[(M.event == "Indian Premier League") & ~M.super_over].iloc[-1],
             M[(M.match_type == "T20") & (M.team_type == "international") & (M.gender == "male")].iloc[-1],
             M[(M.gender == "female")].iloc[-1],
             M[M.super_over & (M.xi_size1 == 12)].iloc[0] if (M.super_over & (M.xi_size1 == 12)).any() else M[M.super_over].iloc[0]]
    for m in picks:
        sub = P[P.match_id == m.match_id]
        ref = ref_score(m.path, dict(zip(sub.name, sub.player_id)))
        got = dict(zip(sub.player_id, sub.total))
        assert ref == got, (m.match_id, {k: (ref[k], got[k]) for k in ref if ref[k] != got.get(k)})


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
