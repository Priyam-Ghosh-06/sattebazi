"""Graph ingestion: the batter x bowler delivery graph. Per (match, batter, bowler): balls faced (wides excluded),
runs off the bat, bowler-credited dismissals of that batter; super overs excluded; the kept matches of ingest."""

import json
import os
import pandas as pd
from multiprocessing import Pool


BOWLER_KINDS = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}


PAIRS, SNAPS = "data/interim/pair_match.parquet", "data/processed/matchup_snapshots.pkl"


def parse(args):
    path, mid = args
    d = json.load(open(path, encoding="utf-8"))
    reg = d["info"].get("registry", {}).get("people", {})
    acc = {}
    for inn in d.get("innings", []):
        if inn.get("super_over"):
            continue  # PS: no points for super overs
        for ov in inn.get("overs", []):
            for dl in ov["deliveries"]:
                if "wides" in dl.get("extras", {}):
                    continue  # a wide is not a ball faced
                i, j = reg.get(dl["batter"]), reg.get(dl["bowler"])
                if i is None or j is None:
                    continue
                a = acc.setdefault((i, j), [0, 0, 0])
                a[0] += 1; a[1] += dl["runs"]["batter"]
                a[2] += sum(w["kind"] in BOWLER_KINDS and w.get("player_out") == dl["batter"] for w in dl.get("wickets", []))
    return [(mid, i, j, n, r, w) for (i, j), (n, r, w) in acc.items()]


def ingest(workers):
    M = pd.read_parquet("data/interim/matches.parquet", columns=["match_id", "date", "path"])
    with Pool(workers) as pool:
        rows = [r for part in pool.imap(parse, zip(M.path.map(os.path.normpath), M.match_id), chunksize=64) for r in part]
    D = pd.DataFrame(rows, columns=["match_id", "bat", "bowl", "n", "r", "w"]).merge(M[["match_id", "date"]], on="match_id")
    D = D.sort_values(["date", "match_id", "bat", "bowl"], kind="stable").reset_index(drop=True)
    D.to_parquet(PAIRS, index=False)
    print(json.dumps({"pair_match_rows": len(D), "matches": int(D.match_id.nunique()), "balls": int(D.n.sum()), "max_date": D.date.max()}))
