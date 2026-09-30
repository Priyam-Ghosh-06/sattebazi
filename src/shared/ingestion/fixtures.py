"""Test-time ingestion: the fixture file adapter (the organisers' schema was unpublished, so it accepts several layouts),
name -> registry-id resolution (exact, registry variants, fuzzy within the team) and the fixture's pre-match context."""
import difflib, json, os, re

import pandas as pd


def _get(d, *keys, default=None):
    for k in keys:
        for kk in d:
            if kk.lower().replace(" ", "_") == k:
                return d[kk]
    return default


def load_fixtures(path):
    """Schema is not published yet (ASK). Accepts a list or {"matches"/"fixtures": [...]}; per fixture:
    date|match_date, team1/team2 or teams[], squad1/squad2 or squads{team: [...]}/players{team: [...]},
    optional roles{name: role}, gender, venue, match_type, team_type. Squad entries may be names or {"name":..., "role":...}."""
    raw = json.load(open(path, encoding="utf-8"))
    if isinstance(raw, dict):
        raw = _get(raw, "matches", "fixtures", "data", default=[raw])
    out = []
    for fx in raw:
        teams = _get(fx, "teams") or [_get(fx, "team1", "team_1", "home_team"), _get(fx, "team2", "team_2", "away_team")]
        sq = _get(fx, "squads", "players") or {}
        squads = [sq.get(teams[0]) if isinstance(sq, dict) else None, sq.get(teams[1]) if isinstance(sq, dict) else None]
        squads[0] = squads[0] or _get(fx, "squad1", "team1_squad", "squad_1", default=[])
        squads[1] = squads[1] or _get(fx, "squad2", "team2_squad", "squad_2", default=[])
        roles = dict(_get(fx, "roles", default={}) or {})
        names = []
        for s in squads:
            ns = []
            for p in s:
                if isinstance(p, dict):
                    n = _get(p, "name", "player", "player_name")
                    if _get(p, "role"):
                        roles[n] = _get(p, "role")
                    ns.append(n)
                else:
                    ns.append(p)
            names.append(ns)
        out.append({"date": str(_get(fx, "date", "match_date", "start_date"))[:10], "team1": teams[0], "team2": teams[1],
                    "squad1": names[0], "squad2": names[1], "roles": roles,
                    "gender": _get(fx, "gender"), "venue": _get(fx, "venue"), "match_type": _get(fx, "match_type"),
                    "team_type": _get(fx, "team_type"), "event": _get(fx, "event", "competition", "series", "tournament")})
    return out


ROLE_MAP = {"wk": "WK", "wicketkeeper": "WK", "wicket-keeper": "WK", "keeper": "WK", "wk-batsman": "WK", "wicketkeeper batter": "WK",
            "bat": "BAT", "batsman": "BAT", "batter": "BAT", "ar": "AR", "all-rounder": "AR", "allrounder": "AR", "all rounder": "AR",
            "bowl": "BOWL", "bowler": "BOWL"}


class Resolver:
    def __init__(self, P, registry_dir):
        self.P = P
        people = pd.read_csv(os.path.join(registry_dir, "people.csv"), usecols=["identifier", "name", "unique_name"])
        names = pd.read_csv(os.path.join(registry_dir, "names.csv"))
        self.ids = set(people.identifier)
        pairs = pd.concat([people[["identifier", "name"]], people[["identifier", "unique_name"]].rename(columns={"unique_name": "name"}), names])
        self.by_name = pairs.dropna().assign(k=lambda d: d.name.map(self.key)).groupby("k").identifier.agg(lambda s: sorted(set(s)))
        last = P.sort_values("date").groupby("player_id").agg(last=("date", "last"), name=("name", "last"))
        self.last = last["last"].to_dict()
        self.team_players = P.groupby("team").player_id.agg(set).to_dict()
        self.name_of = last["name"].to_dict()
        self.variants = pairs.dropna().groupby("identifier").name.agg(list).to_dict()
        self.teams = sorted(P.team.unique())

    @staticmethod
    def key(n):
        return " ".join(re.sub(r"[^a-z ]", "", str(n).lower().replace(".", " ")).split())

    def team(self, t):
        if t in self.team_players:
            return t, "exact"
        m = difflib.get_close_matches(str(t), self.teams, 1, 0.8)
        return (m[0], "fuzzy") if m else (t, "unresolved")

    def player(self, name, team):
        if name in self.ids:
            return name, "id"
        pool = self.team_players.get(team, set())
        cands = self.by_name.get(self.key(name), [])
        if cands:
            best = sorted(cands, key=lambda i: (i in pool, self.last.get(i, "")), reverse=True)[0]
            return best, "registry"
        lookup = {self.key(v): i for i in pool for v in self.variants.get(i, [self.name_of.get(i, "")])}
        m = difflib.get_close_matches(self.key(name), list(lookup), 1, 0.85)
        if m:
            return lookup[m[0]], "fuzzy_team"
        return "UNRES:" + str(name), "unresolved"


def team_ctx(P, M, team, date):
    h = M[((M.team1 == team) | (M.team2 == team)) & (M.date < date)].tail(1)
    if not len(h):
        return {"gender": "male", "team_type": "international", "match_type": "T20", "sched_overs": 20, "balls_per_over": 6}
    r = h.iloc[0]
    return {"gender": r.gender, "team_type": r.team_type, "match_type": r.match_type, "sched_overs": r.sched_overs, "balls_per_over": r.balls_per_over}
