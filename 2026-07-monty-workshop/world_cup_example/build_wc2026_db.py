#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build a single SQLite database of all 2026 FIFA World Cup data.

Two public datasets are combined (both cloned into wc2026_raw_data/ on first run):

  worldcup.json/2026/   openfootball (public domain): schedule, results, goals,
                        groups, teams, stadiums, squads, qualifying play-offs.
  wc2026-events/        WhoScored event data (Noah Bair): one CSV per match with
                        per-event detail (passes, shots, cards, xG qualifiers...)
                        plus a team-name/colour metadata CSV.

The script is idempotent: it shallow-clones any missing source repo, then
rebuilds every table from scratch.

    uv run world_cup_example/build_wc2026_db.py            # -> worldcup2026.db (alongside this script)
    uv run world_cup_example/build_wc2026_db.py --db wc.db
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "wc2026_raw_data"
JSON_DIR = RAW_DIR / "worldcup.json" / "2026"
EVENTS_DIR = RAW_DIR / "wc2026-events" / "data"

# Source repos, shallow-cloned into RAW_DIR if not already present.
REPOS: list[tuple[str, str]] = [
    ("worldcup.json", "https://github.com/openfootball/worldcup.json"),
    ("wc2026-events", "https://github.com/nlbair/wc2026-events"),
]

# CSV fields can be wider than the default limit (qualifier blobs are huge).
csv.field_size_limit(1 << 24)


# --------------------------------------------------------------------------- #
# source repos
# --------------------------------------------------------------------------- #
def ensure_repos() -> None:
    """Shallow-clone (depth=1) each source repo into RAW_DIR if it's missing."""
    RAW_DIR.mkdir(exist_ok=True)
    for name, url in REPOS:
        dest = RAW_DIR / name
        if dest.exists():
            print(f"  {name}: present")
            continue
        print(f"  {name}: cloning {url} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def to_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_bool(v: Any) -> int:
    if isinstance(v, bool):
        return int(v)
    if v in ("True", "true", "1"):
        return 1
    if v in ("False", "false", "0", "", None):
        return 0
    return 0


def _pair(v: Any) -> tuple[Any, Any]:
    """Split a 2-element score list into a pair, else (None, None)."""
    try:
        if v is not None and len(v) == 2:
            return v[0], v[1]
    except TypeError:
        pass
    return None, None


def split_score(score: Any) -> tuple[Any, ...]:
    """Flatten an openfootball `score` into (ft1, ft2, ht1, ht2, et1, et2, p1, p2).

    `score` may be a dict with ft/ht/et/p keys, or a bare [a, b] list (quali),
    or missing entirely.
    """
    out: dict[str, Any] = {k: None for k in ("ft", "ht", "et", "p")}
    s: Any = score  # re-bind as Any so isinstance narrowing doesn't leak Unknown
    if isinstance(score, dict):
        for k in out:
            out[k] = s.get(k)
    elif isinstance(score, list):
        out["ft"] = s
    return (*_pair(out["ft"]), *_pair(out["ht"]), *_pair(out["et"]), *_pair(out["p"]))


def canon(name: str) -> str:
    """Canonical team key for cross-dataset matching (accent/alias-insensitive)."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = "".join(c for c in s.lower() if c.isalnum())
    for token in ("and", "republicof", "republic", "ir"):  # noise words
        if token != s:  # never blank out the whole name
            s = s.replace(token, "")
    aliases = {
        "czech": "czechia",
        "turkey": "turkiye",
        "unitedstates": "usa",
        "korea": "southkorea",
        "southsouthkorea": "southkorea",
        "caboverde": "capeverde",
        "congodr": "drcongo",
        "cotedivoire": "ivorycoast",
    }
    return aliases.get(s, s)


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
SCHEMA = """
DROP TABLE IF EXISTS events;
DROP TABLE IF EXISTS event_matches;
DROP TABLE IF EXISTS team_meta;
DROP TABLE IF EXISTS quali_goals;
DROP TABLE IF EXISTS quali_playoffs;
DROP TABLE IF EXISTS players;
DROP TABLE IF EXISTS goals;
DROP TABLE IF EXISTS matches;
DROP TABLE IF EXISTS stadiums;
DROP TABLE IF EXISTS group_teams;
DROP TABLE IF EXISTS teams;

CREATE TABLE teams (
    name            TEXT PRIMARY KEY,
    name_normalised TEXT,
    continent       TEXT,
    confederation   TEXT,
    fifa_code       TEXT,
    group_letter    TEXT,
    flag_icon       TEXT,
    flag_unicode    TEXT
);

CREATE TABLE group_teams (
    group_name TEXT,
    team_name  TEXT,
    PRIMARY KEY (group_name, team_name)
);

CREATE TABLE stadiums (
    name     TEXT PRIMARY KEY,
    city     TEXT,
    country  TEXT,
    timezone TEXT,
    capacity INTEGER,
    coords   TEXT
);

CREATE TABLE matches (
    id           INTEGER PRIMARY KEY,
    stage        TEXT,            -- 'group' | 'knockout'
    round        TEXT,
    group_letter TEXT,
    date         TEXT,
    time         TEXT,
    team1        TEXT,
    team2        TEXT,
    ground       TEXT,
    ft1 INTEGER, ft2 INTEGER,
    ht1 INTEGER, ht2 INTEGER,
    et1 INTEGER, et2 INTEGER,
    pen1 INTEGER, pen2 INTEGER
);

CREATE TABLE goals (
    match_id  INTEGER REFERENCES matches(id),
    team_side INTEGER,            -- 1 or 2
    team_name TEXT,
    scorer    TEXT,
    minute    TEXT,
    penalty   INTEGER,
    own_goal  INTEGER
);

CREATE TABLE players (
    team_name     TEXT,
    fifa_code     TEXT,
    group_letter  TEXT,
    number        INTEGER,
    position      TEXT,
    name          TEXT,
    club_name     TEXT,
    club_country  TEXT,
    date_of_birth TEXT
);

CREATE TABLE quali_playoffs (
    id         INTEGER PRIMARY KEY,
    round      TEXT,
    date       TEXT,
    time       TEXT,
    time_local TEXT,
    team1      TEXT,
    team2      TEXT,
    ground     TEXT,
    ft1 INTEGER, ft2 INTEGER,
    ht1 INTEGER, ht2 INTEGER,
    et1 INTEGER, et2 INTEGER,
    pen1 INTEGER, pen2 INTEGER
);

CREATE TABLE quali_goals (
    quali_id  INTEGER REFERENCES quali_playoffs(id),
    team_side INTEGER,
    team_name TEXT,
    scorer    TEXT,
    minute    TEXT,
    penalty   INTEGER,
    own_goal  INTEGER
);

CREATE TABLE team_meta (
    whoscored_name TEXT PRIMARY KEY,
    display_name   TEXT,
    group_letter   TEXT,
    primary_color  TEXT,
    flag_code      TEXT
);

CREATE TABLE event_matches (
    match_id    INTEGER PRIMARY KEY,   -- WhoScored match id
    home_team   TEXT,
    away_team   TEXT,
    home_score  INTEGER,
    away_score  INTEGER,
    match_date  TEXT,
    source_file TEXT,
    wc_match_id INTEGER REFERENCES matches(id),  -- link to openfootball, may be NULL
    n_events    INTEGER
);

CREATE TABLE events (
    match_id        INTEGER REFERENCES event_matches(match_id),
    event_id        INTEGER,           -- WhoScored global event id
    event_type_id   INTEGER,
    minute          INTEGER,
    second          INTEGER,
    expanded_minute INTEGER,
    period_value    INTEGER,
    period_name     TEXT,
    team_id         INTEGER,
    team            TEXT,
    player_id       INTEGER,
    player          TEXT,
    event           TEXT,
    outcome         TEXT,
    card_type       TEXT,
    x  REAL, y  REAL,
    end_x REAL, end_y REAL,
    blocked_x REAL, blocked_y REAL,
    goal_mouth_y REAL, goal_mouth_z REAL,
    related_event_id  INTEGER,
    related_player_id INTEGER,
    is_shot  INTEGER,
    is_goal  INTEGER,
    is_touch INTEGER,
    qualifiers TEXT                    -- JSON of all non-empty qual_* columns
);
"""

INDEXES = """
CREATE INDEX idx_matches_date    ON matches(date);
CREATE INDEX idx_goals_match     ON goals(match_id);
CREATE INDEX idx_players_team    ON players(team_name);
CREATE INDEX idx_events_match    ON events(match_id);
CREATE INDEX idx_events_event    ON events(event);
CREATE INDEX idx_events_player   ON events(player);
CREATE INDEX idx_events_team     ON events(team);
CREATE INDEX idx_events_shot     ON events(is_shot);
CREATE INDEX idx_events_goal     ON events(is_goal);
"""


# --------------------------------------------------------------------------- #
# openfootball ingestion
# --------------------------------------------------------------------------- #
def load_teams(con: sqlite3.Connection) -> int:
    rows: list[tuple[Any, ...]] = []
    for t in load_json(JSON_DIR / "worldcup.teams.json"):
        rows.append((
            t["name"], t.get("name_normalised"), t.get("continent"),
            t.get("confed"), t.get("fifa_code"), t.get("group"),
            t.get("flag_icon"), t.get("flag_unicode"),
        ))
    con.executemany("INSERT INTO teams VALUES (?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def load_groups(con: sqlite3.Connection) -> int:
    rows: list[tuple[Any, ...]] = []
    for g in load_json(JSON_DIR / "worldcup.groups.json")["groups"]:
        for team in g["teams"]:
            rows.append((g["name"], team))
    con.executemany("INSERT INTO group_teams VALUES (?,?)", rows)
    return len(rows)


def load_stadiums(con: sqlite3.Connection) -> int:
    rows: list[tuple[Any, ...]] = []
    for s in load_json(JSON_DIR / "worldcup.stadiums.json")["stadiums"]:
        rows.append((
            s["name"], s.get("city"), s.get("cc"), s.get("timezone"),
            to_int(s.get("capacity")), s.get("coords"),
        ))
    con.executemany("INSERT INTO stadiums VALUES (?,?,?,?,?,?)", rows)
    return len(rows)


def _insert_goals(con: sqlite3.Connection, table: str, fk_col: str, match_id: int, match: Any) -> int:
    rows: list[tuple[Any, ...]] = []
    for side in (1, 2):
        team_name = match.get(f"team{side}")
        goals_list: Any = match.get(f"goals{side}") or []
        for g in goals_list:
            rows.append((
                match_id, side, team_name, g.get("name"), str(g.get("minute", "")),
                to_bool(g.get("penalty")), to_bool(g.get("owngoal")),
            ))
    if rows:
        con.executemany(
            f"INSERT INTO {table} ({fk_col},team_side,team_name,scorer,minute,penalty,own_goal)"
            " VALUES (?,?,?,?,?,?,?)",
            rows,
        )
    return len(rows)


def load_matches(con: sqlite3.Connection) -> tuple[int, int]:
    data = load_json(JSON_DIR / "worldcup.json")
    n_goals = 0
    for i, m in enumerate(data["matches"], start=1):
        ft1, ft2, ht1, ht2, et1, et2, p1, p2 = split_score(m.get("score"))
        stage = "group" if m.get("group") else "knockout"
        con.execute(
            "INSERT INTO matches VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, stage, m.get("round"), m.get("group"), m.get("date"), m.get("time"),
             m.get("team1"), m.get("team2"), m.get("ground"),
             ft1, ft2, ht1, ht2, et1, et2, p1, p2),
        )
        n_goals += _insert_goals(con, "goals", "match_id", i, m)
    return len(data["matches"]), n_goals


def load_quali(con: sqlite3.Connection) -> tuple[int, int]:
    path = JSON_DIR / "worldcup.quali_playoffs.json"
    if not path.exists():
        return 0, 0
    data = load_json(path)
    n_goals = 0
    for i, m in enumerate(data["matches"], start=1):
        ft1, ft2, ht1, ht2, et1, et2, p1, p2 = split_score(m.get("score"))
        con.execute(
            "INSERT INTO quali_playoffs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (i, m.get("round"), m.get("date"), m.get("time"), m.get("time_local"),
             m.get("team1"), m.get("team2"), m.get("ground"),
             ft1, ft2, ht1, ht2, et1, et2, p1, p2),
        )
        n_goals += _insert_goals(con, "quali_goals", "quali_id", i, m)
    return len(data["matches"]), n_goals


def load_squads(con: sqlite3.Connection) -> int:
    path = JSON_DIR / "worldcup.squads.json"
    if not path.exists():
        return 0
    rows: list[tuple[Any, ...]] = []
    for team in load_json(path):
        players: Any = team.get("players") or []
        for p in players:
            club: Any = p.get("club") or {}
            rows.append((
                team.get("name"), team.get("fifa_code"), team.get("group"),
                to_int(p.get("number")), p.get("pos"), p.get("name"),
                club.get("name"), club.get("country"), p.get("date_of_birth"),
            ))
    con.executemany("INSERT INTO players VALUES (?,?,?,?,?,?,?,?,?)", rows)
    return len(rows)


# --------------------------------------------------------------------------- #
# events ingestion
# --------------------------------------------------------------------------- #
def load_team_meta(con: sqlite3.Connection) -> int:
    path = EVENTS_DIR / "metadata" / "team_meta.csv"
    rows: list[tuple[Any, ...]] = []
    seen: set[Any] = set()
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = r["whoscored_name"]
            if key in seen:
                continue
            seen.add(key)
            rows.append((key, r["display_name"], r["group"],
                         r["primary_color"], r["flag_code"]))
    con.executemany("INSERT INTO team_meta VALUES (?,?,?,?,?)", rows)
    return len(rows)


# Core event columns: CSV header name -> events table column.
EVENT_CORE: dict[str, str] = {
    "id": "event_id",
    "eventId": "event_type_id",
    "minute": "minute",
    "second": "second",
    "expandedMinute": "expanded_minute",
    "period_value": "period_value",
    "period_name": "period_name",
    "teamId": "team_id",
    "team": "team",
    "playerId": "player_id",
    "player": "player",
    "event": "event",
    "outcome": "outcome",
    "cardType": "card_type",
    "x": "x", "y": "y",
    "endX": "end_x", "endY": "end_y",
    "blockedX": "blocked_x", "blockedY": "blocked_y",
    "goalMouthY": "goal_mouth_y", "goalMouthZ": "goal_mouth_z",
    "relatedEventId": "related_event_id",
    "relatedPlayerId": "related_player_id",
    "isShot": "is_shot",
    "isGoal": "is_goal",
    "isTouch": "is_touch",
}
INT_COLS = {"event_id", "event_type_id", "minute", "second", "expanded_minute",
            "period_value", "team_id", "player_id", "related_event_id",
            "related_player_id"}
FLOAT_COLS = {"x", "y", "end_x", "end_y", "blocked_x", "blocked_y",
              "goal_mouth_y", "goal_mouth_z"}
BOOL_COLS = {"is_shot", "is_goal", "is_touch"}

EVENT_INSERT_COLS = [
    "match_id", "event_id", "event_type_id", "minute", "second",
    "expanded_minute", "period_value", "period_name", "team_id", "team",
    "player_id", "player", "event", "outcome", "card_type", "x", "y",
    "end_x", "end_y", "blocked_x", "blocked_y", "goal_mouth_y", "goal_mouth_z",
    "related_event_id", "related_player_id", "is_shot", "is_goal", "is_touch",
    "qualifiers",
]


def _convert(col: str, value: Any) -> Any:
    if col in INT_COLS:
        return to_int(value)
    if col in FLOAT_COLS:
        return to_float(value)
    if col in BOOL_COLS:
        return to_bool(value)
    return value if value not in ("", None) else None


def load_one_event_file(con: sqlite3.Connection, path: Path) -> tuple[tuple[Any, ...] | None, int]:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        qual_cols = [c for c in header if c.startswith("qual_")]

        first = next(reader, None)
        if first is None:
            return None, 0

        meta: tuple[Any, ...] = (
            to_int(first.get("match_id")),
            first.get("home_team"),
            first.get("away_team"),
            to_int(first.get("home_score")),
            to_int(first.get("away_score")),
            first.get("match_date"),
        )
        match_id = meta[0]

        batch: list[list[Any]] = []
        for row in _iter_rows(first, reader):
            quals: dict[str, Any] = {}
            for qc in qual_cols:
                v = row.get(qc)
                if v in ("", None, "False"):
                    continue
                quals[qc[len("qual_"):]] = True if v == "True" else v
            values: list[Any] = [match_id]
            for csv_col, db_col in EVENT_CORE.items():
                values.append(_convert(db_col, row.get(csv_col)))
            values.append(json.dumps(quals, ensure_ascii=False) if quals else None)
            batch.append(values)

    placeholders = ",".join("?" * len(EVENT_INSERT_COLS))
    con.executemany(
        f"INSERT INTO events ({','.join(EVENT_INSERT_COLS)}) VALUES ({placeholders})",
        batch,
    )
    return meta, len(batch)


def _iter_rows(first: Any, reader: Iterator[Any]) -> Iterator[Any]:
    yield first
    yield from reader


def _date_diff(a: str, b: str) -> int:
    """Absolute difference in days between two YYYY-MM-DD strings (big if unparsable)."""
    from datetime import date

    try:
        ya, ma, da = map(int, a.split("-"))
        yb, mb, db = map(int, b.split("-"))
        return abs((date(ya, ma, da) - date(yb, mb, db)).days)
    except (ValueError, AttributeError):
        return 99


def build_wc_link(con: sqlite3.Connection) -> dict[frozenset[str], list[tuple[Any, Any]]]:
    """Map team-pair -> list of (date, matches.id) for cross-linking.

    WhoScored stamps evening kickoffs with the US-local *next* day, so events
    and openfootball can differ by a day; we match on the team pair and pick the
    closest date (within 1 day).
    """
    link: dict[frozenset[str], list[tuple[Any, Any]]] = {}
    for mid, date, t1, t2 in con.execute(
        "SELECT id, date, team1, team2 FROM matches"
    ):
        link.setdefault(frozenset({canon(t1), canon(t2)}), []).append((date, mid))
    return link


def lookup_wc_match(
    link: dict[frozenset[str], list[tuple[Any, Any]]], home: Any, away: Any, date: Any
) -> int | None:
    candidates = link.get(frozenset({canon(home), canon(away)}))
    if not candidates:
        return None
    best_date, best_id = min(candidates, key=lambda c: _date_diff(c[0], date))
    return best_id if _date_diff(best_date, date) <= 1 else None


def load_events(con: sqlite3.Connection) -> tuple[int, int, int]:
    raw_dir = EVENTS_DIR / "raw"
    files = sorted(raw_dir.glob("wc2026_*_events.csv"))
    link = build_wc_link(con)

    total_events = 0
    linked = 0
    for i, path in enumerate(files, start=1):
        meta, n = load_one_event_file(con, path)
        if meta is None:
            print(f"  [{i}/{len(files)}] {path.name}: empty, skipped")
            continue
        match_id, home, away, hs, as_, date = meta
        wc_id = lookup_wc_match(link, home, away, date)
        if wc_id is not None:
            linked += 1
        con.execute(
            "INSERT INTO event_matches VALUES (?,?,?,?,?,?,?,?,?)",
            (match_id, home, away, hs, as_, date, path.name, wc_id, n),
        )
        total_events += n
        print(f"  [{i}/{len(files)}] {home} vs {away} ({date}): "
              f"{n} events{'' if wc_id else '  [unlinked]'}")
    return len(files), total_events, linked


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(ROOT / "worldcup2026.db"),
                    help="output SQLite file (default: worldcup2026.db)")
    args = ap.parse_args()

    print("source repos:")
    ensure_repos()
    if not JSON_DIR.exists():
        sys.exit(f"missing data dir: {JSON_DIR}")
    if not EVENTS_DIR.exists():
        sys.exit(f"missing data dir: {EVENTS_DIR}")

    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)

        print("openfootball / worldcup.json:")
        print(f"  teams        {load_teams(con)}")
        print(f"  group_teams  {load_groups(con)}")
        print(f"  stadiums     {load_stadiums(con)}")
        n_m, n_g = load_matches(con)
        print(f"  matches      {n_m}  (goals {n_g})")
        n_q, n_qg = load_quali(con)
        print(f"  quali        {n_q}  (goals {n_qg})")
        print(f"  players      {load_squads(con)}")

        print("\nwc2026-events / WhoScored:")
        print(f"  team_meta    {load_team_meta(con)}")
        n_files, n_ev, n_linked = load_events(con)

        con.executescript(INDEXES)
        con.commit()

        print(f"\nDone -> {db_path}")
        print(f"  {n_files} event files, {n_ev:,} events, "
              f"{n_linked}/{n_files} linked to openfootball matches")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
