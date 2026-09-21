"""Generate the synthetic BSE ticketing database.

Deterministic: a fixed RNG seed plus an explicit --anchor-date means the same
invocation always produces the same database.

    uv run python data/seed.py [--anchor-date YYYY-MM-DD] [--out data/bse.db]

The generator is anchored on "today" so that time-relative demo questions
("...last month") return data no matter when the repo is run. See the
DEMO BACKFILL note on `_recent_backfill` for the tradeoff this makes.
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

RNG_SEED = 20260918
HERE = Path(__file__).parent

VENUES = [
    (1, "Barclays Center", "Brooklyn", "NY", 19000),
    (2, "Prudential Center", "Newark", "NJ", 16500),
]

TEAMS = [
    (1, "Brooklyn Nets", "NBA", 1),
    (2, "New York Liberty", "WNBA", 1),
    (3, "Boston Celtics", "NBA", 0),
    (4, "Philadelphia 76ers", "NBA", 0),
    (5, "Miami Heat", "NBA", 0),
    (6, "Chicago Bulls", "NBA", 0),
    (7, "Milwaukee Bucks", "NBA", 0),
    (8, "Toronto Raptors", "NBA", 0),
    (9, "Golden State Warriors", "NBA", 0),
    (10, "Los Angeles Lakers", "NBA", 0),
    (11, "Las Vegas Aces", "WNBA", 0),
    (12, "Connecticut Sun", "WNBA", 0),
    (13, "Chicago Sky", "WNBA", 0),
    (14, "St. John's Red Storm", "NCAA", 0),
]
NBA_VISITORS = [3, 4, 5, 6, 7, 8, 9, 10]
WNBA_VISITORS = [11, 12, 13]

CATEGORIES = [
    (1, "NBA Regular Season"),
    (2, "NBA Playoffs"),
    (3, "WNBA"),
    (4, "Concert"),
    (5, "Comedy"),
    (6, "Family Show"),
    (7, "College Basketball"),
]

CONCERT_ACTS = [
    "Aurora Bay", "The Fulton Street Revival", "Nine Mile Radio", "Vela & the Tide",
    "Kingsland Collective", "Marcy Avenue", "Gowanus Gold", "Neon Cathedral",
    "The Red Hook Sessions", "Paloma Rivers", "Static Meridian", "Coney Electric",
]
COMEDY_ACTS = ["Dee Vance Live", "The Late Shift Tour", "Mira Okonjo: Unfiltered", "Brooklyn Roast Night"]
FAMILY_SHOWS = ["Ice Spectacular", "Monster Truck Mayhem", "Sesame Street Live", "Cirque Lumina"]

FIRST_NAMES = ["James", "Maria", "Andre", "Priya", "Daniel", "Sofia", "Marcus", "Aisha", "Tyler", "Grace",
               "Luis", "Nina", "Omar", "Hannah", "Jordan", "Chloe", "Victor", "Elena", "Kevin", "Rosa"]
LAST_NAMES = ["Rivera", "Chen", "Okafor", "Patel", "Nguyen", "Brooks", "Silva", "Hassan", "Morgan", "Delgado",
              "Kim", "Walsh", "Bennett", "Adeyemi", "Russo", "Fitzgerald", "Park", "Santos", "Doyle", "Ferreira"]
CITIES = [("Brooklyn", "NY"), ("Manhattan", "NY"), ("Queens", "NY"), ("Jersey City", "NJ"),
          ("Newark", "NJ"), ("Hoboken", "NJ"), ("Staten Island", "NY"), ("Yonkers", "NY")]
TIERS = ["none", "none", "none", "silver", "silver", "gold", "season_ticket_holder"]
CHANNELS = ["web", "web", "web", "mobile_app", "mobile_app", "box_office", "resale", "group_sales"]

#: Relative ticket demand per event kind, driving order volume and pricing.
DEMAND = {"nba": 1.0, "playoff": 1.45, "wnba": 0.72, "concert": 0.95,
          "comedy": 0.48, "family": 0.62, "college": 0.4}

# (name, level, base_price, seats_per_row, rows)
SECTION_TEMPLATE = [
    ("Courtside {}", "Courtside", 1250.0, 10, 2),
    ("Lower {}", "Lower", 210.0, 20, 12),
    ("Club {}", "Club", 385.0, 14, 6),
    ("Suite {}", "Suite", 900.0, 8, 2),
    ("Upper {}", "Upper", 65.0, 22, 14),
]


def _daterange_events(rng, anchor: date, years: int):
    """Realistic multi-season schedule going back `years` from the anchor."""
    events = []
    for season_offset in range(years, -1, -1):
        season_start_year = anchor.year - season_offset
        # NBA regular season: Oct -> Apr
        d = date(season_start_year, 10, 15)
        end = date(season_start_year + 1, 4, 12)
        while d <= end:
            if d <= anchor and rng.random() < 0.34:
                events.append(("nba", d))
            d += timedelta(days=1)
        # NBA playoffs: late Apr -> early Jun
        d = date(season_start_year + 1, 4, 20)
        end = date(season_start_year + 1, 6, 5)
        while d <= end:
            if d <= anchor and rng.random() < 0.10:
                events.append(("playoff", d))
            d += timedelta(days=1)
        # WNBA: May -> Sep
        d = date(season_start_year + 1, 5, 10)
        end = date(season_start_year + 1, 9, 20)
        while d <= end:
            if d <= anchor and rng.random() < 0.13:
                events.append(("wnba", d))
            d += timedelta(days=1)
    # Non-sporting events year-round across the whole window
    d = anchor - timedelta(days=365 * years + 180)
    while d <= anchor:
        if rng.random() < 0.17:
            events.append((rng.choice(["concert", "concert", "comedy", "family", "college"]), d))
        d += timedelta(days=1)
    return events


def _recent_backfill(rng, anchor: date):
    """DEMO BACKFILL — guarantees the trailing 120 days contain Brooklyn Nets home
    games and a spread of other categories.

    Tradeoff: real NBA games do not happen in August, so this is deliberately
    unrealistic. Without it, "How many tickets were sold for Brooklyn Nets home
    games last month?" returns 0 rows whenever the repo is run in the off-season,
    which makes the headline demo look broken. Documented in the README.
    """
    events = []
    d = anchor - timedelta(days=120)
    while d <= anchor:
        if rng.random() < 0.22:
            events.append(("nba", d))
        if rng.random() < 0.12:
            events.append((rng.choice(["concert", "comedy", "family", "wnba"]), d))
        d += timedelta(days=1)
    return events


def _make_sections(rng) -> tuple[list, dict]:
    """Seating sections plus a section_id -> (rows, seats_per_row) map."""
    sections, seats_by_section, sid = [], {}, 1
    for venue_id, *_ in VENUES:
        for tmpl, level, price, seats, rows in SECTION_TEMPLATE:
            for n in range(1, 5):
                sections.append((sid, venue_id, tmpl.format(n), level,
                                 round(price * rng.uniform(0.9, 1.15), 2)))
                seats_by_section[sid] = (rows, seats)
                sid += 1
    return sections, seats_by_section


def _make_customers(rng, anchor: date, years: int, count: int = 4000) -> list:
    customers = []
    for cid in range(1, count + 1):
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        city, state = rng.choice(CITIES)
        signup = anchor - timedelta(days=rng.randint(30, 365 * years + 400))
        customers.append((cid, first, last,
                          f"{first.lower()}.{last.lower()}{cid}@example.com",
                          city, state, signup.isoformat(), rng.choice(TIERS)))
    return customers


def _make_events(rng, anchor: date, years: int) -> tuple[list, dict]:
    """Event rows plus an event_id -> (venue_id, kind, date) lookup."""
    team_names = {t[0]: t[1] for t in TEAMS}
    raw = _daterange_events(rng, anchor, years) + _recent_backfill(rng, anchor)
    raw.sort(key=lambda item: item[1])

    events, meta, eid = [], {}, 1
    raw = _daterange_events(rng, anchor, years) + _recent_backfill(rng, anchor)
    raw.sort(key=lambda x: x[1])

    for kind, d in raw:
        venue_id = 1 if kind in ("nba", "playoff", "wnba") or rng.random() < 0.75 else 2
        home = away = None
        if kind == "nba":
            cat, home, away = 1, 1, rng.choice(NBA_VISITORS)
            name = f"Brooklyn Nets vs. {team_names[away]}"
        elif kind == "playoff":
            cat, home, away = 2, 1, rng.choice(NBA_VISITORS)
            name = f"NBA Playoffs: Nets vs. {team_names[away]}"
        elif kind == "wnba":
            cat, home, away = 3, 2, rng.choice(WNBA_VISITORS)
            name = f"New York Liberty vs. {team_names[away]}"
        elif kind == "concert":
            cat, name = 4, f"{rng.choice(CONCERT_ACTS)} — Live"
        elif kind == "comedy":
            cat, name = 5, rng.choice(COMEDY_ACTS)
        elif kind == "family":
            cat, name = 6, rng.choice(FAMILY_SHOWS)
        else:
            cat, home, away = 7, 14, None
            name = "St. John's Red Storm — College Basketball"

        if d > anchor:
            status = "scheduled"
        elif rng.random() < 0.012:
            status = rng.choice(["cancelled", "postponed"])
        else:
            status = "completed"

        announced = d - timedelta(days=rng.randint(30, 210))
        doors = rng.choice(["17:30", "18:00", "18:30", "19:00"])
        events.append((eid, venue_id, cat, home, away, name, d.isoformat(),
                       doors, announced.isoformat(), status))
        meta[eid] = (venue_id, kind, d)
        eid += 1
    return events, meta


def _make_orders_and_tickets(rng, events, event_meta, sections_by_venue,
                             seats_by_section, anchor: date) -> tuple[list, list]:
    orders, tickets = [], []
    oid = tid = 1
    for ev in events:
        event_id, venue_id, _, _, _, _, ev_date, _, _, status = ev
        if status in ("cancelled", "postponed"):
            continue
        kind = event_meta[event_id][1]
        n_orders = int(rng.gauss(420, 90) * DEMAND[kind])
        n_orders = max(40, n_orders)
        ed = date.fromisoformat(ev_date)
        for _ in range(n_orders):
            lead = rng.randint(0, 150)
            ots = datetime.combine(ed - timedelta(days=lead),
                                   datetime.min.time()) + timedelta(
                                       hours=rng.randint(8, 23), minutes=rng.randint(0, 59))
            channel = rng.choice(CHANNELS)
            r = rng.random()
            ostatus = "completed" if r < 0.94 else ("refunded" if r < 0.985 else "cancelled")
            orders.append((oid, rng.randint(1, 4000), event_id, ots.strftime("%Y-%m-%d %H:%M:%S"),
                           channel, ostatus))

            for _ in range(rng.choice([1, 2, 2, 2, 3, 4, 4, 6])):
                section = rng.choice(sections_by_venue[venue_id])
                section_id, _, _, _, base = section
                rows, seats = seats_by_section[section_id]
                face = round(base * DEMAND[kind] * rng.uniform(0.85, 1.3), 2)
                is_comp = 1 if rng.random() < 0.015 else 0
                if is_comp:
                    paid, fees = 0.0, 0.0
                elif channel == "resale":
                    paid, fees = round(face * rng.uniform(0.75, 1.9), 2), round(face * 0.14, 2)
                else:
                    paid, fees = round(face * rng.uniform(0.8, 1.0), 2), round(face * 0.11, 2)
                scanned = None
                if ostatus == "completed" and ed <= anchor and rng.random() < 0.9:
                    scanned = f"{ev_date} {rng.choice(['18:12', '18:44', '19:03', '19:21'])}:00"
                tickets.append((tid, oid, event_id, section_id,
                                chr(ord("A") + rng.randrange(rows)), rng.randint(1, seats),
                                face, paid, fees, is_comp, scanned))
                tid += 1
            oid += 1

    return orders, tickets


def build(out_path: Path, anchor: date, years: int = 3) -> None:
    rng = random.Random(RNG_SEED)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)

    conn = sqlite3.connect(out_path)
    conn.executescript((HERE / "schema.sql").read_text())
    conn.executemany("INSERT INTO venues VALUES (?,?,?,?,?)", VENUES)
    conn.executemany("INSERT INTO teams VALUES (?,?,?,?)", TEAMS)
    conn.executemany("INSERT INTO event_categories VALUES (?,?)", CATEGORIES)

    sections, seats_by_section = _make_sections(rng)
    conn.executemany("INSERT INTO seating_sections VALUES (?,?,?,?,?)", sections)
    sections_by_venue: dict[int, list] = {}
    for section in sections:
        sections_by_venue.setdefault(section[1], []).append(section)

    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?,?)",
                     _make_customers(rng, anchor, years))

    events, event_meta = _make_events(rng, anchor, years)
    conn.executemany("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)", events)

    orders, tickets = _make_orders_and_tickets(
        rng, events, event_meta, sections_by_venue, seats_by_section, anchor)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", orders)
    conn.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?,?,?)", tickets)
    conn.commit()

    print(f"Wrote {out_path}")
    for table in ("venues", "teams", "event_categories", "seating_sections",
                  "customers", "events", "orders", "tickets"):
        n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<18} {n:>9,}")
    lo, hi = conn.execute("SELECT MIN(event_date), MAX(event_date) FROM events").fetchone()
    print(f"  event_date range   {lo} .. {hi}   (anchor {anchor.isoformat()})")
    conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Seed the synthetic BSE ticketing database.")
    p.add_argument("--anchor-date", default=date.today().isoformat(),
                   help="Treat this date as 'today' (default: today).")
    p.add_argument("--out", default=str(HERE / "bse.db"), help="Output SQLite file.")
    p.add_argument("--years", type=int, default=3, help="Seasons of history to generate.")
    a = p.parse_args()
    build(Path(a.out), date.fromisoformat(a.anchor_date), a.years)


if __name__ == "__main__":
    main()
