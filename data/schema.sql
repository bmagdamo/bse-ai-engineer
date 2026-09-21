-- BSE ticketing schema (synthetic). Single source of truth for the database.
-- All dates are ISO-8601 TEXT so SQLite date()/strftime()/range comparisons work.

PRAGMA foreign_keys = ON;

CREATE TABLE venues (
    venue_id   INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    city       TEXT    NOT NULL,
    state      TEXT    NOT NULL,
    capacity   INTEGER NOT NULL
);

CREATE TABLE teams (
    team_id      INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    league       TEXT    NOT NULL,   -- 'NBA' | 'WNBA' | 'NCAA'
    is_bse_owned INTEGER NOT NULL    -- 1 for Brooklyn Nets / New York Liberty
);

CREATE TABLE event_categories (
    category_id INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL     -- 'NBA Regular Season', 'Concert', ...
);

CREATE TABLE events (
    event_id       INTEGER PRIMARY KEY,
    venue_id       INTEGER NOT NULL REFERENCES venues(venue_id),
    category_id    INTEGER NOT NULL REFERENCES event_categories(category_id),
    home_team_id   INTEGER REFERENCES teams(team_id),  -- NULL for non-sporting events
    away_team_id   INTEGER REFERENCES teams(team_id),
    name           TEXT    NOT NULL,
    event_date     TEXT    NOT NULL,                   -- 'YYYY-MM-DD'
    doors_time     TEXT,                               -- 'HH:MM'
    announced_date TEXT,
    status         TEXT    NOT NULL                    -- 'completed'|'scheduled'|'cancelled'|'postponed'
);

CREATE TABLE seating_sections (
    section_id INTEGER PRIMARY KEY,
    venue_id   INTEGER NOT NULL REFERENCES venues(venue_id),
    name       TEXT    NOT NULL,   -- 'Courtside A', 'Lower 12', ...
    level      TEXT    NOT NULL,   -- 'Courtside'|'Lower'|'Club'|'Suite'|'Upper'
    base_price REAL    NOT NULL
);

CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    first_name   TEXT NOT NULL,
    last_name    TEXT NOT NULL,
    email        TEXT NOT NULL,
    city         TEXT,
    state        TEXT,
    signup_date  TEXT,
    loyalty_tier TEXT NOT NULL      -- 'none'|'silver'|'gold'|'season_ticket_holder'
);

CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
    event_id    INTEGER NOT NULL REFERENCES events(event_id),
    order_ts    TEXT    NOT NULL,   -- 'YYYY-MM-DD HH:MM:SS'
    channel     TEXT    NOT NULL,   -- 'web'|'mobile_app'|'box_office'|'resale'|'group_sales'
    status      TEXT    NOT NULL    -- 'completed'|'refunded'|'cancelled'
);

CREATE TABLE tickets (
    ticket_id   INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(order_id),
    event_id    INTEGER NOT NULL REFERENCES events(event_id),
    section_id  INTEGER NOT NULL REFERENCES seating_sections(section_id),
    seat_row    TEXT,
    seat_number INTEGER,
    face_value  REAL    NOT NULL,   -- list price
    price_paid  REAL    NOT NULL,   -- what the customer actually paid (0.00 for comps)
    fees        REAL    NOT NULL,   -- service + facility fees
    is_comp     INTEGER NOT NULL DEFAULT 0,
    scanned_at  TEXT                -- NULL = ticket never scanned (no-show)
);

CREATE INDEX idx_events_date     ON events(event_date);
CREATE INDEX idx_events_home     ON events(home_team_id);
CREATE INDEX idx_orders_event    ON orders(event_id);
CREATE INDEX idx_tickets_order   ON tickets(order_id);
CREATE INDEX idx_tickets_event   ON tickets(event_id);
