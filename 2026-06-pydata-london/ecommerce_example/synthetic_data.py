# ---------------------------------------------------------------------------
# Seed data — richer e-commerce schema engineered for multi-step DS tasks
# ---------------------------------------------------------------------------

# A fixed seed makes every eval run see the same data, so prompts can be
# scored against deterministic ground truth.
import json
from datetime import date, timedelta
from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .database import Database


_SEED = 42

# Row tuple shapes (positional, matching the INSERT statements below).
CustomerRow = tuple[int, str, str, str, str, str]  # id, name, email, state, segment, signup
ProductRow = tuple[int, str, str, float, int, int]  # id, name, category, price, stock, threshold
OrderRow = tuple[int, int, int, int, float, str, str]  # id, customer_id, product_id, qty, amount, placed_at, status

# Schema constants
_STATES = ['CA', 'NY', 'TX', 'WA', 'FL', 'IL', 'MA', 'GA']
_SEGMENTS = ['consumer', 'small_biz', 'enterprise']
_PRODUCT_CATEGORIES = ['Electronics', 'Books', 'Clothing', 'Office', 'Home']

# Reference date — the "today" against which days-since-last-order is measured.
REFERENCE_DATE = date(2024, 7, 1)

__all__ = ('seed_database',)


def seed_database(db: Database) -> None:
    """Seed `db` with realistic, deterministic e-commerce data.

    The schema and contents are fixed (seed=42) so eval cases are
    reproducible across runs — essential for prompt optimization.

    NOTE: the column names here are the POST-migration ("new") schema. See
    the migration note above; the demo's managed-variable prompt deliberately
    documents the pre-migration names.
    """
    rng = Random(_SEED)
    conn = db.conn

    conn.execute(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            attributes TEXT,
            signup_date TEXT
        )
        """
    )
    customers = _build_customers(rng)
    conn.executemany(
        'INSERT INTO customers VALUES (?, ?, ?, ?, ?)',
        [
            (cid, name, email, _customer_attributes_json(cid, state, segment), signup)
            for (cid, name, email, state, segment, signup) in customers
        ],
    )

    conn.execute(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock_on_hand INTEGER,
            reorder_threshold INTEGER
        )
        """
    )
    products = _build_products(rng)
    conn.executemany(
        'INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)',
        products,
    )

    conn.execute(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            total_amount REAL,
            placed_at TEXT,
            status TEXT
        )
        """
    )
    orders = _build_orders(rng, customers, products)
    conn.executemany(
        'INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)',
        orders,
    )

    # Reference date for evaluators that need to know what "today" is.
    conn.execute(
        """
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.execute('INSERT INTO meta VALUES (?, ?)', ('reference_date', str(REFERENCE_DATE)))

    conn.commit()


# Customer-attributes migration: `state` and `segment` used to be plain
# columns; they now live — together with new fields Tom added (`region` and a
# nested `acquisition` object) — inside a JSON-serialized `attributes` TEXT
# column. `describe_table` only reveals `attributes TEXT`, NOT its keys, so
# an agent must SELECT rows and json.loads / json_extract them to discover where
# the data went, several follow-up queries deep. See the migration note on
# `seed_database` and the demo prompts.
_STATE_TO_REGION = {
    'CA': 'West',
    'WA': 'West',
    'TX': 'South',
    'FL': 'South',
    'GA': 'South',
    'NY': 'Northeast',
    'MA': 'Northeast',
    'IL': 'Midwest',
}
_ACQUISITION_CHANNELS = ['referral', 'organic', 'paid_search', 'partner']
_ACQUISITION_CAMPAIGNS = ['q1-launch', 'spring-promo', 'retargeting', 'none']


def _customer_attributes_json(cid: int, state: str, segment: str) -> str:
    """Build the JSON `attributes` blob for a customer (deterministic by id)."""
    return json.dumps(
        {
            'state': state,
            'segment': segment,
            'region': _STATE_TO_REGION[state],
            'acquisition': {
                'channel': _ACQUISITION_CHANNELS[cid % len(_ACQUISITION_CHANNELS)],
                'campaign': _ACQUISITION_CAMPAIGNS[(cid * 7) % len(_ACQUISITION_CAMPAIGNS)],
            },
        }
    )


def _build_customers(rng: Random) -> list[CustomerRow]:
    """50 customers signed up across 18 months, segmented by state/segment."""
    rows: list[CustomerRow] = []
    earliest = date(2023, 1, 1)
    for i in range(1, 51):
        offset_days = rng.randint(0, 540)
        signup = earliest + timedelta(days=offset_days)
        state = rng.choice(_STATES)
        segment = rng.choices(_SEGMENTS, weights=[0.6, 0.3, 0.1])[0]
        rows.append(
            (
                i,
                f'Customer {i:02d}',
                f'customer{i:02d}@example.com',
                state,
                segment,
                str(signup),
            )
        )
    return rows


def _build_products(rng: Random) -> list[ProductRow]:
    """25 products across 5 categories with mixed stock urgency.

    Some products are deliberately low-stock + popular, so reorder-priority
    tasks have real signal. Others are well-stocked + slow movers.
    """
    rows: list[ProductRow] = []
    name_idx = 0
    base_prices = {
        'Electronics': (29.0, 199.0),
        'Books': (12.0, 49.0),
        'Clothing': (15.0, 89.0),
        'Office': (5.0, 39.0),
        'Home': (19.0, 149.0),
    }
    for cat in _PRODUCT_CATEGORIES:
        per_cat = 5
        for _ in range(per_cat):
            name_idx += 1
            low, high = base_prices[cat]
            price = round(rng.uniform(low, high), 2)
            # Mix of stock urgency: some intentionally low + below threshold
            if rng.random() < 0.25:
                stock = rng.randint(2, 15)
                threshold = rng.randint(15, 30)
            else:
                stock = rng.randint(40, 300)
                threshold = rng.randint(20, 60)
            rows.append((name_idx, f'{cat} Item {name_idx:02d}', cat, price, stock, threshold))
    return rows


def _build_orders(
    rng: Random,
    customers: list[CustomerRow],
    products: list[ProductRow],
) -> list[OrderRow]:
    """~300 orders, with deliberate patterns:

    - Most customers order 3-10 times. ~20% are one-shot.
    - ~20% of customers go silent after ~90+ days (churn signal).
    - 3 hand-built pricing anomalies (effective unit price differs from catalog).
    - One orphaned order pointing to a non-existent customer.
    - Order volume skews toward Electronics in CA, Books in NY (subgroup pattern).
    """
    rows: list[OrderRow] = []
    order_id = 0
    prod_by_id = {p[0]: p for p in products}

    for cust in customers:
        cid, _name, _email, state, _segment, signup_str = cust
        signup = date.fromisoformat(signup_str)
        # Order frequency: most customers active, some churned
        is_churned = rng.random() < 0.20
        is_one_shot = rng.random() < 0.20

        if is_one_shot:
            n_orders = 1
        elif is_churned:
            n_orders = rng.randint(2, 5)
        else:
            n_orders = rng.randint(3, 12)

        # Customer's preferred category — state-conditioned for subgroup signal
        if state == 'CA':
            pref_cat = rng.choices(_PRODUCT_CATEGORIES, weights=[0.45, 0.10, 0.15, 0.10, 0.20])[0]
        elif state == 'NY':
            pref_cat = rng.choices(_PRODUCT_CATEGORIES, weights=[0.10, 0.45, 0.15, 0.15, 0.15])[0]
        else:
            pref_cat = rng.choice(_PRODUCT_CATEGORIES)

        category_products = [p for p in products if p[2] == pref_cat]

        # Time window for this customer's orders
        active_window_end = (
            REFERENCE_DATE - timedelta(days=rng.randint(91, 180))
            if is_churned
            else REFERENCE_DATE - timedelta(days=rng.randint(0, 60))
        )
        active_window_start = max(signup, active_window_end - timedelta(days=240))
        window_days = max((active_window_end - active_window_start).days, 1)

        for _ in range(n_orders):
            order_id += 1
            # Prefer category 70% of the time, otherwise random product
            if rng.random() < 0.70 and category_products:
                product = rng.choice(category_products)
            else:
                product = rng.choice(products)
            pid, _pname, _pcat, price, _stock, _thresh = product

            qty = rng.choices([1, 1, 1, 2, 2, 3, 4, 5, 10], k=1)[0]
            amount = round(qty * price, 2)
            order_offset = rng.randint(0, window_days)
            order_date = active_window_start + timedelta(days=order_offset)
            status = rng.choices(['delivered', 'shipped', 'pending'], weights=[0.80, 0.13, 0.07])[0]
            rows.append((order_id, cid, pid, qty, amount, str(order_date), status))

    # Inject 3 pricing anomalies: amount doesn't match qty * catalog price.
    # These are real-world data-quality bugs (data entry error, secret discount,
    # promo period not recorded). Multi-step anomaly tasks should find these.
    anomaly_targets = [r for r in rows if r[3] == 1][:3]  # 3 single-unit orders
    for i, row in enumerate(anomaly_targets):
        oid, cid, pid, qty, _amt, odate, status = row
        catalog_price = prod_by_id[pid][3]
        if i == 0:
            anomalous_amount = round(catalog_price * 5.2, 2)  # bug: charged 5x
        elif i == 1:
            anomalous_amount = round(catalog_price * 0.18, 2)  # 82% off promo
        else:
            anomalous_amount = round(catalog_price + 0.01, 2)  # off by penny (data entry)
        idx = rows.index(row)
        rows[idx] = (oid, cid, pid, qty, anomalous_amount, odate, status)

    # Inject one orphan: customer_id that doesn't exist
    order_id += 1
    orphan_product = rng.choice(products)
    rows.append(
        (
            order_id,
            999,  # nonexistent
            orphan_product[0],
            1,
            orphan_product[3],
            str(REFERENCE_DATE - timedelta(days=30)),
            'pending',
        )
    )

    return rows


# ---------------------------------------------------------------------------
# THE MIGRATION  ("Tom moved customer attributes into a JSON column")
# ---------------------------------------------------------------------------
#
# The demo's premise: a teammate ran a migration. `customers.state` and
# `customers.segment` used to be plain columns; they were consolidated into a
# JSON-serialized `attributes` TEXT column (SQLite has no JSONB), and Tom
# added new fields while he was at it — `region` and a nested `acquisition`
# object (`channel` / `campaign`). The agent's managed-variable system
# prompt still documents the OLD flat columns.
#
# Why JSON instead of a column rename: a rename is too cheap to recover from —
# one `describe_table` reveals the new column name and the agent is back on
# track in a single turn, so it barely moves tokens/latency. With JSON,
# `describe_table` only reveals `attributes TEXT` — it CANNOT show the keys
# inside the blob. So the agent has to SELECT rows and `json.loads` /
# `json_extract` them, often sampling several rows to see the full structure
# (including Tom's new fields), before it can rewrite its query. That's a real
# multi-step investigation on EVERY run — the "expensive moment" that makes the
# wasted work show up as meaningful tokens/latency/cost.
#
# Live `customers` schema: (id, name, email, attributes, signup_date), where
# `attributes` is JSON like:
#   {"state": "CA", "segment": "enterprise", "region": "West",
#    "acquisition": {"channel": "referral", "campaign": "q1-launch"}}
# The other tables (orders, products, meta) are documented correctly in the
# "before" prompt — the customers JSON is the sole mismatch, so the demo turns
# on one clean, believable issue.
#
# The fix the optimizer should propose: bake the discovered JSON structure into
# the prompt (the attributes keys + how to extract them) so the agent stops
# re-investigating it every run.
#
# Tuning: to make the investigation deeper/wider, add more nested fields, make
# the structure vary across rows, migrate an `orders` field too, or bump the
# row counts (`range(1, 51)` in `_build_customers`) so each sampled-rows
# query pulls more context. Inserts below stay POSITIONAL.
# ---------------------------------------------------------------------------
