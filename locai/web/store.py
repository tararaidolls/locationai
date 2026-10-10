"""Orders in SQLite. One row per scored address; it becomes a paid report after checkout."""
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    token          TEXT PRIMARY KEY,
    created_at     REAL NOT NULL,
    city           TEXT NOT NULL,
    profile        TEXT NOT NULL,
    query          TEXT NOT NULL,
    address        TEXT NOT NULL,
    address_exact  INTEGER NOT NULL,
    lat            REAL NOT NULL,
    lon            REAL NOT NULL,
    score          REAL NOT NULL,
    grade          TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'preview',  -- preview | paid
    price_cents    INTEGER NOT NULL,
    currency       TEXT NOT NULL,
    payment_ref    TEXT,
    email          TEXT,
    paid_at        REAL
);
CREATE INDEX IF NOT EXISTS orders_payment_ref ON orders(payment_ref);
"""


@dataclass
class Order:
    token: str
    created_at: float
    city: str
    profile: str
    query: str
    address: str
    address_exact: bool
    lat: float
    lon: float
    score: float
    grade: str
    status: str
    price_cents: int
    currency: str
    payment_ref: str | None
    email: str | None
    paid_at: float | None

    @property
    def paid(self) -> bool:
        return self.status == "paid"


class OrderStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(SCHEMA)

    def create(self, **fields) -> Order:
        token = secrets.token_urlsafe(16)
        fields = {"token": token, "created_at": time.time(), **fields}
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        with self._lock, self._db:
            self._db.execute(f"INSERT INTO orders ({cols}) VALUES ({marks})", list(fields.values()))
        return self.get(token)

    def get(self, token: str) -> Order | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM orders WHERE token = ?", (token,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["address_exact"] = bool(d["address_exact"])
        return Order(**d)

    def set_payment_ref(self, token: str, ref: str) -> None:
        with self._lock, self._db:
            self._db.execute("UPDATE orders SET payment_ref = ? WHERE token = ?", (ref, token))

    def mark_paid(self, token: str, payment_ref: str | None, email: str | None) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE orders SET status = 'paid', paid_at = ?, payment_ref = COALESCE(?, payment_ref), "
                "email = COALESCE(?, email) WHERE token = ? AND status != 'paid'",
                (time.time(), payment_ref, email, token),
            )
