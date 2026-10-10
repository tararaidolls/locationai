"""Orders, feedback and the beta waitlist in SQLite.

One order row per scored address; it becomes a full report after payment (or a free beta claim).
"""
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
CREATE TABLE IF NOT EXISTS feedback (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    token          TEXT NOT NULL REFERENCES orders(token),
    created_at     REAL NOT NULL,
    usefulness     INTEGER NOT NULL,           -- 1..5
    decision       TEXT NOT NULL,              -- yes | no | unsure
    would_pay      TEXT NOT NULL,              -- yes | no | unsure
    comment        TEXT
);
CREATE TABLE IF NOT EXISTS waitlist (
    email          TEXT PRIMARY KEY,
    created_at     REAL NOT NULL,
    token          TEXT
);
"""
FREE_REF = "free"
MIGRATIONS = ["ALTER TABLE orders ADD COLUMN contact_ok INTEGER NOT NULL DEFAULT 0"]


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
    contact_ok: bool = False

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
            for sql in MIGRATIONS:
                try:
                    self._db.execute(sql)
                except sqlite3.OperationalError:  # already applied
                    pass
            self._db.commit()

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
        d["contact_ok"] = bool(d["contact_ok"])
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

    # --- free beta ---------------------------------------------------------------------
    def free_used(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM orders WHERE payment_ref = ?", (FREE_REF,)).fetchone()[0]

    def claim_free(self, token: str, email: str, contact_ok: bool, limit: int) -> bool:
        """Unlock the report for free if the beta quota is not used up. Atomic under concurrent claims."""
        with self._lock, self._db:
            row = self._db.execute("SELECT status FROM orders WHERE token = ?", (token,)).fetchone()
            if row is None:
                return False
            if row["status"] == "paid":
                return True
            used = self._db.execute("SELECT COUNT(*) FROM orders WHERE payment_ref = ?", (FREE_REF,)).fetchone()[0]
            if used >= limit:
                return False
            self._db.execute(
                "UPDATE orders SET status = 'paid', paid_at = ?, payment_ref = ?, email = ?, contact_ok = ? "
                "WHERE token = ?", (time.time(), FREE_REF, email, int(contact_ok), token))
            return True

    def add_waitlist(self, email: str, token: str | None) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO waitlist (email, created_at, token) VALUES (?, ?, ?)",
                             (email, time.time(), token))

    # --- feedback ------------------------------------------------------------------------
    def add_feedback(self, token: str, usefulness: int, decision: str, would_pay: str, comment: str | None) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO feedback (token, created_at, usefulness, decision, would_pay, comment) "
                "VALUES (?, ?, ?, ?, ?, ?)", (token, time.time(), usefulness, decision, would_pay, comment))

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, params).fetchall()]
