"""Beta statistics for the site owner: python -m locai beta-stats [--csv DIR]."""
import csv
import datetime as dt
from collections import Counter
from pathlib import Path

from .store import FREE_REF, OrderStore

LABELS = {"yes": "так", "no": "ні", "unsure": "не знаю"}


def _when(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def beta_stats(db_path: Path, free_total: int, csv_dir: Path | None = None) -> str:
    if not db_path.exists():
        return f"Ще немає даних: {db_path} не існує (сайт ще не запускали)."
    store = OrderStore(db_path)
    orders = store.query("SELECT * FROM orders ORDER BY created_at")
    feedback = store.query(
        "SELECT f.*, o.address, o.profile, o.score, o.email FROM feedback f JOIN orders o USING (token) "
        "ORDER BY f.created_at")
    waitlist = store.query("SELECT * FROM waitlist ORDER BY created_at")
    free = [o for o in orders if o["payment_ref"] == FREE_REF]
    paid = [o for o in orders if o["status"] == "paid" and o["payment_ref"] != FREE_REF]

    lines = [
        "LOCATION AI — бета-статистика",
        "=" * 40,
        f"Оцінок адрес (превʼю):        {len(orders)}",
        f"Безкоштовних повних звітів:   {len(free)} з {free_total}",
        f"Оплачених звітів:             {len(paid)}",
        f"Погодились на контакт:        {sum(o['contact_ok'] for o in free)}",
        f"Лист очікування:              {len(waitlist)}",
        f"Відгуків:                     {len(feedback)}",
    ]
    if orders:
        by_profile = Counter(o["profile"] for o in orders)
        lines.append("Типи бізнесу:                 " + ", ".join(f"{k} {v}" for k, v in by_profile.most_common()))
    if feedback:
        avg = sum(f["usefulness"] for f in feedback) / len(feedback)
        dec = Counter(LABELS[f["decision"]] for f in feedback)
        pay = Counter(LABELS[f["would_pay"]] for f in feedback)
        lines += [
            "",
            f"Корисність (1–5), середня:    {avg:.1f}",
            "Вплинув на рішення:           " + ", ".join(f"{k} {v}" for k, v in dec.most_common()),
            "Заплатили б:                  " + ", ".join(f"{k} {v}" for k, v in pay.most_common()),
            "",
            "Коментарі:",
        ]
        for f in feedback:
            if f["comment"]:
                lines.append(f"  [{_when(f['created_at'])}] {f['profile']}, {f['address']} "
                             f"(бал {f['score']:.0f}, корисність {f['usefulness']}): {f['comment']}")

    if csv_dir:
        csv_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in (("orders", orders), ("feedback", feedback), ("waitlist", waitlist)):
            with open(csv_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
                if rows:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
        lines += ["", f"CSV збережено в {csv_dir}/ (orders.csv, feedback.csv, waitlist.csv)"]
    return "\n".join(lines)
