"""Console (Ukrainian) and JSON rendering of a score result."""
from .geocode import GeocodeResult
from .metrics import CATALOG, DIMENSIONS
from .profiles import Profile
from .scoring import ScoreResult

GRADE_TEXT = {
    "A": "Сильна локація",
    "B": "Добра локація з 1–2 слабкими місцями",
    "C": "Середня: успіх залежить від концепції та маркетингу",
    "D": "Слабка: суттєві ризики",
    "E": "Не рекомендовано",
}


def _bar(v: float, width: int = 20) -> str:
    n = int(round(v / 100 * width))
    return "█" * n + "░" * (width - n)


def _fmt(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") if abs(v) >= 100 else f"{v:.1f}"


def _conf_label(c: float) -> str:
    return "висока" if c >= 0.75 else "середня" if c >= 0.5 else "низька"


def to_text(geo: GeocodeResult, profile: Profile, r: ScoreResult, city_name: str, nearby: list[dict]) -> str:
    L = []
    L.append(f"LOCATION AI · {profile.label} · {geo.label}")
    L.append("=" * 72)
    if not geo.exact:
        L.append("⚠ Точну адресу не знайдено, використано найближчу — перевірте її.")
    L.append(f"Загальний бал: {r.total:.0f}/100  клас {r.grade} — {GRADE_TEXT[r.grade]}")
    L.append(f"Краще, ніж {r.city_percentile:.0f}% комерційних локацій ({city_name}) для цього типу бізнесу")
    L.append(f"Впевненість даних: {_conf_label(r.confidence)} ({r.confidence:.2f})")
    for c in r.caps_applied:
        L.append(f"⚠ {c}")
    L.append("")
    L.append("Підоцінки")
    for d, name in DIMENSIONS.items():
        v = r.dimensions.get(d)
        w = profile.weights.get(d, 0)
        if v is None:
            L.append(f"  {name:<24} {'—':>5}  (вага {w}%) немає даних у прототипі")
        else:
            L.append(f"  {name:<24} {v:5.0f}  {_bar(v)}  (вага {w}%)")
    L.append("")
    L.append("Метрики (сире значення → бал відносно міста)")
    for m, raw in r.raw.items():
        md = CATALOG[m]
        L.append(f"  {m} {md.label:<40} {_fmt(raw):>10} {md.unit:<12} → {r.metric_scores[m]:3.0f}  [{md.quality}]")
    L.append("")
    if r.strengths:
        L.append("Сильні сторони: " + "; ".join(CATALOG[m].label for m in r.strengths))
    if r.weaknesses:
        L.append("Слабкі сторони: " + "; ".join(CATALOG[m].label for m in r.weaknesses))
    if nearby:
        L.append("")
        L.append("Кращі локації поруч (до 1 км):")
        for n in nearby:
            L.append(f"  • {n['address']} — {n['score']:.0f} балів (+{n['score'] - r.total:.0f}), {n['distance_m']} м, "
                     f"{n['lat']:.5f}, {n['lon']:.5f}")
    L.append("")
    L.append("Прототип: дані Overture Maps + Meta HRSL. Ризики (R), доходи, оренда, рейтинги конкурентів "
             "ще не підключені. Оцінка аналітична і не є гарантією успіху.")
    return "\n".join(L)


def to_json(geo: GeocodeResult, profile: Profile, r: ScoreResult, nearby: list[dict]) -> dict:
    return {
        "address": geo.label,
        "address_exact": geo.exact,
        "lat": geo.lat,
        "lon": geo.lon,
        "profile": profile.id,
        "score": r.total,
        "grade": r.grade,
        "city_percentile": r.city_percentile,
        "confidence": r.confidence,
        "caps_applied": r.caps_applied,
        "dimensions": {d: (round(v, 1) if v is not None else None) for d, v in r.dimensions.items()},
        "metrics": {
            m: {"label": CATALOG[m].label, "value": round(v, 2), "unit": CATALOG[m].unit,
                "score": round(r.metric_scores[m], 1), "quality": CATALOG[m].quality, "source": CATALOG[m].source}
            for m, v in r.raw.items()
        },
        "strengths": r.strengths,
        "weaknesses": r.weaknesses,
        "better_nearby": [{k: n[k] for k in ("address", "lat", "lon", "score", "distance_m")} for n in nearby],
    }
