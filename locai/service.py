"""Shared entry point for the CLI and the web app: geocode → score → report."""
import re
import threading
import unicodedata
from dataclasses import dataclass

from .city import CityData, load_city
from .config import CITIES
from .geocode import GeocodeResult, geocode, reverse
from .profiles import list_profiles, load_profile
from .scoring import Scorer, ScoreResult


@dataclass
class Scored:
    city: CityData
    scorer: Scorer
    geo: GeocodeResult
    result: ScoreResult
    x: float
    y: float


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s)).strip()


class LocationService:
    """Loads city data and scorers lazily and caches them; safe to share between threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cities: dict[str, CityData] = {}
        self._scorers: dict[tuple[str, str], Scorer] = {}
        self._suggest_index: dict[str, tuple[list[str], list[str]]] = {}

    def city(self, key: str) -> CityData:
        if key not in CITIES:
            raise ValueError(f"Невідоме місто: {key}")
        with self._lock:
            if key not in self._cities:
                self._cities[key] = load_city(CITIES[key])
            return self._cities[key]

    def scorer(self, city_key: str, profile_id: str) -> Scorer:
        if profile_id not in list_profiles():
            raise ValueError(f"Невідомий тип бізнесу: {profile_id}")
        city = self.city(city_key)
        with self._lock:
            k = (city_key, profile_id)
            if k not in self._scorers:
                self._scorers[k] = Scorer(city, load_profile(profile_id))
            return self._scorers[k]

    def geocode(self, city_key: str, query: str) -> GeocodeResult:
        city = self.city(city_key)
        geo = geocode(query, city.addresses, city.city.default_locality)
        if not city.city.contains(geo.lon, geo.lat):
            raise ValueError(f"Адреса поза зоною покриття ({city.city.name}).")
        return geo

    def score(self, city_key: str, profile_id: str, geo: GeocodeResult) -> Scored:
        city = self.city(city_key)
        scorer = self.scorer(city_key, profile_id)
        x, y = (float(v) for v in city.proj.to_xy(geo.lon, geo.lat))
        return Scored(city, scorer, geo, scorer.score_xy(x, y), x, y)

    def nearby(self, s: Scored) -> list[dict]:
        found = s.scorer.better_nearby(s.x, s.y, s.result.total)
        for n in found:
            n["address"] = reverse(n["lat"], n["lon"], s.city.addresses)
        return found

    def report_html(self, s: Scored, **extra) -> str:
        from .pdf_report import build_context, render_html

        ctx = build_context(s.city, s.scorer, s.geo, s.result, self.nearby(s))
        ctx.update(extra)
        return render_html(ctx)

    def suggest(self, city_key: str, q: str, limit: int = 8) -> list[str]:
        """Address autocomplete: every query token must appear in the address."""
        tokens = _norm(q).split()
        if not tokens or len(" ".join(tokens)) < 3:
            return []
        city = self.city(city_key)
        with self._lock:
            if city_key not in self._suggest_index:
                x0, y0, x1, y1 = city.city.bbox
                a = city.addresses.dropna(subset=["street", "number"])
                a = a[a["lon"].between(x0, x1) & a["lat"].between(y0, y1)]
                if "label" in a and a["label"].notna().all():
                    labels = a["label"].tolist()
                else:
                    labels = (a["number"].astype(str) + " " + a["street"] + ", " + a["postcode"].fillna("") + " "
                              + a["locality"].fillna("")).tolist()
                self._suggest_index[city_key] = (labels, [_norm(lbl) for lbl in labels])
            labels, normed = self._suggest_index[city_key]
        hits = [i for i, n in enumerate(normed) if all(t in n for t in tokens)]
        home = _norm(city.city.default_locality)
        numbers = [t for t in tokens if t[:1].isdigit()]

        def exact_number(i: int) -> bool:  # the typed house number appears as a whole token
            words = normed[i].split()
            return all(n in words for n in numbers)

        # Exact house number first, then the main city, then shorter labels.
        hits.sort(key=lambda i: (not exact_number(i), home not in normed[i], len(labels[i]), labels[i]))
        return [labels[i] for i in hits[:limit]]
