"""Normalisation, aggregation and explanation (methodology §4, §5, §8, §9)."""
import logging
import math
from dataclasses import dataclass, field

import h3
import numpy as np
import pandas as pd

from .city import CityData
from .config import H3_RES, city_dir
from .metrics import CATALOG, DIMENSIONS, MetricEngine
from .profiles import Profile

log = logging.getLogger(__name__)

REFERENCE_SAMPLE = 400
ACTIVE_CELL_MIN_PLACES = 5
NEARBY_RADIUS_M = 1000.0
NEARBY_MIN_PLACES = 3
QUALITY_FACTOR = {"A": 1.0, "B": 0.7, "C": 0.4}
GRADES = [(80, "A"), (65, "B"), (50, "C"), (35, "D"), (0, "E")]


@dataclass
class ScoreResult:
    total: float
    grade: str
    city_percentile: float
    dimensions: dict[str, float | None]
    metric_scores: dict[str, float]
    raw: dict[str, float]
    caps_applied: list[str]
    confidence: float
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)


def _grade(score: float) -> str:
    return next(g for lo, g in GRADES if score >= lo)


def _active_cells(city: CityData, min_places: int) -> pd.DataFrame:
    """H3 cells inside the city's core bbox with at least `min_places` open places."""
    p = city.places[city.places["status"] != "permanently_closed"]
    x0, y0, x1, y1 = city.city.bbox
    p = p[p["lon"].between(x0, x1) & p["lat"].between(y0, y1)]
    cells = pd.Series([h3.latlng_to_cell(la, lo, H3_RES) for la, lo in zip(p["lat"], p["lon"])]).value_counts()
    cells = cells[cells >= min_places]
    latlng = np.array([h3.cell_to_latlng(c) for c in cells.index])
    x, y = city.proj.to_xy(latlng[:, 1], latlng[:, 0])
    return pd.DataFrame({"h3": cells.index, "n_places": cells.values, "lat": latlng[:, 0], "lon": latlng[:, 1], "x": x, "y": y})


class Scorer:
    def __init__(self, city: CityData, profile: Profile):
        self.city = city
        self.profile = profile
        self.engine = MetricEngine(city, profile)
        self.reference = self._load_reference()
        self.reference_totals = np.sort([self._aggregate(r)[0] for r in self.reference.to_dict("records")])

    # --- reference distribution -------------------------------------------------------
    def _load_reference(self) -> pd.DataFrame:
        path = city_dir(self.city.city) / f"reference_{self.profile.id}_{self.profile.fingerprint}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        cells = _active_cells(self.city, ACTIVE_CELL_MIN_PLACES)
        sample = cells.sample(min(REFERENCE_SAMPLE, len(cells)), random_state=42)
        log.info("Building reference distribution for '%s' from %d active cells…", self.profile.id, len(sample))
        ref = pd.DataFrame([self.engine.compute(x, y) for x, y in zip(sample["x"], sample["y"])])
        ref.to_parquet(path)
        return ref

    def _percentile(self, metric: str, value: float) -> float:
        ref = np.sort(self.reference[metric].to_numpy())
        lo, hi = np.searchsorted(ref, value, "left"), np.searchsorted(ref, value, "right")
        return 100.0 * (lo + hi) / 2 / len(ref)

    # --- normalisation & aggregation --------------------------------------------------
    def metric_scores(self, raw: dict[str, float]) -> dict[str, float]:
        out = {}
        for m, v in raw.items():
            pct = self._percentile(m, v)
            curve = self.profile.competition_curve
            if m == "C1" and curve["type"] == "optimum":
                s = 100.0 * math.exp(-(((pct - curve["p_star"]) / curve["sigma"]) ** 2))
            else:
                s = pct if CATALOG[m].direction > 0 else 100.0 - pct
            out[m] = s
        return out

    def _aggregate(self, raw: dict[str, float]):
        ms = self.metric_scores(raw)
        dims: dict[str, float | None] = {}
        for d in DIMENSIONS:
            mw = {m: w for m, w in self.profile.metrics.get(d, {}).items() if m in ms}
            dims[d] = sum(ms[m] * w for m, w in mw.items()) / sum(mw.values()) if mw else None
        avail = {d: w for d, w in self.profile.weights.items() if dims.get(d) is not None}
        total = sum(dims[d] * w for d, w in avail.items()) / sum(avail.values())
        caps = []
        for cap in self.profile.caps:
            dv = dims.get(cap["dimension"])
            if dv is not None and dv < cap["below"] and total > cap["max"]:
                total = cap["max"]
                caps.append(f"{DIMENSIONS[cap['dimension']]} нижче {cap['below']} → бал обмежено до {cap['max']}")
        return total, dims, ms, caps

    def _confidence(self) -> float:
        num = den = 0.0
        for d, wd in self.profile.weights.items():
            mets = self.profile.metrics.get(d, {})
            if not mets:
                den += wd  # dimension without data counts as zero confidence
                continue
            tot = sum(mets.values())
            for m, wm in mets.items():
                num += wd * wm / tot * QUALITY_FACTOR[CATALOG[m].quality]
                den += wd * wm / tot
        return num / den

    def score_xy(self, x: float, y: float) -> ScoreResult:
        raw = self.engine.compute(x, y)
        total, dims, ms, caps = self._aggregate(raw)
        pct = 100.0 * np.searchsorted(self.reference_totals, total) / len(self.reference_totals)
        res = ScoreResult(round(total, 1), _grade(total), round(pct, 1), dims, ms, raw, caps, round(self._confidence(), 2))
        res.strengths, res.weaknesses = self._explain(ms)
        return res

    def _explain(self, ms: dict[str, float]):
        imp = {}
        for d, wd in self.profile.weights.items():
            mets = self.profile.metrics.get(d, {})
            tot = sum(mets.values()) or 1
            for m, wm in mets.items():
                imp[m] = wd * wm / tot
        pos = sorted((imp[m] * (s - 50), m) for m, s in ms.items() if m in imp and s >= 65)
        neg = sorted((imp[m] * (50 - s), m) for m, s in ms.items() if m in imp and s <= 35)
        return [m for _, m in reversed(pos)][:3], [m for _, m in reversed(neg)][:3]

    # --- "better spot nearby" ------------------------------------------------------------
    def better_nearby(self, x: float, y: float, base: float, top: int = 3) -> list[dict]:
        cells = _active_cells(self.city, NEARBY_MIN_PLACES)
        cells = cells[np.hypot(cells["x"] - x, cells["y"] - y) <= NEARBY_RADIUS_M]
        found = []
        for c in cells.itertuples():
            r = self.score_xy(c.x, c.y)
            if r.total > base + 2:
                found.append({"lat": c.lat, "lon": c.lon, "x": c.x, "y": c.y, "score": r.total,
                              "distance_m": round(float(np.hypot(c.x - x, c.y - y))), "result": r})
        return sorted(found, key=lambda f: -f["score"])[:top]
