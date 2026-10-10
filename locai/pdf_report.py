"""PDF report: a Jinja2 HTML template with an inline SVG map, printed to PDF by headless Chromium.

The same HTML is the web version of the report; the PDF is just its print rendering.
"""
import datetime as dt
import html
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import h3
import jinja2
import numpy as np

from .city import CityData
from .config import WALK_SPEED_M_PER_MIN
from .geocode import GeocodeResult
from .metrics import CATALOG, DIMENSIONS
from .profiles import Profile
from .report import GRADE_TEXT, _conf_label
from .scoring import Scorer, ScoreResult

TEMPLATES = Path(__file__).resolve().parent / "templates"

MAP_W, MAP_H = 680, 480
# Walking-time bands for the street layer: sequential blue, darker = closer.
TIME_BANDS = [(5, "#1c5cab"), (10, "#3987e5"), (15, "#86b6ef")]
STREET_OUTSIDE = "#dcdbd6"
COMPETITOR = "#eb6834"   # categorical slot 2 (orange)
SYNERGY = "#1baf7a"      # categorical slot 3 (aqua)
INK = "#0b0b0b"
SURFACE = "#ffffff"

GROUP_LABELS = {
    "offices": "офіси", "education": "навчання", "entertainment": "дозвілля", "lodging": "готелі",
    "fitness": "фітнес", "bar": "бари", "fashion": "одяг", "cafe": "кав'ярні", "groceries": "продукти",
    "veterinary": "ветклініки", "pet_services": "послуги для тварин", "parks": "парки",
}

ABSTRACT_UNITS = {"індекс", "зваж. к-сть", "betweenness"}

CHROME_CANDIDATES = [
    os.environ.get("LOCAI_CHROME", ""),
    "/opt/pw-browsers/chromium",
    "chromium", "chromium-browser", "google-chrome", "google-chrome-stable",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def _fmt_int(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ")


def _fmt(v: float) -> str:
    return _fmt_int(v) if abs(v) >= 100 else f"{v:.1f}".replace(".", ",")


# --- map --------------------------------------------------------------------------------
class _Viewport:
    def __init__(self, cx: float, cy: float, half_w_m: float):
        self.cx, self.cy = cx, cy
        self.k = MAP_W / (2 * half_w_m)
        self.half_w, self.half_h = half_w_m, half_w_m * MAP_H / MAP_W

    def px(self, x, y):
        return (np.asarray(x) - self.cx) * self.k + MAP_W / 2, MAP_H / 2 - (np.asarray(y) - self.cy) * self.k

    def inside(self, x, y, margin_m: float = 0.0):
        return (np.abs(np.asarray(x) - self.cx) <= self.half_w + margin_m) & (np.abs(np.asarray(y) - self.cy) <= self.half_h + margin_m)


def _street_layer(city: CityData, vp: _Viewport, node_dist: np.ndarray, walk_min: float) -> str:
    net = city.net
    keep = vp.inside(net.sample_x, net.sample_y, 50)
    edges = np.unique(net.sample_edge[keep])
    t_edge = np.minimum(node_dist[net.edges_u[edges]], node_dist[net.edges_v[edges]]) / WALK_SPEED_M_PER_MIN

    # Samples are stored contiguously per edge in offset order.
    starts = np.searchsorted(net.sample_edge, edges, "left")
    ends = np.searchsorted(net.sample_edge, edges, "right")
    by_color: dict[str, list[str]] = {}
    for e_start, e_end, t in zip(starts, ends, t_edge):
        color = STREET_OUTSIDE
        for lim, c in TIME_BANDS:
            if t <= lim and lim <= walk_min:
                color = c
                break
        px, py = vp.px(net.sample_x[e_start:e_end], net.sample_y[e_start:e_end])
        by_color.setdefault(color, []).append("M" + "L".join(f"{a:.1f} {b:.1f}" for a, b in zip(px, py)))
    # Draw far streets first so the catchment sits on top.
    order = [STREET_OUTSIDE] + [c for _, c in reversed(TIME_BANDS)]
    out = []
    for color in order:
        if color in by_color:
            width = 0.8 if color == STREET_OUTSIDE else 1.6
            out.append(f'<path d="{" ".join(by_color[color])}" fill="none" stroke="{color}" stroke-width="{width}" '
                       'stroke-linecap="round" stroke-linejoin="round"/>')
    return "\n".join(out)


def _population_layer(city: CityData, vp: _Viewport) -> str:
    c = city.cells
    keep = vp.inside(c["x"].to_numpy(), c["y"].to_numpy(), 200)
    cells = c[keep]
    if cells.empty:
        return ""
    vmax = np.quantile(c["pop"], 0.98)
    out = []
    for h, pop in zip(cells["h3"], cells["pop"]):
        lat_lng = np.array(h3.cell_to_boundary(h))
        x, y = city.proj.to_xy(lat_lng[:, 1], lat_lng[:, 0])
        px, py = vp.px(x, y)
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py))
        out.append(f'<polygon points="{pts}" fill="#8a8984" fill-opacity="{min(pop / vmax, 1) * 0.22:.3f}"/>')
    return "\n".join(out)


def _points(xs, ys, vp: _Viewport, color: str, r: float) -> str:
    px, py = vp.px(xs, ys)
    return "\n".join(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{r}" fill="{color}" stroke="{SURFACE}" stroke-width="2"/>'
                     for a, b in zip(px, py))


def _scale_bar(vp: _Viewport) -> str:
    meters = 250 if vp.half_w < 1200 else 500
    w = meters * vp.k
    x0, y0 = 16, MAP_H - 18
    return (f'<rect x="{x0 - 6}" y="{y0 - 18}" width="{w + 12}" height="28" rx="4" fill="{SURFACE}" fill-opacity="0.85"/>'
            f'<line x1="{x0}" y1="{y0}" x2="{x0 + w:.1f}" y2="{y0}" stroke="{INK}" stroke-width="2"/>'
            f'<line x1="{x0}" y1="{y0 - 4}" x2="{x0}" y2="{y0 + 1}" stroke="{INK}" stroke-width="2"/>'
            f'<line x1="{x0 + w:.1f}" y1="{y0 - 4}" x2="{x0 + w:.1f}" y2="{y0 + 1}" stroke="{INK}" stroke-width="2"/>'
            f'<text x="{x0}" y="{y0 - 6}" font-size="10" fill="{INK}">{meters} м</text>')


def build_map_svg(city: CityData, scorer: Scorer, x: float, y: float, nearby: list[dict],
                  competitors_idx: np.ndarray, synergy_idx: np.ndarray, node_dist: np.ndarray) -> str:
    walk = scorer.profile.walk_minutes
    # Frame the catchment and every "better nearby" marker with a margin.
    reach = [walk * WALK_SPEED_M_PER_MIN * 1.05, 800]
    for n in nearby:
        reach += [abs(n["x"] - x) + 80, (abs(n["y"] - y) + 80) * MAP_W / MAP_H]
    vp = _Viewport(x, y, max(reach))
    places = city.places
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {MAP_W} {MAP_H}" width="100%" role="img" '
        f'aria-label="Карта зони обслуговування">',
        f'<rect width="{MAP_W}" height="{MAP_H}" fill="{SURFACE}"/>',
        _population_layer(city, vp),
        _street_layer(city, vp, node_dist, walk),
    ]
    syn = synergy_idx[vp.inside(places["x"].to_numpy()[synergy_idx], places["y"].to_numpy()[synergy_idx])]
    comp = competitors_idx[vp.inside(places["x"].to_numpy()[competitors_idx], places["y"].to_numpy()[competitors_idx])]
    parts.append(_points(places["x"].to_numpy()[syn], places["y"].to_numpy()[syn], vp, SYNERGY, 3.5))
    parts.append(_points(places["x"].to_numpy()[comp], places["y"].to_numpy()[comp], vp, COMPETITOR, 4.5))

    for i, n in enumerate(nearby, 1):
        nx, ny = vp.px(n["x"], n["y"])
        if 0 <= nx <= MAP_W and 0 <= ny <= MAP_H:
            parts.append(f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="10" fill="#52514e" stroke="{SURFACE}" stroke-width="2"/>'
                         f'<text x="{nx:.1f}" y="{ny + 3.5:.1f}" text-anchor="middle" font-size="11" font-weight="700" '
                         f'fill="{SURFACE}">{i}</text>')

    ax, ay = vp.px(x, y)
    star = []
    for k in range(10):
        ang = -np.pi / 2 + k * np.pi / 5
        rad = 13 if k % 2 == 0 else 5.5
        star.append(f"{ax + rad * np.cos(ang):.1f},{ay + rad * np.sin(ang):.1f}")
    parts.append(f'<polygon points="{" ".join(star)}" fill="{INK}" stroke="{SURFACE}" stroke-width="2.5" stroke-linejoin="round"/>')
    parts.append(_scale_bar(vp))
    parts.append(f'<text x="{MAP_W - 8}" y="{MAP_H - 8}" text-anchor="end" font-size="8.5" fill="#52514e">'
                 '© Overture Maps Foundation · © OpenStreetMap contributors · Meta HRSL</text>')
    parts.append("</svg>")
    return "\n".join(p for p in parts if p)


# --- context ------------------------------------------------------------------------------
def _metric_sentence(m: str, raw: float, score: float, strong: bool) -> str:
    md = CATALOG[m]
    # Abstract units (indices, weighted counts) mean little to a reader; show only the comparison.
    head = md.label if md.unit in ABSTRACT_UNITS else f"{md.label}: {_fmt(raw)} {md.unit}"
    share = score if strong else 100 - score
    if share >= 99:
        extreme = "найкращих" if strong else "найслабших"
        return f"{head}. Один із {extreme} показників серед комерційних локацій міста."
    if md.direction < 0:
        word = "нижчий" if strong else "вищий"
        return f"{head}. Конкурентний тиск {word}, ніж у {share:.0f}% комерційних локацій міста."
    word = "Краще" if strong else "Гірше"
    return f"{head}. {word}, ніж у {share:.0f}% комерційних локацій міста."


def _verdict(r: ScoreResult, profile: Profile) -> str:
    dims = {d: v for d, v in r.dimensions.items() if v is not None}
    best = max(dims, key=dims.get)
    worst = min(dims, key=dims.get)
    return (f"Локація набирає {r.total:.0f} балів зі 100 для профілю «{profile.label.lower()}» і входить до "
            f"{100 - r.city_percentile:.0f}% найкращих комерційних локацій міста. Найсильніший вимір — "
            f"«{DIMENSIONS[best].lower()}» ({dims[best]:.0f}), найслабший — «{DIMENSIONS[worst].lower()}» "
            f"({dims[worst]:.0f}).")


def build_context(city: CityData, scorer: Scorer, geo: GeocodeResult, r: ScoreResult, nearby: list[dict]) -> dict:
    profile = scorer.profile
    eng = scorer.engine
    x, y = (float(v) for v in city.proj.to_xy(geo.lon, geo.lat))
    walk = profile.walk_minutes
    t_places, t_cells = eng.times(x, y, max(walk, 15) * 1.6)
    node_dist = city.net.distances_from_point(x, y, max(walk, 15) * 1.6 * WALK_SPEED_M_PER_MIN)

    in_zone = t_places <= walk
    comp_idx = np.flatnonzero(eng.comp_mask & in_zone)
    comp_all = np.flatnonzero(eng.comp_mask)
    syn_mask = np.zeros(len(city.places), dtype=bool)
    for g in profile.synergy:
        syn_mask |= eng.masks[g]
    syn_idx = np.flatnonzero(syn_mask & ~eng.comp_mask & in_zone)  # only those that count toward S1

    order = comp_idx[np.argsort(t_places[comp_idx])][:7]
    competitors = [{"name": city.places["name"].iat[i] or "Без назви",
                    "category": (city.places["category"].iat[i] or "").replace("_", " "),
                    "minutes": f"{t_places[i]:.0f}"} for i in order]

    dims = []
    for d, name in DIMENSIONS.items():
        v = r.dimensions.get(d)
        dims.append({"name": name, "weight": profile.weights.get(d, 0), "score": None if v is None else round(v)})

    metrics = []
    for m, raw in r.raw.items():
        md = CATALOG[m]
        metrics.append({"code": m, "label": md.label, "value": _fmt(raw), "unit": md.unit,
                        "score": round(r.metric_scores[m]), "quality": md.quality, "source": md.source})

    pop_zone = float(city.cells["pop"].to_numpy()[t_cells <= walk].sum())
    tiles = [
        {"value": _fmt_int(pop_zone), "label": f"мешканців у {walk:.0f} хв пішки"},
        {"value": str(len(comp_idx)), "label": f"конкурентів у {walk:.0f} хв пішки"},
        {"value": _fmt_int(r.raw["C4"]), "label": "потенційних клієнтів (модель Хаффа)"},
        {"value": f"{r.metric_scores['A2']:.0f}", "label": "бал центральності вулиці (0–100)"},
    ]

    return {
        "title": "Location AI — звіт про локацію",
        "date": dt.date.today().strftime("%d.%m.%Y"),
        "address": geo.label,
        "address_exact": geo.exact,
        "coords": f"{geo.lat:.5f}, {geo.lon:.5f}",
        "city": city.city.name,
        "profile": profile.label,
        "walk_minutes": f"{walk:.0f}",
        "score": f"{r.total:.0f}",
        "grade": r.grade,
        "grade_text": GRADE_TEXT[r.grade],
        "percentile": f"{r.city_percentile:.0f}",
        "confidence": _conf_label(r.confidence),
        "confidence_value": f"{r.confidence:.2f}".replace(".", ","),
        "verdict": _verdict(r, profile),
        "caps": r.caps_applied,
        "dimensions": dims,
        "tiles": tiles,
        "strengths": [_metric_sentence(m, r.raw[m], r.metric_scores[m], True) for m in r.strengths],
        "weaknesses": [_metric_sentence(m, r.raw[m], r.metric_scores[m], False) for m in r.weaknesses],
        "metrics": metrics,
        "competitors": competitors,
        "competitors_total": len(comp_idx),
        "nearby": [{"n": i, "address": n["address"], "score": f"{n['score']:.0f}",
                    "delta": f"+{n['score'] - r.total:.0f}", "distance": n["distance_m"]}
                   for i, n in enumerate(nearby, 1)],
        "map_svg": build_map_svg(city, scorer, x, y, nearby, comp_all, syn_idx, node_dist),
        "time_bands": [{"label": f"до {lim} хв", "color": c} for lim, c in TIME_BANDS if lim <= walk],
        "colors": {"competitor": COMPETITOR, "synergy": SYNERGY, "outside": STREET_OUTSIDE},
        "synergy_groups": ", ".join(GROUP_LABELS.get(g, g.replace("_", " ")) for g in profile.synergy),
    }


# --- rendering ----------------------------------------------------------------------------
def render_html(ctx: dict) -> str:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATES), autoescape=jinja2.select_autoescape(["html"]))
    return env.get_template("report.html").render(**ctx)


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if c and (Path(c).exists() or shutil.which(c)):
            return shutil.which(c) or c
    raise RuntimeError("Не знайдено Chromium/Chrome. Встановіть його або вкажіть шлях у змінній LOCAI_CHROME.")


def html_to_pdf(html_text: str, pdf_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "report.html"
        src.write_text(html_text, encoding="utf-8")
        cmd = [find_chrome(), "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
               f"--user-data-dir={tmp}/profile", f"--print-to-pdf={Path(pdf_path).resolve()}", src.as_uri()]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0 or not Path(pdf_path).exists():
            raise RuntimeError(f"Chromium не зміг створити PDF:\n{html.escape(proc.stderr[-2000:])}")
