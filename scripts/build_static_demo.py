"""Export precomputed examples for the static demo page (no server needed to browse it).

  python scripts/build_static_demo.py out_dir/

Writes examples.json (preview data), reports/<n>.html (full report fragments, loaded on demand)
and report.css (report styles scoped under .rep), all produced by the real engine and template.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from locai.metrics import DIMENSIONS  # noqa: E402
from locai.profiles import list_profiles  # noqa: E402
from locai.report import GRADE_TEXT  # noqa: E402
from locai.service import LocationService  # noqa: E402

ADDRESSES = [
    "12 rue Paul Bert",
    "20 boulevard de la Croix-Rousse",
    "300 avenue Jean Jaurès",
    "25 cours Émile Zola, Villeurbanne",
]


def scope_css(css: str, scope: str = ".rep") -> str:
    """Prefix every selector with `scope` so the report styles cannot leak into the page."""
    out, i = [], 0
    while i < len(css):
        j = css.find("{", i)
        if j < 0:
            break
        head = css[i:j].strip()
        if head.startswith("@media"):
            depth, k = 1, j + 1
            while depth:
                depth += {"{": 1, "}": -1}.get(css[k], 0)
                k += 1
            out.append(f"{head} {{{scope_css(css[j + 1:k - 1], scope)}}}")
            i = k
            continue
        k = css.find("}", j)
        body = css[j + 1:k]
        i = k + 1
        if head.startswith("@page"):
            continue
        sels = []
        for sel in head.split(","):
            sel = sel.strip()
            if sel in (":root", "html", "body"):
                sels.append(scope)
            elif sel == "*":
                sels.append(f"{scope} *")
            else:
                sels.append(f"{scope} {sel}")
        out.append(f"{', '.join(dict.fromkeys(sels))} {{{body}}}")
    return "\n".join(out)


def main(out_dir: Path) -> None:
    svc = LocationService()
    examples, css = [], None
    for q in ADDRESSES:
        geo = svc.geocode("lyon", q)
        for pid in list_profiles():
            s = svc.score("lyon", pid, geo)
            walk = s.scorer.profile.walk_minutes
            t_places, t_cells = s.scorer.engine.times(s.x, s.y, walk)
            html = svc.report_html(s)
            style = re.search(r"<style>(.*?)</style>", html, re.S).group(1)
            body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
            css = css or scope_css(style)
            rng = np.random.default_rng(abs(hash((q, pid))) % 2**32)
            examples.append({
                "id": len(examples),
                "query": q,
                "address": geo.label,
                "profile": pid,
                "profile_label": s.scorer.profile.label,
                "walk": f"{walk:.0f}",
                "score": round(s.result.total),
                "grade": s.result.grade,
                "grade_text": GRADE_TEXT[s.result.grade],
                "percentile": f"{s.result.city_percentile:.0f}",
                "population": f"{s.city.cells['pop'].to_numpy()[t_cells <= walk].sum():,.0f}".replace(",", " "),
                "competitors": int((s.scorer.engine.comp_mask & (t_places <= walk)).sum()),
                "locked": [{"name": n, "width": int(rng.integers(30, 90))} for n in DIMENSIONS.values()],
                "report": body.strip(),
            })
            print(f"{geo.label:<55} {pid:<13} {s.result.total:5.1f}")
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    for e in examples:
        (out_dir / "reports" / f"{e['id']}.html").write_text(e.pop("report"), encoding="utf-8")
    profiles = [{"id": p, "label": next(e["profile_label"] for e in examples if e["profile"] == p)} for p in list_profiles()]
    (out_dir / "examples.json").write_text(json.dumps({"profiles": profiles, "examples": examples}, ensure_ascii=False))
    (out_dir / "report.css").write_text(css)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "demo_build"))
