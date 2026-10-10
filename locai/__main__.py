"""CLI.

  python -m locai build-city lyon
  python -m locai score "10 rue de la République" --profile cafe [--city lyon] [--json out.json] [--pdf out.pdf]
"""
import argparse
import json
import logging
import sys

from .city import build_city, load_city
from .config import CITIES
from .geocode import geocode, reverse
from .profiles import list_profiles, load_profile
from .report import to_json, to_text
from .scoring import Scorer


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="locai", description="Location AI — оцінка локації для малого бізнесу (прототип)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-city", help="завантажити й підготувати дані міста")
    b.add_argument("city", choices=sorted(CITIES))
    s = sub.add_parser("score", help="оцінити адресу")
    s.add_argument("address", help="«10 rue de la République» або «45.7640, 4.8357»")
    s.add_argument("--profile", required=True, choices=list_profiles())
    s.add_argument("--city", default="lyon", choices=sorted(CITIES))
    s.add_argument("--json", help="зберегти результат у JSON")
    s.add_argument("--pdf", help="зберегти звіт у PDF (потрібен Chromium/Chrome)")
    s.add_argument("--html", help="зберегти веб-версію звіту в HTML")
    s.add_argument("--no-nearby", action="store_true", help="не шукати кращих локацій поруч (швидше)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(message)s")

    if args.cmd == "build-city":
        build_city(CITIES[args.city])
        return 0

    city = load_city(CITIES[args.city])
    profile = load_profile(args.profile)
    try:
        geo = geocode(args.address, city.addresses, CITIES[args.city].default_locality)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if not city.city.contains(geo.lon, geo.lat):
        print(f"Адреса поза зоною покриття ({city.city.name}).", file=sys.stderr)
        return 2
    x, y = city.proj.to_xy(geo.lon, geo.lat)
    scorer = Scorer(city, profile)
    result = scorer.score_xy(float(x), float(y))
    nearby = [] if args.no_nearby else scorer.better_nearby(float(x), float(y), result.total)
    for n in nearby:
        n["address"] = reverse(n["lat"], n["lon"], city.addresses)
    print(to_text(geo, profile, result, city.city.name, nearby))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(to_json(geo, profile, result, nearby), f, ensure_ascii=False, indent=2)
    if args.pdf or args.html:
        from .pdf_report import build_context, html_to_pdf, render_html

        page = render_html(build_context(city, scorer, geo, result, nearby))
        if args.html:
            with open(args.html, "w", encoding="utf-8") as f:
                f.write(page)
        if args.pdf:
            html_to_pdf(page, args.pdf)
            print(f"PDF: {args.pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
