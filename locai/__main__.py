"""CLI.

  python -m locai build-city lyon
  python -m locai score "10 rue de la République" --profile cafe [--city lyon] [--json out.json] [--pdf out.pdf]
  python -m locai serve [--port 8000]
"""
import argparse
import json
import logging
import sys

from .city import build_city
from .config import CITIES
from .profiles import list_profiles
from .report import to_json, to_text
from .service import LocationService


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
    w = sub.add_parser("serve", help="запустити сайт")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(message)s")

    if args.cmd == "build-city":
        build_city(CITIES[args.city])
        return 0

    if args.cmd == "serve":
        import uvicorn

        from .web.app import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    svc = LocationService()
    try:
        geo = svc.geocode(args.city, args.address)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    s = svc.score(args.city, args.profile, geo)
    nearby = [] if args.no_nearby else svc.nearby(s)
    print(to_text(geo, s.scorer.profile, s.result, s.city.city.name, nearby))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(to_json(geo, s.scorer.profile, s.result, nearby), f, ensure_ascii=False, indent=2)
    if args.pdf or args.html:
        from .pdf_report import build_context, html_to_pdf, render_html

        page = render_html(build_context(s.city, s.scorer, geo, s.result, nearby))
        if args.html:
            with open(args.html, "w", encoding="utf-8") as f:
                f.write(page)
        if args.pdf:
            html_to_pdf(page, args.pdf)
            print(f"PDF: {args.pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
