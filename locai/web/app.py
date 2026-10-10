"""FastAPI app: address form → free preview → checkout → paid report (HTML + PDF).

Run: python -m locai serve   (or: uvicorn locai.web.app:create_app --factory)

Environment:
  LOCAI_PRICE_CENTS      report price in minor units (default 3900)
  LOCAI_CURRENCY         ISO currency (default eur)
  LOCAI_BASE_URL         public URL used in payment redirects (default: from the request)
  STRIPE_SECRET_KEY      enables real Stripe Checkout; without it the demo payment flow is used
  STRIPE_WEBHOOK_SECRET  signing secret of the /stripe/webhook endpoint
"""
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import CITIES, DATA_DIR
from ..metrics import DIMENSIONS
from ..profiles import list_profiles, load_profile
from ..report import GRADE_TEXT
from ..service import LocationService, Scored
from .payments import DemoPayments, StripePayments
from .store import Order, OrderStore

log = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent
PRODUCT_NAME = "Location AI — звіт про локацію"


@dataclass
class Settings:
    db_path: Path = DATA_DIR / "web" / "orders.sqlite"
    reports_dir: Path = DATA_DIR / "web" / "reports"
    price_cents: int = int(os.environ.get("LOCAI_PRICE_CENTS", "3900"))
    currency: str = os.environ.get("LOCAI_CURRENCY", "eur")
    base_url: str | None = os.environ.get("LOCAI_BASE_URL")
    stripe_secret_key: str | None = os.environ.get("STRIPE_SECRET_KEY")
    stripe_webhook_secret: str | None = os.environ.get("STRIPE_WEBHOOK_SECRET")
    enabled_cities: list[str] = field(default_factory=lambda: sorted(CITIES))


def _price_label(cents: int, currency: str) -> str:
    symbol = {"eur": "€", "pln": "zł", "ron": "lei", "uah": "грн"}.get(currency, currency.upper())
    amount = f"{cents / 100:.2f}".rstrip("0").rstrip(".")
    return f"{amount} {symbol}"


def create_app(settings: Settings | None = None, service: LocationService | None = None) -> FastAPI:
    settings = settings or Settings()
    svc = service or LocationService()
    store = OrderStore(settings.db_path)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    if settings.stripe_secret_key:
        if not settings.stripe_webhook_secret:
            raise RuntimeError("STRIPE_WEBHOOK_SECRET is required when STRIPE_SECRET_KEY is set")
        payments = StripePayments(settings.stripe_secret_key, settings.stripe_webhook_secret)
    else:
        payments = DemoPayments()
        log.warning("Stripe is not configured: using DEMO payments (no money is charged).")

    app = FastAPI(title="Location AI", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    profiles = [{"id": p, "label": load_profile(p).label} for p in list_profiles()]
    price = _price_label(settings.price_cents, settings.currency)
    report_locks: dict[str, threading.Lock] = {}
    locks_guard = threading.Lock()

    def page(request: Request, name: str, status: int = 200, **ctx) -> HTMLResponse:
        base = {"price": price, "demo": payments.name == "demo", "profiles": profiles,
                "cities": [{"key": k, "name": CITIES[k].name} for k in settings.enabled_cities]}
        return templates.TemplateResponse(request, name, {**base, **ctx}, status_code=status)

    def base_url(request: Request) -> str:
        return (settings.base_url or str(request.base_url)).rstrip("/")

    def order_or_404(token: str) -> Order:
        order = store.get(token)
        if order is None:
            raise HTTPException(404, "Замовлення не знайдено")
        return order

    def rescore(order: Order) -> Scored:
        geo = svc.geocode(order.city, order.query)
        return svc.score(order.city, order.profile, geo)

    def report_files(order: Order) -> tuple[Path, Path]:
        """Build (once) and cache the paid report as HTML and PDF."""
        html_path = settings.reports_dir / f"{order.token}.html"
        pdf_path = settings.reports_dir / f"{order.token}.pdf"
        with locks_guard:
            lock = report_locks.setdefault(order.token, threading.Lock())
        with lock:
            if not html_path.exists():
                html_text = svc.report_html(rescore(order), download_url=f"/report/{order.token}.pdf")
                html_path.write_text(html_text, encoding="utf-8")
            if not pdf_path.exists():
                from ..pdf_report import html_to_pdf

                html_to_pdf(html_path.read_text(encoding="utf-8"), pdf_path)
        return html_path, pdf_path

    # --- pages -----------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return page(request, "index.html")

    @app.post("/score")
    def score(request: Request, address: str = Form(...), profile: str = Form(...), city: str = Form("lyon")):
        address = address.strip()[:200]
        if city not in settings.enabled_cities or profile not in list_profiles():
            raise HTTPException(400, "Невідоме місто або тип бізнесу")
        try:
            geo = svc.geocode(city, address)
        except ValueError as e:
            return page(request, "index.html", status=422, error=str(e), address=address, profile=profile)
        s = svc.score(city, profile, geo)
        order = store.create(
            city=city, profile=profile, query=address, address=geo.label, address_exact=int(geo.exact),
            lat=float(geo.lat), lon=float(geo.lon), score=s.result.total, grade=s.result.grade,
            price_cents=settings.price_cents, currency=settings.currency,
        )
        return RedirectResponse(f"/preview/{order.token}", status_code=303)

    @app.get("/preview/{token}", response_class=HTMLResponse)
    def preview(request: Request, token: str):
        order = order_or_404(token)
        s = rescore(order)
        walk = s.scorer.profile.walk_minutes
        t_places, t_cells = s.scorer.engine.times(s.x, s.y, walk)
        teaser = {
            "population": f"{s.city.cells['pop'].to_numpy()[t_cells <= walk].sum():,.0f}".replace(",", " "),
            "competitors": int((s.scorer.engine.comp_mask & (t_places <= walk)).sum()),
            "walk": f"{walk:.0f}",
        }
        # Locked dimensions get placeholder widths so the real values never reach the page.
        rng = np.random.default_rng(len(token))
        locked = [{"name": name, "width": int(rng.integers(30, 90))} for name in DIMENSIONS.values()]
        return page(request, "preview.html", order=order, profile_label=s.scorer.profile.label,
                    grade_text=GRADE_TEXT[order.grade], percentile=f"{s.result.city_percentile:.0f}",
                    teaser=teaser, locked=locked)

    @app.post("/checkout/{token}")
    def checkout(request: Request, token: str):
        order = order_or_404(token)
        if order.paid:
            return RedirectResponse(f"/report/{token}", status_code=303)
        result = payments.create_checkout(order, base_url(request), PRODUCT_NAME)
        store.set_payment_ref(token, result.payment_ref)
        return RedirectResponse(result.redirect_url, status_code=303)

    # Registered before /report/{token}, which would otherwise capture "<token>.pdf".
    @app.get("/report/{token}.pdf")
    def report_pdf(token: str):
        order = order_or_404(token)
        if not order.paid:
            raise HTTPException(402, "Звіт ще не оплачено")
        _, pdf_path = report_files(order)
        return FileResponse(pdf_path, media_type="application/pdf",
                            filename=f"location-ai-{order.profile}-{token[:6]}.pdf")

    @app.get("/report/{token}", response_class=HTMLResponse)
    def report(request: Request, token: str):
        order = order_or_404(token)
        if not order.paid:
            if payments.name == "stripe" and request.query_params.get("paid"):
                # Returned from Stripe before the webhook arrived: wait and refresh.
                return page(request, "pending.html", order=order)
            return RedirectResponse(f"/preview/{token}", status_code=303)
        html_path, _ = report_files(order)
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    # --- API ---------------------------------------------------------------------------
    @app.get("/api/suggest")
    def suggest(q: str = "", city: str = "lyon"):
        if city not in settings.enabled_cities:
            return JSONResponse([])
        return JSONResponse(svc.suggest(city, q[:100]))

    @app.get("/api/orders/{token}")
    def order_status(token: str):
        order = order_or_404(token)
        return {"status": order.status}

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "payments": payments.name}

    # --- payments ----------------------------------------------------------------------
    if isinstance(payments, StripePayments):
        @app.post("/stripe/webhook")
        async def stripe_webhook(request: Request):
            payload = await request.body()
            try:
                event = payments.parse_webhook(payload, request.headers.get("stripe-signature", ""))
            except Exception as e:  # signature or payload error
                log.warning("Rejected Stripe webhook: %s", e)
                raise HTTPException(400, "invalid webhook")
            if event and store.get(event.token):
                store.mark_paid(event.token, event.payment_ref, event.email)
            return {"received": True}
    else:
        @app.get("/demo-pay/{token}", response_class=HTMLResponse)
        def demo_pay_form(request: Request, token: str):
            order = order_or_404(token)
            if order.paid:
                return RedirectResponse(f"/report/{token}", status_code=303)
            return page(request, "demo_pay.html", order=order)

        @app.post("/demo-pay/{token}")
        def demo_pay(token: str, email: str = Form("")):
            order_or_404(token)
            store.mark_paid(token, f"demo_{token}", email.strip()[:200] or None)
            return RedirectResponse(f"/report/{token}", status_code=303)

    return app
