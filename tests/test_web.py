"""End-to-end tests of the web flow. Need the Lyon data (python -m locai build-city lyon)."""
import hashlib
import hmac
import json
import time
from types import SimpleNamespace

import pytest

from locai.config import CITIES, city_dir

pytestmark = pytest.mark.skipif(not (city_dir(CITIES["lyon"]) / "addresses.parquet").exists(),
                                reason="Lyon data not built")

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from locai.service import LocationService  # noqa: E402
from locai.web.app import Settings, create_app  # noqa: E402

ADDRESS = "12 rue Paul Bert"
SERVICE = LocationService()  # share loaded city data between tests


def make_client(tmp_path, **overrides) -> TestClient:
    settings = Settings(db_path=tmp_path / "orders.sqlite", reports_dir=tmp_path / "reports",
                        stripe_secret_key=None, stripe_webhook_secret=None, base_url="http://testserver")
    for k, v in overrides.items():
        setattr(settings, k, v)
    return TestClient(create_app(settings, SERVICE), follow_redirects=False)


def start_order(client) -> str:
    r = client.post("/score", data={"address": ADDRESS, "profile": "beauty_salon", "city": "lyon"})
    assert r.status_code == 303
    return r.headers["location"].rsplit("/", 1)[1]


def test_index_and_suggest(tmp_path):
    c = make_client(tmp_path)
    assert c.get("/").status_code == 200
    s = c.get("/api/suggest", params={"q": "12 rue paul bert"}).json()
    assert s[0].startswith("12 Rue Paul Bert")


def test_unknown_address_shows_form_error(tmp_path):
    r = make_client(tmp_path).post("/score", data={"address": "5 avenue nulle part", "profile": "cafe"})
    assert r.status_code == 422
    assert "не знайдено" in r.text


def test_preview_does_not_leak_paid_content(tmp_path):
    c = make_client(tmp_path)
    token = start_order(c)
    r = c.get(f"/preview/{token}")
    assert r.status_code == 200
    assert "12 Rue Paul Bert" in r.text and "/100" in r.text
    # The map and per-metric content belong to the paid report only.
    assert "<svg" not in r.text and "Хаффа" not in r.text
    assert c.get(f"/report/{token}").status_code == 303
    assert c.get(f"/report/{token}.pdf").status_code == 402
    assert c.get("/preview/nope").status_code == 404


def test_demo_payment_unlocks_report(tmp_path):
    c = make_client(tmp_path)
    token = start_order(c)
    r = c.post(f"/checkout/{token}")
    assert r.headers["location"].endswith(f"/demo-pay/{token}")
    r = c.post(f"/demo-pay/{token}", data={"email": "a@example.com"})
    assert r.headers["location"] == f"/report/{token}"
    r = c.get(f"/report/{token}")
    assert r.status_code == 200
    assert "Завантажити PDF" in r.text and "<svg" in r.text
    assert c.get(f"/api/orders/{token}").json() == {"status": "paid"}


def test_demo_pdf_download(tmp_path):
    from locai.pdf_report import find_chrome

    try:
        find_chrome()
    except RuntimeError:
        pytest.skip("Chromium not installed")
    c = make_client(tmp_path)
    token = start_order(c)
    c.post(f"/demo-pay/{token}")
    r = c.get(f"/report/{token}.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


def _signed(payload: dict, secret: str) -> tuple[bytes, str]:
    body = json.dumps(payload).encode()
    ts = int(time.time())
    sig = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
    return body, f"t={ts},v1={sig}"


def test_stripe_checkout_and_webhook(tmp_path, monkeypatch):
    secret = "whsec_test"
    c = make_client(tmp_path, stripe_secret_key="sk_test_dummy", stripe_webhook_secret=secret)
    assert c.get("/demo-pay/x").status_code == 404  # demo flow is off when Stripe is configured
    token = start_order(c)

    created = {}

    def fake_create(self, params=None, options=None):
        created.update(params)
        return SimpleNamespace(url="https://checkout.stripe.test/s/cs_1", id="cs_1")

    import stripe

    session_service = type(stripe.StripeClient("sk_test_dummy").v1.checkout.sessions)
    monkeypatch.setattr(session_service, "create", fake_create)
    r = c.post(f"/checkout/{token}")
    assert r.headers["location"] == "https://checkout.stripe.test/s/cs_1"
    assert created["client_reference_id"] == token
    assert created["line_items"][0]["price_data"]["unit_amount"] == 3900

    # Back from Stripe before the webhook: waiting page, not the report.
    assert "Підтверджуємо оплату" in c.get(f"/report/{token}?paid=1").text

    event = {"id": "evt_1", "object": "event", "type": "checkout.session.completed",
             "data": {"object": {"id": "cs_1", "object": "checkout.session", "client_reference_id": token,
                                 "payment_status": "paid", "customer_details": {"email": "b@example.com"}}}}
    body, bad_sig = _signed(event, "wrong")
    assert c.post("/stripe/webhook", content=body, headers={"stripe-signature": bad_sig}).status_code == 400
    assert c.get(f"/api/orders/{token}").json()["status"] == "preview"

    body, sig = _signed(event, secret)
    assert c.post("/stripe/webhook", content=body, headers={"stripe-signature": sig}).status_code == 200
    assert c.get(f"/api/orders/{token}").json()["status"] == "paid"
