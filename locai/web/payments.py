"""Payment providers.

StripePayments — real Stripe Checkout (set STRIPE_SECRET_KEY and STRIPE_WEBHOOK_SECRET).
DemoPayments   — no money moves; a "pay" button marks the order paid. Only for development.
"""
from dataclasses import dataclass

from .store import Order


@dataclass
class CheckoutResult:
    redirect_url: str
    payment_ref: str


@dataclass
class PaidEvent:
    token: str
    payment_ref: str
    email: str | None


class DemoPayments:
    name = "demo"

    def create_checkout(self, order: Order, base_url: str, product_name: str) -> CheckoutResult:
        return CheckoutResult(f"{base_url}/demo-pay/{order.token}", f"demo_{order.token}")


class StripePayments:
    name = "stripe"

    def __init__(self, secret_key: str, webhook_secret: str):
        import stripe

        self._stripe = stripe
        self._client = stripe.StripeClient(secret_key)
        self._webhook_secret = webhook_secret

    def create_checkout(self, order: Order, base_url: str, product_name: str) -> CheckoutResult:
        session = self._client.v1.checkout.sessions.create(params={
            "mode": "payment",
            "line_items": [{
                "quantity": 1,
                "price_data": {
                    "currency": order.currency,
                    "unit_amount": order.price_cents,
                    "product_data": {"name": product_name, "description": order.address},
                },
            }],
            "client_reference_id": order.token,
            "metadata": {"order_token": order.token},
            "success_url": f"{base_url}/report/{order.token}?paid=1",
            "cancel_url": f"{base_url}/preview/{order.token}",
        })
        return CheckoutResult(session.url, session.id)

    def parse_webhook(self, payload: bytes, signature: str) -> PaidEvent | None:
        """Verify the Stripe signature; return the paid order for checkout.session.completed."""
        event = self._stripe.Webhook.construct_event(payload, signature, self._webhook_secret).to_dict()
        if event["type"] != "checkout.session.completed":
            return None
        s = event["data"]["object"]
        if s.get("payment_status") != "paid":
            return None
        email = (s.get("customer_details") or {}).get("email")
        return PaidEvent(s["client_reference_id"], s["id"], email)
