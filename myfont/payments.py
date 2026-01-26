"""Stripe payment verification for credit purchases."""

import streamlit as st
import stripe


def verify_payment(session_id: str) -> bool:
    """Verify Stripe session was paid."""
    stripe.api_key = st.secrets["STRIPE_SECRET_KEY"]
    try:
        session = stripe.checkout.Session.retrieve(session_id)
        return session.payment_status == "paid"
    except Exception:
        return False
