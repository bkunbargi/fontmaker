"""Credit code generation and management for paid AI generation."""

import streamlit as st
from supabase import create_client
import secrets
import string


def get_supabase():
    """Get Supabase client using Streamlit secrets."""
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_KEY"]
    )


def generate_code() -> str:
    """Generate a unique credit code like FONT-A7X9K2."""
    chars = string.ascii_uppercase + string.digits
    suffix = ''.join(secrets.choice(chars) for _ in range(6))
    return f"FONT-{suffix}"


def create_credit_code(amount: int = 3) -> str:
    """Create a new credit code with given amount."""
    code = generate_code()
    # Ensure uniqueness
    while get_credits(code) is not None:
        code = generate_code()
    get_supabase().table("credits").insert({
        "code": code,
        "amount": amount
    }).execute()
    return code


def get_credits(code: str) -> int | None:
    """Get credits for a code. Returns None if code doesn't exist."""
    result = get_supabase().table("credits").select("amount").eq("code", code.upper()).execute()
    return result.data[0]["amount"] if result.data else None


def use_credit(code: str) -> bool:
    """Use 1 credit. Returns True if successful."""
    current = get_credits(code)
    if current and current > 0:
        get_supabase().table("credits").update({"amount": current - 1}).eq("code", code.upper()).execute()
        return True
    return False
