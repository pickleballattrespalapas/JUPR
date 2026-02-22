import os

from supabase import Client, create_client

from jupr_app.utils.write_guard import install_match_insert_guard


def make_supabase(url: str, key: str) -> Client:
    """
    Minimal Supabase client factory.

    Do NOT pass ClientOptions here.
    Streamlit server-side apps manage session in st.session_state.
    """
    client = create_client(url, key)
    env = str(os.getenv("JUPR_ENV") or os.getenv("ENV") or "").lower()
    if env in {"dev", "development", "local"}:
        install_match_insert_guard(client)
    return client
