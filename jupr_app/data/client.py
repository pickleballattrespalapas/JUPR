from supabase import create_client, Client


def make_supabase(url: str, key: str) -> Client:
    """
    Minimal Supabase client factory.

    Do NOT pass ClientOptions here.
    Streamlit server-side apps manage session in st.session_state.
    """
    return create_client(url, key)
