from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions


def make_supabase(url: str, key: str) -> Client:
    options = ClientOptions(
        persist_session=True,
        auto_refresh_token=True,
    )

    return create_client(
        url,
        key,
        options=options,
    )
