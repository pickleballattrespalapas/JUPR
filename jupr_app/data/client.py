from supabase import create_client

def make_supabase(url: str, key: str):
    return create_client(url, key)
import httpx

from supabase.lib.client_options import ClientOptions

def get_supabase_client():
    http_client = httpx.Client(
        http2=False,  # 🔥 critical for replay stability
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(
            max_connections=5,
            max_keepalive_connections=2,
        ),
    )

    options = ClientOptions(
        httpx_client=http_client,
        postgrest_client_timeout=60,
        storage_client_timeout=60,
    )

    return create_client(
        SUPABASE_URL,
        SUPABASE_KEY,
        options=options,
    )
