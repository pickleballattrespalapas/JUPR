import httpx
from supabase import create_client
from supabase.lib.client_options import ClientOptions


def make_supabase(url: str, key: str):
    """
    Centralized Supabase client factory.
    Disables HTTP/2 to stabilize long-running admin jobs (replay).
    """

    http_client = httpx.Client(
        http2=False,
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(
            max_connections=5,
            max_keepalive_connections=2,
        ),
    )

    options = ClientOptions(
        postgrest_client_timeout=60,
        storage_client_timeout=60,
    )

    client = create_client(url, key, options=options)

    # 🔥 Inject custom httpx client into underlying Postgrest client
    client.postgrest._session = http_client

    return client
