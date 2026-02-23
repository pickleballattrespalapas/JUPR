from supabase import create_client
import os


def get_service_supabase():
    """
    Returns Supabase client using service_role key.
    Used ONLY for canonical domain-level writes.
    """

    url = os.environ.get("SUPABASE_URL")
    service_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not service_key:
        raise RuntimeError("Missing SUPABASE service role credentials")

    return create_client(url, service_key)
