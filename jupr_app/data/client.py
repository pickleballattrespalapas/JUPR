from supabase import Client, create_client


def make_supabase(url: str, key: str) -> Client:
    return create_client(
        url,
        key,
        options={
            "persist_session": True,
            "auto_refresh_token": True,
        },
    )
