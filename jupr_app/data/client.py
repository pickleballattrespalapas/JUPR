from supabase import create_client

def make_supabase(url: str, key: str):
    return create_client(url, key)
