import pandas as pd


def fetch_recent_jobs(supabase, club_id: str, limit: int = 25):
    try:
        resp = (
            supabase.table("jobs")
            .select("*")
            .eq("club_id", club_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return pd.DataFrame(resp.data or [])
    except Exception:
        return pd.DataFrame()
