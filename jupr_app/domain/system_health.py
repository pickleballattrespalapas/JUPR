from datetime import datetime, timezone


def get_system_health(supabase, club_id: str):
    health = {}

    try:
        matches = (
            supabase.table("matches")
            .select("id", count="exact")
            .eq("club_id", club_id)
            .execute()
        )
        health["match_count"] = matches.count or 0
    except Exception:
        health["match_count"] = "ERR"

    try:
        q = (
            supabase.table("badge_eval_queue")
            .select("id", count="exact")
            .eq("club_id", club_id)
            .eq("status", "pending")
            .execute()
        )
        health["pending_badge_jobs"] = q.count or 0
    except Exception:
        health["pending_badge_jobs"] = "N/A"

    health["timestamp"] = datetime.now(timezone.utc).isoformat()

    return health
