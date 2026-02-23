from jupr_app.data.sb_write import sb_insert, sb_update, sb_upsert
from typing import Any


def upsert_or_get_active_event(
    supabase: Any,
    club_id: str,
    name: str,
    event_type: str = "popup_rr",
) -> str:
    if supabase is None:
        raise ValueError("Supabase client is required to upsert events.")
    if not club_id:
        raise ValueError("club_id is required to upsert events.")
    if not name:
        raise ValueError("name is required to upsert events.")

    lookup = (
        supabase.table("events")
        .select("id")
        .eq("club_id", club_id)
        .eq("name", name)
        .eq("event_type", event_type)
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    if lookup.data:
        return str(lookup.data[0]["id"])

    payload = {
        "club_id": club_id,
        "name": name,
        "event_type": event_type,
        "is_active": True,
    }
    inserted = sb_insert(supabase, "events", payload)
    if not inserted.data:
        raise RuntimeError("Failed to insert event record.")
    return str(inserted.data[0]["id"])
