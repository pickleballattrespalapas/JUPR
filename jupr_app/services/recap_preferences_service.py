"""Club layouts are versioned independently from individual recap editions."""
from datetime import datetime, timezone
from jupr_app.domain.recaps.recap_builder import normalize_config
from jupr_app.domain.recaps.weekly_report_public import text
from jupr_app.services.staging_write_guard import require_staging_communications_mutations


def get_preferences(db, club_id):
    rows = db.table("club_recap_preferences").select("revision,settings").eq("club_id", club_id).limit(1).execute().data or []
    row = rows[0] if rows else {"revision": 0, "settings": {"templates": [], "default_template_id": ""}}
    leagues = db.table("leagues_metadata").select("league_name").eq("club_id", club_id).execute().data or []
    return {**row, "leagues": sorted({r["league_name"] for r in leagues if r.get("league_name")})}


def save_preferences(db, club_id, revision, settings):
    require_staging_communications_mutations()
    from jupr_app.services.admin_weekly_recap_service import StaleWeeklyRecapStateError
    if len(settings.get("templates") or []) > 12:
        raise ValueError("A club can save up to 12 recap layouts.")
    templates = []
    for item in (settings.get("templates") or [])[:12]:
        if not isinstance(item, dict) or not text(item.get("name"), 80) or not text(item.get("id"), 100):
            raise ValueError("Each template needs a name and ID.")
        templates.append({"id": text(item["id"], 100), "name": text(item["name"], 80), "builder": normalize_config(item.get("builder"), template=True)})
    if len({t["id"] for t in templates}) != len(templates):
        raise ValueError("Template IDs must be unique.")
    default_id = text(settings.get("default_template_id"), 100)
    if default_id and default_id not in {t["id"] for t in templates}:
        raise ValueError("Choose a saved default template.")
    payload = {"club_id": club_id, "revision": revision + 1, "settings": {"templates": templates, "default_template_id": default_id}, "updated_at": datetime.now(timezone.utc).isoformat()}
    if revision:
        rows = db.table("club_recap_preferences").update(payload).eq("club_id", club_id).eq("revision", revision).execute().data or []
        if not rows:
            raise StaleWeeklyRecapStateError("Club templates changed. Reload before saving.")
    else:
        try:
            db.table("club_recap_preferences").insert(payload).execute()
        except Exception as exc:
            if "duplicate" in str(exc).lower() or "23505" in str(exc):
                raise StaleWeeklyRecapStateError("Club templates changed. Reload before saving.") from exc
            raise
    return get_preferences(db, club_id)
