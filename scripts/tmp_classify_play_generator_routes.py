from pathlib import Path

path = Path(__file__).resolve().parents[1] / "scripts/staging_write_waves.py"
text = path.read_text(encoding="utf-8")
old = '''    "jupr-live": (
        ("POST", "/admin/clubs/{club_id}/jupr-live/sessions"),
        ("PATCH", "/admin/clubs/{club_id}/jupr-live/sessions/{session_key}"),
        ("PATCH", "/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/scores"),
        ("POST", "/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/advance"),
        ("POST", "/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/publish"),
        ("POST", "/admin/clubs/{club_id}/jupr-live/operations/{operation_key}/reconcile"),
    ),
'''
new = '''    "jupr-live": (
        # Legacy API paths remain classified during the generator migration.
        ("POST", "/admin/clubs/{club_id}/jupr-live/sessions"),
        ("PATCH", "/admin/clubs/{club_id}/jupr-live/sessions/{session_key}"),
        ("PATCH", "/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/scores"),
        ("POST", "/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/advance"),
        ("POST", "/admin/clubs/{club_id}/jupr-live/sessions/{session_key}/publish"),
        ("POST", "/admin/clubs/{club_id}/jupr-live/operations/{operation_key}/reconcile"),
        # Round-Robin Generator and Ladder Generator share the reviewed
        # permanent-open staging gate while using their own product-facing API.
        ("POST", "/admin/clubs/{club_id}/play-generators/preview"),
        ("POST", "/admin/clubs/{club_id}/play-generators/sessions"),
        ("PATCH", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/scores"),
        ("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/skip"),
        ("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/advance"),
        ("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/roster"),
        ("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/complete"),
        ("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/publish"),
    ),
'''
if text.count(old) != 1:
    raise SystemExit(f"jupr-live route block match count={text.count(old)}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
