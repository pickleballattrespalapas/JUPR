from pathlib import Path

path = Path("scripts/staging_write_waves.py")
text = path.read_text(encoding="utf-8")
replacements = {
    '        ("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/skip"),\n': (
        '        ("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/played"),\n'
        '        ("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/skip"),\n'
    ),
    '        ("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/skip"),\n': (
        '        ("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/played"),\n'
        '        ("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/skip"),\n'
    ),
}
for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"Expected one route marker, found {count}: {old!r}")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
print("Classified Round Played routes in the staging write manifest.")
