from pathlib import Path
import re


BADGE_RENDER_FILES = [
    Path("jupr_app/ui/pages/badge_codex.py"),
    Path("jupr_app/ui/pages/players.py"),
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_badge_renderers_do_not_label_requirements():
    for path in BADGE_RENDER_FILES:
        content = _read(path)
        assert "Requirements:" not in content, f"Found Requirements: label in {path}"


def test_badge_renderers_do_not_use_lore_or_hint_fields():
    patterns = [
        r"badge\.get\([\"']lore[\"']\)",
        r"badge\.get\([\"']hint[\"']\)",
        r"badge\.get\([\"']description[\"']\)",
        r"badge\.get\([\"']flavor[\"']\)",
        r"badge\.get\([\"']tagline[\"']\)",
    ]
    for path in BADGE_RENDER_FILES:
        content = _read(path)
        for pattern in patterns:
            assert re.search(pattern, content) is None, f"Unexpected lore/hint usage in {path}: {pattern}"
