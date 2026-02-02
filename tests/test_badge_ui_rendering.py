import re
from pathlib import Path

from jupr_app.domain.gamification.badge_copy import BadgeCopyPlain
from jupr_app.ui.components.badge_cards import render_badge_card_html


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


def test_badge_card_html_has_no_blank_lines():
    copy_plain = BadgeCopyPlain(desc_text="Desc", req_text="Do thing", meta_text="Meta")
    html = render_badge_card_html(name="Test Badge", icon="🏆", copy_plain=copy_plain, state_label="Live")
    lines = html.splitlines()
    assert all(line.strip() for line in lines), f"Found blank lines in badge card HTML: {lines}"
