from __future__ import annotations

import re
from pathlib import Path


HEX_OR_RGB_PATTERN = re.compile(r"(#[0-9a-fA-F]{3,8}|rgba?\()")


def _iter_ui_files(root: Path) -> list[Path]:
    ui_root = root / "jupr_app" / "ui"
    return [path for path in ui_root.rglob("*.py")]


def test_ui_files_do_not_use_hardcoded_colors():
    repo_root = Path(__file__).resolve().parents[1]
    # Token migration is incremental; enforce on the migrated surface.
    enforced_paths = {
        repo_root / "jupr_app" / "ui" / "components" / "skeleton.py",
        repo_root / "jupr_app" / "ui" / "components" / "badge_cards.py",
        repo_root / "jupr_app" / "ui" / "pages" / "players.py",
        repo_root / "jupr_app" / "ui" / "pages" / "admin_tools.py",
    }
    allowlist = {
        repo_root / "jupr_app" / "ui" / "theme_clean.py",
    }
    violations: list[str] = []

    for path in _iter_ui_files(repo_root):
        if path not in enforced_paths:
            continue
        if path in allowlist:
            continue
        contents = path.read_text(encoding="utf-8")
        for match in HEX_OR_RGB_PATTERN.finditer(contents):
            line_no = contents.count("\n", 0, match.start()) + 1
            violations.append(f"{path.relative_to(repo_root)}:{line_no} -> {match.group(0)}")

    if violations:
        joined = "\n".join(violations)
        raise AssertionError(
            "Hard-coded color literals detected in UI files. Use theme tokens instead:\n" + joined
        )
