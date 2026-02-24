import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {
    "domain/match_processing.py",
    "domain/replay_history.py",
    # Session ladder applies ratings from canonical session game history.
    "domain/session_ladder_service.py",
    # Player merge reassigns historical rating records during account consolidation.
    "domain/player_merge.py",
}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "build", "dist"}

MUTATION_PATTERNS = {
    "sb_update_players_rating": re.compile(r'sb_update\s*\([\s\S]{0,400}?"players"\s*,\s*\{\s*"rating"', re.MULTILINE),
    "sb_update_league_ratings": re.compile(r'sb_update\s*\([\s\S]{0,400}?"league_ratings"', re.MULTILINE),
    "supabase_players_update_rating": re.compile(r'supabase\.table\("players"\)\.update\(\{\s*"rating"', re.MULTILINE),
    "supabase_league_ratings_update_rating": re.compile(r'supabase\.table\("league_ratings"\)\.update\(\{\s*"rating"', re.MULTILINE),
}


def _is_allowed_file(path: Path) -> bool:
    path_str = path.as_posix()
    return any(path_str.endswith(suffix) for suffix in ALLOWED_SUFFIXES)


def test_no_ui_rating_mutations_outside_domain_guards():
    violations: list[str] = []

    for file_path in REPO_ROOT.rglob("*.py"):
        rel_path = file_path.relative_to(REPO_ROOT)

        if any(part in SKIP_DIRS for part in rel_path.parts):
            continue
        if _is_allowed_file(rel_path):
            continue
        if rel_path == Path("tests/test_no_ui_rating_mutations.py"):
            continue

        content = file_path.read_text(encoding="utf-8")
        for pattern_name, pattern in MUTATION_PATTERNS.items():
            if pattern.search(content):
                violations.append(f"{rel_path.as_posix()} ({pattern_name})")

    if violations:
        print("Rating mutation violations found:")
        for violation in violations:
            print(f" - {violation}")

    assert not violations, "Found forbidden rating mutations outside allowed domain files."
