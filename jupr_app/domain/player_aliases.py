"""Public presentation only: canonical player names and IDs never change."""
from __future__ import annotations

import re
from typing import Any

from jupr_app.data.paged_reads import read_all_rows


def public_player_name(row: dict) -> str:
    return str(row.get("public_alias") or row.get("name") or "Player").strip()


def load_player_aliases(supabase: Any, club_id: str) -> dict[str, str]:
    rows = read_all_rows(lambda: supabase.table("players")
                         .select("id,name,public_alias").eq("club_id", club_id)
                         .neq("public_alias", ""))
    return {str(row["name"]).strip(): str(row["public_alias"]).strip()
            for row in rows if row.get("name") and str(row.get("public_alias") or "").strip()}


def public_alias_projection(value: Any, aliases: dict[str, str]) -> Any:
    """Also cover saved name snapshots, team labels, and authored recap text.

    A single pass avoids cascading aliases. Full-name word boundaries avoid
    rewriting another person's longer surname. Does not mutate stored objects.
    """
    if not aliases:
        return value
    replacements = {re.sub(r"\s+", " ", name).casefold(): alias for name, alias in aliases.items()}
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(
        r"\s+".join(re.escape(part) for part in name.split())
        for name in sorted(aliases, key=len, reverse=True)
    ) + r")(?!\w)", re.IGNORECASE)

    def replace(text: str) -> str:
        return pattern.sub(lambda match: replacements[re.sub(r"\s+", " ", match.group()).casefold()], text)

    def visit(item: Any) -> Any:
        if isinstance(item, str):
            return replace(item)
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, dict):
            return {replace(key): visit(child) for key, child in item.items()
                    if key != "public_alias"}
        return item

    return visit(value)
