from __future__ import annotations

import pandas as pd


def preview_week_tag_update(
    df_matches: pd.DataFrame,
    selected_ids: list[int],
    new_week_tag: str,
) -> dict[str, object]:
    if df_matches is None or df_matches.empty:
        return {"count": 0, "old_tags": [], "new_tag": new_week_tag}

    if not selected_ids:
        return {"count": 0, "old_tags": [], "new_tag": new_week_tag}

    subset = df_matches[df_matches["id"].isin(selected_ids)].copy()
    if subset.empty:
        return {"count": 0, "old_tags": [], "new_tag": new_week_tag}

    old_tags = (
        subset.get("week_tag", "")
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "(blank)")
        .unique()
        .tolist()
    )
    old_tags_sorted = sorted(old_tags)

    return {
        "count": int(len(subset)),
        "old_tags": old_tags_sorted,
        "new_tag": new_week_tag,
    }
