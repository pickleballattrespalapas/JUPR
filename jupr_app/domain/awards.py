from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class TopPerformerSpec:
    category_key: str
    label: str
    sort_key: str
    value_fn: Callable[[pd.Series], float | int | None]
    display_fn: Callable[[pd.Series, float | int | None], str]


def _numeric_or_zero(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _int_or_zero(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _win_pct_value(row: pd.Series) -> float | None:
    value = row.get("Win %")
    if pd.isna(value):
        return None
    return _numeric_or_zero(value)


TOP_PERFORMER_SPECS: list[TopPerformerSpec] = [
    TopPerformerSpec(
        category_key="highest_rating",
        label="Highest Rating",
        sort_key="rating",
        value_fn=lambda row: _numeric_or_zero(row.get("JUPR")),
        display_fn=lambda _row, value: f"{_numeric_or_zero(value):.3f}",
    ),
    TopPerformerSpec(
        category_key="most_improved",
        label="Most Improved",
        sort_key="rating_gain",
        value_fn=lambda row: _numeric_or_zero(row.get("rating_gain")) / 400.0,
        display_fn=lambda _row, value: f"{_numeric_or_zero(value):+.3f}",
    ),
    TopPerformerSpec(
        category_key="best_win_pct",
        label="Best Win %",
        sort_key="Win %",
        value_fn=_win_pct_value,
        display_fn=lambda _row, value: (
            f"{_numeric_or_zero(value):.1f}%"
            if value is not None and not pd.isna(value)
            else "—"
        ),
    ),
    TopPerformerSpec(
        category_key="most_wins",
        label="Most Wins",
        sort_key="wins",
        value_fn=lambda row: _int_or_zero(row.get("wins")),
        display_fn=lambda _row, value: f"{_int_or_zero(value)}",
    ),
]


def build_top_performer_entries(
    qualified_df: pd.DataFrame | None,
    limit: int = 5,
) -> list[dict]:
    if qualified_df is None or qualified_df.empty:
        return []

    df = qualified_df.copy()
    results: list[dict] = []
    for spec in TOP_PERFORMER_SPECS:
        if spec.sort_key not in df.columns:
            sort_df = df.copy()
            sort_df[spec.sort_key] = 0
        else:
            sort_df = df
        top = sort_df.sort_values(spec.sort_key, ascending=False).head(int(limit))
        entries: list[dict] = []
        for _, row in top.iterrows():
            player_id = row.get("_pid")
            if pd.isna(player_id):
                player_id = row.get("player_id")
            name = str(row.get("name", "") or "")
            metric_value = spec.value_fn(row)
            metric_display = spec.display_fn(row, metric_value)
            entries.append(
                {
                    "player_id": int(player_id) if player_id is not None and not pd.isna(player_id) else None,
                    "name": name,
                    "metric_value": metric_value,
                    "metric_display": metric_display,
                }
            )
        results.append(
            {
                "category_key": spec.category_key,
                "label": spec.label,
                "entries": entries,
            }
        )
    return results


def compute_top_performer_awards(
    qualified_df: pd.DataFrame | None,
    min_games: int,
    winners_per_category: int = 1,
) -> list[dict]:
    entries = build_top_performer_entries(qualified_df, limit=max(1, winners_per_category))
    awards: list[dict] = []
    for category in entries:
        for rank, entry in enumerate(category.get("entries", [])[:winners_per_category], start=1):
            player_id = entry.get("player_id")
            if player_id is None:
                continue
            awards.append(
                {
                    "category_key": category.get("category_key"),
                    "category_label": category.get("label"),
                    "player_id": int(player_id),
                    "metric_value": entry.get("metric_value"),
                    "metric_display": entry.get("metric_display"),
                    "rank": int(rank),
                    "min_games": int(min_games),
                }
            )
    return awards
