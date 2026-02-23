from __future__ import annotations

from dataclasses import dataclass

from jupr_app.ui.components.skeleton import (
    render_header_skeleton,
    render_table_row_skeleton,
)


@dataclass(frozen=True)
class LeaderboardPlayer:
    name: str
    rating: int
    movement: int
    division: str | None = None


def _placeholder_leaderboard_players() -> list[LeaderboardPlayer]:
    """Temporary placeholder leaderboard snapshot until live query integration."""
    return [
        LeaderboardPlayer(name="M. Santos", rating=1985, movement=2, division="D1"),
        LeaderboardPlayer(name="R. Patel", rating=1962, movement=-1, division="D1"),
        LeaderboardPlayer(name="K. Nguyen", rating=1948, movement=1, division="D2"),
        LeaderboardPlayer(name="A. Gomez", rating=1931, movement=0, division="D2"),
        LeaderboardPlayer(name="L. Kim", rating=1914, movement=-2),
    ]


def _movement_badge(player: LeaderboardPlayer) -> str:
    if player.movement > 0:
        return (
            '<span class="cc-lb-move cc-lb-move-up" '
            f'aria-label="Moved up {player.movement} positions">↑ {player.movement}</span>'
        )
    if player.movement < 0:
        movement = abs(player.movement)
        return (
            '<span class="cc-lb-move cc-lb-move-down" '
            f'aria-label="Moved down {movement} positions">↓ {movement}</span>'
        )
    return (
        '<span class="cc-lb-move cc-lb-move-flat" aria-label="No movement">→ 0</span>'
    )


def render_leaderboards_snapshot_html(is_loading: bool = False) -> str:
    if is_loading:
        return (
            """
            <section class="cc-leaderboards" aria-label="Leaderboards snapshot">
              <div class="cc-leaderboards-header">
            """
            + render_header_skeleton()
            + """
              </div>
              <div class="cc-lb-grid">
            """
            + "".join(render_table_row_skeleton() for _ in range(5))
            + """
              </div>
            </section>
            """
        )

    rows: list[str] = []
    for index, player in enumerate(_placeholder_leaderboard_players(), start=1):
        division = (
            f'<span class="cc-lb-division">{player.division}</span>'
            if player.division
            else ""
        )
        rows.append(
            f"""
            <article class="cc-lb-row" aria-label="Leaderboard player {player.name}">
              <div class="cc-lb-player-wrap">
                <span class="cc-lb-rank">#{index}</span>
                <div>
                  <p class="cc-lb-player">{player.name}</p>
                  <p class="cc-lb-meta">Rating {player.rating}</p>
                </div>
              </div>
              <div class="cc-lb-right">
                {division}
                {_movement_badge(player)}
              </div>
            </article>
            """.strip()
        )

    return (
        """
        <section class="cc-leaderboards" aria-label="Leaderboards snapshot">
          <div class="cc-leaderboards-header">
            <h3>Leaderboards Snapshot</h3>
            <p>Top 5 players across all competitions (placeholder source data).</p>
          </div>
          <div class="cc-lb-grid">
        """
        + "".join(rows)
        + """
          </div>
        </section>
        """
    )
