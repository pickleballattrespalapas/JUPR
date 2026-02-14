from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitionAction:
    label: str
    href: str
    style: str = "secondary"


@dataclass(frozen=True)
class CompetitionCard:
    name: str
    status: str
    actions: tuple[CompetitionAction, ...]


def _placeholder_competitions() -> list[CompetitionCard]:
    """Temporary placeholder competition metadata until live data is wired."""
    return [
        CompetitionCard(
            name="Ladder League",
            status="Active",
            actions=(
                CompetitionAction("Enter Result", "/?page=league_results", style="primary"),
                CompetitionAction("View Standings", "/?page=leaderboards"),
            ),
        ),
        CompetitionCard(
            name="Challenge Ladder",
            status="Active",
            actions=(
                CompetitionAction("Enter Result", "/?page=challenge_ladder_admin", style="primary"),
                CompetitionAction("View Standings", "/?page=challenge_ladder"),
            ),
        ),
        CompetitionCard(
            name="Tournament",
            status="Upcoming",
            actions=(
                CompetitionAction("Enter Result", "/?page=tournament_manager", style="primary"),
                CompetitionAction("View Standings", "/?page=tournaments"),
            ),
        ),
        CompetitionCard(
            name="Round Robin",
            status="Active",
            actions=(
                CompetitionAction("Enter Result", "/?page=tournaments", style="primary"),
                CompetitionAction("View Standings", "/?page=leaderboards"),
            ),
        ),
        CompetitionCard(
            name="Moneyball",
            status="Closed",
            actions=(
                CompetitionAction("Enter Result", "/?page=moneyball", style="primary"),
                CompetitionAction("View Standings", "/?page=moneyball"),
            ),
        ),
    ]


def render_active_competitions_html() -> str:
    cards: list[str] = []
    for card in _placeholder_competitions():
        actions = "".join(
            (
                f'<a class="cc-competition-btn cc-competition-btn-{action.style}" '
                f'href="{action.href}" target="_self">{action.label}</a>'
            )
            for action in card.actions
        )

        cards.append(
            f"""
            <article class="cc-competition-card" aria-label="{card.name}">
              <div class="cc-competition-top">
                <h4>{card.name}</h4>
                <span class="cc-competition-status cc-competition-status-{card.status.lower()}">{card.status}</span>
              </div>
              <div class="cc-competition-actions">
                {actions}
              </div>
            </article>
            """.strip()
        )

    return (
        """
        <section class="cc-competitions" aria-label="Active competitions">
          <div class="cc-competitions-header">
            <h3>Active Competitions</h3>
            <p>Monitor live programs and jump directly to key admin actions.</p>
          </div>
          <div class="cc-competition-grid">
        """
        + "".join(cards)
        + """
          </div>
        </section>
        """
    )
