from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicNavCard:
    title: str
    description: str
    href: str


def _public_nav_cards() -> list[PublicNavCard]:
    return [
        PublicNavCard(
            title="Leaderboards",
            description="View current rankings and key performance trends.",
            href="/?page=leaderboards&public=1",
        ),
        PublicNavCard(
            title="Weekly Recap",
            description="Catch up on the latest highlights and club storylines.",
            href="/?page=weekly_recap&public=1",
        ),
        PublicNavCard(
            title="Player Profiles",
            description="Look up player bios, ratings history, and match activity.",
            href="/?page=players&public=1",
        ),
        PublicNavCard(
            title="Standings",
            description="Browse full standings snapshots for active programs.",
            href="/?page=leaderboards&public=1",
        ),
    ]


def render_public_navigation_html() -> str:
    cards = "".join(
        (
            f"""
            <article class=\"cc-public-card\" aria-label=\"Public link {card.title}\">
              <h4>{card.title}</h4>
              <p>{card.description}</p>
              <a class=\"cc-public-link\" href=\"{card.href}\" target=\"_self\">Open view →</a>
            </article>
            """.strip()
        )
        for card in _public_nav_cards()
    )

    return (
        """
        <section class="cc-public-nav" aria-label="Public navigation">
          <div class="cc-public-header">
            <h3>Public Navigation</h3>
            <p>Quick links to fan-facing views outside admin workflows.</p>
          </div>
          <div class="cc-public-grid">
        """
        + cards
        + """
          </div>
        </section>
        """
    )
