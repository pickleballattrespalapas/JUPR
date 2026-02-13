from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AlertItem:
    title: str
    count: int
    subtitle: str
    href: str
    cta_label: str
    state: str


def _placeholder_alert_items() -> list[AlertItem]:
    """Temporary placeholder data until live admin metrics are wired."""
    return [
        AlertItem(
            title="Pending Ladder Challenges",
            count=4,
            subtitle="Challenges awaiting admin resolution.",
            href="/?page=challenge_ladder_admin",
            cta_label="Open ladder admin",
            state="warning",
        ),
        AlertItem(
            title="Incomplete Competition Rounds",
            count=3,
            subtitle="Active competition rounds still missing final entries.",
            href="/?page=tournament_manager",
            cta_label="Review tournaments",
            state="warning",
        ),
        AlertItem(
            title="Moneyball Events Missing Scores",
            count=2,
            subtitle="Moneyball sessions without complete scores.",
            href="/?page=moneyball",
            cta_label="Update moneyball",
            state="danger",
        ),
        AlertItem(
            title="Weekly Recap Not Published",
            count=1,
            subtitle="Latest recap is drafted but not yet published.",
            href="/?page=weekly_recap_admin",
            cta_label="Publish recap",
            state="info",
        ),
    ]


def render_alerts_html() -> str:
    cards: list[str] = []
    for item in _placeholder_alert_items():
        cards.append(
            f"""
            <article class=\"cc-alert-card cc-alert-{item.state}\">
              <div class=\"cc-alert-top\">
                <p class=\"cc-alert-title\">{item.title}</p>
                <p class=\"cc-alert-count\">{item.count}</p>
              </div>
              <p class=\"cc-alert-subtitle\">{item.subtitle}</p>
              <a class=\"cc-alert-link\" href=\"{item.href}\" target=\"_self\">{item.cta_label} →</a>
            </article>
            """.strip()
        )

    return (
        """
        <section class="cc-alerts" aria-label="Admin alerts">
          <div class="cc-alerts-header">
            <h3>Alerts</h3>
            <p>Operational items requiring follow-up.</p>
          </div>
          <div class="cc-alert-grid">
        """
        + "".join(cards)
        + """
          </div>
        </section>
        """
    )
