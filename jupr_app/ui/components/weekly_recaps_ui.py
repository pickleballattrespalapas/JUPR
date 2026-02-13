from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecapAction:
    label: str
    href: str
    style: str = "secondary"


def render_weekly_recap_builder_html() -> str:
    """Render command-center weekly recap controls (UI only, no action handling)."""
    actions = [
        RecapAction("Build Recap", "/?page=weekly_recap_admin&recap_action=build", style="primary"),
        RecapAction("Preview Recap", "/?page=weekly_recap_admin&recap_action=preview"),
        RecapAction("Publish Recap", "/?page=weekly_recap_admin&recap_action=publish"),
    ]

    action_links = "".join(
        (
            f'<a class="cc-recap-btn cc-recap-btn-{action.style}" '
            f'href="{action.href}" target="_self">{action.label}</a>'
        )
        for action in actions
    )

    return (
        """
        <section class="cc-recap" aria-label="Weekly recap builder">
          <div class="cc-recap-header">
            <h3>Weekly Recap Builder</h3>
            <p>Prepare, review, and publish the weekly summary.</p>
          </div>
          <div class="cc-recap-status-wrap">
            <p class="cc-recap-status-label">Current recap status</p>
            <p class="cc-recap-status-value">Draft not started</p>
          </div>
          <div class="cc-recap-actions">
        """
        + action_links
        + """
          </div>
        </section>
        """
    )
