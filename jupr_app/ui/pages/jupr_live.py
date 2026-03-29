from __future__ import annotations

from jupr_app.ui.layout import page_shell
from jupr_app.ui.live import LivePageConfig, render_live_page


PUBLIC_CONFIG = LivePageConfig(
    state_key="jupr_live_public_state",
    intro_markdown=(
        "Run a lightweight JUPR Live session with session-only scoring. "
        "No league context, ratings effects, or official saves are used on this page."
    ),
    event_types=("Round Robin", "League / Ladder"),
    mode_pill_label="Session",
    allow_official=False,
    allow_tournament=False,
    show_official_context=False,
)


def render(ctx):
    mode_label = (
        "Public"
        if bool(ctx.public_mode)
        else ("Admin" if bool(ctx.admin_logged_in) else "Guest")
    )
    page_shell(
        "🔴 JUPR Live",
        "Run public Round Robin or League / Ladder sessions without admin save controls.",
        mode_label=mode_label,
    )
    render_live_page(ctx, PUBLIC_CONFIG)
