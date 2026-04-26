from __future__ import annotations

from jupr_app.ui.pages import tournament_roster


def render(ctx):
    tournament_roster.render(ctx, focus_partners=True, legacy_partner_board=True)
