from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageDefinition:
    key: str
    label: str
    public: bool = False
    admin_only: bool = False


PAGE_DEFINITIONS: tuple[PageDefinition, ...] = (
    PageDefinition("leaderboards", "🏆 Leaderboards", public=True),
    PageDefinition("league_results", "📊 League Results", public=True),
    PageDefinition("league_printout", "🖨️ League Night Printout", admin_only=True),
    PageDefinition("match_explorer", "🎯 Match Explorer", public=True),
    PageDefinition("players", "🔍 Player Search", public=True),
    PageDefinition("badge_codex", "📼 Badge Codex", public=True),
    PageDefinition("badge_debug", "🧪 Badge Debug", admin_only=True),
    PageDefinition("challenge_ladder", "🪜 Challenge Ladder", public=True),
    PageDefinition("faqs", "❓ FAQs", public=True),
    PageDefinition("league_manager", "🏟️ League Manager", admin_only=True),
    PageDefinition("match_uploader", "📝 Match Uploader", admin_only=True),
    PageDefinition("match_log", "📝 Match Log", admin_only=True),
    PageDefinition("player_editor", "👥 Player Editor", admin_only=True),
    PageDefinition("admin_tools", "⚙️ Admin Tools", admin_only=True),
    PageDefinition("admin_guide", "📘 Admin Guide", admin_only=True),
    PageDefinition(
        "challenge_ladder_admin",
        "🛠️ Challenge Ladder Admin",
        admin_only=True,
    ),
    PageDefinition("moneyball", "💰 Moneyball", admin_only=True),
    PageDefinition("jupr_live", "🔴 JUPR Live", public=True),
    PageDefinition("jupr_live_admin", "🔴 JUPR Live Admin", admin_only=True),
    PageDefinition("theme_qa", "🎨 Theme QA", admin_only=True),
    PageDefinition("tournaments", "🏆 Tournaments", admin_only=True),
    PageDefinition("tournament_manager", "🏆 Tournament Manager", admin_only=True),
    PageDefinition("tournament_registration", "📝 Tournament Registration", public=True),
    PageDefinition("tournament_partner_board", "🤝 Partner Board", public=True),
    PageDefinition("weekly_recap", "🗞️ Weekly Recap", public=True),
    PageDefinition("top_players_printable", "🧾 Top Active Players PDF", admin_only=True),
    PageDefinition("weekly_recap_admin", "🗞️ Weekly Recap Admin", admin_only=True),
)

PAGE_KEY_TO_LABEL = {page.key: page.label for page in PAGE_DEFINITIONS}
LABEL_TO_PAGE_KEY = {page.label: page.key for page in PAGE_DEFINITIONS}
ADMIN_ONLY_PAGE_KEYS = frozenset(
    page.key for page in PAGE_DEFINITIONS if page.admin_only
)
ADMIN_ONLY_LABELS = frozenset(
    page.label for page in PAGE_DEFINITIONS if page.admin_only
)
PUBLIC_NAV_KEYS = (
    "leaderboards",
    "league_results",
    "weekly_recap",
    "tournament_registration",
    "tournament_partner_board",
    "match_explorer",
    "players",
    "badge_codex",
    "jupr_live",
    "challenge_ladder",
    "faqs",
)


def labels_for_keys(page_keys: list[str] | tuple[str, ...]) -> list[str]:
    return [PAGE_KEY_TO_LABEL[key] for key in page_keys if key in PAGE_KEY_TO_LABEL]
