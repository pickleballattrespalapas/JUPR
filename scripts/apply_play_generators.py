from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label} marker count={count}")
    return text.replace(old, new, 1)


def write_bundle() -> None:
    bundle = "".join(
        (ROOT / f"scripts/.play_generator_bundle_{index:02d}.txt").read_text(
            encoding="utf-8"
        ).strip()
        for index in range(8)
    )
    payload = json.loads(
        zlib.decompress(base64.b64decode(bundle)).decode("utf-8")
    )
    for relative, content in payload.items():
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(content).rstrip() + "\n", encoding="utf-8")


def patch_main() -> None:
    path = ROOT / "services/api/main.py"
    text = path.read_text(encoding="utf-8")
    import_marker = (
        "from services.api.admin_team_league_routes import "
        "install_admin_team_league_routes\n"
    )
    import_text = (
        import_marker
        + "from services.api.admin_play_generator_routes import "
        "install_admin_play_generator_routes\n"
    )
    if (
        "from services.api.admin_play_generator_routes import "
        "install_admin_play_generator_routes"
        not in text
    ):
        text = replace_once(text, import_marker, import_text, "main import")
    call_marker = (
        "install_admin_team_league_routes(app, "
        "get_supabase_client=get_supabase_client)\n"
    )
    call_text = (
        call_marker
        + "install_admin_play_generator_routes(app, "
        "get_supabase_client=get_supabase_client)\n"
    )
    if (
        "install_admin_play_generator_routes(app, "
        "get_supabase_client=get_supabase_client)"
        not in text
    ):
        text = replace_once(text, call_marker, call_text, "main installer")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def patch_sidebar() -> None:
    path = ROOT / "apps/web/components/AdminShell.tsx"
    text = path.read_text(encoding="utf-8")
    old = '''      {
        label: "JUPR Live",
        href: "/admin/jupr-live",
        active: (pathname) => pathname.startsWith("/admin/jupr-live")
      },
'''
    new = '''      {
        label: "Round-Robin Generator",
        href: "/admin/round-robin-generator",
        active: (pathname) =>
          pathname.startsWith("/admin/round-robin-generator")
      },
      {
        label: "Ladder Generator",
        href: "/admin/ladder-generator",
        active: (pathname) => pathname.startsWith("/admin/ladder-generator")
      },
'''
    text = replace_once(text, old, new, "admin sidebar JUPR Live link")
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def patch_package() -> None:
    path = ROOT / "apps/web/package.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("dependencies", {})["jspdf"] = "4.2.1"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def patch_public_names() -> None:
    page = ROOT / "apps/web/app/clubs/[clubSlug]/live/page.tsx"
    text = page.read_text(encoding="utf-8")
    replacements = {
        "Live Events": "Play Generators",
        "{clubName} live sessions": "{clubName} play sessions",
        "Start a durable Round Robin, League / Ladder, or Club Social session; resume after refresh, substitute players, export results, and share a view-only scoreboard. Official rated workflows remain separate in JUPR Live Admin.": "Create a Round-Robin or Ladder session, resume after refresh, manage players, export results, and share a view-only scoreboard. Official rated publishing remains in the staff generators.",
        "New public sessions are paused here": "New public generators are paused here",
        "Open the Streamlit JUPR Live fallback": "Open the legacy live-session fallback",
        "Live sessions are temporarily unavailable.": "Play sessions are temporarily unavailable.",
        "No shared live sessions right now": "No shared play sessions right now",
        "Create a public live event above, or open a shared session link from an organizer.": "Create a play session above, or open a shared session link from an organizer.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    page.write_text(text.rstrip() + "\n", encoding="utf-8")

    creator = ROOT / "apps/web/app/clubs/[clubSlug]/live/PublicLiveCreator.tsx"
    text = creator.read_text(encoding="utf-8")
    replacements = {
        "🔴 JUPR Live": "Round-Robin and Ladder Generators",
        "Run a durable Round Robin or League / Ladder session. Quick sessions stay unrated; Club Social sends completed results to moderation without changing ratings.": "Create a durable Round-Robin or Ladder session. Quick sessions stay unrated; Club Social sends completed results to moderation without changing ratings.",
        "Public JUPR Live supports up to 20 players.": "Public play generators support up to 20 players.",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    creator.write_text(text.rstrip() + "\n", encoding="utf-8")

    detail = ROOT / "apps/web/app/clubs/[clubSlug]/live/[sessionKey]/page.tsx"
    text = detail.read_text(encoding="utf-8")
    text = text.replace("Live Events", "Play Generators")
    text = text.replace("Live session unavailable", "Play session unavailable")
    text = text.replace("Back to live sessions", "Back to play sessions")
    detail.write_text(text.rstrip() + "\n", encoding="utf-8")

    runner = (
        ROOT
        / "apps/web/app/clubs/[clubSlug]/live/[sessionKey]/LiveSessionRunner.tsx"
    )
    text = runner.read_text(encoding="utf-8")
    text = text.replace("JUPR Live", "play generator")
    text = text.replace("live session", "play session")
    text = text.replace("Live Event", "Play Session")
    runner.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    write_bundle()
    patch_main()
    patch_sidebar()
    patch_package()
    patch_public_names()


if __name__ == "__main__":
    main()
