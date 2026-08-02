from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, label: str, flags: int = 0) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected one regex match, found {count}")
    return updated


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Install public API routes.
# ---------------------------------------------------------------------------
main_path = ROOT / "services" / "api" / "main.py"
main = main_path.read_text(encoding="utf-8")
main = replace_once(
    main,
    "from services.api.public_match_explorer_routes import install_public_match_explorer_routes\n",
    "from services.api.public_match_explorer_routes import install_public_match_explorer_routes\n"
    "from services.api.public_play_generator_routes import install_public_play_generator_routes\n",
    "main public generator import",
)
install_marker = "install_public_tournament_team_routes(app, get_club=get_club, get_supabase_client=get_supabase_client, public_club_payload=_public_club_payload)\n"
install_block = install_marker + "install_public_play_generator_routes(\n    app,\n    get_club=get_club,\n    get_supabase_client=get_supabase_client,\n    public_club_payload=_public_club_payload,\n    require_public_writes=_require_public_live_writes,\n    require_service_role=_require_live_sessions_service_role,\n    requester_hash=_public_live_requester_hash,\n    raise_public_error=_raise_public_live_write_error,\n    public_writes_enabled=is_public_live_write_enabled,\n    service_role_configured=_has_supabase_service_role_key,\n)\n"
main = replace_once(main, install_marker, install_block, "main public generator install")
main_path.write_text(main, encoding="utf-8")

# Classify every public generator mutation in the permanent public-live wave.
wave_path = ROOT / "scripts" / "staging_write_waves.py"
wave = wave_path.read_text(encoding="utf-8")
old_public_routes = '''    "public-live": (
        ("POST", "/clubs/{club_slug}/live-sessions"),
        ("PATCH", "/clubs/{club_slug}/live-sessions/{session_key}/scores"),
        ("POST", "/clubs/{club_slug}/live-sessions/{session_key}/advance"),
        ("POST", "/clubs/{club_slug}/live-sessions/{session_key}/substitutions"),
        ("POST", "/clubs/{club_slug}/live-sessions/{session_key}/complete"),
    ),
'''
new_public_routes = '''    "public-live": (
        ("POST", "/clubs/{club_slug}/live-sessions"),
        ("PATCH", "/clubs/{club_slug}/live-sessions/{session_key}/scores"),
        ("POST", "/clubs/{club_slug}/live-sessions/{session_key}/advance"),
        ("POST", "/clubs/{club_slug}/live-sessions/{session_key}/substitutions"),
        ("POST", "/clubs/{club_slug}/live-sessions/{session_key}/complete"),
        ("POST", "/clubs/{club_slug}/play-generators/preview"),
        ("POST", "/clubs/{club_slug}/play-generators/sessions"),
        ("PATCH", "/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/scores"),
        ("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/skip"),
        ("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/advance"),
        ("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/roster"),
        ("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/complete"),
    ),
'''
wave = replace_once(wave, old_public_routes, new_public_routes, "public generator route classification")
wave_path.write_text(wave, encoding="utf-8")

# ---------------------------------------------------------------------------
# Public navigation and hub.
# ---------------------------------------------------------------------------
header_path = WEB / "components" / "PublicSiteHeader.tsx"
header = header_path.read_text(encoding="utf-8")
header = replace_once(
    header,
    '''  {
    label: "Live",
    href: `${clubBase}/live`,
    active: (pathname) => pathname.startsWith(`${clubBase}/live`)
  },
''',
    '''  {
    label: "Play",
    href: `${clubBase}/play`,
    active: (pathname) =>
      pathname.startsWith(`${clubBase}/play`) ||
      pathname.startsWith(`${clubBase}/round-robin-generator`) ||
      pathname.startsWith(`${clubBase}/ladder-generator`) ||
      pathname.startsWith(`${clubBase}/live`)
  },
''',
    "public header Play navigation",
)
header_path.write_text(header, encoding="utf-8")

write(
    WEB / "app" / "clubs" / "[clubSlug]" / "play" / "page.tsx",
    '''import Link from "next/link";

type Props = { params: { clubSlug: string } };

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white",
  color: "#0f172a",
  textDecoration: "none"
};

export default function PublicPlayHub({ params }: Props) {
  const base = `/clubs/${params.clubSlug}`;
  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Play tools
      </p>
      <h1 style={{ marginTop: 0 }}>Create and run play</h1>
      <p style={{ color: "#334155", maxWidth: "860px" }}>
        Build a printable schedule, share a view-only link, and run one round at a time. Public sessions are unrated; official match publishing remains a staff action.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
        <Link href={`${base}/round-robin-generator`} style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Round-Robin Generator</h2>
          <p style={{ color: "#475569" }}>
            Singles or doubles. Preview every planned round and bye, export the schedule, then score one round at a time.
          </p>
          <strong>Open Round-Robin Generator →</strong>
        </Link>
        <Link href={`${base}/ladder-generator`} style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Ladder Generator</h2>
          <p style={{ color: "#475569" }}>
            Preview Round 1, record results, and generate later court assignments adaptively from play.
          </p>
          <strong>Open Ladder Generator →</strong>
        </Link>
      </div>
    </section>
  );
}
''',
)

# Retain old shared-session URLs while retiring the old landing page.
write(
    WEB / "app" / "clubs" / "[clubSlug]" / "live" / "page.tsx",
    '''import { redirect } from "next/navigation";

type Props = { params: { clubSlug: string } };

export default function RetiredPublicLiveLanding({ params }: Props) {
  redirect(`/clubs/${params.clubSlug}/play`);
}
''',
)

# ---------------------------------------------------------------------------
# Public workspace, copied from the verified admin workspace and adapted to
# unauthenticated public operations protected by an organizer edit token.
# ---------------------------------------------------------------------------
admin_workspace_path = WEB / "app" / "admin" / "play-generators" / "GeneratorWorkspace.tsx"
workspace = admin_workspace_path.read_text(encoding="utf-8")
workspace = workspace.replace('import { useAdminSession } from "@/lib/useAdminSession";\n', "")
workspace = workspace.replace("  const { accessToken } = useAdminSession();\n", "")
workspace = replace_once(
    workspace,
    '''type StartResponse = {
  ok: boolean;
  session?: GeneratorSession;
};
''',
    '''type StartResponse = {
  ok: boolean;
  edit_token?: string;
  session?: GeneratorSession;
};
''',
    "public workspace start response",
)
workspace = regex_once(
    workspace,
    r'''  async function requestJson<T>\(path: string, options\?: RequestInit\): Promise<T> \{\n    if \(!apiBase\) throw new Error\("Missing API base URL\."\);\n    if \(!accessToken\) throw new Error\("Sign in before using the generator\."\);\n    const headers = new Headers\(options\?\.headers\);\n    headers\.set\("Authorization", `Bearer \$\{accessToken\}`\);\n    if \(options\?\.body\) headers\.set\("Content-Type", "application/json"\);''',
    '''  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing API base URL.");
    const headers = new Headers(options?.headers);
    if (options?.body) headers.set("Content-Type", "application/json");''',
    "public workspace request helper",
)
workspace = workspace.replace("    if (!accessToken || !status?.enabled) return;", "    if (!status?.enabled) return;")
workspace = workspace.replace("  }, [accessToken, generatorKind]);", "  }, [generatorKind, status?.enabled]);")
workspace = workspace.replace("/admin/clubs/${encodeURIComponent(clubId)}/play-generators", "/clubs/${encodeURIComponent(clubId)}/play-generators")
workspace = workspace.replace("      player_ids: hasAnyId ? rawIds.map(Number) : [],", "      participant_player_ids: hasAnyId ? Object.fromEntries(cleanRows.map((row, index) => [row.name.trim(), Number(rawIds[index])])) : {},")
workspace = workspace.replace("        expected_version: \"new\",\n", "")
workspace = regex_once(
    workspace,
    r'''      const session = payload\.session;\n      if \(!session\?\.session_key\) throw new Error\("The session was created without a session key\."\);\n      const path = `/admin/\$\{generatorSlug\(generatorKind\)\}/sessions/\$\{encodeURIComponent\(\n        session\.session_key\n      \)\}/rounds/\$\{session\.current_round_number \|\| 1\}`;\n      router\.push\(path\);''',
    '''      const session = payload.session;
      const editToken = String(payload.edit_token || "");
      if (!session?.session_key || !editToken) throw new Error("The session was created without a private organizer link.");
      sessionStorage.setItem(`public-generator-edit:${clubId}:${session.session_key}`, editToken);
      const path = `/clubs/${clubId}/${generatorSlug(generatorKind)}/sessions/${encodeURIComponent(
        session.session_key
      )}/rounds/${session.current_round_number || 1}`;
      router.push(`${path}#edit=${encodeURIComponent(editToken)}`);''',
    "public workspace start navigation",
    flags=re.MULTILINE,
)
workspace = workspace.replace("/admin/${generatorSlug(generatorKind)}/sessions/", "/clubs/${clubId}/${generatorSlug(generatorKind)}/sessions/")
workspace = workspace.replace("Sign in before using the generator.", "Public generator access is unavailable.")
workspace = workspace.replace("Official ID optional", "Club player ID optional")
workspace = workspace.replace("Official player ID", "Club player ID")
workspace = workspace.replace(
    "Either link every entered player to an official player ID or leave all IDs blank.",
    "Either link every entered player to a club player ID or leave all links blank.",
)
workspace = workspace.replace(
    "Create a durable session from this preview.",
    "Create a shareable unrated session from this preview. Keep the organizer link private.",
)
write(WEB / "app" / "clubs" / "[clubSlug]" / "play-generators" / "PublicGeneratorWorkspace.tsx", workspace)

# ---------------------------------------------------------------------------
# Public round runner, copied from the verified admin runner.
# ---------------------------------------------------------------------------
admin_runner_path = WEB / "app" / "admin" / "play-generators" / "GeneratorRoundRunner.tsx"
runner = admin_runner_path.read_text(encoding="utf-8")
runner = runner.replace('import { useAdminSession } from "@/lib/useAdminSession";\n', "")
runner = runner.replace("  version: string;", "  version: string | number;")
runner = replace_once(
    runner,
    '''function roundPath(kind: GeneratorKind, sessionKey: string, round: number): string {
  return `/admin/${generatorSlug(kind)}/sessions/${encodeURIComponent(sessionKey)}/rounds/${round}`;
}
''',
    '''function roundPath(kind: GeneratorKind, clubSlug: string, sessionKey: string, round: number): string {
  return `/clubs/${clubSlug}/${generatorSlug(kind)}/sessions/${encodeURIComponent(sessionKey)}/rounds/${round}`;
}
''',
    "public runner path helper",
)
runner = runner.replace("  const { accessToken } = useAdminSession();\n", "  const [editToken, setEditToken] = useState(\"\");\n")
runner = regex_once(
    runner,
    r'''  async function requestJson<T>\(path: string, options\?: RequestInit\): Promise<T> \{\n    if \(!apiBase\) throw new Error\("Missing API base URL\."\);\n    if \(!accessToken\) throw new Error\("Sign in before managing this session\."\);\n    const headers = new Headers\(options\?\.headers\);\n    headers\.set\("Authorization", `Bearer \$\{accessToken\}`\);\n    if \(options\?\.body\) headers\.set\("Content-Type", "application/json"\);''',
    '''  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing API base URL.");
    const headers = new Headers(options?.headers);
    if (options?.body) headers.set("Content-Type", "application/json");''',
    "public runner request helper",
)
runner = runner.replace("    if (!accessToken) return;\n", "")
runner = runner.replace("/admin/clubs/${encodeURIComponent(clubId)}/play-generators", "/clubs/${encodeURIComponent(clubId)}/play-generators")
runner = runner.replace("roundPath(generatorKind, sessionKey,", "roundPath(generatorKind, clubId, sessionKey,")
runner = replace_once(
    runner,
    '''  useEffect(() => {
    void loadSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, sessionKey, roundNumber]);
''',
    '''  useEffect(() => {
    const storageKey = `public-generator-edit:${clubId}:${sessionKey}`;
    const hash = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    const discovered = hash.get("edit") || sessionStorage.getItem(storageKey) || "";
    if (discovered) {
      sessionStorage.setItem(storageKey, discovered);
      setEditToken(discovered);
    }
    if (hash.has("edit")) {
      window.history.replaceState({}, "", `${window.location.pathname}${window.location.search}`);
    }
    void loadSession();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clubId, sessionKey, roundNumber]);
''',
    "public runner edit-token discovery",
)
runner = replace_once(
    runner,
    '''  const canEditRound =
    Boolean(session) &&
    session?.status === "active" &&
''',
    '''  const canEditRound =
    Boolean(session) &&
    Boolean(editToken) &&
    session?.status === "active" &&
''',
    "public runner score edit gate",
)
runner = replace_once(
    runner,
    '''      body: JSON.stringify({
        ...body,
        expected_version: session?.version || "",
''',
    '''      body: JSON.stringify({
        ...body,
        edit_token: editToken,
        expected_version: Number(session?.version || 1),
''',
    "public runner generic mutation token",
)
runner = runner.replace(
    '''          body: JSON.stringify({
            scores: scoreRows,
            expected_version: session.version,
''',
    '''          body: JSON.stringify({
            scores: scoreRows,
            edit_token: editToken,
            expected_version: Number(session.version),
''',
)
runner = runner.replace(
    '''          body: JSON.stringify({
            reason: skipReason,
            expected_version: session.version,
''',
    '''          body: JSON.stringify({
            reason: skipReason,
            edit_token: editToken,
            expected_version: Number(session.version),
''',
)
runner = runner.replace("      {isCurrent && session.status === \"active\" ? (", "      {isCurrent && session.status === \"active\" && Boolean(editToken) ? (")
runner = runner.replace("<Link href={`/admin/${generatorSlug(generatorKind)}`}", "<Link href={`/clubs/${clubId}/${generatorSlug(generatorKind)}`}")
runner = runner.replace("<Link href={`/admin/${generatorSlug(generatorKind)}`}>", "<Link href={`/clubs/${clubId}/${generatorSlug(generatorKind)}`}>")
runner = regex_once(
    runner,
    r'''      <article style=\{cardStyle\}>\n        <h2 style=\{\{ marginTop: 0 \}\}>Official results</h2>.*?      </article>\n''',
    '''      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Ratings and official results</h2>
        <p style={{ color: "#475569" }}>
          Public generator sessions are unrated. Share this page for viewing and score entry; staff can publish controlled official matches from the administrative generators.
        </p>
        <a href={`/api/clubs/${clubId}/play-generators/sessions/${sessionKey}/export?format=csv`}>
          Download session CSV
        </a>
      </article>
''',
    "public runner official section",
    flags=re.DOTALL,
)
runner = runner.replace(
    "        <p style={{ margin: 0, color: \"#475569\" }}>\n          {session.play_format === \"singles\" ? \"Singles\" : \"Doubles\"} · Round {roundNumber} of{\" \"}\n          {event.totalRounds} · {round.status}\n        </p>",
    "        <p style={{ margin: 0, color: \"#475569\" }}>\n          {session.play_format === \"singles\" ? \"Singles\" : \"Doubles\"} · Round {roundNumber} of{\" \"}\n          {event.totalRounds} · {round.status} · Unrated public session\n        </p>\n        {!editToken ? <p style={{ color: \"#64748b\" }}>View-only link. The organizer's private link enables scores and roster changes.</p> : null}",
)
write(WEB / "app" / "clubs" / "[clubSlug]" / "play-generators" / "PublicGeneratorRoundRunner.tsx", runner)

# ---------------------------------------------------------------------------
# Public generator pages and one-round routes.
# ---------------------------------------------------------------------------
def generator_page(kind: str, title: str, description: str) -> str:
    return f'''import PublicGeneratorWorkspace from "@/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace";

type Props = {{ params: {{ clubSlug: string }} }};
type StatusResponse = {{ enabled: boolean; writes_enabled?: boolean; status: string; warnings?: string[] }};

function serverApiBase(): string | null {{
  return process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;
}}

async function loadStatus(clubSlug: string): Promise<StatusResponse | null> {{
  const base = serverApiBase();
  if (!base) return null;
  try {{
    const response = await fetch(`${{base.replace(/\\/$/, "")}}/clubs/${{encodeURIComponent(clubSlug)}}/play-generators/status`, {{ cache: "no-store" }});
    return response.ok ? ((await response.json()) as StatusResponse) : null;
  }} catch {{
    return null;
  }}
}}

export default async function PublicGeneratorPage({{ params }}: Props) {{
  const status = await loadStatus(params.clubSlug);
  return (
    <section>
      <p style={{{{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}}}>
        Public play tools
      </p>
      <h1 style={{{{ marginTop: 0 }}}}>{title}</h1>
      <p style={{{{ color: "#334155", maxWidth: "900px" }}}}>{description}</p>
      <p style={{{{ color: "#475569", maxWidth: "900px" }}}}>
        Public sessions are unrated. Keep the organizer link private; share the clean page URL as the view-only scoreboard.
      </p>
      <PublicGeneratorWorkspace generatorKind="{kind}" apiBase="/api" clubId={{params.clubSlug}} status={{status}} />
    </section>
  );
}}
'''

write(
    WEB / "app" / "clubs" / "[clubSlug]" / "round-robin-generator" / "page.tsx",
    generator_page(
        "round_robin",
        "Round-Robin Generator",
        "Build singles or doubles schedules, preview every planned matchup and bye, download a paper schedule, and run one round at a time.",
    ),
)
write(
    WEB / "app" / "clubs" / "[clubSlug]" / "ladder-generator" / "page.tsx",
    generator_page(
        "ladder",
        "Ladder Generator",
        "Preview Round 1, record results, and generate each later court assignment adaptively from the completed or skipped round.",
    ),
)


def round_page(kind: str, component_name: str) -> str:
    return f'''import PublicGeneratorRoundRunner from "@/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner";

type Props = {{
  params: {{ clubSlug: string; sessionKey: string; roundNumber: string }};
}};

export default function {component_name}({{ params }}: Props) {{
  return (
    <PublicGeneratorRoundRunner
      apiBase="/api"
      clubId={{params.clubSlug}}
      generatorKind="{kind}"
      sessionKey={{params.sessionKey}}
      roundNumber={{Math.max(1, Number(params.roundNumber) || 1)}}
    />
  );
}}
'''

write(
    WEB / "app" / "clubs" / "[clubSlug]" / "round-robin-generator" / "sessions" / "[sessionKey]" / "rounds" / "[roundNumber]" / "page.tsx",
    round_page("round_robin", "PublicRoundRobinRoundPage"),
)
write(
    WEB / "app" / "clubs" / "[clubSlug]" / "ladder-generator" / "sessions" / "[sessionKey]" / "rounds" / "[roundNumber]" / "page.tsx",
    round_page("ladder", "PublicLadderRoundPage"),
)

# ---------------------------------------------------------------------------
# Focused contracts.
# ---------------------------------------------------------------------------
write(
    ROOT / "tests" / "test_api_contract_public_play_generators.py",
    '''from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_public_play_generator_routes_and_safety_are_installed() -> None:
    main = read("services/api/main.py")
    routes = read("services/api/public_play_generator_routes.py")
    waves = read("scripts/staging_write_waves.py")
    assert "install_public_play_generator_routes" in main
    for route in (
        "/play-generators/preview",
        "/play-generators/sessions",
        "/rounds/{round_number}/scores",
        "/rounds/{round_number}/skip",
        "/sessions/{session_key}/advance",
        "/sessions/{session_key}/roster",
        "/sessions/{session_key}/complete",
    ):
        assert route in routes
    for route in (
        '("POST", "/clubs/{club_slug}/play-generators/preview")',
        '("POST", "/clubs/{club_slug}/play-generators/sessions")',
        '("PATCH", "/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/scores")',
        '("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/skip")',
        '("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/advance")',
        '("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/roster")',
        '("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/complete")',
    ):
        assert route in waves


def test_public_modules_mirror_generator_functionality_without_official_publish() -> None:
    workspace = read("apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx")
    runner = read("apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx")
    header = read("apps/web/components/PublicSiteHeader.tsx")
    live = read("apps/web/app/clubs/[clubSlug]/live/page.tsx")
    assert 'label: "Play"' in header
    assert "Round-Robin Generator" in read("apps/web/app/clubs/[clubSlug]/play/page.tsx")
    assert "Ladder Generator" in read("apps/web/app/clubs/[clubSlug]/play/page.tsx")
    assert "Download CSV" in workspace
    assert "Download one-sheet PDF" in workspace
    assert "Preview matchups" in workspace
    assert "Start session" in workspace
    assert "Save round scores" in runner
    assert "Skip round" in runner
    assert "Adaptive roster" in runner
    assert "Substitute player" in runner
    assert "View-only link" in runner
    assert "Public generator sessions are unrated" in runner
    assert "Publish official matches" not in runner
    assert "redirect(`/clubs/${params.clubSlug}/play`)" in live


def test_public_service_uses_edit_tokens_and_durable_operation_ledger() -> None:
    service = read("jupr_app/services/public_play_generator_service.py")
    assert "begin_public_live_operation" in service
    assert "edit_token_matches" in service
    assert "hash_edit_token" in service
    assert "version" in service
    assert "create_generator_preview" in service
    assert "save_generator_round" in service
    assert "skip_generator_round" in service
    assert "mutate_generator_roster" in service
    assert "advance_generator_event" in service
    assert "public_play_generator" in service
''',
)

write(
    ROOT / "tests" / "test_public_play_generator_service.py",
    '''from types import SimpleNamespace

import pytest

from jupr_app.services.public_play_generator_service import (
    advance_public_play_generator_session,
    create_public_play_generator_session,
    get_public_play_generator_session,
    mutate_public_play_generator_roster,
    preview_public_play_generator,
    save_public_play_generator_round,
    skip_public_play_generator_round,
)


class Query:
    def __init__(self, db, name):
        self.db = db
        self.name = name
        self.filters = []
        self.limit_n = None
        self.order_key = None
        self.desc = False
        self.payload = None
        self.update_payload = None

    def select(self, *_args, **_kwargs): return self
    def eq(self, key, value): self.filters.append((key, value)); return self
    def in_(self, key, values): self.filters.append((key, set(values))); return self
    def gte(self, *_args): return self
    def limit(self, value): self.limit_n = int(value); return self
    def order(self, key, desc=False): self.order_key = key; self.desc = bool(desc); return self
    def insert(self, payload): self.payload = payload; return self
    def update(self, payload): self.update_payload = dict(payload); return self

    def matches(self, row):
        for key, value in self.filters:
            if isinstance(value, set):
                if row.get(key) not in value: return False
            elif str(row.get(key)) != str(value): return False
        return True

    def execute(self):
        rows = self.db.setdefault(self.name, [])
        if self.payload is not None:
            values = self.payload if isinstance(self.payload, list) else [self.payload]
            rows.extend(dict(row) for row in values)
            return SimpleNamespace(data=values)
        selected = [row for row in rows if self.matches(row)]
        if self.update_payload is not None:
            for row in selected: row.update(self.update_payload)
            return SimpleNamespace(data=selected)
        if self.order_key:
            selected = sorted(selected, key=lambda row: str(row.get(self.order_key) or ""), reverse=self.desc)
        if self.limit_n is not None: selected = selected[: self.limit_n]
        return SimpleNamespace(data=selected)


class FakeSupabase:
    def __init__(self):
        self.db = {"live_sessions": [], "public_live_operations": [], "players": []}
    def table(self, name): return Query(self.db, name)


def requester(): return "a" * 64

def token_secret(): return "x" * 48

def key(label): return f"public-generator-{label}-00000001"

def matches(round_row):
    return list(round_row.get("matches") or []) or [match for court in round_row.get("courts") or [] for match in court.get("matches") or []]


def test_public_round_robin_preview_create_score_skip_and_roster():
    supabase = FakeSupabase()
    preview = preview_public_play_generator(
        supabase,
        club_id="club",
        generator_kind="round_robin",
        play_format="singles",
        title="Public Singles",
        participant_names=["A", "B", "C", "D", "E"],
        participant_player_ids={},
        total_rounds=4,
        court_count=2,
    )["preview"]
    assert len(preview["rounds"]) == 4
    assert any(row["byeParticipantIds"] for row in preview["rounds"])

    created = create_public_play_generator_session(
        supabase,
        club_id="club",
        generator_kind="round_robin",
        play_format="singles",
        title="Public Singles",
        participant_names=["A", "B", "C", "D", "E"],
        participant_player_ids={},
        total_rounds=4,
        court_count=2,
        preview_fingerprint=preview["previewFingerprint"],
        idempotency_key=key("create"),
        requester_hash=requester(),
        token_secret=token_secret(),
    )
    session = created["session"]
    edit = created["edit_token"]
    first = session["event"]["rounds"][0]
    scored = save_public_play_generator_round(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        round_number=1,
        scores=[{"match_id": row["id"], "score_a": 11, "score_b": 7} for row in matches(first)],
        edit_token=edit,
        expected_version=session["version"],
        idempotency_key=key("scores"),
        requester_hash=requester(),
    )["session"]
    advanced = advance_public_play_generator_session(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        edit_token=edit,
        expected_version=scored["version"],
        idempotency_key=key("advance"),
        requester_hash=requester(),
    )["session"]
    roster = mutate_public_play_generator_roster(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        action="add",
        participant_id=None,
        name="F",
        player_id=None,
        substitute_scope="rest",
        roster_order=[],
        edit_token=edit,
        expected_version=advanced["version"],
        idempotency_key=key("roster"),
        requester_hash=requester(),
    )["session"]
    skipped = skip_public_play_generator_round(
        supabase,
        club_id="club",
        session_key=session["session_key"],
        round_number=2,
        reason="Weather",
        edit_token=edit,
        expected_version=roster["version"],
        idempotency_key=key("skip"),
        requester_hash=requester(),
    )["session"]
    assert skipped["event"]["rounds"][0]["status"] == "saved"
    assert skipped["event"]["rounds"][1]["status"] == "skipped"
    assert any(row["name"] == "F" for row in skipped["event"]["participants"])
    assert get_public_play_generator_session(supabase, club_id="club", session_key=session["session_key"])["session"]["unrated"] is True


def test_public_ladder_previews_only_round_one_and_requires_results_to_advance():
    supabase = FakeSupabase()
    preview = preview_public_play_generator(
        supabase,
        club_id="club",
        generator_kind="ladder",
        play_format="doubles",
        title="Public Ladder",
        participant_names=[f"P{idx}" for idx in range(1, 10)],
        participant_player_ids={},
        total_rounds=3,
        court_count=2,
    )["preview"]
    assert len(preview["rounds"]) == 1
    created = create_public_play_generator_session(
        supabase,
        club_id="club",
        generator_kind="ladder",
        play_format="doubles",
        title="Public Ladder",
        participant_names=[f"P{idx}" for idx in range(1, 10)],
        participant_player_ids={},
        total_rounds=3,
        court_count=2,
        preview_fingerprint=preview["previewFingerprint"],
        idempotency_key=key("ladder-create"),
        requester_hash=requester(),
        token_secret=token_secret(),
    )
    with pytest.raises(Exception):
        advance_public_play_generator_session(
            supabase,
            club_id="club",
            session_key=created["session"]["session_key"],
            edit_token=created["edit_token"],
            expected_version=created["session"]["version"],
            idempotency_key=key("ladder-early"),
            requester_hash=requester(),
        )
''',
)

# Source formatting and generated-file sanity.
for path in (
    ROOT / "jupr_app" / "services" / "public_play_generator_service.py",
    ROOT / "services" / "api" / "public_play_generator_routes.py",
    WEB / "app" / "clubs" / "[clubSlug]" / "play-generators" / "PublicGeneratorWorkspace.tsx",
    WEB / "app" / "clubs" / "[clubSlug]" / "play-generators" / "PublicGeneratorRoundRunner.tsx",
):
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        raise RuntimeError(f"missing generated file: {path}")
