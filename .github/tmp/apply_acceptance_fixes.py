from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def read(path: str) -> str:
    return (ROOT / path).read_text()


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)


def replace_once(path: str, old: str, new: str) -> None:
    text = read(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one occurrence, found {count}: {old[:100]!r}")
    write(path, text.replace(old, new, 1))


# ---------------------------------------------------------------------------
# League Manager navigation and results
# ---------------------------------------------------------------------------
write(
    "apps/web/app/admin/league-manager/LeagueManagerNav.tsx",
    '''"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  ["/admin/league-manager", "Home"],
  ["/admin/league-manager/results", "Results"],
  ["/admin/league-manager/settings", "Settings"],
  ["/admin/league-manager/roster", "Roster"],
  ["/admin/league-manager/teams", "Team leagues"],
  ["/admin/league-manager/live", "Live rounds"],
  ["/admin/league-manager/awards", "Awards"]
] as const;

export default function LeagueManagerNav() {
  const pathname = usePathname() || "";
  return (
    <nav aria-label="League Manager sections" style={{ display: "flex", gap: "0.55rem", flexWrap: "wrap", margin: "0 0 1rem" }}>
      {links.map(([href, label]) => {
        const active = href === "/admin/league-manager"
          ? pathname === href
          : pathname === href || pathname.startsWith(`${href}/`);
        return (
          <Link
            key={href}
            href={href}
            aria-current={active ? "page" : undefined}
            style={{
              padding: "0.45rem 0.75rem",
              border: `1px solid ${active ? "#2563eb" : "#cbd5e1"}`,
              borderRadius: "999px",
              background: active ? "#dbeafe" : "white",
              textDecoration: "none",
              color: active ? "#1d4ed8" : "#0f172a",
              fontWeight: 700
            }}
          >
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
''',
)

write(
    "apps/web/app/admin/league-manager/results/page.tsx",
    '''import Link from "next/link";
import { getClubLeagueResults } from "@/lib/api";
import LeagueManagerNav from "../LeagueManagerNav";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

function rating(value?: number | null): string {
  return value == null || Number.isNaN(Number(value)) ? "—" : Number(value).toFixed(3);
}

export default async function AdminLeagueResultsPage({ searchParams }: Props) {
  const requestedLeague = first(searchParams?.league).trim();
  const { data, error } = await getClubLeagueResults("tres-palapas", requestedLeague || null);
  const selectedLeague = data?.selected_league || requestedLeague;
  const selectedWeek = data?.selected_week || null;
  const weeklyRows = (data?.weekly_results || []).filter((row) => selectedWeek == null || row.week_num === selectedWeek);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>Admin League Manager</p>
      <h1 style={{ marginTop: 0 }}>League results</h1>
      <LeagueManagerNav />
      <p style={{ color: "#334155", maxWidth: "900px" }}>Review one league's current standings, ratings, record, weekly results, and public result detail from a dedicated league workspace.</p>

      <form method="get" style={{ ...cardStyle, display: "grid", gridTemplateColumns: "minmax(240px, 1fr) auto", gap: "0.75rem", alignItems: "end", marginBottom: "1rem" }}>
        <label><strong>League</strong><br />
          <select name="league" defaultValue={selectedLeague || ""} style={{ width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" }}>
            {(data?.leagues || []).map((league) => <option key={league.name} value={league.name}>{league.name}</option>)}
          </select>
        </label>
        <button type="submit" style={{ padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 }}>Load results</button>
      </form>

      {error ? <p role="alert" style={{ color: "#b91c1c" }}>{error}</p> : null}
      {!error && !data?.leagues?.length ? <p>No leagues are available.</p> : null}

      {data && selectedLeague ? (
        <>
          <article style={{ ...cardStyle, marginBottom: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>{selectedLeague}</h2>
            <p style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
              <span><strong>Players:</strong> {data.standings.length}</span>
              <span><strong>Latest week:</strong> {selectedWeek ? `Week ${selectedWeek}` : "No weekly results"}</span>
              <Link href={`/clubs/tres-palapas/league-results?league=${encodeURIComponent(selectedLeague)}`}>Open full public league results</Link>
            </p>
          </article>

          <article style={{ ...cardStyle, marginBottom: "1rem" }}>
            <h2 style={{ marginTop: 0 }}>Current standings</h2>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "720px" }}>
                <thead><tr>{["Rank", "Player", "Rating", "Matches", "Wins", "Losses", "Win %", "Rating change"].map((heading) => <th key={heading} style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid #cbd5e1" }}>{heading}</th>)}</tr></thead>
                <tbody>{data.standings.map((row) => <tr key={String(row.player_id)}><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.rank ?? "—"}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{rating(row.rating_jupr)}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.matches_played ?? 0}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.losses ?? 0}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.win_pct == null ? "—" : `${Number(row.win_pct).toFixed(1)}%`}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{rating(row.rating_delta_jupr)}</td></tr>)}</tbody>
              </table>
            </div>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>{selectedWeek ? `Week ${selectedWeek} results` : "Weekly results"}</h2>
            {weeklyRows.length ? <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "620px" }}><thead><tr>{["Player", "Games", "Wins", "Losses", "Win %", "Rating change"].map((heading) => <th key={heading} style={{ textAlign: "left", padding: "0.55rem", borderBottom: "1px solid #cbd5e1" }}>{heading}</th>)}</tr></thead><tbody>{weeklyRows.map((row) => <tr key={`${row.week_num}-${row.player_id}`}><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.player_name}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.games ?? 0}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.losses ?? 0}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{row.win_pct == null ? "—" : `${Number(row.win_pct).toFixed(1)}%`}</td><td style={{ padding: "0.55rem", borderBottom: "1px solid #e2e8f0" }}>{rating(row.rating_delta_jupr)}</td></tr>)}</tbody></table></div> : <p style={{ color: "#64748b" }}>No weekly results are available yet.</p>}
          </article>
        </>
      ) : null}
    </section>
  );
}
''',
)

# ---------------------------------------------------------------------------
# League roster UX
# ---------------------------------------------------------------------------
replace_once(
    "apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx",
    'const [filter, setFilter] = useState<RosterFilter>("all");',
    'const [filter, setFilter] = useState<RosterFilter>("in_league");',
)
replace_once(
    "apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx",
    'setDetail(null);\n    setSelectedIds([]);',
    'setDetail(null);\n    setSelectedIds([]);\n    setFilter("in_league");',
)
replace_once(
    "apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx",
    '<option value="all">All club players</option><option value="in_league">In this league</option><option value="not_in_league">Not in this league</option><option value="inactive">Inactive club players</option>',
    '<option value="in_league">In this league</option><option value="not_in_league">Add players</option><option value="all">All club players</option><option value="inactive">Inactive club players</option>',
)
replace_once(
    "apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx",
    '<option value="activate">Add / reactivate</option><option value="deactivate">Deactivate</option>',
    '<option value="activate">Add / reactivate</option><option value="deactivate">Remove from league</option>',
)
replace_once(
    "apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx",
    'triggerLabel={busy ? "Saving…" : `${action === "activate" ? "Add" : "Deactivate"} ${selectedIds.length} selected`}',
    'triggerLabel={busy ? "Saving…" : action === "activate" ? (selectedIds.length === 1 ? "Add Player" : "Add Players") : (selectedIds.length === 1 ? "Remove Player" : "Remove Players")}',
)
replace_once(
    "apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx",
    'title={`${action === "activate" ? "Add" : "Deactivate"} these league players?`}',
    'title={`${action === "activate" ? "Add" : "Remove"} ${selectedIds.length === 1 ? "this player" : "these players"}?`}',
)
replace_once(
    "apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx",
    'confirmLabel="Yes, save roster batch"',
    'confirmLabel={action === "activate" ? "Yes, add players" : "Yes, remove players"}',
)

# ---------------------------------------------------------------------------
# Structural Individual / Team league mode
# ---------------------------------------------------------------------------
replace_once(
    "jupr_app/services/admin_league_manager_create_service.py",
    'def _normalize_match_format(value: Any) -> str:\n',
    '''def _normalize_league_type(value: Any) -> str:\n    clean = str(value or "Individual").strip().casefold()\n    if clean not in {"individual", "team"}:\n        raise ValueError("league_type must be Individual or Team.")\n    return "Team" if clean == "team" else "Individual"\n\n\ndef _normalize_match_format(value: Any) -> str:\n''',
)
replace_once(
    "jupr_app/services/admin_league_manager_create_service.py",
    '    match_format: str = "doubles",\n    source: str = "next_league_manager_create",',
    '    match_format: str = "doubles",\n    league_type: str = "Individual",\n    source: str = "next_league_manager_create",',
)
replace_once(
    "jupr_app/services/admin_league_manager_create_service.py",
    '    clean_match_format = _normalize_match_format(match_format)\n',
    '    clean_match_format = _normalize_match_format(match_format)\n    clean_league_type = _normalize_league_type(league_type)\n',
)
replace_once(
    "jupr_app/services/admin_league_manager_create_service.py",
    '        "description": clean_description,\n        "match_format": clean_match_format,',
    '        "description": clean_description,\n        "league_type": clean_league_type,\n        "match_format": clean_match_format,',
)
replace_once(
    "jupr_app/services/admin_league_manager_create_service.py",
    '        "description": _clean_text(source_league.get("description"), limit=2000),\n        "match_format": _normalize_match_format(source_league.get("match_format")),',
    '        "description": _clean_text(source_league.get("description"), limit=2000),\n        "league_type": _normalize_league_type(source_league.get("league_type")),\n        "match_format": _normalize_match_format(source_league.get("match_format")),',
)
replace_once(
    "services/api/admin_league_manager_routes.py",
    'class AdminLeagueManagerCreateRequest(BaseModel):\n    league_name: str = Field(min_length=1, max_length=120)\n    match_format: str = Field(default="doubles", pattern=r"^(doubles|singles)$")',
    'class AdminLeagueManagerCreateRequest(BaseModel):\n    league_name: str = Field(min_length=1, max_length=120)\n    league_type: str = Field(default="Individual", pattern=r"^(Individual|Team)$")\n    match_format: str = Field(default="doubles", pattern=r"^(doubles|singles)$")',
)
replace_once(
    "services/api/admin_league_manager_routes.py",
    '                match_format=payload.match_format,\n                actor_email=actor_email,',
    '                match_format=payload.match_format,\n                league_type=payload.league_type,\n                actor_email=actor_email,',
)

replace_once(
    "apps/web/app/admin/league-manager/LeagueManagerPanel.tsx",
    '  const [createMatchFormat, setCreateMatchFormat] = useState<"doubles" | "singles">("doubles");',
    '  const [createLeagueType, setCreateLeagueType] = useState<"Individual" | "Team">("Individual");\n  const [createMatchFormat, setCreateMatchFormat] = useState<"doubles" | "singles">("doubles");',
)
replace_once(
    "apps/web/app/admin/league-manager/LeagueManagerPanel.tsx",
    'body: JSON.stringify({ league_name: name, match_format: createMatchFormat, description: createDescription, min_games: minGames, k_factor: kFactor, confirmation_text: confirmationText, source: "next_league_manager_create_form" })',
    'body: JSON.stringify({ league_name: name, league_type: createLeagueType, match_format: createMatchFormat, description: createDescription, min_games: minGames, k_factor: kFactor, confirmation_text: confirmationText, source: "next_league_manager_create_form" })',
)
replace_once(
    "apps/web/app/admin/league-manager/LeagueManagerPanel.tsx",
    'setCreateName(""); setCreateDescription(""); setCreateMatchFormat("doubles"); setCreateMinGames("6"); setCreateKFactor("32");',
    'setCreateName(""); setCreateDescription(""); setCreateLeagueType("Individual"); setCreateMatchFormat("doubles"); setCreateMinGames("6"); setCreateKFactor("32");',
)
replace_once(
    "apps/web/app/admin/league-manager/LeagueManagerPanel.tsx",
    '<label><strong>League format</strong><br /><select value={createMatchFormat}',
    '<label><strong>League mode</strong><br /><select value={createLeagueType} onChange={(event) => setCreateLeagueType(event.target.value as "Individual" | "Team")} style={inputStyle}><option value="Individual">Individual</option><option value="Team">Team</option></select></label>\n          <label><strong>League format</strong><br /><select value={createMatchFormat}',
)
replace_once(
    "apps/web/app/admin/league-manager/LeagueManagerPanel.tsx",
    '{league.league_name} · {league.match_format === "singles" ? "Singles" : "Doubles"} · {league.status}',
    '{league.league_name} · {league.league_type || "Individual"} · {league.match_format === "singles" ? "Singles" : "Doubles"} · {league.status}',
)
replace_once(
    "apps/web/app/admin/league-manager/LeagueManagerPanel.tsx",
    'style={{ ...cardStyle, textAlign: "left", cursor: "pointer" }}><strong>{league.league_name}</strong><br /><span style={{ color: "#475569" }}>{league.match_format === "singles" ? "Singles" : "Doubles"}</span><br />',
    'style={{ ...cardStyle, textAlign: "left", cursor: "pointer", display: "grid", gap: "0.35rem", alignContent: "start", minWidth: 0 }}><strong style={{ overflowWrap: "anywhere" }}>{league.league_name}</strong><span style={{ color: "#475569" }}>{league.league_type || "Individual"} · {league.match_format === "singles" ? "Singles" : "Doubles"}</span>',
)
replace_once(
    "apps/web/app/admin/league-manager/LeagueManagerPanel.tsx",
    '{detail ? <>\n        <article style={cardStyle}>',
    '''{detail ? <>\n        <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>\n          <h2 style={{ marginTop: 0 }}>League home</h2>\n          <p style={{ color: "#475569" }}>Open a focused workspace for <strong>{detail.league.league_name}</strong>.</p>\n          <p style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap" }}>\n            <Link href={`/admin/league-manager/results?league=${encodeURIComponent(detail.league.league_name)}`}>Results</Link>\n            <Link href="/admin/league-manager/roster">Roster</Link>\n            <Link href="/admin/league-manager/settings">Settings</Link>\n            <Link href="/admin/league-manager/live">Live rounds</Link>\n            {String(detail.league.league_type || "Individual") === "Team" ? <Link href="/admin/league-manager/teams">Team league</Link> : null}\n            <Link href="/admin/league-manager/awards">Awards</Link>\n          </p>\n        </article>\n        <article style={cardStyle}>''',
)
replace_once(
    "apps/web/app/admin/league-manager/LeagueManagerPanel.tsx",
    '<div><strong>Status</strong><br />{detail.league.status}</div><div><strong>Format</strong>',
    '<div><strong>Status</strong><br />{detail.league.status}</div><div><strong>Mode</strong><br />{detail.league.league_type || "Individual"}</div><div><strong>Format</strong>',
)

write(
    "supabase/migrations/20260731210000_league_competition_mode.sql",
    '''-- Structural Individual / Team league mode for staging acceptance.
update public.leagues_metadata lm
set league_type = case
  when exists (
    select 1 from public.team_league_settings tls
    where tls.club_id = lm.club_id and tls.league_name = lm.league_name
  ) then 'Team'
  else 'Individual'
end
where coalesce(lm.league_type, '') not in ('Individual', 'Team');

alter table public.leagues_metadata
  drop constraint if exists leagues_metadata_league_type_check;
alter table public.leagues_metadata
  add constraint leagues_metadata_league_type_check
  check (league_type in ('Individual', 'Team'));

create or replace function public.prevent_league_competition_mode_change()
returns trigger
language plpgsql
as $$
begin
  if old.league_type is distinct from new.league_type then
    raise exception 'league competition mode is immutable after creation';
  end if;
  return new;
end;
$$;

drop trigger if exists trg_prevent_league_competition_mode_change on public.leagues_metadata;
create trigger trg_prevent_league_competition_mode_change
before update of league_type on public.leagues_metadata
for each row execute function public.prevent_league_competition_mode_change();

create or replace function public.require_team_league_mode()
returns trigger
language plpgsql
as $$
begin
  if not exists (
    select 1 from public.leagues_metadata lm
    where lm.club_id = new.club_id
      and lm.league_name = new.league_name
      and lm.league_type = 'Team'
  ) then
    raise exception 'team-league setup requires a Team league created in League Manager';
  end if;
  return new;
end;
$$;

drop trigger if exists trg_require_team_league_mode on public.team_league_settings;
create trigger trg_require_team_league_mode
before insert or update of club_id, league_name on public.team_league_settings
for each row execute function public.require_team_league_mode();
''',
)

# ---------------------------------------------------------------------------
# Team league reload and timezone control
# ---------------------------------------------------------------------------
replace_once(
    "jupr_app/services/team_league_service.py",
    '        order="created_at",\n    )\n    players = [',
    '        order="started_at",\n    )\n    players = [',
)
replace_once(
    "jupr_app/services/team_league_service.py",
    '                "created_at",\n                "updated_at",',
    '                "started_at",\n                "updated_at",',
)
replace_once(
    "apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx",
    'type PendingOperation = { id: string; operation_type?: string; status?: string; created_at?: string };',
    'type PendingOperation = { id: string; operation_type?: string; status?: string; started_at?: string };',
)
replace_once(
    "apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx",
    'const gridStyle = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" };',
    '''const gridStyle = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" };
const TIMEZONE_OPTIONS = [
  "America/Mazatlan",
  "America/Phoenix",
  "America/Los_Angeles",
  "America/Denver",
  "America/Chicago",
  "America/New_York",
  "America/Mexico_City",
  "UTC"
];''',
)
replace_once(
    "apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx",
    'timezone: String(settings?.timezone || "America/Chicago"),',
    'timezone: String(settings?.timezone || "America/Mazatlan"),',
)
replace_once(
    "apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx",
    '<label><strong>Timezone</strong><br /><input value={draft.timezone} onChange={(event) => update("timezone", event.target.value)} placeholder="America/Chicago" style={inputStyle} /></label>',
    '<label><strong>Timezone</strong><br /><select value={draft.timezone} onChange={(event) => update("timezone", event.target.value)} style={inputStyle}>{!TIMEZONE_OPTIONS.includes(draft.timezone) ? <option value={draft.timezone}>{draft.timezone}</option> : null}{TIMEZONE_OPTIONS.map((zone) => <option key={zone} value={zone}>{zone}</option>)}</select></label>',
)
replace_once(
    "apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx",
    'const names = Array.from(new Set([...(base.leagues || []).map((league) => league.league_name), ...(team.leagues || []).map((league) => league.league_name)])).sort();',
    'const names = Array.from(new Set([...(base.leagues || []).filter((league) => String(league.league_type || "Individual") === "Team").map((league) => league.league_name), ...(team.leagues || []).map((league) => league.league_name)])).sort();',
)

# ---------------------------------------------------------------------------
# Player Editor result modal
# ---------------------------------------------------------------------------
replace_once(
    "apps/web/app/admin/players/PlayerEditorPanel.tsx",
    'const [saving, setSaving] = useState(false); const [message, setMessage] = useState<string | null>(null);',
    'const [saving, setSaving] = useState(false); const [message, setMessage] = useState<string | null>(null); const [resultDialog, setResultDialog] = useState<string | null>(null);',
)
replace_once(
    "apps/web/app/admin/players/PlayerEditorPanel.tsx",
    'setSaving(false); setMessage(null);\n    setPlayers([]);',
    'setSaving(false); setMessage(null); setResultDialog(null);\n    setPlayers([]);',
)
replace_once(
    "apps/web/app/admin/players/PlayerEditorPanel.tsx",
    'setMessage("Player saved. Use Match Log and Replay History if downstream rating repair is needed.");',
    'setMessage(null); setResultDialog(`Player saved: ${payload.player?.name || editName.trim()}. Use Match Log and Replay History if downstream rating repair is needed.`);',
)
replace_once(
    "apps/web/app/admin/players/PlayerEditorPanel.tsx",
    '  return <section style={{ display: "grid", gap: "1rem" }}>\n    {message ? (',
    '''  return <section style={{ display: "grid", gap: "1rem" }}>
    {resultDialog ? (
      <div role="dialog" aria-modal="true" aria-labelledby="player-save-result-title" style={{ position: "fixed", inset: 0, zIndex: 1000, display: "grid", placeItems: "center", padding: "1rem", background: "rgba(15, 23, 42, 0.55)" }}>
        <article style={{ ...cardStyle, width: "min(560px, 100%)", boxShadow: "0 24px 70px rgba(15, 23, 42, 0.35)" }}>
          <h2 id="player-save-result-title" style={{ marginTop: 0 }}>Player update complete</h2>
          <p role="status" style={{ color: "#166534" }}>{resultDialog}</p>
          <p style={{ textAlign: "right", marginBottom: 0 }}><button type="button" onClick={() => setResultDialog(null)} style={buttonStyle}>OK</button></p>
        </article>
      </div>
    ) : null}
    {message ? (''',
)

# ---------------------------------------------------------------------------
# Match Log final score correction summary
# ---------------------------------------------------------------------------
replace_once(
    "apps/web/app/admin/match-log/MatchLogApplyPanel.tsx",
    '      if (!stagedPatches.length) throw new Error("Stage at least one match edit first.");\n      const payload = await callApi',
    '      if (!stagedPatches.length) throw new Error("Stage at least one match edit first.");\n      const submittedPatches = stagedPatches.map((patch) => ({ ...patch }));\n      const payload = await callApi',
)
replace_once(
    "apps/web/app/admin/match-log/MatchLogApplyPanel.tsx",
    '        patches: stagedPatches,',
    '        patches: submittedPatches,',
)
replace_once(
    "apps/web/app/admin/match-log/MatchLogApplyPanel.tsx",
    '      const summary = resultSummary(payload) || "Match edits completed.";\n      setResult(payload);',
    '''      const summary = resultSummary(payload) || "Match edits completed.";
      const correctionSummaries = submittedPatches.map((patch) => {
        const original = matches.find((match) => match.id != null && Number(match.id) === Number(patch.id));
        const changes: string[] = [];
        if (patch.score_t1 != null || patch.score_t2 != null) {
          const oldTeam1 = Number(original?.score?.team1 ?? 0);
          const oldTeam2 = Number(original?.score?.team2 ?? 0);
          const newTeam1 = patch.score_t1 == null ? oldTeam1 : Number(patch.score_t1);
          const newTeam2 = patch.score_t2 == null ? oldTeam2 : Number(patch.score_t2);
          changes.push(`score ${oldTeam1}-${oldTeam2} → ${newTeam1}-${newTeam2}`);
        }
        const otherFields = patchFields(patch).filter((field) => field !== "score_t1" && field !== "score_t2");
        if (otherFields.length) changes.push(otherFields.join(", "));
        return `Match #${patch.id}: ${changes.join("; ") || "updated"}`;
      });
      setResult(payload);''',
)
replace_once(
    "apps/web/app/admin/match-log/MatchLogApplyPanel.tsx",
    '<p role="status" style={{ color: "#166534" }}><strong>{summary}</strong></p>\n            <dl',
    '<p role="status" style={{ color: "#166534" }}><strong>{summary}</strong></p>\n            {correctionSummaries.length ? <ul>{correctionSummaries.map((item) => <li key={item}>{item}</li>)}</ul> : null}\n            <dl',
)

# ---------------------------------------------------------------------------
# Tournament workspace simplification
# ---------------------------------------------------------------------------
replace_once(
    "apps/web/components/TournamentAdminNav.tsx",
    '{ href: "/admin/tournaments", label: "Registration editor", match: exact("/admin/tournaments") },',
    '{ href: "/admin/tournaments", label: "Tournament home", match: exact("/admin/tournaments") },',
)
replace_once(
    "apps/web/components/TournamentAdminNav.tsx",
    '      <nav aria-label="Tournament operations workflows" className={`${styles.nav} ${styles.subnav}`}>\n        <p className={styles.subnavLabel}>Operations workflows</p>\n        <NavigationLinks items={operationItems} pathname={pathname} />\n      </nav>',
    '      {pathname.startsWith("/admin/tournaments/ops") ? <nav aria-label="Tournament operations workflows" className={`${styles.nav} ${styles.subnav}`}><p className={styles.subnavLabel}>Operations workflows</p><NavigationLinks items={operationItems} pathname={pathname} /></nav> : null}',
)
replace_once(
    "apps/web/app/admin/tournaments/TournamentAdminPanel.tsx",
    '<article style={cardStyle}>\n        <h2 style={{ marginTop: 0 }}>Tournament Admin session</h2>',
    '<article style={{ ...cardStyle, display: detail ? "none" : "block" }}>\n        <h2 style={{ marginTop: 0 }}>Create or open a tournament</h2>',
)
replace_once(
    "apps/web/app/admin/tournaments/TournamentAdminPanel.tsx",
    '          <h2 style={{ marginTop: 0 }}>Tournaments</h2>',
    '          <h2 style={{ marginTop: 0 }}>Open tournament</h2>',
)
replace_once(
    "apps/web/app/admin/tournaments/TournamentAdminPanel.tsx",
    '<article style={cardStyle}>\n          <h2 style={{ marginTop: 0 }}>Tournaments</h2>',
    '<article style={{ ...cardStyle, display: detail ? "none" : "block" }}>\n          <h2 style={{ marginTop: 0 }}>Open tournament</h2>',
)
replace_once(
    "apps/web/app/admin/tournaments/TournamentAdminPanel.tsx",
    '<h2 style={{ marginTop: 0 }}>{detail.tournament.name}</h2>\n            <div style={{ display: "grid"',
    '<h2 style={{ marginTop: 0 }}>{detail.tournament.name}</h2>\n            <p style={{ display: "flex", gap: "0.65rem", flexWrap: "wrap", alignItems: "center" }}><strong>Selected tournament</strong><button type="button" onClick={() => { setDetail(null); setSelectedId(""); setSelectedRegistrationId(""); setSelectedSelectionId(""); }} style={ghostButtonStyle}>Change tournament</button></p>\n            <div style={{ display: "grid"',
)
replace_once(
    "apps/web/app/admin/tournaments/TournamentAdminPanel.tsx",
    '          <article style={{ ...cardStyle, background: "#f8fafc" }}>\n            <h2 style={{ marginTop: 0 }}>Edit tournament shell</h2>',
    '''          <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
            <h2 style={{ marginTop: 0 }}>Tournament home</h2>
            <p style={{ display: "flex", gap: "0.65rem", flexWrap: "wrap" }}>
              <Link href="/admin/tournament-setup">Setup</Link>
              <Link href="/admin/tournaments/registrations">Registrations and reports</Link>
              <Link href="/admin/tournaments/bulk">Bulk actions</Link>
              <Link href="/admin/tournaments/commerce">Extras and fulfillment</Link>
              <Link href="/admin/tournaments/team-competition">Ratings and team play</Link>
              <Link href="/admin/tournaments/ops">Operations and results</Link>
              <Link href="/admin/tournament-live">Live runner</Link>
              <Link href="/admin/tournaments/status">Status and recovery</Link>
            </p>
          </article>

          <article style={{ ...cardStyle, background: "#f8fafc" }}>
            <h2 style={{ marginTop: 0 }}>Tournament settings</h2>''',
)
replace_once(
    "apps/web/app/admin/tournaments/page.tsx",
    '<h1 style={{ marginTop: 0 }}>Tournament registration management</h1>',
    '<h1 style={{ marginTop: 0 }}>Tournament Manager</h1>',
)
replace_once(
    "apps/web/app/admin/tournaments/page.tsx",
    'Read-foundation for moving Streamlit tournament registration management to Next/Vercel. This view is guarded by the stored admin session and FastAPI tournament permissions.',
    'Create a tournament or open an existing tournament, then work from a focused tournament home for setup, registration, commerce, operations, results, live play, publishing, and recovery.',
)

# ---------------------------------------------------------------------------
# Focused source-contract tests
# ---------------------------------------------------------------------------
write(
    "apps/web/tests/acceptance-fixes-20260731.cjs",
    '''const fs = require("fs");

function read(path) { return fs.readFileSync(path, "utf8"); }
function expect(path, text) {
  const body = read(path);
  if (!body.includes(text)) throw new Error(`${path} is missing: ${text}`);
}

expect("app/admin/league-manager/LeagueManagerNav.tsx", 'aria-current={active ? "page" : undefined}');
expect("app/admin/league-manager/LeagueManagerNav.tsx", '"/admin/league-manager/results"');
expect("app/admin/league-manager/roster/LeagueRosterPanel.tsx", 'useState<RosterFilter>("in_league")');
expect("app/admin/league-manager/roster/LeagueRosterPanel.tsx", '"Add Player"');
expect("app/admin/league-manager/LeagueManagerPanel.tsx", 'League mode');
expect("app/admin/league-manager/LeagueManagerPanel.tsx", 'League home');
expect("app/admin/league-manager/results/page.tsx", 'League results');
expect("app/admin/players/PlayerEditorPanel.tsx", 'Player update complete');
expect("app/admin/match-log/MatchLogApplyPanel.tsx", 'score ${oldTeam1}-${oldTeam2} → ${newTeam1}-${newTeam2}');
expect("app/admin/league-manager/teams/TeamLeaguesPanel.tsx", 'America/Mazatlan');
expect("app/admin/league-manager/teams/TeamLeaguesPanel.tsx", 'league.league_type');
expect("app/admin/tournaments/TournamentAdminPanel.tsx", 'Tournament home');
expect("components/TournamentAdminNav.tsx", 'Tournament home');
console.log("July 31 acceptance source contracts passed.");
''',
)

write(
    "tests/test_acceptance_fixes_20260731.py",
    '''from pathlib import Path


def test_team_league_operation_history_uses_real_timestamp_column():
    source = Path("jupr_app/services/team_league_service.py").read_text()
    block = source[source.index('"team_league_operations"'):source.index('players = [', source.index('"team_league_operations"'))]
    assert 'order="started_at"' in block
    assert 'order="created_at"' not in block


def test_league_creation_has_structural_mode():
    service = Path("jupr_app/services/admin_league_manager_create_service.py").read_text()
    routes = Path("services/api/admin_league_manager_routes.py").read_text()
    migration = Path("supabase/migrations/20260731210000_league_competition_mode.sql").read_text()
    assert 'league_type: str = "Individual"' in service
    assert 'pattern=r"^(Individual|Team)$"' in routes
    assert "prevent_league_competition_mode_change" in migration
    assert "require_team_league_mode" in migration
''',
)

print("Applied July 31 acceptance fixes.")
