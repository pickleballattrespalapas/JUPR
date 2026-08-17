"use client";

import Link from "next/link";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, type ActionCompletion } from "@/components/interaction";
import type {
  AdminLeagueManagerListResponse,
  AdminLeagueManagerStatusResponse
} from "@/lib/adminLeagueManagerApi";
import { isTeamLeagueType } from "@/lib/leagueRouteContext";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = { apiBase: string | null; clubId: string; status: AdminLeagueManagerStatusResponse };
type TeamSettings = {
  league_name: string;
  status?: string;
  registration_open?: boolean;
  team_size?: 2 | 3 | 4;
  team_category?: "open" | "mens" | "womens" | "mixed";
  max_alternates?: number;
  substitute_pool_enabled?: boolean;
  mixed_required_men?: number;
  mixed_required_women?: number;
  allow_substitutes?: boolean;
  playoff_format?: string;
  playoff_team_count?: number | null;
  start_date?: string | null;
  weekday?: number;
  start_time?: string;
  timezone?: string;
  venue?: string | null;
  registration_closes_at?: string | null;
  settings_version?: number;
  schedule_version?: number;
  standings_version?: number;
  roster_version?: number;
};
type TeamRow = {
  id: string;
  team_name: string;
  status: string;
  captain_player_id: number;
  partner_player_id?: number | null;
  invitation_delivery_status?: string;
  roster_complete?: boolean;
  members?: TeamMemberRow[];
};
type TeamMemberRow = { id: string; team_id: string; player_id: number; role: "captain" | "primary" | "alternate"; status: string; player_name?: string | null; gender?: string | null };
type SubstitutePoolRow = { id: string; player_id: number; status: "available" | "unavailable" | "withdrawn"; note?: string | null };
type WaitlistRow = { id: string; player_id: number; status: string; note?: string | null };
type PlayerRow = { id: number; name: string; rating?: number | null; gender?: string | null; active?: boolean };
type FixtureRow = {
  id: string;
  phase: "regular" | "playoff";
  round_number: number;
  week_number?: number | null;
  team_a_id?: string | null;
  team_b_id?: string | null;
  status: string;
  team_a_score?: number | null;
  team_b_score?: number | null;
  winner_team_id?: string | null;
  official_match_id?: number | null;
  scheduled_at?: string | null;
};
type StandingRow = {
  rank: number;
  team_id: string;
  team_name: string;
  wins: number;
  losses: number;
  points_for?: number;
  points_against?: number;
  point_differential?: number;
};
type PendingOperation = { id: string; operation_type?: string; status?: string; started_at?: string };
type TeamLeagueDetail = {
  ok: boolean;
  settings: TeamSettings;
  teams: TeamRow[];
  members: TeamMemberRow[];
  substitute_pool: SubstitutePoolRow[];
  waitlist: WaitlistRow[];
  players: PlayerRow[];
  fixtures: FixtureRow[];
  standings: StandingRow[];
  pending_operations: PendingOperation[];
  recovery_required: boolean;
};
type TeamLeagueListResponse = { ok: boolean; leagues: TeamSettings[]; league_count: number };
type SchedulePreview = {
  ok: boolean;
  phase: "regular" | "playoff";
  proposed_fixtures: FixtureRow[];
  team_names: Record<string, string>;
  expected_schedule_version: number;
  expected_standings_version: number;
  expected_roster_version: number;
  confirmed_roster_fingerprint: string;
  preview_fingerprint: string;
};
type RecoveryEvidence = {
  ok: boolean;
  operation: Record<string, unknown>;
  fixture?: Record<string, unknown> | null;
  stable_direct_match_receipt?: { found?: boolean; committed?: boolean; match_ids?: number[] };
  safe_action?: string;
};
type WriteResponse = { ok?: boolean; committed?: boolean; operation_id?: string; message?: string };
type SettingsDraft = {
  registrationOpen: boolean;
  teamSize: string;
  teamCategory: string;
  maxAlternates: string;
  substitutePoolEnabled: boolean;
  mixedRequiredMen: string;
  mixedRequiredWomen: string;
  allowSubstitutes: boolean;
  playoffFormat: string;
  playoffTeamCount: string;
  startDate: string;
  weekday: string;
  startTime: string;
  timezone: string;
  venue: string;
  registrationClosesAt: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const insetStyle = { border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.8rem", background: "#f8fafc" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const gridStyle = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" };
const TIMEZONE_OPTIONS = [
  "America/Mazatlan",
  "America/Phoenix",
  "America/Los_Angeles",
  "America/Denver",
  "America/Chicago",
  "America/New_York",
  "America/Mexico_City",
  "UTC"
];

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function operationKey(kind: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") return `${kind}:${crypto.randomUUID()}`;
  return `${kind}:${Date.now()}:${Math.random().toString(16).slice(2)}`;
}

const confirmedWriteRefreshWarning = " The action completed, but the latest team league view could not be refreshed. Reload this page before making another change; do not repeat the completed action.";

async function refreshAfterConfirmedWrite(refresh: () => Promise<void>): Promise<string> {
  try {
    await refresh();
    return "";
  } catch {
    return confirmedWriteRefreshWarning;
  }
}

function settingsDraft(settings?: TeamSettings | null): SettingsDraft {
  return {
    registrationOpen: Boolean(settings?.registration_open),
    teamSize: String(settings?.team_size || 2),
    teamCategory: String(settings?.team_category || "open"),
    maxAlternates: String(settings?.max_alternates || 0),
    substitutePoolEnabled: Boolean(settings?.substitute_pool_enabled),
    mixedRequiredMen: String(settings?.mixed_required_men || 1),
    mixedRequiredWomen: String(settings?.mixed_required_women || 1),
    allowSubstitutes: Boolean(settings?.allow_substitutes),
    playoffFormat: String(settings?.playoff_format || "none"),
    playoffTeamCount: settings?.playoff_team_count == null ? "" : String(settings.playoff_team_count),
    startDate: String(settings?.start_date || ""),
    weekday: String(settings?.weekday ?? 0),
    startTime: String(settings?.start_time || "18:00").slice(0, 5),
    timezone: String(settings?.timezone || "America/Mazatlan"),
    venue: String(settings?.venue || ""),
    registrationClosesAt: settings?.registration_closes_at ? String(settings.registration_closes_at).slice(0, 16) : ""
  };
}

function teamName(detail: TeamLeagueDetail, teamId?: string | null): string {
  if (!teamId) return "TBD";
  return detail.teams.find((team) => team.id === teamId)?.team_name || teamId;
}

function playerName(detail: TeamLeagueDetail, playerId: number): string {
  return detail.players.find((player) => player.id === playerId)?.name || `Player ${playerId}`;
}

function activeTeamMembers(team?: TeamRow): TeamMemberRow[] {
  return (team?.members || []).filter((member) => member.status === "active");
}

function defaultPlayingLineup(team?: TeamRow): number[] {
  const members = activeTeamMembers(team)
    .filter((member) => member.role !== "alternate")
    .slice(0, 2)
    .map((member) => member.player_id);
  if (members.length === 2) return members;
  return team
    ? [team.captain_player_id, team.partner_player_id]
      .filter((playerId): playerId is number => typeof playerId === "number")
      .slice(0, 2)
    : [];
}

function playingOptions(detail: TeamLeagueDetail, team?: TeamRow): Array<PlayerRow & { sourceLabel: string }> {
  if (!team) return [];
  const roleByPlayer = new Map(
    activeTeamMembers(team).map((member) => [member.player_id, member.role] as const)
  );
  const poolIds = new Set(
    detail.settings.allow_substitutes && detail.settings.substitute_pool_enabled
      ? detail.substitute_pool
        .filter((entry) => entry.status === "available")
        .map((entry) => entry.player_id)
      : []
  );
  return detail.players
    .filter((player) => player.active !== false && (roleByPlayer.has(player.id) || poolIds.has(player.id)))
    .map((player) => ({
      ...player,
      sourceLabel: roleByPlayer.has(player.id)
        ? String(roleByPlayer.get(player.id)).replace("primary", "roster")
        : "substitute pool"
    }));
}

function requestError(payload: unknown, statusCode: number): string {
  if (payload && typeof payload === "object") {
    const detail = (payload as { detail?: unknown }).detail;
    if (detail && typeof detail === "object") {
      const message = String((detail as { message?: unknown }).message || `API error (${statusCode})`);
      const operationId = (detail as { operation_id?: unknown }).operation_id;
      return operationId ? `${message} Recovery operation: ${String(operationId)}.` : message;
    }
    if (detail) return String(detail);
  }
  return `API error (${statusCode})`;
}

function TeamSettingsForm({
  settings,
  busy,
  onDirty,
  onSave
}: {
  settings?: TeamSettings | null;
  busy: boolean;
  onDirty: () => void;
  onSave: (draft: SettingsDraft, confirmationText: string) => Promise<ActionCompletion>;
}) {
  const [draft, setDraft] = useState<SettingsDraft>(() => settingsDraft(settings));
  function update<K extends keyof SettingsDraft>(key: K, value: SettingsDraft[K]) {
    setDraft((current) => ({ ...current, [key]: value }));
    onDirty();
  }
  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Registration and season setup</h2>
      <p style={{ color: "#475569" }}>Configure a 2–4 player primary season roster, assigned alternates, a shared substitute pool, weekly schedule, and optional playoffs. Every match still uses a two-player doubles lineup.</p>
      <div style={gridStyle}>
        <label><input type="checkbox" checked={draft.registrationOpen} onChange={(event) => update("registrationOpen", event.target.checked)} /> <strong>Registration open</strong></label>
        <label><strong>Primary roster size</strong><br /><select value={draft.teamSize} onChange={(event) => { const size = Number(event.target.value); update("teamSize", event.target.value); update("mixedRequiredMen", String(Math.floor(size / 2))); update("mixedRequiredWomen", String(size - Math.floor(size / 2))); }} style={inputStyle}><option value="2">2 players</option><option value="3">3 players</option><option value="4">4 players</option></select></label>
        <label><strong>Maximum alternates</strong><br /><select value={draft.maxAlternates} onChange={(event) => update("maxAlternates", event.target.value)} style={inputStyle}>{[0, 1, 2, 3, 4].map((count) => <option key={count} value={String(count)}>{count}</option>)}</select></label>
        <label><strong>Team eligibility</strong><br /><select value={draft.teamCategory} onChange={(event) => { update("teamCategory", event.target.value); if (event.target.value === "mixed") { const size = Number(draft.teamSize); update("mixedRequiredMen", String(Math.floor(size / 2))); update("mixedRequiredWomen", String(size - Math.floor(size / 2))); } }} style={inputStyle}><option value="open">Open</option><option value="mens">Men&apos;s</option><option value="womens">Women&apos;s</option><option value="mixed">Mixed</option></select></label>
        <label><input type="checkbox" checked={draft.allowSubstitutes} onChange={(event) => { update("allowSubstitutes", event.target.checked); if (!event.target.checked) update("substitutePoolEnabled", false); }} /> <strong>Allow substitutes</strong></label>
        <label><input type="checkbox" checked={draft.substitutePoolEnabled} disabled={!draft.allowSubstitutes} onChange={(event) => update("substitutePoolEnabled", event.target.checked)} /> <strong>Shared substitute pool</strong></label>
        {draft.teamCategory === "mixed" ? <><label><strong>Required men on primary roster</strong><br /><input type="number" min={1} max={3} value={draft.mixedRequiredMen} onChange={(event) => update("mixedRequiredMen", event.target.value)} style={inputStyle} /></label><label><strong>Required women on primary roster</strong><br /><input type="number" min={1} max={3} value={draft.mixedRequiredWomen} onChange={(event) => update("mixedRequiredWomen", event.target.value)} style={inputStyle} /></label></> : null}
        <label><strong>Playoffs</strong><br /><select value={draft.playoffFormat} onChange={(event) => update("playoffFormat", event.target.value)} style={inputStyle}><option value="none">No playoffs</option><option value="top_2_final">Top 2 final</option><option value="top_4_single_elimination">Top 4 single elimination</option><option value="all_team_single_elimination">All-team single elimination</option></select></label>
        {draft.playoffFormat === "all_team_single_elimination" ? <label><strong>Playoff team count</strong><br /><input type="number" min={2} max={128} value={draft.playoffTeamCount} onChange={(event) => update("playoffTeamCount", event.target.value)} style={inputStyle} /></label> : null}
        <label><strong>Season start</strong><br /><input type="date" value={draft.startDate} onChange={(event) => update("startDate", event.target.value)} style={inputStyle} /></label>
        <label><strong>League night</strong><br /><select value={draft.weekday} onChange={(event) => update("weekday", event.target.value)} style={inputStyle}><option value="0">Monday</option><option value="1">Tuesday</option><option value="2">Wednesday</option><option value="3">Thursday</option><option value="4">Friday</option><option value="5">Saturday</option><option value="6">Sunday</option></select></label>
        <label><strong>Start time</strong><br /><input type="time" value={draft.startTime} onChange={(event) => update("startTime", event.target.value)} style={inputStyle} /></label>
        <label><strong>Timezone</strong><br /><select value={draft.timezone} onChange={(event) => update("timezone", event.target.value)} style={inputStyle}>{!TIMEZONE_OPTIONS.includes(draft.timezone) ? <option value={draft.timezone}>{draft.timezone}</option> : null}{TIMEZONE_OPTIONS.map((zone) => <option key={zone} value={zone}>{zone}</option>)}</select></label>
        <label><strong>Venue</strong><br /><input value={draft.venue} onChange={(event) => update("venue", event.target.value)} maxLength={240} style={inputStyle} /></label>
        <label><strong>Registration closes</strong><br /><input type="datetime-local" value={draft.registrationClosesAt} onChange={(event) => update("registrationClosesAt", event.target.value)} style={inputStyle} /></label>
      </div>
      <p><ConfirmAction triggerLabel={busy ? "Saving…" : "Save team league setup"} title="Save this team league setup?" description="This saves roster size, eligibility, alternates, substitute policy, weekly schedule, and playoff choices together." confirmLabel="Yes, save setup" confirmationText="SAVE TEAM LEAGUE" disabled={busy} busy={busy} onConfirm={(confirmationText) => onSave(draft, confirmationText)} /></p>
    </article>
  );
}

function ScoreFixtureCard({
  fixture,
  detail,
  requestWrite,
  onSaved
}: {
  fixture: FixtureRow;
  detail: TeamLeagueDetail;
  requestWrite: <T>(path: string, body: Record<string, unknown>) => Promise<T>;
  onSaved: () => Promise<void>;
}) {
  const teamA = detail.teams.find((team) => team.id === fixture.team_a_id);
  const teamB = detail.teams.find((team) => team.id === fixture.team_b_id);
  const [resultStatus, setResultStatus] = useState<"complete" | "forfeit">("complete");
  const [scoreA, setScoreA] = useState("");
  const [scoreB, setScoreB] = useState("");
  const [winnerId, setWinnerId] = useState(String(fixture.team_a_id || ""));
  const [playersA, setPlayersA] = useState<number[]>(() => defaultPlayingLineup(teamA));
  const [playersB, setPlayersB] = useState<number[]>(() => defaultPlayingLineup(teamB));
  const [note, setNote] = useState("");
  const [scoreKey, setScoreKey] = useState(() => operationKey(`team-score-${fixture.id}`));
  const [reconcileKey, setReconcileKey] = useState(() => operationKey(`team-reconcile-${fixture.id}`));
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const optionsA = playingOptions(detail, teamA);
  const optionsB = playingOptions(detail, teamB);

  function updateLineup(side: "a" | "b", index: number, value: number) {
    const setter = side === "a" ? setPlayersA : setPlayersB;
    setter((current) => current.map((playerId, playerIndex) => playerIndex === index ? value : playerId));
    setScoreKey(operationKey(`team-score-${fixture.id}`));
  }

  async function saveScore(confirmationText: string): Promise<ActionCompletion> {
    setBusy(true);
    setMessage(null);
    try {
      await requestWrite<WriteResponse>(`/fixtures/${encodeURIComponent(fixture.id)}/score`, {
        status: resultStatus,
        team_a_score: resultStatus === "complete" ? Number(scoreA) : null,
        team_b_score: resultStatus === "complete" ? Number(scoreB) : null,
        winner_team_id: winnerId,
        team_a_player_ids: resultStatus === "complete" ? playersA : [],
        team_b_player_ids: resultStatus === "complete" ? playersB : [],
        score_note: note,
        idempotency_key: scoreKey,
        confirmation_text: confirmationText,
        source: "next_team_league_fixture_result"
      });
      setScoreKey(operationKey(`team-score-${fixture.id}`));
      const refreshWarning = await refreshAfterConfirmedWrite(onSaved);
      const successMessage = `${resultStatus === "complete" ? "The canonical doubles match and fixture were saved together." : "The forfeit was saved to the fixture and standings."}${refreshWarning}`;
      setMessage(successMessage);
      return actionSuccess(resultStatus === "complete" ? "Team result saved" : "Forfeit saved", successMessage);
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Unable to save this result."} The same request key is retained for retry.`);
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function reconcile(confirmationText: string): Promise<ActionCompletion> {
    setBusy(true);
    setMessage(null);
    try {
      await requestWrite<WriteResponse>(`/fixtures/${encodeURIComponent(fixture.id)}/reconcile`, {
        idempotency_key: reconcileKey,
        confirmation_text: confirmationText,
        source: "next_team_league_fixture_reconcile"
      });
      setReconcileKey(operationKey(`team-reconcile-${fixture.id}`));
      const refreshWarning = await refreshAfterConfirmedWrite(onSaved);
      const successMessage = `The fixture was reconciled from its canonical match.${refreshWarning}`;
      setMessage(successMessage);
      return actionSuccess("Team result reconciled", successMessage);
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Unable to reconcile this result."} The same request key is retained for retry.`);
      throw error;
    } finally {
      setBusy(false);
    }
  }

  return (
    <article style={insetStyle}>
      <strong>{fixture.phase === "regular" ? `Week ${fixture.week_number || fixture.round_number}` : `Playoff round ${fixture.round_number}`} · {teamName(detail, fixture.team_a_id)} vs {teamName(detail, fixture.team_b_id)}</strong>
      {fixture.scheduled_at ? <p style={{ color: "#64748b" }}>{new Date(fixture.scheduled_at).toLocaleString()}</p> : null}
      {fixture.status === "scheduled" && fixture.team_a_id && fixture.team_b_id ? (
        <>
          <div style={gridStyle}>
            <label><strong>Result type</strong><br /><select value={resultStatus} onChange={(event) => { setResultStatus(event.target.value as "complete" | "forfeit"); setScoreKey(operationKey(`team-score-${fixture.id}`)); }} style={inputStyle}><option value="complete">Played match</option><option value="forfeit">Forfeit</option></select></label>
            <label><strong>Winner</strong><br /><select value={winnerId} onChange={(event) => { setWinnerId(event.target.value); setScoreKey(operationKey(`team-score-${fixture.id}`)); }} style={inputStyle}><option value={fixture.team_a_id}>{teamName(detail, fixture.team_a_id)}</option><option value={fixture.team_b_id}>{teamName(detail, fixture.team_b_id)}</option></select></label>
            {resultStatus === "complete" ? <><label><strong>{teamName(detail, fixture.team_a_id)} score</strong><br /><input type="number" min={0} value={scoreA} onChange={(event) => { setScoreA(event.target.value); setScoreKey(operationKey(`team-score-${fixture.id}`)); }} style={inputStyle} /></label><label><strong>{teamName(detail, fixture.team_b_id)} score</strong><br /><input type="number" min={0} value={scoreB} onChange={(event) => { setScoreB(event.target.value); setScoreKey(operationKey(`team-score-${fixture.id}`)); }} style={inputStyle} /></label></> : null}
          </div>
          {resultStatus === "complete" ? (
            <div style={{ ...gridStyle, marginTop: "0.75rem" }}>
              {[0, 1].map((index) => <label key={`a-${index}`}><strong>{teamName(detail, fixture.team_a_id)} player {index + 1}</strong><br /><select value={playersA[index] || ""} onChange={(event) => updateLineup("a", index, Number(event.target.value))} style={inputStyle}>{optionsA.map((player) => <option key={player.id} value={player.id}>{player.name} · {player.sourceLabel}</option>)}</select></label>)}
              {[0, 1].map((index) => <label key={`b-${index}`}><strong>{teamName(detail, fixture.team_b_id)} player {index + 1}</strong><br /><select value={playersB[index] || ""} onChange={(event) => updateLineup("b", index, Number(event.target.value))} style={inputStyle}>{optionsB.map((player) => <option key={player.id} value={player.id}>{player.name} · {player.sourceLabel}</option>)}</select></label>)}
            </div>
          ) : null}
          <label><strong>Result note</strong><br /><input value={note} onChange={(event) => { setNote(event.target.value); setScoreKey(operationKey(`team-score-${fixture.id}`)); }} maxLength={500} style={inputStyle} /></label>
          <p><ConfirmAction triggerLabel={busy ? "Saving…" : resultStatus === "complete" ? "Save result" : "Save forfeit"} title="Save this team league result?" description={resultStatus === "complete" ? "This publishes one canonical doubles match and updates the fixture together." : "Forfeits update standings but do not create a rated match."} confirmLabel="Yes, save result" confirmationText={resultStatus === "complete" ? "SAVE TEAM LEAGUE RESULT" : "SAVE TEAM LEAGUE FORFEIT"} tone={resultStatus === "forfeit" ? "danger" : "default"} disabled={busy || !winnerId || (resultStatus === "complete" && (!scoreA || !scoreB || playersA.length !== 2 || playersB.length !== 2))} busy={busy} onConfirm={saveScore} /></p>
        </>
      ) : (
        <p style={{ color: "#475569" }}>Status: <strong>{fixture.status}</strong>{fixture.team_a_score != null ? ` · ${fixture.team_a_score}-${fixture.team_b_score}` : ""}{fixture.official_match_id ? ` · Match #${fixture.official_match_id}` : ""}</p>
      )}
      {fixture.official_match_id ? <ConfirmAction triggerLabel={busy ? "Working…" : "Reconcile from match"} title="Reconcile this fixture?" description="Re-read the canonical match and correct this fixture. Regular-season corrections are locked after playoff seeding." confirmLabel="Yes, reconcile" confirmationText="RECONCILE TEAM LEAGUE RESULT" disabled={busy} busy={busy} onConfirm={reconcile} /> : null}
      {message ? <p role="status" style={{ color: /unable|retry|error|recovery/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </article>
  );
}

export default function TeamLeaguesPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [leagueNames, setLeagueNames] = useState<string[]>([]);
  const [teamLeagueRows, setTeamLeagueRows] = useState<TeamSettings[]>([]);
  const [leagueName, setLeagueName] = useState("");
  const [detail, setDetail] = useState<TeamLeagueDetail | null>(null);
  const [preview, setPreview] = useState<SchedulePreview | null>(null);
  const [phase, setPhase] = useState<"regular" | "playoff">("regular");
  const [scheduleKey, setScheduleKey] = useState(() => operationKey("team-schedule"));
  const [settingsKey, setSettingsKey] = useState(() => operationKey("team-settings"));
  const [waitlistIds, setWaitlistIds] = useState<string[]>([]);
  const [waitlistAction, setWaitlistAction] = useState<"pair" | "withdraw">("pair");
  const [waitlistTeamName, setWaitlistTeamName] = useState("");
  const [waitlistKey, setWaitlistKey] = useState(() => operationKey("team-waitlist"));
  const [newTeamName, setNewTeamName] = useState("");
  const [newCaptainId, setNewCaptainId] = useState("");
  const [newCaptainEmail, setNewCaptainEmail] = useState("");
  const [newInitialPrimaryId, setNewInitialPrimaryId] = useState("");
  const [newInitialPrimaryEmail, setNewInitialPrimaryEmail] = useState("");
  const [createTeamKey, setCreateTeamKey] = useState(() => operationKey("team-create"));
  const [rosterTeamId, setRosterTeamId] = useState("");
  const [rosterPlayerId, setRosterPlayerId] = useState("");
  const [rosterRole, setRosterRole] = useState<"primary" | "alternate">("primary");
  const [rosterStatus, setRosterStatus] = useState<"active" | "invited">("active");
  const [rosterKey, setRosterKey] = useState(() => operationKey("team-roster"));
  const [poolPlayerId, setPoolPlayerId] = useState("");
  const [poolStatus, setPoolStatus] = useState<"available" | "unavailable" | "withdrawn">("available");
  const [poolNote, setPoolNote] = useState("");
  const [poolKey, setPoolKey] = useState(() => operationKey("team-substitute-pool"));
  const [recoveryId, setRecoveryId] = useState("");
  const [recoveryEvidence, setRecoveryEvidence] = useState<RecoveryEvidence | null>(null);
  const [recoveryResolution, setRecoveryResolution] = useState<"finalize" | "compensate">("finalize");
  const [recoveryNote, setRecoveryNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const listRequest = useLatestRequestGuard(accessToken, clearProtectedState);
  const detailRequest = useLatestRequestGuard(accessToken);
  const actionRequest = useLatestRequestGuard(accessToken);

  function teamLeaguePath(suffix = ""): string {
    return `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/team-leagues/${encodeURIComponent(leagueName)}${suffix}`;
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in before managing team leagues.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(requestError(payload, response.status));
    return payload as T;
  }

  async function requestTeamWrite<T>(suffix: string, body: Record<string, unknown>): Promise<T> {
    return requestJson<T>(teamLeaguePath(suffix), { method: "POST", body: JSON.stringify(body) });
  }

  function clearProtectedState() {
    detailRequest.invalidate();
    actionRequest.invalidate();
    setLeagueNames([]);
    setTeamLeagueRows([]);
    setLeagueName("");
    setDetail(null);
    setPreview(null);
    setWaitlistIds([]);
    setRecoveryEvidence(null);
    setBusy(false);
    setMessage(null);
  }

  async function loadLeagues() {
    const generation = listRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const [base, team] = await Promise.all([
        requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`),
        requestJson<TeamLeagueListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/team-leagues`)
      ]);
      if (!listRequest.isCurrent(generation)) return;
      const names = Array.from(new Set([...(base.leagues || []).filter((league) => isTeamLeagueType(league.league_type)).map((league) => league.league_name), ...(team.leagues || []).map((league) => league.league_name)])).sort();
      setLeagueNames(names);
      setTeamLeagueRows(team.leagues || []);
      if (leagueName && names.includes(leagueName)) await selectLeague(leagueName, team.leagues || []);
      else if (leagueName) {
        setLeagueName("");
        setDetail(null);
      }
      setMessage(names.length ? `Loaded ${names.length} league(s).` : "Create a standard league draft before setting up a team league.");
    } catch (error) {
      if (listRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load team leagues.");
    } finally {
      if (listRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function selectLeague(selectedLeague: string, rows = teamLeagueRows) {
    const generation = detailRequest.begin();
    setLeagueName(selectedLeague);
    setDetail(null);
    setPreview(null);
    setWaitlistIds([]);
    setRecoveryEvidence(null);
    setScheduleKey(operationKey("team-schedule"));
    setSettingsKey(operationKey("team-settings"));
    setWaitlistKey(operationKey("team-waitlist"));
    setNewTeamName("");
    setNewCaptainId("");
    setNewCaptainEmail("");
    setNewInitialPrimaryId("");
    setNewInitialPrimaryEmail("");
    setCreateTeamKey(operationKey("team-create"));
    setRosterTeamId("");
    setRosterPlayerId("");
    setRosterKey(operationKey("team-roster"));
    setPoolPlayerId("");
    setPoolNote("");
    setPoolKey(operationKey("team-substitute-pool"));
    setMessage(null);
    if (!selectedLeague || !rows.some((row) => row.league_name === selectedLeague)) return;
    setBusy(true);
    try {
      const payload = await requestJson<TeamLeagueDetail>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/team-leagues/${encodeURIComponent(selectedLeague)}`
      );
      if (!detailRequest.isCurrent(generation)) return;
      setDetail(payload);
      setWaitlistAction(payload.settings.team_size === 2 ? "pair" : "withdraw");
    } catch (error) {
      if (detailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load the team league.");
    } finally {
      if (detailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function refreshDetail() {
    if (!leagueName) return;
    const payload = await requestJson<TeamLeagueDetail>(teamLeaguePath());
    setDetail(payload);
    setWaitlistAction(payload.settings.team_size === 2 ? waitlistAction : "withdraw");
    setTeamLeagueRows((current) => {
      const without = current.filter((row) => row.league_name !== leagueName);
      return [...without, payload.settings].sort((a, b) => a.league_name.localeCompare(b.league_name));
    });
  }

  async function saveSettings(draft: SettingsDraft, confirmationText: string): Promise<ActionCompletion> {
    if (!leagueName) {
      const error = new Error("Select a league first.");
      setMessage(error.message);
      throw error;
    }
    const playoffCount = draft.playoffTeamCount ? Number(draft.playoffTeamCount) : null;
    const teamSize = Number(draft.teamSize);
    const maxAlternates = Number(draft.maxAlternates);
    const mixedRequiredMen = Number(draft.mixedRequiredMen);
    const mixedRequiredWomen = Number(draft.mixedRequiredWomen);
    if (![2, 3, 4].includes(teamSize)) {
      const error = new Error("Primary roster size must be 2, 3, or 4 players.");
      setMessage(error.message);
      throw error;
    }
    if (!Number.isInteger(maxAlternates) || maxAlternates < 0 || maxAlternates > 4) {
      const error = new Error("Maximum alternates must be a whole number from 0 to 4.");
      setMessage(error.message);
      throw error;
    }
    if (draft.teamCategory === "mixed" && (
      !Number.isInteger(mixedRequiredMen)
      || !Number.isInteger(mixedRequiredWomen)
      || mixedRequiredMen < 1
      || mixedRequiredWomen < 1
      || mixedRequiredMen + mixedRequiredWomen !== teamSize
    )) {
      const error = new Error("Mixed roster counts must each be at least one and total the primary roster size.");
      setMessage(error.message);
      throw error;
    }
    if (draft.substitutePoolEnabled && !draft.allowSubstitutes) {
      const error = new Error("Enable substitutes before enabling the shared substitute pool.");
      setMessage(error.message);
      throw error;
    }
    if (draft.playoffFormat === "all_team_single_elimination" && (!Number.isInteger(playoffCount) || Number(playoffCount) < 2 || Number(playoffCount) > 128)) {
      const error = new Error("Playoff team count must be a whole number from 2 to 128.");
      setMessage(error.message);
      throw error;
    }
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<WriteResponse>(teamLeaguePath("/settings"), {
        method: "PUT",
        body: JSON.stringify({
          settings: {
            registration_open: draft.registrationOpen,
            team_size: teamSize,
            team_category: draft.teamCategory,
            max_alternates: maxAlternates,
            substitute_pool_enabled: draft.substitutePoolEnabled,
            mixed_required_men: mixedRequiredMen,
            mixed_required_women: mixedRequiredWomen,
            allow_substitutes: draft.allowSubstitutes,
            playoff_format: draft.playoffFormat,
            playoff_team_count: playoffCount,
            start_date: draft.startDate || null,
            weekday: Number(draft.weekday),
            start_time: draft.startTime,
            timezone: draft.timezone,
            venue: draft.venue || null,
            registration_closes_at: draft.registrationClosesAt ? new Date(draft.registrationClosesAt).toISOString() : null
          },
          expected_settings_version: Number(detail?.settings.settings_version || 0),
          idempotency_key: settingsKey,
          confirmation_text: confirmationText,
          source: "next_team_league_settings_page"
        })
      });
      setSettingsKey(operationKey("team-settings"));
      const refreshWarning = await refreshAfterConfirmedWrite(refreshDetail);
      const successMessage = `Team eligibility, registration, substitute, schedule, and playoff settings were saved together.${refreshWarning}`;
      setMessage(successMessage);
      return actionSuccess("Team league setup saved", successMessage);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save team league setup.");
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function buildPreview() {
    if (!leagueName) return;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<SchedulePreview>(teamLeaguePath(`/schedule-preview/${phase}`));
      setPreview(payload);
      setScheduleKey(operationKey(`team-schedule-${phase}`));
      setMessage(`Previewed ${payload.proposed_fixtures.length} ${phase} fixture(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to preview the schedule.");
    } finally {
      setBusy(false);
    }
  }

  async function commitSchedule(confirmationText: string): Promise<ActionCompletion> {
    if (!preview || preview.phase !== phase) {
      const error = new Error("Preview the current schedule phase before publishing it.");
      setMessage(error.message);
      throw error;
    }
    setBusy(true);
    setMessage(null);
    try {
      await requestTeamWrite<WriteResponse>("/schedule", {
        phase,
        fixtures: preview.proposed_fixtures,
        expected_schedule_version: preview.expected_schedule_version,
        expected_standings_version: preview.expected_standings_version,
        expected_roster_version: preview.expected_roster_version,
        confirmed_roster_fingerprint: preview.confirmed_roster_fingerprint,
        preview_fingerprint: preview.preview_fingerprint,
        idempotency_key: scheduleKey,
        confirmation_text: confirmationText,
        source: "next_team_league_schedule_page"
      });
      setPreview(null);
      setScheduleKey(operationKey(`team-schedule-${phase}`));
      const refreshWarning = await refreshAfterConfirmedWrite(refreshDetail);
      const successMessage = `${phase === "regular" ? "Regular-season schedule" : "Playoff bracket"} published.${refreshWarning}`;
      setMessage(successMessage);
      return actionSuccess(phase === "regular" ? "Schedule published" : "Playoff bracket published", successMessage);
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Unable to publish the schedule."} The same request key is retained for retry.`);
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function applyWaitlist(confirmationText: string): Promise<ActionCompletion> {
    if ((waitlistAction === "pair" && waitlistIds.length !== 2) || (waitlistAction === "withdraw" && !waitlistIds.length)) {
      const error = new Error(waitlistAction === "pair" ? "Select exactly two waiting players." : "Select at least one waiting player.");
      setMessage(error.message);
      throw error;
    }
    setBusy(true);
    setMessage(null);
    try {
      await requestTeamWrite<WriteResponse>("/waitlist-actions", {
        action: waitlistAction,
        waitlist_ids: waitlistIds,
        team_name: waitlistTeamName,
        idempotency_key: waitlistKey,
        confirmation_text: confirmationText,
        source: "next_team_league_waitlist_page"
      });
      setWaitlistIds([]);
      setWaitlistTeamName("");
      setWaitlistKey(operationKey("team-waitlist"));
      const refreshWarning = await refreshAfterConfirmedWrite(refreshDetail);
      const successMessage = `${waitlistAction === "pair" ? "Waitlisted players paired." : "Waitlist entries withdrawn."}${refreshWarning}`;
      setMessage(successMessage);
      return actionSuccess(waitlistAction === "pair" ? "Waitlisted players paired" : "Waitlist entries withdrawn", successMessage);
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Unable to update the waitlist."} The same request key is retained for retry.`);
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function createTeam(confirmationText: string): Promise<ActionCompletion> {
    if (!detail || !newTeamName.trim() || !newCaptainId || !newCaptainEmail.trim()) {
      const error = new Error("Enter a team name, captain, and captain email.");
      setMessage(error.message);
      throw error;
    }
    setBusy(true);
    setMessage(null);
    try {
      await requestTeamWrite<WriteResponse>("/teams", {
        team_name: newTeamName,
        captain_player_id: Number(newCaptainId),
        captain_contact_email: newCaptainEmail,
        initial_primary_player_id: newInitialPrimaryId ? Number(newInitialPrimaryId) : null,
        initial_primary_contact_email: newInitialPrimaryEmail,
        expected_roster_version: Number(detail.settings.roster_version || 0),
        idempotency_key: createTeamKey,
        confirmation_text: confirmationText,
        source: "next_team_league_create_team_page"
      });
      setNewTeamName("");
      setNewCaptainId("");
      setNewCaptainEmail("");
      setNewInitialPrimaryId("");
      setNewInitialPrimaryEmail("");
      setCreateTeamKey(operationKey("team-create"));
      const refreshWarning = await refreshAfterConfirmedWrite(refreshDetail);
      const successMessage = `The forming team was created.${refreshWarning}`;
      setMessage(successMessage);
      return actionSuccess("Team created", successMessage);
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Unable to create the team."} The same request key is retained for retry.`);
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function addRosterMember(confirmationText: string): Promise<ActionCompletion> {
    if (!detail || !rosterTeamId || !rosterPlayerId) {
      const error = new Error("Choose a team and player before updating the roster.");
      setMessage(error.message);
      throw error;
    }
    setBusy(true);
    setMessage(null);
    try {
      await requestTeamWrite<WriteResponse>("/roster-actions", {
        action: "add_member",
        team_id: rosterTeamId,
        player_id: Number(rosterPlayerId),
        member_role: rosterRole,
        member_status: rosterStatus,
        expected_roster_version: Number(detail.settings.roster_version || 0),
        idempotency_key: rosterKey,
        confirmation_text: confirmationText,
        source: "next_team_league_roster_page"
      });
      setRosterPlayerId("");
      setRosterKey(operationKey("team-roster"));
      const refreshWarning = await refreshAfterConfirmedWrite(refreshDetail);
      const successMessage = `The assigned team roster was updated.${refreshWarning}`;
      setMessage(successMessage);
      return actionSuccess("Team roster updated", successMessage);
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Unable to update the roster."} The same request key is retained for retry.`);
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function removeRosterMember(teamId: string, playerId: number, confirmationText: string): Promise<ActionCompletion> {
    if (!detail) throw new Error("Reload the team league before updating its roster.");
    const key = operationKey("team-roster-remove");
    setBusy(true);
    setMessage(null);
    try {
      await requestTeamWrite<WriteResponse>("/roster-actions", {
        action: "remove_member",
        team_id: teamId,
        player_id: playerId,
        member_status: "active",
        expected_roster_version: Number(detail.settings.roster_version || 0),
        idempotency_key: key,
        confirmation_text: confirmationText,
        source: "next_team_league_roster_page"
      });
      const refreshWarning = await refreshAfterConfirmedWrite(refreshDetail);
      const successMessage = `The player was removed from the assigned roster.${refreshWarning}`;
      setMessage(successMessage);
      return actionSuccess("Roster member removed", successMessage);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to remove the roster member.");
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function updateSubstitutePool(confirmationText: string): Promise<ActionCompletion> {
    if (!detail || !poolPlayerId) {
      const error = new Error("Choose a player before updating the substitute pool.");
      setMessage(error.message);
      throw error;
    }
    setBusy(true);
    setMessage(null);
    try {
      await requestTeamWrite<WriteResponse>("/roster-actions", {
        action: "set_pool",
        player_id: Number(poolPlayerId),
        member_status: poolStatus,
        note: poolNote,
        expected_roster_version: Number(detail.settings.roster_version || 0),
        idempotency_key: poolKey,
        confirmation_text: confirmationText,
        source: "next_team_league_substitute_pool_page"
      });
      setPoolPlayerId("");
      setPoolNote("");
      setPoolKey(operationKey("team-substitute-pool"));
      const refreshWarning = await refreshAfterConfirmedWrite(refreshDetail);
      const successMessage = `The shared substitute pool was updated.${refreshWarning}`;
      setMessage(successMessage);
      return actionSuccess("Substitute pool updated", successMessage);
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Unable to update the substitute pool."} The same request key is retained for retry.`);
      throw error;
    } finally {
      setBusy(false);
    }
  }

  async function inspectRecovery(operationId = recoveryId) {
    const cleanId = operationId.trim();
    if (!cleanId) {
      setMessage("Choose or enter an operation ID.");
      return;
    }
    setRecoveryId(cleanId);
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<RecoveryEvidence>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/team-leagues/operations/${encodeURIComponent(cleanId)}`
      );
      setRecoveryEvidence(payload);
      setRecoveryResolution(payload.safe_action === "finalize" ? "finalize" : "compensate");
      setMessage(`Loaded recovery evidence. Recommended action: ${payload.safe_action || "review"}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to inspect the operation.");
    } finally {
      setBusy(false);
    }
  }

  async function resolveRecovery(confirmationText: string): Promise<ActionCompletion> {
    if (!recoveryId || recoveryNote.trim().length < 5) {
      const error = new Error("Add a recovery note of at least five characters.");
      setMessage(error.message);
      throw error;
    }
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<WriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/team-leagues/operations/${encodeURIComponent(recoveryId)}/resolve`,
        {
          method: "POST",
          body: JSON.stringify({
            resolution: recoveryResolution,
            recovery_note: recoveryNote,
            confirmation_text: confirmationText,
            source: "next_team_league_recovery_page"
          })
        }
      );
      setRecoveryEvidence(null);
      setRecoveryId("");
      setRecoveryNote("");
      const refreshWarning = await refreshAfterConfirmedWrite(refreshDetail);
      const successMessage = `The interrupted operation was resolved and verified against canonical evidence.${refreshWarning}`;
      setMessage(successMessage);
      return actionSuccess("Recovery completed", successMessage);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to resolve the operation.");
      throw error;
    } finally {
      setBusy(false);
    }
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadLeagues);

  const selectedSettings = detail?.settings || teamLeagueRows.find((row) => row.league_name === leagueName);
  const waiting = (detail?.waitlist || []).filter((row) => row.status === "waiting");

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Choose a league</h2>
        <p style={{ color: "#475569" }}>Signed in as {adminSessionLabel(session)}. Payment remains offline; team-league writes remain staging-only.</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#b91c1c" }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p><Link href="/admin/login">Open admin login</Link></p> : null}
        <div style={gridStyle}>
          <label><strong>League</strong><br /><select value={leagueName} onChange={(event) => void selectLeague(event.target.value)} disabled={busy || !accessToken} style={inputStyle}><option value="">Select a league</option>{leagueNames.map((name) => <option key={name} value={name}>{name}{teamLeagueRows.some((row) => row.league_name === name) ? " · team setup saved" : " · not set up"}</option>)}</select></label>
          <button type="button" onClick={() => void loadLeagues()} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Working…" : "Refresh leagues"}</button>
        </div>
        {message ? <p role="status" style={{ color: /unable|error|required|stale|retry|recovery/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>

      {leagueName ? <TeamSettingsForm key={`${leagueName}:${selectedSettings?.settings_version || 0}`} settings={selectedSettings} busy={busy} onDirty={() => setSettingsKey(operationKey("team-settings"))} onSave={saveSettings} /> : null}

      {detail ? (
        <>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Teams and standings</h2>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "680px" }}>
                <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Team</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Players</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Registration</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Standing</th></tr></thead>
                <tbody>{detail.teams.map((team) => {
                  const standing = detail.standings.find((row) => row.team_id === team.id);
                  const members = (team.members || []).filter((member) => ["invited", "active"].includes(member.status));
                  return <tr key={team.id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><strong>{team.team_name}</strong></td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{members.map((member) => <div key={member.id}>{playerName(detail, member.player_id)} · {member.role}{member.status === "invited" ? " · invited" : ""}{member.role !== "captain" ? <> · <ConfirmAction triggerLabel="Remove" title={`Remove ${playerName(detail, member.player_id)} from ${team.team_name}?`} description="The team becomes incomplete if it no longer has the configured number of active primary players." confirmLabel="Yes, remove player" confirmationText="UPDATE TEAM ROSTER" tone="danger" disabled={busy} busy={busy} onConfirm={(confirmationText) => removeRosterMember(team.id, member.player_id, confirmationText)} /></> : null}</div>)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{team.status}{team.roster_complete ? " · roster ready" : ` · needs ${detail.settings.team_size || 2} active primary players`}{team.invitation_delivery_status ? ` · ${team.invitation_delivery_status}` : ""}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{standing ? `#${standing.rank} · ${standing.wins}-${standing.losses} · ${standing.point_differential && standing.point_differential > 0 ? "+" : ""}${standing.point_differential || 0}` : "—"}</td></tr>;
                })}</tbody>
              </table>
            </div>
            {!detail.teams.length ? <p style={{ color: "#64748b" }}>No teams have registered yet.</p> : null}
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Assigned rosters and substitute pool</h2>
            <p style={{ color: "#475569" }}>Primary players and alternates stay assigned for the season. Match-day replacements must come from the shared pool and still satisfy the league&apos;s lineup eligibility.</p>
            <h3>Create a forming team</h3>
            <p style={{ color: "#475569" }}>Start with a captain and optionally one initial primary. Add the remaining primary players below; the team becomes schedule-ready only when all {detail.settings.team_size || 2} primary slots are active.</p>
            <div style={gridStyle}>
              <label><strong>Team name</strong><br /><input value={newTeamName} onChange={(event) => { setNewTeamName(event.target.value); setCreateTeamKey(operationKey("team-create")); }} maxLength={120} style={inputStyle} /></label>
              <label><strong>Captain</strong><br /><select value={newCaptainId} onChange={(event) => { setNewCaptainId(event.target.value); if (newInitialPrimaryId === event.target.value) setNewInitialPrimaryId(""); setCreateTeamKey(operationKey("team-create")); }} style={inputStyle}><option value="">Choose an active player</option>{detail.players.filter((player) => player.active !== false).map((player) => <option key={player.id} value={player.id}>{player.name}</option>)}</select></label>
              <label><strong>Captain email</strong><br /><input type="email" value={newCaptainEmail} onChange={(event) => { setNewCaptainEmail(event.target.value); setCreateTeamKey(operationKey("team-create")); }} style={inputStyle} /></label>
              <label><strong>Initial primary (optional)</strong><br /><select value={newInitialPrimaryId} onChange={(event) => { setNewInitialPrimaryId(event.target.value); setCreateTeamKey(operationKey("team-create")); }} style={inputStyle}><option value="">Add later</option>{detail.players.filter((player) => player.active !== false && String(player.id) !== newCaptainId).map((player) => <option key={player.id} value={player.id}>{player.name}</option>)}</select></label>
              {newInitialPrimaryId ? <label><strong>Initial primary email (optional)</strong><br /><input type="email" value={newInitialPrimaryEmail} onChange={(event) => { setNewInitialPrimaryEmail(event.target.value); setCreateTeamKey(operationKey("team-create")); }} style={inputStyle} /></label> : null}
            </div>
            <p><ConfirmAction triggerLabel={busy ? "Creating…" : "Create forming team"} title="Create this forming team?" description="The team starts incomplete unless its active primary roster already matches the configured size. Player uniqueness, substitute-pool exclusion, and eligibility are rechecked atomically." confirmLabel="Yes, create team" confirmationText="CREATE TEAM" disabled={busy || !newTeamName.trim() || !newCaptainId || !newCaptainEmail.trim()} busy={busy} onConfirm={createTeam} /></p>
            <h3>Assign a team member</h3>
            <div style={gridStyle}>
              <label><strong>Team</strong><br /><select value={rosterTeamId} onChange={(event) => { setRosterTeamId(event.target.value); setRosterKey(operationKey("team-roster")); }} style={inputStyle}><option value="">Choose a team</option>{detail.teams.filter((team) => ["pending_partner", "confirmed"].includes(team.status)).map((team) => <option key={team.id} value={team.id}>{team.team_name}{team.roster_complete ? " · ready" : " · incomplete"}</option>)}</select></label>
              <label><strong>Player</strong><br /><select value={rosterPlayerId} onChange={(event) => { setRosterPlayerId(event.target.value); setRosterKey(operationKey("team-roster")); }} style={inputStyle}><option value="">Choose an active player</option>{detail.players.filter((player) => player.active !== false).map((player) => <option key={player.id} value={player.id}>{player.name}{player.gender ? ` · ${player.gender}` : ""}</option>)}</select></label>
              <label><strong>Season role</strong><br /><select value={rosterRole} onChange={(event) => { setRosterRole(event.target.value as "primary" | "alternate"); setRosterKey(operationKey("team-roster")); }} style={inputStyle}><option value="primary">Primary player</option><option value="alternate">Assigned alternate</option></select></label>
              <label><strong>Member status</strong><br /><select value={rosterStatus} onChange={(event) => { setRosterStatus(event.target.value as "active" | "invited"); setRosterKey(operationKey("team-roster")); }} style={inputStyle}><option value="active">Active</option><option value="invited">Invited</option></select></label>
            </div>
            <p><ConfirmAction triggerLabel={busy ? "Saving…" : "Assign player"} title="Update this assigned roster?" description="The server rechecks player uniqueness, roster capacity, category eligibility, and the current roster version before committing." confirmLabel="Yes, update roster" confirmationText="UPDATE TEAM ROSTER" disabled={busy || !rosterTeamId || !rosterPlayerId} busy={busy} onConfirm={addRosterMember} /></p>

            <h3>Shared substitute pool</h3>
            {detail.settings.substitute_pool_enabled ? <>
              <div style={gridStyle}>
                <label><strong>Player</strong><br /><select value={poolPlayerId} onChange={(event) => { setPoolPlayerId(event.target.value); setPoolKey(operationKey("team-substitute-pool")); }} style={inputStyle}><option value="">Choose an active player</option>{detail.players.filter((player) => player.active !== false).map((player) => <option key={player.id} value={player.id}>{player.name}</option>)}</select></label>
                <label><strong>Availability</strong><br /><select value={poolStatus} onChange={(event) => { setPoolStatus(event.target.value as "available" | "unavailable" | "withdrawn"); setPoolKey(operationKey("team-substitute-pool")); }} style={inputStyle}><option value="available">Available</option><option value="unavailable">Unavailable</option><option value="withdrawn">Withdrawn</option></select></label>
                <label><strong>Note</strong><br /><input value={poolNote} onChange={(event) => { setPoolNote(event.target.value); setPoolKey(operationKey("team-substitute-pool")); }} maxLength={500} style={inputStyle} /></label>
              </div>
              <p><ConfirmAction triggerLabel={busy ? "Saving…" : "Update substitute pool"} title="Update this substitute pool entry?" description="Assigned team members cannot also be available in the shared substitute pool." confirmLabel="Yes, update pool" confirmationText="UPDATE SUBSTITUTE POOL" disabled={busy || !poolPlayerId} busy={busy} onConfirm={updateSubstitutePool} /></p>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>{detail.substitute_pool.filter((entry) => entry.status !== "withdrawn").map((entry) => <span key={entry.id} style={insetStyle}><strong>{playerName(detail, entry.player_id)}</strong> · {entry.status}{entry.note ? ` · ${entry.note}` : ""}</span>)}</div>
              {!detail.substitute_pool.some((entry) => entry.status !== "withdrawn") ? <p style={{ color: "#64748b" }}>No players are in the shared substitute pool.</p> : null}
            </> : <p style={{ color: "#64748b" }}>Enable substitutes and the shared substitute pool in season setup before adding pool players.</p>}
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Partner waitlist</h2>
            {detail.settings.team_size !== 2 ? <p style={{ color: "#475569" }}>Two-person auto-pairing is closed for this roster size. Create a forming team above, then assign its remaining primary players. Existing waitlist entries can still be withdrawn safely.</p> : null}
            <div style={gridStyle}>
              <label><strong>Action</strong><br /><select value={waitlistAction} onChange={(event) => { setWaitlistAction(event.target.value as "pair" | "withdraw"); setWaitlistIds([]); setWaitlistKey(operationKey("team-waitlist")); }} style={inputStyle}>{detail.settings.team_size === 2 ? <option value="pair">Pair two players</option> : null}<option value="withdraw">Withdraw selected</option></select></label>
              {waitlistAction === "pair" ? <label><strong>New team name</strong><br /><input value={waitlistTeamName} onChange={(event) => { setWaitlistTeamName(event.target.value); setWaitlistKey(operationKey("team-waitlist")); }} maxLength={120} style={inputStyle} /></label> : null}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.6rem", marginTop: "0.75rem" }}>
              {waiting.map((row) => <label key={row.id} style={insetStyle}><input type="checkbox" checked={waitlistIds.includes(row.id)} onChange={(event) => { setWaitlistIds((current) => event.target.checked ? [...current, row.id] : current.filter((id) => id !== row.id)); setWaitlistKey(operationKey("team-waitlist")); }} /> <strong>{playerName(detail, row.player_id)}</strong>{row.note ? <><br /><small>{row.note}</small></> : null}</label>)}
            </div>
            {!waiting.length ? <p style={{ color: "#64748b" }}>No players are waiting for a partner.</p> : null}
            {waiting.length ? <p><ConfirmAction triggerLabel={busy ? "Working…" : waitlistAction === "pair" ? `Pair ${waitlistIds.length} selected` : `Withdraw ${waitlistIds.length} selected`} title={waitlistAction === "pair" ? "Pair these waitlisted players?" : "Withdraw these waitlist entries?"} description={waitlistAction === "pair" ? "Exactly two selected players become one confirmed fixed-partner team." : "Selected waiting entries are withdrawn without deleting their audit history."} confirmLabel="Yes, update waitlist" confirmationText={waitlistAction === "pair" ? "PAIR WAITLIST PLAYERS" : "WITHDRAW WAITLIST PLAYERS"} tone={waitlistAction === "withdraw" ? "danger" : "default"} disabled={busy || (waitlistAction === "pair" ? waitlistIds.length !== 2 || !waitlistTeamName.trim() : !waitlistIds.length)} busy={busy} onConfirm={applyWaitlist} /></p> : null}
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Schedule and playoffs</h2>
            <p style={{ color: "#475569" }}>Regular season schedules one match per team each week and one meeting against every opponent. Playoffs use the saved format and current standings.</p>
            <div style={gridStyle}>
              <label><strong>Schedule phase</strong><br /><select value={phase} onChange={(event) => { setPhase(event.target.value as "regular" | "playoff"); setPreview(null); setScheduleKey(operationKey("team-schedule")); }} style={inputStyle}><option value="regular">Regular season</option><option value="playoff">Playoffs</option></select></label>
              <button type="button" onClick={() => void buildPreview()} disabled={busy || (phase === "playoff" && detail.settings.playoff_format === "none")} style={ghostButtonStyle}>Preview {phase === "regular" ? "weekly schedule" : "playoff bracket"}</button>
            </div>
            {preview ? <div style={{ marginTop: "0.75rem" }}><div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "580px" }}><thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Round</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Team A</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Team B</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Status</th></tr></thead><tbody>{preview.proposed_fixtures.map((fixture, index) => <tr key={`${fixture.round_number}-${index}`}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{fixture.round_number}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{fixture.team_a_id ? preview.team_names[fixture.team_a_id] || fixture.team_a_id : "TBD"}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{fixture.team_b_id ? preview.team_names[fixture.team_b_id] || fixture.team_b_id : "Bye / TBD"}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{fixture.status}</td></tr>)}</tbody></table></div><p><ConfirmAction triggerLabel={busy ? "Publishing…" : `Publish ${phase === "regular" ? "schedule" : "playoffs"}`} title={`Publish this ${phase === "regular" ? "team league schedule" : "playoff bracket"}?`} description={`Replace the unscored ${phase} fixtures with this exact reviewed preview. Team, roster, schedule, and standings versions must still match.`} confirmLabel="Yes, publish reviewed fixtures" confirmationText={phase === "regular" ? "PUBLISH TEAM LEAGUE SCHEDULE" : "PUBLISH TEAM LEAGUE PLAYOFFS"} disabled={busy} busy={busy} onConfirm={commitSchedule} /></p></div> : null}
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Fixtures and results</h2>
            <div style={{ display: "grid", gap: "0.75rem" }}>
              {detail.fixtures.map((fixture) => <ScoreFixtureCard key={fixture.id} fixture={fixture} detail={detail} requestWrite={requestTeamWrite} onSaved={refreshDetail} />)}
            </div>
            {!detail.fixtures.length ? <p style={{ color: "#64748b" }}>No fixtures are published yet.</p> : null}
          </article>

          <article style={{ ...cardStyle, borderColor: detail.recovery_required ? "#fecaca" : "#e2e8f0" }}>
            <h2 style={{ marginTop: 0 }}>Safe recovery</h2>
            <p style={{ color: "#475569" }}>Inspect durable evidence before finalizing or compensating an interrupted result. A committed canonical match can only be finalized.</p>
            {detail.pending_operations.length ? <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>{detail.pending_operations.map((operation) => <button type="button" key={operation.id} onClick={() => void inspectRecovery(operation.id)} disabled={busy} style={ghostButtonStyle}>{operation.operation_type || "Operation"} · {operation.id.slice(0, 8)}</button>)}</div> : <p style={{ color: "#64748b" }}>No unfinished team-league operations.</p>}
            <div style={{ ...gridStyle, marginTop: "0.75rem" }}>
              <label><strong>Operation ID</strong><br /><input value={recoveryId} onChange={(event) => { setRecoveryId(event.target.value); setRecoveryEvidence(null); }} style={inputStyle} /></label>
              <button type="button" onClick={() => void inspectRecovery()} disabled={busy || !recoveryId.trim()} style={ghostButtonStyle}>Inspect evidence</button>
            </div>
            {recoveryEvidence ? <div style={{ ...insetStyle, marginTop: "0.75rem" }}><p><strong>Safe action:</strong> {recoveryEvidence.safe_action || "review"} · Canonical commit: {recoveryEvidence.stable_direct_match_receipt?.committed ? "yes" : "no"}</p><div style={gridStyle}><label><strong>Resolution</strong><br /><select value={recoveryResolution} onChange={(event) => setRecoveryResolution(event.target.value as "finalize" | "compensate")} style={inputStyle}><option value="finalize">Finalize from committed evidence</option><option value="compensate">Compensate uncommitted operation</option></select></label><label><strong>Recovery note</strong><br /><input value={recoveryNote} onChange={(event) => setRecoveryNote(event.target.value)} minLength={5} maxLength={500} style={inputStyle} /></label></div><p><ConfirmAction triggerLabel={busy ? "Working…" : recoveryResolution === "finalize" ? "Finalize recovery" : "Compensate operation"} title="Resolve this interrupted operation?" description="The server rechecks the canonical match receipt and refuses an unsafe resolution." confirmLabel="Yes, resolve operation" confirmationText={recoveryResolution === "finalize" ? "FINALIZE TEAM LEAGUE RECOVERY" : "COMPENSATE TEAM LEAGUE RECOVERY"} tone={recoveryResolution === "compensate" ? "danger" : "default"} disabled={busy || recoveryNote.trim().length < 5} busy={busy} onConfirm={resolveRecovery} /></p></div> : null}
          </article>
        </>
      ) : leagueName ? <p style={{ color: "#475569" }}>Save team league setup to enable registration, scheduling, results, and recovery.</p> : null}
    </div>
  );
}
