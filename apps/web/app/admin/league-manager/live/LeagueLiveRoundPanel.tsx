"use client";

import { useMemo, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type { AdminLeagueLiveStatusResponse, AdminLeagueManagerDetailResponse, AdminLeagueManagerListResponse, AdminLeagueManagerStatusResponse } from "@/lib/adminLeagueManagerApi";
import type { AdminMatchUploaderRoundRobinCourt, AdminMatchUploaderRoundRobinPreview, AdminMatchUploaderStatusResponse } from "@/lib/adminMatchUploaderApi";
import type { PublicPlayer } from "@/lib/api";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  leagueStatus: AdminLeagueManagerStatusResponse;
  uploaderStatus: AdminMatchUploaderStatusResponse;
  players: PublicPlayer[];
  liveDomainStatus: AdminLeagueLiveStatusResponse;
};

type CourtDraft = { court: string; formatType: string; playerNames: string };
type ScoreDraft = { scoreT1: string; scoreT2: string };
type LeagueLiveSession = {
  id: string;
  league_name: string;
  week_tag: string;
  status: string;
  total_rounds: number;
  current_round: number;
  roster_json?: Array<Record<string, unknown>>;
  current_court_state_json?: Array<Record<string, unknown>>;
  notes?: string | null;
  updated_at: string | null;
};
type LeagueLiveCourt = { round_number: number; court_number: number; format_type: string; player_names: string[]; players_json?: Array<Record<string, unknown>> };
type LeagueLiveRosterRow = { player_id: number; player_name: string; rating?: number | null; status: "active" | "bench"; court_number?: number | null; slot?: number | null; bench_reason?: string | null };
type LeagueMovementRow = { player_id: number; player_name: string; from_court: number; suggested_court: number; to_court: number; wins: number; differential: number; points: number; direction: "up" | "down" | "stay"; overridden: boolean };
type LeagueMovementPayload = { strategy: string; authority: "python_fastapi"; applied: boolean; override_applied: boolean; override_reason?: string | null; next_round: number; rows: LeagueMovementRow[]; next_courts: LeagueLiveCourt[]; operation_key: string };
type LeagueLiveRound = { round_number: number; round_label?: string | null; status: string; submitted_match_count?: number | null; match_date?: string | null; updated_at?: string | null; movement_json?: LeagueMovementPayload | Record<string, unknown> | null };
type LeagueLivePublishOperation = { id: string; round_number: number; status: string; attempt_count: number; published_match_ids?: unknown[]; error_text?: string | null; request_fingerprint?: string; completed_at?: string | null };
type LeagueLiveRatingRow = { player_id: number; player_name: string; rating_before?: number | null; rating_after?: number | null; rating_delta?: number | null; matches_played_before?: number | null; matches_played_after?: number | null };
type LeagueLiveRatingReview = { status: string; requires_replay_review: boolean; published_match_count: number; rows: LeagueLiveRatingRow[]; warnings: string[] };
type LeagueLiveListResponse = { ok: boolean; sessions: LeagueLiveSession[]; count: number };
type LeagueLiveDetailResponse = { ok: boolean; session: LeagueLiveSession; rounds: LeagueLiveRound[]; courts: LeagueLiveCourt[]; publish_operations?: LeagueLivePublishOperation[] };
type LeagueLiveWriteResponse = { ok: boolean; idempotent_replay?: boolean; operation_key?: string; session: LeagueLiveSession; round?: LeagueLiveRound; rounds?: LeagueLiveRound[]; courts?: LeagueLiveCourt[]; movement?: LeagueMovementPayload; bench?: LeagueLiveRosterRow[]; publish_operation?: LeagueLivePublishOperation; rating_review?: LeagueLiveRatingReview; published_match_ids?: unknown[]; warnings?: string[] };
type LeagueLiveRosterSuggestion = { ok: boolean; roster: LeagueLiveRosterRow[]; active_roster: LeagueLiveRosterRow[]; bench: LeagueLiveRosterRow[]; bench_player_ids: number[]; default_bench_player_ids: number[]; bench_override_applied: boolean; court_sizes: number[]; courts: LeagueLiveCourt[]; suggestion_note?: string | null; fingerprint: string };
type LeagueLiveRoundPlan = { ok: boolean; operation_key: string; session_updated_at: string; round_number: number; next_round: number; ready_to_save: boolean; scored_match_count: number; warnings: string[]; movement: LeagueMovementPayload; next_roster: LeagueLiveRosterRow[]; next_courts: LeagueLiveCourt[]; bench: LeagueLiveRosterRow[]; bench_player_ids: number[] };
type LeagueLiveGuestResponse = { ok: boolean; idempotent_replay: boolean; player: { id: number; name: string; rating?: number | null; rating_jupr?: number | null }; guest_operation_id: string };
type LeagueLiveExportResponse = { ok: boolean; kind: string; filename: string; content_type: string; row_count: number; csv_text: string };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

function splitNames(value: string): string[] {
  return String(value || "").replace(/,/g, "\n").split("\n").map((item) => item.trim()).filter(Boolean);
}

function scoreIsValid(score: ScoreDraft): boolean {
  const a = Number(score.scoreT1);
  const b = Number(score.scoreT2);
  return Number.isInteger(a) && Number.isInteger(b) && a >= 0 && b >= 0 && a !== b && a + b > 0;
}

function courtsToPayload(courts: CourtDraft[], currentRound: number) {
  return courts.map((court, index) => ({
    round_number: currentRound,
    court_number: Number(court.court) || index + 1,
    court: Number(court.court) || index + 1,
    format_type: court.formatType,
    player_names: splitNames(court.playerNames)
  }));
}

function courtsFromPersisted(courts: LeagueLiveCourt[], currentRound: number, fallback: Array<Record<string, unknown>> = []): CourtDraft[] {
  const scoped = courts.filter((court) => Number(court.round_number) === Number(currentRound));
  const rows = scoped.length ? scoped : fallback.map((row, index) => ({
    round_number: currentRound,
    court_number: Number(row.court_number || row.court || index + 1),
    format_type: String(row.format_type || row.formatType || "4-player"),
    player_names: Array.isArray(row.player_names) ? row.player_names.map(String) : []
  }));
  if (!rows.length) return [{ court: "1", formatType: "4-player", playerNames: "" }];
  return rows
    .sort((a, b) => Number(a.court_number) - Number(b.court_number))
    .map((row) => ({ court: String(row.court_number), formatType: String(row.format_type || "4-player"), playerNames: (row.player_names || []).join("\n") }));
}

function activeRosterPayload(detail: AdminLeagueManagerDetailResponse | null) {
  return (detail?.roster || []).filter((row) => row.in_league).map((row) => ({
    player_id: row.player_id,
    player_name: row.player_name,
    rating: row.rating,
    rating_jupr: row.rating_jupr,
    wins: row.wins,
    losses: row.losses,
    matches_played: row.matches_played
  }));
}

function movementSummary(movement?: LeagueMovementPayload | Record<string, unknown> | null): string {
  if (!movement || typeof movement !== "object") return "—";
  const rows = Array.isArray((movement as LeagueMovementPayload).rows) ? (movement as LeagueMovementPayload).rows : [];
  const moved = rows.filter((row) => row.direction !== "stay").length;
  return moved ? `${moved} move(s)` : "No movement";
}

export default function LeagueLiveRoundPanel({ apiBase, clubId, leagueStatus, uploaderStatus, players, liveDomainStatus }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [leagues, setLeagues] = useState<string[]>([]);
  const [leagueName, setLeagueName] = useState("");
  const [loadedLeagueName, setLoadedLeagueName] = useState("");
  const [weekTag, setWeekTag] = useState("Week 1");
  const [roundNumber, setRoundNumber] = useState("1");
  const [totalRounds, setTotalRounds] = useState("5");
  const [roundLabel, setRoundLabel] = useState("Round 1");
  const [matchDate, setMatchDate] = useState(todayIso());
  const [detail, setDetail] = useState<AdminLeagueManagerDetailResponse | null>(null);
  const [sessionRoster, setSessionRoster] = useState<LeagueLiveRosterRow[]>([]);
  const [liveSessions, setLiveSessions] = useState<LeagueLiveSession[]>([]);
  const [sessionId, setSessionId] = useState("");
  const [loadedSessionId, setLoadedSessionId] = useState("");
  const [sessionLeagueName, setSessionLeagueName] = useState("");
  const [sessionUpdatedAt, setSessionUpdatedAt] = useState("");
  const [sessionStatus, setSessionStatus] = useState("active");
  const [sessionNotes, setSessionNotes] = useState("");
  const [roundHistory, setRoundHistory] = useState<LeagueLiveRound[]>([]);
  const [publishOperations, setPublishOperations] = useState<LeagueLivePublishOperation[]>([]);
  const [ratingReview, setRatingReview] = useState<LeagueLiveRatingReview | null>(null);
  const [courts, setCourts] = useState<CourtDraft[]>([{ court: "1", formatType: "4-player", playerNames: "" }]);
  const [preview, setPreview] = useState<AdminMatchUploaderRoundRobinPreview | null>(null);
  const [scores, setScores] = useState<Record<string, ScoreDraft>>({});
  const [rosterSuggestion, setRosterSuggestion] = useState<LeagueLiveRosterSuggestion | null>(null);
  const [movementPlan, setMovementPlan] = useState<LeagueLiveRoundPlan | null>(null);
  const [movementPlanStale, setMovementPlanStale] = useState(false);
  const [movementOverrides, setMovementOverrides] = useState<Record<number, string>>({});
  const [overrideReason, setOverrideReason] = useState("");
  const [rosterAction, setRosterAction] = useState<"none" | "add" | "substitute">("none");
  const [incomingPlayerId, setIncomingPlayerId] = useState("");
  const [replacedPlayerId, setReplacedPlayerId] = useState("");
  const [guestPlayers, setGuestPlayers] = useState<Array<{ id: number; name: string; rating?: number | null }>>([]);
  const [guestName, setGuestName] = useState("");
  const [guestJupr, setGuestJupr] = useState("3.5");
  const [guestReason, setGuestReason] = useState("");
  const [compensationReference, setCompensationReference] = useState("");
  const [compensationReason, setCompensationReason] = useState("");
  const [benchOverrideIds, setBenchOverrideIds] = useState<number[]>([]);
  const [benchOverrideReason, setBenchOverrideReason] = useState("");
  const [loadingLeagueName, setLoadingLeagueName] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const currentRound = Math.max(1, Number(roundNumber) || 1);
  const safeTotalRounds = Math.max(currentRound, Number(totalRounds) || currentRound);
  const playerOptions = useMemo(() => players.map((player) => player.name).filter(Boolean).sort((a, b) => a.localeCompare(b)), [players]);
  const rosterPlayerIds = useMemo(() => new Set(sessionRoster.map((row) => Number(row.player_id))), [sessionRoster]);
  const incomingPlayerOptions = useMemo(
    () => [
      ...players.map((player) => ({ id: Number(player.id), name: player.name, rating: player.rating })),
      ...guestPlayers
    ].filter((player, index, rows) => !rosterPlayerIds.has(Number(player.id)) && rows.findIndex((candidate) => Number(candidate.id) === Number(player.id)) === index).sort((a, b) => a.name.localeCompare(b.name)),
    [players, guestPlayers, rosterPlayerIds]
  );
  const activeSessionRoster = useMemo(() => sessionRoster.filter((row) => row.status === "active"), [sessionRoster]);
  const allPreviewMatches = (preview?.courts || []).flatMap((court) => court.matches || []);
  const validScoreCount = allPreviewMatches.filter((match) => scoreIsValid(scores[match.row_id] || { scoreT1: "", scoreT2: "" })).length;
  const sessionIsCurrentLeague = Boolean(
    loadedSessionId
    && loadedSessionId === sessionId
    && sessionUpdatedAt
    && sessionLeagueName
    && sessionLeagueName === leagueName
    && sessionLeagueName === loadedLeagueName
  );

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using League Live.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function applySession(
    sessionRow: LeagueLiveSession,
    courtsRows: LeagueLiveCourt[] = [],
    rounds: LeagueLiveRound[] = [],
    operations: LeagueLivePublishOperation[] = []
  ) {
    setSessionId(sessionRow.id);
    setLoadedSessionId(sessionRow.id);
    setSessionLeagueName(sessionRow.league_name);
    setSessionUpdatedAt(sessionRow.updated_at || "");
    setLeagueName(sessionRow.league_name);
    setLoadedLeagueName(sessionRow.league_name);
    if (detail?.league.league_name !== sessionRow.league_name) setDetail(null);
    setRosterSuggestion(null);
    setWeekTag(sessionRow.week_tag || "Week 1");
    setSessionStatus(sessionRow.status || "active");
    setTotalRounds(String(sessionRow.total_rounds || 1));
    setRoundNumber(String(sessionRow.current_round || 1));
    setRoundLabel(`Round ${sessionRow.current_round || 1}`);
    setSessionNotes(sessionRow.notes || "");
    setSessionRoster((sessionRow.roster_json || []) as LeagueLiveRosterRow[]);
    setRoundHistory(rounds || []);
    setPublishOperations(operations || []);
    setCourts(courtsFromPersisted(courtsRows, sessionRow.current_round || 1, sessionRow.current_court_state_json || []));
    setPreview(null);
    setScores({});
    setMovementPlan(null);
    setMovementPlanStale(false);
    setMovementOverrides({});
    setRatingReview(null);
    setBenchOverrideIds(((sessionRow.roster_json || []) as LeagueLiveRosterRow[]).filter((row) => row.status === "bench").map((row) => Number(row.player_id)));
  }

  function clearPersistedSessionBinding() {
    setSessionId("");
    setLoadedSessionId("");
    setSessionLeagueName("");
    setSessionUpdatedAt("");
    setSessionStatus("active");
    setSessionNotes("");
    setRoundHistory([]);
    setPublishOperations([]);
    setRatingReview(null);
    setMovementPlan(null);
    setMovementPlanStale(false);
    setMovementOverrides({});
    setCompensationReference("");
    setCompensationReason("");
  }

  function requireCurrentSession(action: string): boolean {
    if (sessionIsCurrentLeague) return true;
    setMessage(`Resume a persisted session for ${loadedLeagueName || leagueName || "the selected league"} before ${action}.`);
    return false;
  }

  function rosterForWrite(): Array<Record<string, unknown>> {
    return sessionRoster.length ? sessionRoster : activeRosterPayload(detail);
  }

  function markPlanStale() {
    if (movementPlan) setMovementPlanStale(true);
  }

  async function fetchRosterSuggestion(
    leagueDetail: AdminLeagueManagerDetailResponse,
    requestedBenchIds: number[] = [],
    requestedBenchReason = ""
  ): Promise<LeagueLiveRosterSuggestion> {
    return requestJson<LeagueLiveRosterSuggestion>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live/roster-suggestion`, {
      method: "POST",
      body: JSON.stringify({
        roster: activeRosterPayload(leagueDetail),
        bench_player_ids: requestedBenchIds,
        bench_override_reason: requestedBenchReason || null,
        round_number: currentRound
      })
    });
  }

  function applyRosterSuggestion(payload: LeagueLiveRosterSuggestion) {
    setRosterSuggestion(payload);
    setSessionRoster(payload.roster || []);
    setBenchOverrideIds(payload.bench_player_ids || []);
    setCourts(courtsFromPersisted(payload.courts || [], currentRound));
    setPreview(null);
    setScores({});
    setMovementPlan(null);
    setMovementPlanStale(false);
  }

  async function requestRosterSuggestion(
    leagueDetail: AdminLeagueManagerDetailResponse,
    requestedBenchIds: number[] = [],
    requestedBenchReason = ""
  ): Promise<LeagueLiveRosterSuggestion> {
    const payload = await fetchRosterSuggestion(leagueDetail, requestedBenchIds, requestedBenchReason);
    applyRosterSuggestion(payload);
    return payload;
  }

  async function loadLeagues() {
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      const names = (payload.leagues || []).map((league) => league.league_name).filter(Boolean);
      setLeagues(names);
      const selectedLeague = names.includes(leagueName) ? leagueName : (names[0] || "");
      setLeagueName(selectedLeague);
      if (selectedLeague) await loadLeagueDetail(selectedLeague);
      else setMessage("No leagues are available.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load leagues.");
    } finally {
      setBusy(false);
    }
  }

  async function loadSessions() {
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions?limit=100`);
      setLiveSessions(payload.sessions || []);
      setMessage(`Loaded ${payload.count ?? payload.sessions?.length ?? 0} persisted live session(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load live sessions.");
    } finally {
      setBusy(false);
    }
  }

  async function loadLeagueDetail(selectedLeague = leagueName) {
    if (!selectedLeague) {
      setMessage("Select a league first.");
      return;
    }
    setBusy(true);
    setLoadingLeagueName(selectedLeague);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}`);
      const suggestion = await fetchRosterSuggestion(payload);
      if (sessionId && (sessionId !== loadedSessionId || sessionLeagueName !== selectedLeague)) clearPersistedSessionBinding();
      setDetail(payload);
      setLoadedLeagueName(selectedLeague);
      applyRosterSuggestion(suggestion);
      setMessage(`Python suggested ${suggestion.courts.length} court(s) and ${suggestion.bench.length} bench player(s). Review before creating the session.`);
    } catch (error) {
      const reason = error instanceof Error ? error.message : "Unable to load league detail.";
      if (loadedLeagueName) setLeagueName(loadedLeagueName);
      setMessage(detail || loadedLeagueName ? `${reason} The previous league roster remains visible and selected.` : reason);
    } finally {
      setLoadingLeagueName(null);
      setBusy(false);
    }
  }

  function selectLeague(selectedLeague: string) {
    setLeagueName(selectedLeague);
    void loadLeagueDetail(selectedLeague);
  }

  async function refreshRosterSuggestion() {
    if (!detail || detail.league.league_name !== leagueName) {
      setMessage("Load a league roster before requesting another Python roster suggestion.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const suggestion = await requestRosterSuggestion(detail, benchOverrideIds, benchOverrideReason);
      setMessage(`Python refreshed ${suggestion.courts.length} court(s); ${suggestion.bench.length} player(s) are benched.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to refresh the Python roster suggestion.");
    } finally {
      setBusy(false);
    }
  }

  async function createSession(confirmationText: string) {
    if (!leagueName) {
      setMessage("Select a league before creating a persisted session.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions`, {
        method: "POST",
        body: JSON.stringify({
          league_name: leagueName,
          week_tag: weekTag,
          total_rounds: safeTotalRounds,
          current_round: currentRound,
          roster: rosterForWrite(),
          courts: courtsToPayload(courts, currentRound),
          bench_player_ids: benchOverrideIds,
          bench_override_reason: benchOverrideReason || null,
          notes: sessionNotes,
          confirmation_text: confirmationText,
          source: "next_league_live_session_create"
        })
      });
      applySession(payload.session, payload.courts || [], []);
      await loadSessions();
      setMessage("Persisted League Live session created. You can now resume it later from this page.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to create persisted session.");
    } finally {
      setBusy(false);
    }
  }

  async function loadSessionDetail(selectedSessionId = sessionId) {
    if (!selectedSessionId) {
      setMessage("Select a persisted session first.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(selectedSessionId)}`);
      applySession(payload.session, payload.courts || [], payload.rounds || [], payload.publish_operations || []);
      setMessage("Persisted League Live session loaded.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load persisted session.");
    } finally {
      setBusy(false);
    }
  }

  async function saveSessionSnapshot(confirmationText: string) {
    if (!requireCurrentSession("saving a snapshot")) return;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(loadedSessionId)}/snapshot`, {
        method: "PATCH",
        body: JSON.stringify({
          status: sessionStatus,
          week_tag: weekTag,
          total_rounds: safeTotalRounds,
          current_round: currentRound,
          roster: rosterForWrite(),
          courts: courtsToPayload(courts, currentRound),
          bench_player_ids: benchOverrideIds,
          bench_override_reason: benchOverrideReason || null,
          notes: sessionNotes,
          expected_updated_at: sessionUpdatedAt,
          confirmation_text: confirmationText,
          source: "next_league_live_session_snapshot"
        })
      });
      applySession(payload.session, payload.courts || [], roundHistory);
      setMessage("League Live session snapshot saved.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save session snapshot.");
    } finally {
      setBusy(false);
    }
  }

  function updateCourt(index: number, patch: Partial<CourtDraft>) {
    setCourts((current) => current.map((row, idx) => idx === index ? { ...row, ...patch } : row));
    setPreview(null);
    setScores({});
    markPlanStale();
  }

  function addCourt() {
    setCourts((current) => [...current, { court: String(current.length + 1), formatType: "4-player", playerNames: "" }]);
    setPreview(null);
    setScores({});
    markPlanStale();
  }

  function removeCourt(index: number) {
    setCourts((current) => current.filter((_, idx) => idx !== index).map((row, idx) => ({ ...row, court: String(idx + 1) })));
    setPreview(null);
    setScores({});
    markPlanStale();
  }

  async function generatePreview() {
    setBusy(true);
    setMessage(null);
    try {
      const courtPayload = courtsToPayload(courts, currentRound).map((court) => ({ court: court.court, format_type: court.format_type, player_names: court.player_names }));
      const payload = await requestJson<AdminMatchUploaderRoundRobinPreview>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/round-robin/preview`, {
        method: "POST",
        body: JSON.stringify({ courts: courtPayload, schedule_mode: "full", source: "next_league_manager_live_preview" })
      });
      setPreview(payload);
      const nextScores: Record<string, ScoreDraft> = {};
      for (const match of (payload.courts || []).flatMap((court) => court.matches || [])) nextScores[match.row_id] = { scoreT1: "", scoreT2: "" };
      setScores(nextScores);
      setMovementPlan(null);
      setMovementPlanStale(false);
      setMovementOverrides({});
      if (payload.missing_players?.length) setMessage(`Missing players: ${payload.missing_players.join(", ")}`);
      else setMessage(`Generated ${payload.match_count || 0} match slot(s). Save the session snapshot before leaving this page.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to generate round preview.");
    } finally {
      setBusy(false);
    }
  }

  function buildScoredMatches(): Array<Record<string, unknown>> {
    return allPreviewMatches
      .filter((match) => scoreIsValid(scores[match.row_id] || { scoreT1: "", scoreT2: "" }))
      .map((match) => ({
        date: matchDate,
        league: leagueName,
        week_tag: weekTag,
        match_type: "League Manager Live",
        court: match.court,
        t1_p1: match.t1_p1,
        t1_p2: match.t1_p2,
        t2_p1: match.t2_p1,
        t2_p2: match.t2_p2,
        score_t1: Number(scores[match.row_id]?.scoreT1 || 0),
        score_t2: Number(scores[match.row_id]?.scoreT2 || 0)
      }));
  }

  function movementOverridePayload(): Array<{ player_id: number; to_court: number }> {
    return Object.entries(movementOverrides)
      .filter(([, toCourt]) => Number(toCourt) > 0)
      .map(([playerId, toCourt]) => ({ player_id: Number(playerId), to_court: Number(toCourt) }));
  }

  function rosterChangePayload(): Record<string, unknown> | null {
    if (rosterAction === "none") return null;
    const player = incomingPlayerOptions.find((row) => Number(row.id) === Number(incomingPlayerId));
    if (!player) throw new Error("Select a valid incoming player before previewing movement.");
    if (rosterAction === "substitute" && !replacedPlayerId) throw new Error("Select the player being replaced before previewing movement.");
    return {
      action: rosterAction,
      replaced_player_id: rosterAction === "substitute" ? Number(replacedPlayerId) : null,
      player: { player_id: Number(player.id), player_name: player.name, rating: player.rating ?? 1200 }
    };
  }

  async function previewPythonMovement() {
    if (!requireCurrentSession("previewing Python movement")) return;
    const matches = buildScoredMatches();
    if (!matches.length) {
      setMessage("Enter at least one valid non-tied score before previewing Python movement.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveRoundPlan>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(loadedSessionId)}/rounds/${encodeURIComponent(String(currentRound))}/plan`, {
        method: "POST",
        body: JSON.stringify({
          expected_updated_at: sessionUpdatedAt,
          matches,
          courts: courtsToPayload(courts, currentRound),
          movement_overrides: movementOverridePayload(),
          override_reason: overrideReason || null,
          roster_change: rosterChangePayload(),
          bench_player_ids: benchOverrideIds,
          bench_override_reason: benchOverrideReason || null
        })
      });
      setMovementPlan(payload);
      setMovementPlanStale(false);
      setMessage(`Python-authoritative plan ${payload.operation_key.slice(0, 12)}… is ready for Round ${payload.next_round}.`);
    } catch (error) {
      setMovementPlanStale(true);
      setMessage(error instanceof Error ? error.message : "Unable to preview Python court movement.");
    } finally {
      setBusy(false);
    }
  }

  async function submitRound(confirmationText: string) {
    if (!requireCurrentSession("submitting official scores")) return;
    const matches = buildScoredMatches();
    if (!matches.length || matches.length !== allPreviewMatches.length) {
      setMessage(`Score every generated match before publishing (${matches.length} of ${allPreviewMatches.length} complete).`);
      return;
    }
    if (!movementPlan || movementPlanStale) {
      setMessage("Preview the current Python movement plan before submitting. Any score, roster, bench, or override change makes the previous plan stale.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const plannedMovement = movementPlan.movement;
      const payload = await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(loadedSessionId)}/rounds/${encodeURIComponent(String(currentRound))}/submit`, {
        method: "POST",
        body: JSON.stringify({
          round_label: roundLabel,
          match_date: matchDate,
          preview: preview || {},
          matches,
          expected_match_count: allPreviewMatches.length,
          movement_overrides: movementOverridePayload(),
          override_reason: overrideReason || null,
          roster_change: rosterChangePayload(),
          bench_player_ids: benchOverrideIds,
          bench_override_reason: benchOverrideReason || null,
          expected_updated_at: sessionUpdatedAt,
          expected_operation_key: movementPlan.operation_key,
          idempotency_key: movementPlan.operation_key,
          courts: courtsToPayload(courts, currentRound),
          confirmation_text: confirmationText,
          source: "next_league_live_round_submit"
        })
      });
      if (payload.session) {
        applySession(payload.session, payload.courts || [], payload.rounds || [...roundHistory, ...(payload.round ? [payload.round] : [])]);
        setRoundNumber(String(payload.session.current_round || currentRound));
        setRoundLabel(`Round ${payload.session.current_round || currentRound}`);
      }
      setPreview(null);
      setScores({});
      if (loadedSessionId) await loadSessionDetail(loadedSessionId);
      setRatingReview(payload.rating_review || null);
      const movementText = plannedMovement?.applied ? ` Applied ${plannedMovement.rows.filter((row) => row.direction !== "stay").length} court movement(s) for the next round.` : " No court movement was required.";
      setMessage(`${payload.idempotent_replay ? "Reconciled" : "Published"} ${payload.published_match_ids?.length ?? matches.length} league match(es) through one durable Python operation.${movementText}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to submit league round.");
    } finally {
      setBusy(false);
    }
  }

  async function createGuest(confirmationText: string) {
    if (!requireCurrentSession("creating a guest")) return;
    setBusy(true);
    setMessage(null);
    try {
      const idempotencyKey = `guest:${loadedSessionId}:${guestName.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 60)}`;
      const payload = await requestJson<LeagueLiveGuestResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(loadedSessionId)}/guests`, {
        method: "POST",
        body: JSON.stringify({
          guest_name: guestName,
          starting_jupr: Number(guestJupr),
          reason: guestReason,
          expected_updated_at: sessionUpdatedAt,
          idempotency_key: idempotencyKey,
          confirmation_text: confirmationText,
          source: "next_league_live_guest_create"
        })
      });
      const guest = { id: Number(payload.player.id), name: payload.player.name, rating: payload.player.rating };
      setGuestPlayers((current) => current.some((row) => row.id === guest.id) ? current : [...current, guest]);
      setIncomingPlayerId(String(guest.id));
      setGuestName("");
      setGuestReason("");
      markPlanStale();
      setMessage(`${payload.idempotent_replay ? "Recovered" : "Created"} guest ${guest.name}. Select add or substitute, then preview Python movement again.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to create League Live guest.");
    } finally {
      setBusy(false);
    }
  }

  async function reconcileRound(round: number, confirmationText: string) {
    if (!requireCurrentSession("reconciling a round")) return;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(loadedSessionId)}/rounds/${encodeURIComponent(String(round))}/reconcile`, {
        method: "POST",
        body: JSON.stringify({ confirmation_text: confirmationText, source: "next_league_live_round_reconcile" })
      });
      await loadSessionDetail(loadedSessionId);
      setRatingReview(payload.rating_review || null);
      setMessage(`Round ${round} publish and League Live snapshot are reconciled. No match was republished.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to reconcile League Live round.");
    } finally {
      setBusy(false);
    }
  }

  async function verifyCompensation(round: number, confirmationText: string) {
    if (!requireCurrentSession("verifying compensation")) return;
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(loadedSessionId)}/rounds/${encodeURIComponent(String(round))}/compensate`, {
        method: "POST",
        body: JSON.stringify({
          recovery_reference: compensationReference,
          reason: compensationReason,
          confirmation_text: confirmationText,
          source: "next_league_live_round_compensate"
        })
      });
      setCompensationReference("");
      setCompensationReason("");
      await loadSessionDetail(loadedSessionId);
      setMessage(`Round ${round} recovery is recorded as compensated. No active deterministic match context remained.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to verify League Live compensation.");
    } finally {
      setBusy(false);
    }
  }

  async function downloadExport(kind: "matches" | "ratings" | "roster" | "rounds") {
    if (!requireCurrentSession("exporting session data")) return;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveExportResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(loadedSessionId)}/export?kind=${encodeURIComponent(kind)}`);
      const blob = new Blob([payload.csv_text], { type: payload.content_type });
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = payload.filename;
      anchor.click();
      URL.revokeObjectURL(href);
      setMessage(`Exported ${payload.row_count} ${kind} row(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : `Unable to export ${kind}.`);
    } finally {
      setBusy(false);
    }
  }

  if (!leagueStatus.enabled || !uploaderStatus.enabled || !liveDomainStatus.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>League Live is disabled</h2>
        <p style={{ color: "#475569" }}>League Live fails closed until League Manager, Match Uploader, the Python Live domain flag, and FastAPI service-role storage are all ready. Keep using the Streamlit League Manager fallback meanwhile.</p>
        <ul style={{ color: "#475569" }}>
          <li>League Manager: {leagueStatus.status}</li>
          <li>Match Uploader: {uploaderStatus.status}</li>
          <li>Python Live domain: {liveDomainStatus.status}</li>
          <li>Service role: {liveDomainStatus.service_role_configured ? "configured on FastAPI" : "not configured"}</li>
        </ul>
        <p style={{ color: "#475569" }}>Required staging flag: <code>JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN=true</code></p>
      </article>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2>
        <p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>
        <p style={{ color: "#166534" }}><strong>Movement authority:</strong> {liveDomainStatus.movement_authority === "python_fastapi" ? "Python / FastAPI" : liveDomainStatus.movement_authority || "unavailable"}. The browser displays plans but never ranks players.</p>
        <p style={{ color: liveDomainStatus.submit_enabled ? "#166534" : "#92400e" }}><strong>Publish authority:</strong> {liveDomainStatus.submit_enabled ? "Python / FastAPI durable all-match operation" : "guarded off; staging must apply both League Live migrations and enable JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT"}.</p>
        <p style={{ color: "#475569" }}><strong>Recovery:</strong> Streamlit League Manager remains the fallback until this staging workflow is manually accepted.</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>1. League and persisted session</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>League<br /><select value={leagueName} onChange={(event) => selectLeague(event.target.value)} disabled={busy} style={inputStyle}>{leagues.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
          <label>Week<br /><input value={weekTag} onChange={(event) => setWeekTag(event.target.value)} disabled={busy} style={inputStyle} /></label>
          <label>Round #<br /><input value={roundNumber} onChange={(event) => { setRoundNumber(event.target.value); setRoundLabel(`Round ${event.target.value || 1}`); }} disabled={busy} style={inputStyle} /></label>
          <label>Total rounds<br /><input value={totalRounds} onChange={(event) => setTotalRounds(event.target.value)} disabled={busy} style={inputStyle} /></label>
          <label>Date<br /><input type="date" value={matchDate} onChange={(event) => setMatchDate(event.target.value)} disabled={busy} style={inputStyle} /></label>
          <button type="button" onClick={loadLeagues} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Loading…" : "Load leagues"}</button>
          <button type="button" onClick={() => void loadLeagueDetail()} disabled={busy || !leagueName} style={buttonStyle}>{loadingLeagueName ? "Loading roster…" : "Reload roster"}</button>
        </div>
        {loadingLeagueName ? <p role="status" style={{ color: "#475569" }}>Loading {loadingLeagueName}. The current league roster will remain visible until the replacement is ready.</p> : null}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "0.75rem", marginTop: "0.75rem", alignItems: "end" }}>
          <label>Existing sessions<br />
            <select value={sessionId} onChange={(event) => setSessionId(event.target.value)} disabled={busy} style={inputStyle}>
              <option value="">Select session…</option>
              {liveSessions.map((row) => <option key={row.id} value={row.id}>{row.league_name} · {row.week_tag} · R{row.current_round}/{row.total_rounds} · {row.status}</option>)}
            </select>
          </label>
          <button type="button" onClick={loadSessions} disabled={busy || !accessToken} style={ghostButtonStyle}>Load sessions</button>
          <button type="button" onClick={() => void loadSessionDetail()} disabled={busy || !sessionId} style={ghostButtonStyle}>Resume selected</button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", marginTop: "0.75rem", alignItems: "end" }}>
          <label>Session status<br /><select value={sessionStatus} onChange={(event) => setSessionStatus(event.target.value)} disabled={busy} style={inputStyle}><option>active</option><option>paused</option><option>complete</option><option>archived</option></select></label>
          <label>Round label<br /><input value={roundLabel} onChange={(event) => setRoundLabel(event.target.value)} disabled={busy} style={inputStyle} /></label>
          <label>Notes<br /><input value={sessionNotes} onChange={(event) => setSessionNotes(event.target.value)} disabled={busy} style={inputStyle} /></label>
          <ConfirmAction
            triggerLabel="Create persisted session"
            title="Create this League Live session?"
            description="This saves the current league, roster, bench, court, and round settings as a resumable session."
            confirmLabel="Yes, create session"
            confirmationText="CREATE LIVE SESSION"
            disabled={busy || !leagueName}
            busy={busy}
            onConfirm={createSession}
          />
        </div>
        {rosterSuggestion ? (
          <section style={{ marginTop: "1rem", padding: "0.75rem", border: "1px solid #dbeafe", borderRadius: "12px", background: "#eff6ff" }}>
            <h3 style={{ marginTop: 0 }}>Python roster and bench suggestion</h3>
            <p style={{ color: "#475569" }}>Uncheck or check bench assignments, explain any non-default choice, then ask Python to validate and rebuild the courts.</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.35rem" }}>
              {rosterSuggestion.roster.map((row) => (
                <label key={row.player_id} style={{ display: "flex", gap: "0.45rem", alignItems: "center" }}>
                  <input
                    type="checkbox"
                    checked={benchOverrideIds.includes(Number(row.player_id))}
                    disabled={busy}
                    onChange={(event) => {
                      setBenchOverrideIds((current) => event.target.checked ? [...new Set([...current, Number(row.player_id)])] : current.filter((id) => id !== Number(row.player_id)));
                      markPlanStale();
                    }}
                  />
                  Bench {row.player_name}{row.court_number ? ` (currently Court ${row.court_number})` : ""}
                </label>
              ))}
            </div>
            <div data-responsive-bench-controls style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 240px), 1fr))", gap: "0.75rem", alignItems: "end", marginTop: "0.75rem" }}>
              <label>Bench override reason<br /><input value={benchOverrideReason} onChange={(event) => { setBenchOverrideReason(event.target.value); markPlanStale(); }} disabled={busy} placeholder="Required when changing Python's default bench" style={inputStyle} /></label>
              <button type="button" onClick={refreshRosterSuggestion} disabled={busy} style={ghostButtonStyle}>Refresh Python roster suggestion</button>
            </div>
            <small style={{ color: "#64748b" }}>Roster fingerprint: {rosterSuggestion.fingerprint.slice(0, 16)}…</small>
          </section>
        ) : null}
        {sessionIsCurrentLeague ? <p style={{ color: "#475569" }}><strong>Active persisted session:</strong> {loadedSessionId}</p> : <p style={{ color: "#92400e" }}>Create or resume a persisted session for the selected league before submitting official scores.</p>}
        {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("missing") || message.toLowerCase().includes("type") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>2. Courts</h2>
        <p style={{ color: "#475569" }}>Roster players are seeded from the selected league. Edit courts, then save a session snapshot so the night can be resumed.</p>
        <datalist id="league-live-players">{playerOptions.map((name) => <option key={name} value={name} />)}</datalist>
        {courts.map((court, index) => (
          <div key={index} style={{ borderTop: index ? "1px solid #e2e8f0" : undefined, paddingTop: index ? "0.75rem" : 0, marginTop: index ? "0.75rem" : 0 }}>
            <div data-responsive-court-grid style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 180px), 1fr))", gap: "0.75rem", alignItems: "start" }}>
              <label>Court<br /><input value={court.court} onChange={(event) => updateCourt(index, { court: event.target.value })} disabled={busy} style={inputStyle} /></label>
              <label>Format<br /><select value={court.formatType} onChange={(event) => updateCourt(index, { formatType: event.target.value })} disabled={busy} style={inputStyle}>{(uploaderStatus.round_robin_format_options || ["4-player"]).map((option) => <option key={option} value={option}>{option}</option>)}</select></label>
              <label>Players, one per line<br /><textarea value={court.playerNames} onChange={(event) => updateCourt(index, { playerNames: event.target.value })} disabled={busy} rows={4} style={inputStyle} /></label>
              <button type="button" onClick={() => removeCourt(index)} disabled={busy} style={ghostButtonStyle}>Remove</button>
            </div>
          </div>
        ))}
        <p><button type="button" onClick={addCourt} disabled={busy} style={ghostButtonStyle}>Add court</button> <button type="button" onClick={generatePreview} disabled={busy || !leagueName} style={buttonStyle}>Generate match slots</button></p>
        <ConfirmAction
          triggerLabel="Save session snapshot"
          title="Save this League Live snapshot?"
          description="This persists the current session status, roster, bench, courts, round, and notes."
          confirmLabel="Yes, save snapshot"
          confirmationText="SAVE SESSION"
          disabled={busy || !sessionIsCurrentLeague}
          busy={busy}
          onConfirm={saveSessionSnapshot}
        />
      </article>

      {roundHistory.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Round history</h2>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th align="left">Round</th><th align="left">Label</th><th align="left">Status</th><th align="right">Submitted</th><th align="left">Movement</th><th align="left">Date</th></tr></thead>
              <tbody>{roundHistory.map((round) => <tr key={`${round.round_number}-${round.status}`}><td>{round.round_number}</td><td>{round.round_label || "—"}</td><td>{round.status}</td><td align="right">{round.submitted_match_count ?? 0}</td><td>{movementSummary(round.movement_json)}</td><td>{round.match_date || "—"}</td></tr>)}</tbody>
            </table>
          </div>
        </article>
      ) : null}

      {sessionIsCurrentLeague && liveDomainStatus.submit_enabled ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Publish recovery and exports</h2>
          <p style={{ color: "#475569" }}>A publish operation records intent before scores, verifies every deterministic match context, then reconciles the League Live snapshot. Retry or reconcile never republishes verified matches.</p>
          {publishOperations.length ? (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead><tr><th align="left">Round</th><th align="left">State</th><th align="right">Attempts</th><th align="right">Matches</th><th align="left">Recovery</th></tr></thead>
                <tbody>{publishOperations.map((operation) => (
                  <tr key={operation.id}>
                    <td>{operation.round_number}</td><td>{operation.status}</td><td align="right">{operation.attempt_count}</td><td align="right">{operation.published_match_ids?.length || 0}</td>
                    <td>{operation.status === "completed" ? "Verified" : operation.error_text || "Retry the original publish with the same plan key."}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ) : <p>No publish operations recorded for this session.</p>}
          {publishOperations.some((operation) => operation.status !== "completed") ? (
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.75rem" }}>
              {publishOperations.filter((operation) => operation.status !== "completed").map((operation) => <ConfirmAction
                key={operation.id}
                triggerLabel={`Reconcile R${operation.round_number}`}
                title={`Reconcile League Live round ${operation.round_number}?`}
                description="This checks the durable publish record and reconciles the session snapshot without republishing verified matches."
                confirmLabel="Yes, reconcile round"
                confirmationText="RECONCILE LEAGUE ROUND"
                disabled={busy}
                busy={busy}
                onConfirm={(confirmationText) => reconcileRound(operation.round_number, confirmationText)}
              />)}
            </div>
          ) : null}
          {publishOperations.some((operation) => ["published", "reconciling", "recovery_required"].includes(operation.status)) ? (
            <details style={{ marginTop: "0.75rem" }}>
              <summary>Record completed Match Log / Replay History compensation</summary>
              <p style={{ color: "#92400e" }}>Use this only after recovery removed or soft-excluded every deterministic match context and ratings replay is complete. FastAPI verifies that no active context remains.</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
                <label>Recovery reference<br /><input value={compensationReference} onChange={(event) => setCompensationReference(event.target.value)} disabled={busy} placeholder="Match Log / replay operation ID" style={inputStyle} /></label>
                <label>Reason<br /><input value={compensationReason} onChange={(event) => setCompensationReason(event.target.value)} disabled={busy} placeholder="At least 10 characters" style={inputStyle} /></label>
                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>{publishOperations.filter((operation) => ["published", "reconciling", "recovery_required"].includes(operation.status)).map((operation) => <ConfirmAction
                  key={operation.id}
                  triggerLabel={`Verify R${operation.round_number}`}
                  title={`Verify compensation for round ${operation.round_number}?`}
                  description="Use this only after every deterministic match context was removed or excluded and ratings replay completed."
                  confirmLabel="Yes, verify compensation"
                  confirmationText="VERIFY LEAGUE COMPENSATION"
                  tone="danger"
                  disabled={busy || !sessionIsCurrentLeague || !compensationReference || compensationReason.trim().length < 10}
                  busy={busy}
                  onConfirm={(confirmationText) => verifyCompensation(operation.round_number, confirmationText)}
                />)}</div>
              </div>
            </details>
          ) : null}
          <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            {(["matches", "ratings", "roster", "rounds"] as const).map((kind) => <button key={kind} type="button" onClick={() => downloadExport(kind)} disabled={busy || !sessionIsCurrentLeague} style={ghostButtonStyle}>Export {kind} CSV</button>)}
          </p>
        </article>
      ) : null}

      {ratingReview ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Rating refresh review</h2>
          <p style={{ color: ratingReview.requires_replay_review ? "#92400e" : "#166534" }}>{ratingReview.requires_replay_review ? "Readback is incomplete. Stop and review Match Log / Replay History." : `Verified rating readback for ${ratingReview.rows.length} affected player(s).`}</p>
          <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Player</th><th align="right">Before</th><th align="right">After</th><th align="right">Delta</th><th align="right">Games</th></tr></thead><tbody>{ratingReview.rows.map((row) => <tr key={row.player_id}><td>{row.player_name}</td><td align="right">{row.rating_before ?? "—"}</td><td align="right">{row.rating_after ?? "—"}</td><td align="right">{row.rating_delta == null ? "—" : row.rating_delta.toFixed(1)}</td><td align="right">{row.matches_played_before ?? "—"} → {row.matches_played_after ?? "—"}</td></tr>)}</tbody></table></div>
        </article>
      ) : null}

      {preview?.courts?.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>3. Enter scores</h2>
          <p style={{ color: "#475569" }}>Every generated match must have a valid non-tied score. FastAPI/Python publishes the whole round, recomputes statistics and movement, and reconciles its snapshot under one durable idempotency key.</p>
          <div style={{ display: "grid", gap: "0.75rem" }}>
            {(preview.courts as AdminMatchUploaderRoundRobinCourt[]).map((court) => (
              <section key={court.court} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
                <h3 style={{ marginTop: 0 }}>Court {court.court} · {court.format_type}</h3>
                {(court.matches || []).map((match) => (
                  <div key={match.row_id} style={{ display: "grid", gridTemplateColumns: "1fr 90px 90px", gap: "0.75rem", alignItems: "center", borderTop: "1px solid #f1f5f9", padding: "0.5rem 0" }}>
                    <div><strong>{match.label}</strong><br />{match.t1.map((p) => p.name).join(" / ")} vs {match.t2.map((p) => p.name).join(" / ")}</div>
                    <input value={scores[match.row_id]?.scoreT1 || ""} onChange={(event) => { setScores((current) => ({ ...current, [match.row_id]: { ...(current[match.row_id] || { scoreT1: "", scoreT2: "" }), scoreT1: event.target.value } })); markPlanStale(); }} disabled={busy} placeholder="Team 1" style={inputStyle} />
                    <input value={scores[match.row_id]?.scoreT2 || ""} onChange={(event) => { setScores((current) => ({ ...current, [match.row_id]: { ...(current[match.row_id] || { scoreT1: "", scoreT2: "" }), scoreT2: event.target.value } })); markPlanStale(); }} disabled={busy} placeholder="Team 2" style={inputStyle} />
                  </div>
                ))}
              </section>
            ))}
          </div>
          <p style={{ color: "#475569" }}>Valid scored matches: {validScoreCount} / {allPreviewMatches.length}</p>
          <section style={{ padding: "0.75rem", border: "1px solid #e2e8f0", borderRadius: "12px", marginBottom: "0.75rem" }}>
            <h3 style={{ marginTop: 0 }}>Next-round roster change</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem" }}>
              <label>Action<br /><select value={rosterAction} onChange={(event) => { setRosterAction(event.target.value as "none" | "add" | "substitute"); markPlanStale(); }} disabled={busy} style={inputStyle}><option value="none">No roster change</option><option value="add">Add player</option><option value="substitute">Substitute player</option></select></label>
              {rosterAction !== "none" ? <label>Incoming player<br /><select value={incomingPlayerId} onChange={(event) => { setIncomingPlayerId(event.target.value); markPlanStale(); }} disabled={busy} style={inputStyle}><option value="">Select player…</option>{incomingPlayerOptions.map((player) => <option key={String(player.id)} value={String(player.id)}>{player.name}</option>)}</select></label> : null}
              {rosterAction === "substitute" ? <label>Replace active player<br /><select value={replacedPlayerId} onChange={(event) => { setReplacedPlayerId(event.target.value); markPlanStale(); }} disabled={busy} style={inputStyle}><option value="">Select player…</option>{activeSessionRoster.map((player) => <option key={player.player_id} value={String(player.player_id)}>{player.player_name}</option>)}</select></label> : null}
            </div>
            {liveDomainStatus.submit_enabled ? (
              <details style={{ marginTop: "0.75rem" }}>
                <summary>Create a guest player for this session</summary>
                <p style={{ color: "#475569" }}>Guest creation makes a real club player record so ratings and Match Log recovery remain authoritative. Use Player Editor to retire an abandoned guest.</p>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
                  <label>Guest name<br /><input value={guestName} onChange={(event) => setGuestName(event.target.value)} disabled={busy} style={inputStyle} /></label>
                  <label>Starting JUPR<br /><input value={guestJupr} onChange={(event) => setGuestJupr(event.target.value)} disabled={busy} type="number" min="1" max="7" step="0.1" style={inputStyle} /></label>
                  <label>Operator reason<br /><input value={guestReason} onChange={(event) => setGuestReason(event.target.value)} disabled={busy} placeholder="At least 10 characters" style={inputStyle} /></label>
                  <ConfirmAction
                    triggerLabel="Create guest"
                    title="Create this guest player?"
                    description="This creates a real club player record for authoritative ratings and Match Log recovery."
                    confirmLabel="Yes, create guest"
                    confirmationText="CREATE LIVE GUEST"
                    disabled={busy || !sessionIsCurrentLeague || !guestName.trim() || guestReason.trim().length < 10}
                    busy={busy}
                    onConfirm={createGuest}
                  />
                </div>
              </details>
            ) : null}
            {sessionRoster.length ? (
              <details style={{ marginTop: "0.75rem" }}>
                <summary>Review next-round bench assignments</summary>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.35rem", marginTop: "0.5rem" }}>
                  {sessionRoster.map((row) => <label key={row.player_id}><input type="checkbox" checked={benchOverrideIds.includes(Number(row.player_id))} onChange={(event) => { setBenchOverrideIds((current) => event.target.checked ? [...new Set([...current, Number(row.player_id)])] : current.filter((id) => id !== Number(row.player_id))); markPlanStale(); }} disabled={busy} /> Bench {row.player_name}</label>)}
                </div>
                <label style={{ display: "block", marginTop: "0.5rem" }}>Bench override reason<br /><input value={benchOverrideReason} onChange={(event) => { setBenchOverrideReason(event.target.value); markPlanStale(); }} disabled={busy} placeholder="Required for a non-default bench" style={inputStyle} /></label>
              </details>
            ) : null}
          </section>
          {movementPlan ? (
            <article style={{ ...cardStyle, background: "#f8fafc", marginBottom: "0.75rem" }}>
              <strong>Python next-round movement plan:</strong> {movementPlan.movement.applied ? `${movementPlan.movement.rows.filter((row) => row.direction !== "stay").length} player movement(s)` : "no court movement required"} for Round {movementPlan.movement.next_round}.
              <p style={{ color: movementPlanStale ? "#b91c1c" : "#166534" }}>{movementPlanStale ? "This plan is stale. Preview Python movement again before submitting." : `Verified operation key ${movementPlan.operation_key.slice(0, 16)}…`}</p>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead><tr><th align="left">Player</th><th align="right">Wins</th><th align="right">Diff</th><th align="left">From</th><th align="left">Python</th><th align="left">Final target</th></tr></thead>
                  <tbody>{movementPlan.movement.rows.map((row) => <tr key={row.player_id}><td>{row.player_name}</td><td align="right">{row.wins}</td><td align="right">{row.differential}</td><td>Court {row.from_court}</td><td>Court {row.suggested_court}</td><td><select aria-label={`Final court for ${row.player_name}`} value={movementOverrides[row.player_id] || String(row.to_court)} onChange={(event) => { setMovementOverrides((current) => ({ ...current, [row.player_id]: event.target.value })); setMovementPlanStale(true); }} disabled={busy} style={inputStyle}>{courts.map((court) => <option key={court.court} value={court.court}>Court {court.court}</option>)}</select></td></tr>)}</tbody>
                </table>
              </div>
              <label style={{ display: "block", marginTop: "0.75rem" }}>Manual movement override reason<br /><input value={overrideReason} onChange={(event) => { setOverrideReason(event.target.value); markPlanStale(); }} disabled={busy} placeholder="At least 10 characters when changing a Python target" style={inputStyle} /></label>
            </article>
          ) : null}
          <p><button type="button" onClick={previewPythonMovement} disabled={busy || validScoreCount !== allPreviewMatches.length || !sessionIsCurrentLeague} style={ghostButtonStyle}>{busy ? "Planning…" : "Preview Python movement"}</button></p>
          <ConfirmAction
            triggerLabel="Publish complete league round"
            title="Publish this complete league round?"
            description="This publishes every scored match as official, recalculates ratings and movement, and advances the persisted session."
            confirmLabel="Yes, publish round"
            confirmationText="SUBMIT LEAGUE ROUND"
            tone="danger"
            disabled={busy || !liveDomainStatus.submit_enabled || validScoreCount !== allPreviewMatches.length || !sessionIsCurrentLeague || !movementPlan || movementPlanStale}
            busy={busy}
            onConfirm={submitRound}
          />
        </article>
      ) : null}
    </div>
  );
}
