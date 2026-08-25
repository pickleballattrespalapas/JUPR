"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, actionUncertain, type ActionCompletion } from "@/components/interaction";
import type { AdminLeagueLiveStatusResponse, AdminLeagueManagerDetailResponse, AdminLeagueManagerListResponse, AdminLeagueManagerStatusResponse } from "@/lib/adminLeagueManagerApi";
import type { AdminMatchUploaderCreatePlayersResult, AdminMatchUploaderRoundRobinCourt, AdminMatchUploaderRoundRobinMatch, AdminMatchUploaderRoundRobinPreview, AdminMatchUploaderStatusResponse } from "@/lib/adminMatchUploaderApi";
import type { PublicPlayer } from "@/lib/api";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  selectedLeagueName: string;
  leagueStatus: AdminLeagueManagerStatusResponse;
  uploaderStatus: AdminMatchUploaderStatusResponse;
  players: PublicPlayer[];
  liveDomainStatus: AdminLeagueLiveStatusResponse;
};

type CourtDraft = { court: string; formatType: string; playerNames: string };
type ScoreDraft = { scoreT1: string; scoreT2: string };
type WorkflowStep = 1 | 2 | 3 | 4 | 5 | 6;
type PastedPlayerResolution = {
  inputName: string;
  status: "existing" | "missing" | "ambiguous";
  playerId?: number;
  playerName?: string;
};
type MissingPlayerDraft = { name: string; startingJupr: string };
type MatchStructure = {
  kind: "fixed_games" | "best_of";
  games: number;
  result_counting: "each_game";
  completion: "all_games" | "clinch";
};
type SeriesScoreState = {
  complete: boolean;
  playedGames: number;
  clinchedAt: number | null;
  hasGap: boolean;
  hasScoreAfterClinch: boolean;
};
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
type LeagueLiveRoundPlan = { ok: boolean; operation_key: string; session_updated_at: string; round_number: number; next_round: number; ready_to_save: boolean; scored_match_count: number; warnings: string[]; movement: LeagueMovementPayload; next_roster: LeagueLiveRosterRow[]; next_courts: LeagueLiveCourt[]; bench: LeagueLiveRosterRow[]; bench_player_ids: number[]; match_structure?: MatchStructure };
type LeagueLiveGuestResponse = { ok: boolean; idempotent_replay: boolean; player: { id: number; name: string; rating?: number | null; rating_jupr?: number | null }; guest_operation_id: string };
type LeagueLiveExportResponse = { ok: boolean; kind: string; filename: string; content_type: string; row_count: number; csv_text: string };
type LeagueLiveCreateRecovery = { operationKey: string; status: string; message: string };
type StoredLeagueLiveCreateRecovery = LeagueLiveCreateRecovery & { version: 1 };
type LeagueLiveCreateOperationResponse = { ok: boolean; operation_key: string; status: string; result?: LeagueLiveWriteResponse | null; error?: string | null; recovery_required?: boolean; updated_at?: string | null };
type LeagueLiveCreateReconcileResponse = Partial<LeagueLiveWriteResponse> & { ok: boolean; status?: string; recovery_required?: boolean; error?: string | null };
type LeagueLivePlayerCreateRecovery = { operationKey: string; status: string; message: string; leagueName?: string };
type StoredLeagueLivePlayerCreateRecovery = LeagueLivePlayerCreateRecovery & { version: 1 };
type StoredLeagueLiveRoundDraft = {
  version: 1;
  sessionId: string;
  roundNumber: number;
  sessionUpdatedAt: string;
  courtFingerprint: string;
  workflowStep: WorkflowStep;
  matchDate: string;
  weekTag: string;
  roundLabel: string;
  preview: AdminMatchUploaderRoundRobinPreview;
  scores: Record<string, ScoreDraft>;
  scoreReviewMode: boolean;
  scoresReviewed: boolean;
  movementPlan: LeagueLiveRoundPlan | null;
  movementPlanStale: boolean;
  movementOverrides: Record<number, string>;
  overrideReason: string;
  rosterAction: "none" | "add" | "substitute";
  incomingPlayerId: string;
  replacedPlayerId: string;
  guestPlayers: Array<{ id: number; name: string; rating?: number | null }>;
  benchOverrideIds: number[];
  benchOverrideReason: string;
};
type LeagueLivePlayerOperationResponse = {
  ok: boolean;
  operation_key?: string;
  status: string;
  result?: AdminMatchUploaderCreatePlayersResult | null;
  result_json?: AdminMatchUploaderCreatePlayersResult | null;
  error?: string | null;
  error_text?: string | null;
  recovery_required?: boolean;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", minWidth: 0, boxSizing: "border-box" as const, padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const DEFAULT_MATCH_STRUCTURE: MatchStructure = { kind: "fixed_games", games: 1, result_counting: "each_game", completion: "all_games" };
const RESUMABLE_SESSION_STATUSES = new Set(["setup", "active", "paused"]);
const RETRYABLE_PUBLISH_STATUSES = new Set(["intent", "publishing", "retryable"]);
const RECONCILABLE_PUBLISH_STATUSES = new Set(["published", "reconciling", "recovery_required"]);
const WORKFLOW_STEPS: Array<{ id: WorkflowStep; label: string }> = [
  { id: 1, label: "Setup" },
  { id: 2, label: "Players" },
  { id: 3, label: "Courts and Preview" },
  { id: 4, label: "Score Entry with Review" },
  { id: 5, label: "Movement" },
  { id: 6, label: "Repeat or Finish" }
];

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function todayIso(): string {
  const today = new Date();
  const year = today.getFullYear();
  const month = String(today.getMonth() + 1).padStart(2, "0");
  const day = String(today.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function splitNames(value: string): string[] {
  return String(value || "").replace(/,/g, "\n").split("\n").map((item) => item.trim()).filter(Boolean);
}

function normalizePlayerName(value: string): string {
  return String(value || "").replace(/\s+/gu, " ").trim();
}

function playerNameKey(value: string): string {
  return normalizePlayerName(value).toLocaleLowerCase("en-US");
}

function parsePastedPlayerNames(value: string): string[] {
  return String(value || "")
    .replace(/,/g, "\n")
    .split("\n")
    .map(normalizePlayerName)
    .filter(Boolean);
}

function duplicatePastedPlayerNames(names: string[]): string[] {
  const seen = new Set<string>();
  const duplicates = new Map<string, string>();
  for (const name of names) {
    const key = playerNameKey(name);
    if (seen.has(key)) duplicates.set(key, name);
    seen.add(key);
  }
  return [...duplicates.values()];
}

function sameNumberSet(left: number[], right: number[]): boolean {
  const leftSet = new Set(left.map(Number));
  const rightSet = new Set(right.map(Number));
  return leftSet.size === rightSet.size && [...leftSet].every((value) => rightSet.has(value));
}

function resolvePastedPlayers(names: string[], knownPlayers: PublicPlayer[]): PastedPlayerResolution[] {
  const existingPlayersByName = new Map<string, PublicPlayer[]>();
  for (const player of knownPlayers) {
    const key = playerNameKey(player.name);
    if (!key) continue;
    existingPlayersByName.set(key, [...(existingPlayersByName.get(key) || []), player]);
  }
  return names.map((inputName) => {
    const matches = existingPlayersByName.get(playerNameKey(inputName)) || [];
    if (matches.length > 1) return { inputName, status: "ambiguous" as const, playerName: `${matches.length} club players share this name` };
    if (matches.length === 1) return { inputName, status: "existing" as const, playerId: Number(matches[0].id), playerName: matches[0].name };
    return { inputName, status: "missing" as const };
  });
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    const object = value as Record<string, unknown>;
    return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(object[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

async function reviewedPlayerBatchFingerprint(players: Array<{ name: string; starting_jupr: number }>): Promise<string> {
  const reviewed = {
    players: players.map((player) => ({
      name: normalizePlayerName(player.name),
      starting_jupr: Number(player.starting_jupr).toFixed(4)
    }))
  };
  const digest = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(canonicalJson(reviewed)));
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function scoreIsValid(score: ScoreDraft): boolean {
  const left = score.scoreT1.trim();
  const right = score.scoreT2.trim();
  if (!/^(0|[1-9]\d*)$/.test(left) || !/^(0|[1-9]\d*)$/.test(right)) return false;
  const a = Number(left);
  const b = Number(right);
  return Number.isInteger(a) && Number.isInteger(b) && a >= 0 && b >= 0 && a !== b && a + b > 0;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function matchStructureFromDetail(detail: AdminLeagueManagerDetailResponse | null): MatchStructure {
  const rules = asRecord(detail?.league.rules_config);
  const competition = asRecord(rules.competition);
  const raw = asRecord(competition.match_structure);
  const kind = raw.kind === "best_of" ? "best_of" : "fixed_games";
  const games = Math.max(1, Math.min(9, Math.trunc(Number(raw.games) || 1)));
  if (kind === "best_of" && (games < 3 || games % 2 === 0)) return DEFAULT_MATCH_STRUCTURE;
  return {
    kind,
    games,
    result_counting: "each_game",
    completion: kind === "best_of" ? "clinch" : "all_games"
  };
}

function participationModeFromDetail(detail: AdminLeagueManagerDetailResponse | null): "flex" | "set" {
  const rules = asRecord(detail?.league.rules_config);
  const operation = asRecord(rules.operation);
  return operation.participation_mode === "flex" ? "flex" : "set";
}

function matchStructureLabel(structure: MatchStructure): string {
  if (structure.kind === "best_of") return `Best ${Math.floor(structure.games / 2) + 1} out of ${structure.games}`;
  return structure.games === 1 ? "1 game" : `${structure.games} games`;
}

function scoreKey(rowId: string, gameNumber: number): string {
  return `${rowId}:game:${gameNumber}`;
}

function scoreHasValue(score: ScoreDraft | undefined): boolean {
  return Boolean(score && (score.scoreT1.trim() || score.scoreT2.trim()));
}

function seriesScoreState(rowId: string, scores: Record<string, ScoreDraft>, structure: MatchStructure): SeriesScoreState {
  let teamOneWins = 0;
  let teamTwoWins = 0;
  let playedGames = 0;
  let clinchedAt: number | null = null;
  let hasGap = false;
  let hasScoreAfterClinch = false;
  const winsNeeded = Math.floor(structure.games / 2) + 1;

  for (let gameNumber = 1; gameNumber <= structure.games; gameNumber += 1) {
    const score = scores[scoreKey(rowId, gameNumber)];
    if (clinchedAt != null) {
      if (scoreHasValue(score)) hasScoreAfterClinch = true;
      continue;
    }
    if (!scoreIsValid(score || { scoreT1: "", scoreT2: "" })) {
      if (scoreHasValue(score)) hasGap = true;
      for (let later = gameNumber + 1; later <= structure.games; later += 1) {
        if (scoreHasValue(scores[scoreKey(rowId, later)])) hasGap = true;
      }
      break;
    }
    playedGames += 1;
    if (Number(score?.scoreT1) > Number(score?.scoreT2)) teamOneWins += 1;
    else teamTwoWins += 1;
    if (structure.kind === "best_of" && Math.max(teamOneWins, teamTwoWins) === winsNeeded) {
      clinchedAt = gameNumber;
    }
  }

  return {
    playedGames,
    clinchedAt,
    hasGap,
    hasScoreAfterClinch,
    complete: structure.kind === "fixed_games"
      ? playedGames === structure.games && !hasGap
      : clinchedAt != null && !hasGap && !hasScoreAfterClinch
  };
}

function leagueLiveOperatorMessage(value: unknown, fallback: string): string {
  const raw = value instanceof Error ? value.message : typeof value === "string" ? value : fallback;
  return raw
    .replace(/FastAPI\s*\/\s*Python|Python\s*\/\s*FastAPI/gi, "League Live")
    .replace(/Python-authoritative League Live/gi, "League Live")
    .replace(/\bPython\b/gi, "League Live")
    .replace(/\bFastAPI\b/gi, "the League Live service");
}

class LeagueLiveRequestError extends Error {
  readonly status: number;
  readonly uncertain: boolean;
  readonly operationKey: string | null;

  constructor(message: string, status: number, uncertain = false, operationKey: string | null = null) {
    super(message);
    this.name = "LeagueLiveRequestError";
    this.status = status;
    this.uncertain = uncertain;
    this.operationKey = operationKey;
  }
}

function leagueLiveWriteIsUncertain(error: unknown): boolean {
  return !(error instanceof LeagueLiveRequestError) || error.uncertain;
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

function courtDraftFingerprint(courts: CourtDraft[]): string {
  return canonicalJson(courts.map((court) => ({
    court: court.court.trim(),
    format_type: court.formatType.trim().toLowerCase(),
    player_names: splitNames(court.playerNames).map(playerNameKey)
  })));
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

function activeRosterPayload(detail: AdminLeagueManagerDetailResponse | null, attendeeIds?: Set<number>) {
  return (detail?.roster || []).filter((row) => row.in_league && (!attendeeIds || attendeeIds.has(Number(row.player_id)))).map((row) => ({
    player_id: row.player_id,
    player_name: row.player_name,
    rating: row.rating,
    rating_jupr: row.rating_jupr,
    wins: row.wins,
    losses: row.losses,
    matches_played: row.matches_played
  }));
}

function selectedSessionRosterPayload(
  detail: AdminLeagueManagerDetailResponse | null,
  knownPlayers: PublicPlayer[],
  selectedPlayerIds: number[]
): Array<Record<string, unknown>> {
  const leagueRowsById = new Map((detail?.roster || []).map((row) => [Number(row.player_id), row]));
  const clubPlayersById = new Map(knownPlayers.map((player) => [Number(player.id), player]));
  return selectedPlayerIds.flatMap((playerId) => {
    const leagueRow = leagueRowsById.get(Number(playerId));
    if (leagueRow) {
      return [{
        player_id: Number(leagueRow.player_id),
        player_name: leagueRow.player_name,
        rating: leagueRow.rating,
        rating_jupr: leagueRow.rating_jupr,
        wins: leagueRow.wins,
        losses: leagueRow.losses,
        matches_played: leagueRow.matches_played
      }];
    }
    const player = clubPlayersById.get(Number(playerId));
    if (!player) return [];
    return [{
      player_id: Number(player.id),
      player_name: player.name,
      rating: player.rating,
      rating_jupr: player.rating_jupr,
      wins: player.wins,
      losses: player.losses,
      matches_played: player.matches_played
    }];
  });
}

function playerJuprLabel(player: PublicPlayer): string {
  const jupr = player.rating_jupr ?? (player.rating == null ? null : Number(player.rating) / 400);
  return jupr == null || !Number.isFinite(Number(jupr)) ? "—" : Number(jupr).toFixed(2);
}

function publicPlayersFromBatch(payload: AdminMatchUploaderCreatePlayersResult | null | undefined): PublicPlayer[] {
  return (payload?.players || []).map((player) => ({
    id: Number(player.id),
    name: player.name,
    rating: player.rating,
    rating_jupr: player.rating == null ? null : Number(player.rating) / 400,
    wins: player.wins,
    losses: player.losses,
    matches_played: player.matches_played,
    is_active: player.is_active
  }));
}

function movementSummary(movement?: LeagueMovementPayload | Record<string, unknown> | null): string {
  if (!movement || typeof movement !== "object") return "—";
  const rows = Array.isArray((movement as LeagueMovementPayload).rows) ? (movement as LeagueMovementPayload).rows : [];
  const moved = rows.filter((row) => row.direction !== "stay").length;
  return moved ? `${moved} move(s)` : "No movement";
}

function resumableSessionsForLeague(sessions: LeagueLiveSession[], leagueName: string): LeagueLiveSession[] {
  return sessions.filter((session) => (
    session.league_name === leagueName
    && RESUMABLE_SESSION_STATUSES.has(String(session.status || "").trim().toLowerCase())
  ));
}

function persistedPublishedRoundNumber(
  session: LeagueLiveSession,
  rounds: LeagueLiveRound[],
  operations: LeagueLivePublishOperation[]
): number | null {
  const submittedRounds = rounds
    .filter((round) => round.status === "submitted")
    .map((round) => Number(round.round_number))
    .filter((roundNumber) => Number.isFinite(roundNumber));
  if (session.status === "complete") {
    return submittedRounds.length ? Math.max(...submittedRounds) : Number(session.current_round || 1);
  }
  const currentRoundNumber = Number(session.current_round || 1);
  if (!submittedRounds.includes(currentRoundNumber)) return null;
  const currentRoundOperations = operations.filter((operation) => Number(operation.round_number) === currentRoundNumber);
  if (currentRoundOperations.length && !currentRoundOperations.some((operation) => operation.status === "completed")) return null;
  return currentRoundNumber;
}

export default function LeagueLiveRoundPanel({ apiBase, clubId, selectedLeagueName, leagueStatus, uploaderStatus, players, liveDomainStatus }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [workflowStep, setWorkflowStep] = useState<WorkflowStep>(1);
  const [leagues, setLeagues] = useState<string[]>([]);
  const [leagueName, setLeagueName] = useState(selectedLeagueName);
  const [loadedLeagueName, setLoadedLeagueName] = useState("");
  const [weekTag, setWeekTag] = useState("Week 1");
  const [roundNumber, setRoundNumber] = useState("1");
  const [totalRounds, setTotalRounds] = useState("5");
  const [roundLabel, setRoundLabel] = useState("Round 1");
  const [matchDate, setMatchDate] = useState(todayIso());
  const [detail, setDetail] = useState<AdminLeagueManagerDetailResponse | null>(null);
  const [matchStructure, setMatchStructure] = useState<MatchStructure>(DEFAULT_MATCH_STRUCTURE);
  const [sessionRoster, setSessionRoster] = useState<LeagueLiveRosterRow[]>([]);
  const [attendeePlayerIds, setAttendeePlayerIds] = useState<number[]>([]);
  const [knownPlayers, setKnownPlayers] = useState<PublicPlayer[]>(players);
  const [pastedPlayerText, setPastedPlayerText] = useState("");
  const [pastedPlayerResolutions, setPastedPlayerResolutions] = useState<PastedPlayerResolution[]>([]);
  const [pasteDuplicateNames, setPasteDuplicateNames] = useState<string[]>([]);
  const [pasteResolutionCurrent, setPasteResolutionCurrent] = useState(true);
  const [pastedSelectedPlayerIds, setPastedSelectedPlayerIds] = useState<number[]>([]);
  const [missingPlayerDrafts, setMissingPlayerDrafts] = useState<MissingPlayerDraft[]>([]);
  const [creatingMissingPlayers, setCreatingMissingPlayers] = useState(false);
  const [playerCreateRecovery, setPlayerCreateRecovery] = useState<LeagueLivePlayerCreateRecovery | null>(null);
  const [checkingPlayerCreateRecovery, setCheckingPlayerCreateRecovery] = useState(false);
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
  const [scoreReviewMode, setScoreReviewMode] = useState(false);
  const [scoresReviewed, setScoresReviewed] = useState(false);
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
  const [guestJupr, setGuestJupr] = useState("");
  const [guestReason, setGuestReason] = useState("");
  const [compensationReference, setCompensationReference] = useState("");
  const [compensationReason, setCompensationReason] = useState("");
  const [benchOverrideIds, setBenchOverrideIds] = useState<number[]>([]);
  const [benchOverrideReason, setBenchOverrideReason] = useState("");
  const [loadingLeagueName, setLoadingLeagueName] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [createRecovery, setCreateRecovery] = useState<LeagueLiveCreateRecovery | null>(null);
  const [checkingCreateRecovery, setCheckingCreateRecovery] = useState(false);
  const [roundPublished, setRoundPublished] = useState(false);
  const [lastPublishedRound, setLastPublishedRound] = useState<number | null>(null);
  const createSessionOperationRef = useRef<{ fingerprint: string; key: string } | null>(null);
  const createPlayersOperationRef = useRef<{ fingerprint: string; key: string } | null>(null);
  const leagueListRequest = useLatestRequestGuard(accessToken, clearProtectedLiveWorkspace);
  const leagueDetailRequest = useLatestRequestGuard(accessToken);
  const sessionListRequest = useLatestRequestGuard(accessToken);
  const sessionDetailRequest = useLatestRequestGuard(accessToken);
  const actionRequest = useLatestRequestGuard(accessToken);
  const createRecoveryStorageKey = `jupr-league-live-create-recovery:${clubId}`;
  const playerCreateRecoveryStorageKey = `jupr-league-live-player-create-recovery:${clubId}`;

  function roundDraftStorageKey(targetSessionId: string, targetRound: number): string {
    return `jupr-league-live-round-draft:${clubId}:${targetSessionId}:${targetRound}`;
  }

  function clearStoredRoundDraft(targetSessionId: string, targetRound: number) {
    if (!targetSessionId) return;
    try {
      globalThis.sessionStorage?.removeItem(roundDraftStorageKey(targetSessionId, targetRound));
    } catch {
      // A successful durable publish remains authoritative if local cleanup is blocked.
    }
  }

  useEffect(() => {
    setKnownPlayers((current) => {
      const byId = new Map(current.map((player) => [Number(player.id), player]));
      for (const player of players) byId.set(Number(player.id), player);
      return [...byId.values()].sort((left, right) => left.name.localeCompare(right.name));
    });
  }, [players]);

  useEffect(() => {
    setPlayerCreateRecovery(null);
    try {
      const raw = globalThis.sessionStorage?.getItem(playerCreateRecoveryStorageKey);
      if (!raw) return;
      const stored = JSON.parse(raw) as Partial<StoredLeagueLivePlayerCreateRecovery>;
      if (stored.version === 1 && typeof stored.operationKey === "string" && typeof stored.status === "string" && typeof stored.message === "string") {
        setPlayerCreateRecovery(stored as StoredLeagueLivePlayerCreateRecovery);
      }
    } catch {
      // In-memory recovery remains available when session storage is blocked.
    }
  }, [playerCreateRecoveryStorageKey]);

  useEffect(() => {
    setCreateRecovery(null);
    try {
      const raw = globalThis.sessionStorage?.getItem(createRecoveryStorageKey);
      if (!raw) return;
      const stored = JSON.parse(raw) as Partial<StoredLeagueLiveCreateRecovery>;
      if (stored.version === 1 && typeof stored.operationKey === "string" && typeof stored.status === "string" && typeof stored.message === "string") {
        setCreateRecovery(stored as StoredLeagueLiveCreateRecovery);
      }
    } catch {
      // The in-memory guard remains available when session storage is blocked.
    }
  }, [createRecoveryStorageKey]);

  function retainCreateRecovery(recovery: LeagueLiveCreateRecovery) {
    setCreateRecovery(recovery);
    try {
      globalThis.sessionStorage?.setItem(
        createRecoveryStorageKey,
        JSON.stringify({ version: 1, ...recovery } satisfies StoredLeagueLiveCreateRecovery),
      );
    } catch {
      // The in-memory state still blocks another session create in this page session.
    }
  }

  function clearCreateRecovery() {
    setCreateRecovery(null);
    createSessionOperationRef.current = null;
    try {
      globalThis.sessionStorage?.removeItem(createRecoveryStorageKey);
    } catch {
      // A conclusive server result remains authoritative if cleanup is blocked.
    }
  }

  function retainPlayerCreateRecovery(recovery: LeagueLivePlayerCreateRecovery) {
    setPlayerCreateRecovery(recovery);
    try {
      globalThis.sessionStorage?.setItem(
        playerCreateRecoveryStorageKey,
        JSON.stringify({ version: 1, ...recovery } satisfies StoredLeagueLivePlayerCreateRecovery)
      );
    } catch {
      // The in-memory guard still prevents a second player batch.
    }
  }

  function clearPlayerCreateRecovery() {
    setPlayerCreateRecovery(null);
    createPlayersOperationRef.current = null;
    try {
      globalThis.sessionStorage?.removeItem(playerCreateRecoveryStorageKey);
    } catch {
      // A conclusive server result remains authoritative.
    }
  }

  function restoreStoredRoundDraft(sessionRow: LeagueLiveSession, persistedCourts: LeagueLiveCourt[]): WorkflowStep | null {
    const targetRound = Number(sessionRow.current_round || 1);
    const targetCourts = courtsFromPersisted(persistedCourts, targetRound, sessionRow.current_court_state_json || []);
    const key = roundDraftStorageKey(sessionRow.id, targetRound);
    try {
      const raw = globalThis.sessionStorage?.getItem(key);
      if (!raw) return null;
      const stored = JSON.parse(raw) as Partial<StoredLeagueLiveRoundDraft>;
      const valid = stored.version === 1
        && stored.sessionId === sessionRow.id
        && Number(stored.roundNumber) === targetRound
        && stored.sessionUpdatedAt === String(sessionRow.updated_at || "")
        && stored.courtFingerprint === courtDraftFingerprint(targetCourts)
        && stored.preview && typeof stored.preview === "object"
        && stored.scores && typeof stored.scores === "object";
      if (!valid) {
        globalThis.sessionStorage?.removeItem(key);
        return null;
      }
      setPreview(stored.preview as AdminMatchUploaderRoundRobinPreview);
      setScores(stored.scores as Record<string, ScoreDraft>);
      setScoreReviewMode(stored.scoreReviewMode === true);
      setScoresReviewed(stored.scoresReviewed === true);
      setMovementPlan((stored.movementPlan as LeagueLiveRoundPlan | null | undefined) || null);
      setMovementPlanStale(stored.movementPlanStale === true);
      setMovementOverrides((stored.movementOverrides as Record<number, string> | undefined) || {});
      setOverrideReason(String(stored.overrideReason || ""));
      setRosterAction(["add", "substitute"].includes(String(stored.rosterAction)) ? stored.rosterAction as "add" | "substitute" : "none");
      setIncomingPlayerId(String(stored.incomingPlayerId || ""));
      setReplacedPlayerId(String(stored.replacedPlayerId || ""));
      setGuestPlayers(Array.isArray(stored.guestPlayers) ? stored.guestPlayers.map((player) => ({ id: Number(player.id), name: String(player.name), rating: player.rating == null ? null : Number(player.rating) })) : []);
      setBenchOverrideIds(Array.isArray(stored.benchOverrideIds) ? stored.benchOverrideIds.map(Number) : []);
      setBenchOverrideReason(String(stored.benchOverrideReason || ""));
      setMatchDate(String(stored.matchDate || ""));
      setWeekTag(String(stored.weekTag || sessionRow.week_tag || "Week 1"));
      setRoundLabel(String(stored.roundLabel || `Round ${targetRound}`));
      if (!/^\d{4}-\d{2}-\d{2}$/.test(String(stored.matchDate || ""))) return 1;
      const requestedStep = Number(stored.workflowStep);
      if (requestedStep === 6 && stored.scoresReviewed && stored.movementPlan && !stored.movementPlanStale) return 6;
      if (requestedStep >= 5 && stored.scoresReviewed) return 5;
      return 4;
    } catch {
      try {
        globalThis.sessionStorage?.removeItem(key);
      } catch {
        // Corrupt local draft stays ignored if browser storage is unavailable.
      }
      return null;
    }
  }

  const parsedCurrentRound = Number(roundNumber);
  const parsedTotalRounds = Number(totalRounds);
  const roundContextValid = Number.isInteger(parsedCurrentRound)
    && Number.isInteger(parsedTotalRounds)
    && parsedCurrentRound >= 1
    && parsedCurrentRound <= parsedTotalRounds
    && parsedTotalRounds <= 50;
  const currentRound = Number.isInteger(parsedCurrentRound) && parsedCurrentRound >= 1 && parsedCurrentRound <= 50 ? parsedCurrentRound : 1;
  const safeTotalRounds = Number.isInteger(parsedTotalRounds) && parsedTotalRounds >= currentRound && parsedTotalRounds <= 50 ? parsedTotalRounds : currentRound;
  const matchDateValid = /^\d{4}-\d{2}-\d{2}$/.test(matchDate);
  const participationMode = participationModeFromDetail(detail);
  const activeLeagueMembers = useMemo(
    () => (detail?.roster || []).filter((row) => row.in_league),
    [detail?.roster]
  );
  const playerOptions = useMemo(() => knownPlayers.map((player) => player.name).filter(Boolean).sort((a, b) => a.localeCompare(b)), [knownPlayers]);
  const selectedSessionPlayers = useMemo(() => {
    const byId = new Map(knownPlayers.map((player) => [Number(player.id), player]));
    return attendeePlayerIds.flatMap((playerId) => {
      const player = byId.get(Number(playerId));
      return player ? [player] : [];
    });
  }, [knownPlayers, attendeePlayerIds]);
  const rosterPlayerIds = useMemo(() => new Set(sessionRoster.map((row) => Number(row.player_id))), [sessionRoster]);
  const incomingPlayerOptions = useMemo(
    () => [
      ...knownPlayers.map((player) => ({ id: Number(player.id), name: player.name, rating: player.rating })),
      ...guestPlayers
    ].filter((player, index, rows) => !rosterPlayerIds.has(Number(player.id)) && rows.findIndex((candidate) => Number(candidate.id) === Number(player.id)) === index).sort((a, b) => a.name.localeCompare(b.name)),
    [knownPlayers, guestPlayers, rosterPlayerIds]
  );
  const activeSessionRoster = useMemo(() => sessionRoster.filter((row) => row.status === "active"), [sessionRoster]);
  const allPreviewMatches = (preview?.courts || []).flatMap((court) => court.matches || []);
  const seriesStates = allPreviewMatches.map((match) => seriesScoreState(match.row_id, scores, matchStructure));
  const completeSeriesCount = seriesStates.filter((state) => state.complete).length;
  const validScoreCount = seriesStates.reduce((total, state) => total + state.playedGames, 0);
  const allSeriesComplete = allPreviewMatches.length > 0 && completeSeriesCount === allPreviewMatches.length;
  const scoreReviewRows = allPreviewMatches.flatMap((match) => {
    const state = seriesScoreState(match.row_id, scores, matchStructure);
    return Array.from({ length: state.playedGames }, (_, index) => {
      const gameNumber = index + 1;
      const score = scores[scoreKey(match.row_id, gameNumber)] || { scoreT1: "", scoreT2: "" };
      return {
        key: scoreKey(match.row_id, gameNumber),
        court: match.court,
        game: gameNumber,
        teamOne: match.t1.map((player) => player.name).join(" / "),
        score: `${score.scoreT1}-${score.scoreT2}`,
        teamTwo: match.t2.map((player) => player.name).join(" / ")
      };
    });
  });
  const sessionIsCurrentLeague = Boolean(
    loadedSessionId
    && loadedSessionId === sessionId
    && sessionUpdatedAt
    && sessionLeagueName
    && sessionLeagueName === leagueName
    && sessionLeagueName === loadedLeagueName
  );
  const blockingCurrentRoundPublishOperation = publishOperations.find(
    (operation) => Number(operation.round_number) === currentRound && operation.status !== "completed"
  ) || null;
  const hasEnteredRoundScores = Object.values(scores).some(scoreHasValue);
  const hasUnsubmittedRoundWork = Boolean(sessionIsCurrentLeague && preview && (hasEnteredRoundScores || scoresReviewed || movementPlan));

  useEffect(() => {
    if (!sessionIsCurrentLeague || !loadedSessionId || !sessionUpdatedAt) return;
    const key = `jupr-league-live-round-draft:${clubId}:${loadedSessionId}:${currentRound}`;
    try {
      if (roundPublished) {
        globalThis.sessionStorage?.removeItem(key);
        return;
      }
      if (!preview) {
        globalThis.sessionStorage?.removeItem(key);
        return;
      }
      const draft: StoredLeagueLiveRoundDraft = {
        version: 1,
        sessionId: loadedSessionId,
        roundNumber: currentRound,
        sessionUpdatedAt,
        courtFingerprint: courtDraftFingerprint(courts),
        workflowStep,
        matchDate,
        weekTag,
        roundLabel,
        preview,
        scores,
        scoreReviewMode,
        scoresReviewed,
        movementPlan,
        movementPlanStale,
        movementOverrides,
        overrideReason,
        rosterAction,
        incomingPlayerId,
        replacedPlayerId,
        guestPlayers,
        benchOverrideIds,
        benchOverrideReason
      };
      globalThis.sessionStorage?.setItem(key, JSON.stringify(draft));
    } catch {
      // The in-memory workflow remains usable when browser storage is unavailable.
    }
  }, [benchOverrideIds, benchOverrideReason, clubId, courts, currentRound, guestPlayers, incomingPlayerId, loadedSessionId, matchDate, movementOverrides, movementPlan, movementPlanStale, overrideReason, preview, replacedPlayerId, rosterAction, roundLabel, roundPublished, scoreReviewMode, scores, scoresReviewed, sessionIsCurrentLeague, sessionUpdatedAt, weekTag, workflowStep]);

  useEffect(() => {
    if (!hasUnsubmittedRoundWork) return;
    const warnBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    globalThis.addEventListener("beforeunload", warnBeforeUnload);
    return () => globalThis.removeEventListener("beforeunload", warnBeforeUnload);
  }, [hasUnsubmittedRoundWork]);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using League Live.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload?.detail;
      const detailRecord = detail && typeof detail === "object" ? detail as Record<string, unknown> : null;
      const errorMessage = leagueLiveOperatorMessage(
        typeof detail === "string" ? detail : detailRecord?.message,
        `League Live error (${response.status})`
      );
      const explicitlyFailed = detailRecord?.kind === "failed" && detailRecord?.recovery_required !== true;
      const uncertainStatus = response.status >= 500 || [408, 425, 429].includes(response.status);
      throw new LeagueLiveRequestError(
        errorMessage,
        response.status,
        detailRecord?.kind === "uncertain" || detailRecord?.recovery_required === true || (!explicitlyFailed && uncertainStatus),
        typeof detailRecord?.operation_key === "string" ? detailRecord.operation_key : null,
      );
    }
    return payload as T;
  }

  function applySession(
    sessionRow: LeagueLiveSession,
    courtsRows: LeagueLiveCourt[] = [],
    rounds: LeagueLiveRound[] = [],
    operations: LeagueLivePublishOperation[] = [],
    leagueDetail: AdminLeagueManagerDetailResponse | null = null
  ): number | null {
    const publishedRoundNumber = persistedPublishedRoundNumber(sessionRow, rounds, operations);
    setSessionId(sessionRow.id);
    setLoadedSessionId(sessionRow.id);
    setSessionLeagueName(sessionRow.league_name);
    setSessionUpdatedAt(sessionRow.updated_at || "");
    setLeagueName(sessionRow.league_name);
    setLoadedLeagueName(sessionRow.league_name);
    setDetail(leagueDetail);
    setMatchStructure(matchStructureFromDetail(leagueDetail));
    setRosterSuggestion(null);
    setWeekTag(sessionRow.week_tag || "Week 1");
    setSessionStatus(sessionRow.status || "active");
    setTotalRounds(String(sessionRow.total_rounds || 1));
    setRoundNumber(String(sessionRow.current_round || 1));
    setRoundLabel(`Round ${sessionRow.current_round || 1}`);
    setSessionNotes(sessionRow.notes || "");
    setSessionRoster((sessionRow.roster_json || []) as LeagueLiveRosterRow[]);
    setAttendeePlayerIds(((sessionRow.roster_json || []) as LeagueLiveRosterRow[]).map((row) => Number(row.player_id)));
    setPastedPlayerText("");
    setPastedPlayerResolutions([]);
    setPasteDuplicateNames([]);
    setPasteResolutionCurrent(true);
    setPastedSelectedPlayerIds([]);
    setMissingPlayerDrafts([]);
    setRoundHistory(rounds || []);
    setPublishOperations(operations || []);
    setCourts(courtsFromPersisted(courtsRows, sessionRow.current_round || 1, sessionRow.current_court_state_json || []));
    setPreview(null);
    setScores({});
    setScoreReviewMode(false);
    setScoresReviewed(false);
    setMovementPlan(null);
    setMovementPlanStale(false);
    setMovementOverrides({});
    setRatingReview(null);
    setBenchOverrideIds(((sessionRow.roster_json || []) as LeagueLiveRosterRow[]).filter((row) => row.status === "bench").map((row) => Number(row.player_id)));
    setRoundPublished(publishedRoundNumber != null);
    setLastPublishedRound(publishedRoundNumber);
    return publishedRoundNumber;
  }

  function clearRoundDerivedState() {
    setDetail(null);
    setMatchStructure(DEFAULT_MATCH_STRUCTURE);
    setLoadedLeagueName("");
    setSessionRoster([]);
    setAttendeePlayerIds([]);
    setPastedPlayerText("");
    setPastedPlayerResolutions([]);
    setPasteDuplicateNames([]);
    setPasteResolutionCurrent(true);
    setPastedSelectedPlayerIds([]);
    setMissingPlayerDrafts([]);
    setRosterSuggestion(null);
    setCourts([{ court: "1", formatType: "4-player", playerNames: "" }]);
    setPreview(null);
    setScores({});
    setScoreReviewMode(false);
    setScoresReviewed(false);
    setMovementPlan(null);
    setMovementPlanStale(false);
    setMovementOverrides({});
    setOverrideReason("");
    setRatingReview(null);
    setBenchOverrideIds([]);
    setBenchOverrideReason("");
    setRosterAction("none");
    setIncomingPlayerId("");
    setReplacedPlayerId("");
    setGuestPlayers([]);
    setGuestName("");
    setGuestJupr("");
    setGuestReason("");
    setRoundPublished(false);
    setLastPublishedRound(null);
  }

  function clearPersistedSessionBinding(selectedSessionId = "") {
    setSessionId(selectedSessionId);
    setLoadedSessionId("");
    setSessionLeagueName("");
    setSessionUpdatedAt("");
    setWeekTag("Week 1");
    setRoundNumber("1");
    setTotalRounds("5");
    setRoundLabel("Round 1");
    setMatchDate(todayIso());
    setSessionStatus("active");
    setSessionNotes("");
    setRoundHistory([]);
    setPublishOperations([]);
    setCompensationReference("");
    setCompensationReason("");
    setWorkflowStep(1);
    clearRoundDerivedState();
  }

  function clearProtectedLiveWorkspace() {
    leagueDetailRequest.invalidate();
    sessionListRequest.invalidate();
    sessionDetailRequest.invalidate();
    setLeagues([]);
    setLeagueName("");
    setLiveSessions([]);
    setLoadingLeagueName(null);
    setBusy(false);
    setMessage(null);
    clearPersistedSessionBinding();
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

  function invalidateScoreReview() {
    setScoreReviewMode(false);
    setScoresReviewed(false);
    setMovementPlan(null);
    setMovementPlanStale(false);
    setMovementOverrides({});
  }

  function invalidateRoundDraft(message: string) {
    setPreview(null);
    setScores({});
    invalidateScoreReview();
    setRoundPublished(false);
    setLastPublishedRound(null);
    createSessionOperationRef.current = null;
    setWorkflowStep(1);
    setMessage(message);
  }

  function setupContextError(): string | null {
    if (!roundContextValid) return "Use whole numbers with 1 ≤ Round # ≤ Total rounds ≤ 50 before continuing.";
    if (!matchDateValid) return "Confirm this round's date before continuing; a resumed session never assumes today's date.";
    if (sessionIsCurrentLeague && sessionStatus !== "active") return `This persisted session is ${sessionStatus.replace(/_/g, " ")}; activate it before continuing.`;
    return null;
  }

  function invalidateFlexRosterPlan(message: string) {
    setRosterSuggestion(null);
    setSessionRoster([]);
    setBenchOverrideIds([]);
    setBenchOverrideReason("");
    setCourts([{ court: "1", formatType: "4-player", playerNames: "" }]);
    setPreview(null);
    setScores({});
    setScoreReviewMode(false);
    setScoresReviewed(false);
    setMovementPlan(null);
    setMovementPlanStale(false);
    setMovementOverrides({});
    setOverrideReason("");
    createSessionOperationRef.current = null;
    setMessage(message);
  }

  function replaceFlexAttendance(playerIds: number[], message: string) {
    setAttendeePlayerIds([...new Set(playerIds)]);
    setPastedSelectedPlayerIds([]);
    invalidateFlexRosterPlan(message);
  }

  function changeFlexAttendance(playerId: number, attending: boolean) {
    setAttendeePlayerIds((current) => attending
      ? [...new Set([...current, playerId])]
      : current.filter((id) => id !== playerId));
    if (!attending) setPastedSelectedPlayerIds((current) => current.filter((id) => id !== playerId));
    invalidateFlexRosterPlan(participationMode === "flex"
      ? "Attendance changed. Build fresh Flex courts before creating the session."
      : "Set session players changed. Rebuild the court plan before continuing.");
  }

  function appendSelectedPlayerIds(playerIds: number[]) {
    setAttendeePlayerIds((current) => [...new Set([...current, ...playerIds.map(Number)])]);
  }

  function appendPastedPlayerIds(playerIds: number[]) {
    const normalizedIds = [...new Set(playerIds.map(Number))];
    const newlySelectedIds = normalizedIds.filter((playerId) => !attendeePlayerIds.includes(playerId));
    setPastedSelectedPlayerIds((current) => [...new Set([...current, ...newlySelectedIds])]);
    appendSelectedPlayerIds(normalizedIds);
  }

  function changePastedPlayerText(nextText: string) {
    const pastedSelection = new Set(pastedSelectedPlayerIds.map(Number));
    if (pastedSelection.size) {
      setAttendeePlayerIds((current) => current.filter((playerId) => !pastedSelection.has(Number(playerId))));
    }
    setPastedSelectedPlayerIds([]);
    setPastedPlayerText(nextText);
    setPasteDuplicateNames([]);
    setPastedPlayerResolutions([]);
    setMissingPlayerDrafts([]);
    setPasteResolutionCurrent(!nextText.trim());
    invalidateFlexRosterPlan(nextText.trim()
      ? "Pasted player names changed. Resolve the complete list again before building courts."
      : "Pasted player list cleared. Review the selected session players before rebuilding courts.");
  }

  function removeSelectedPlayer(playerId: number) {
    setAttendeePlayerIds((current) => current.filter((id) => Number(id) !== Number(playerId)));
    setPastedSelectedPlayerIds((current) => current.filter((id) => Number(id) !== Number(playerId)));
    invalidateFlexRosterPlan("Session players changed. Build a fresh court plan before continuing.");
  }

  function resolvePlayerNames() {
    const parsedNames = parsePastedPlayerNames(pastedPlayerText);
    const duplicates = duplicatePastedPlayerNames(parsedNames);
    setPasteDuplicateNames(duplicates);
    setPasteResolutionCurrent(false);
    setPastedPlayerResolutions([]);
    setMissingPlayerDrafts([]);
    if (!parsedNames.length) {
      setMessage("Paste at least one player name, separated by commas or new lines.");
      return;
    }
    if (duplicates.length) {
      setMessage(`Remove duplicate pasted name${duplicates.length === 1 ? "" : "s"}: ${duplicates.join(", ")}.`);
      return;
    }
    const resolutions = resolvePastedPlayers(parsedNames, knownPlayers);
    const resolvedIds = resolutions.flatMap((row) => row.status === "existing" && row.playerId ? [Number(row.playerId)] : []);
    const missing = resolutions.filter((row) => row.status === "missing").map((row) => ({ name: row.inputName, startingJupr: "" }));
    setPastedPlayerResolutions(resolutions);
    setMissingPlayerDrafts(missing);
    appendPastedPlayerIds(resolvedIds);
    setPasteResolutionCurrent(true);
    invalidateFlexRosterPlan(
      missing.length
        ? `Matched ${resolvedIds.length} existing player${resolvedIds.length === 1 ? "" : "s"}. Enter a Starting JUPR for ${missing.length} new player${missing.length === 1 ? "" : "s"}.`
        : `Resolved and added ${resolvedIds.length} existing player${resolvedIds.length === 1 ? "" : "s"}.`
    );
  }

  async function createMissingPlayers() {
    if (playerCreateRecovery) {
      setMessage(`Check exact player operation ${playerCreateRecovery.operationKey} before creating another batch.`);
      return;
    }
    const reviewedPlayers = missingPlayerDrafts.map((draft) => ({
      name: normalizePlayerName(draft.name),
      starting_jupr: Number(draft.startingJupr)
    }));
    const invalid = reviewedPlayers.find((player) => !player.name || !Number.isFinite(player.starting_jupr) || player.starting_jupr < 1 || player.starting_jupr > 7);
    if (invalid) {
      setMessage("Every new player needs an explicit Starting JUPR between 1.0 and 7.0; no default is assumed.");
      return;
    }
    const duplicateNames = duplicatePastedPlayerNames(reviewedPlayers.map((player) => player.name));
    if (duplicateNames.length) {
      setMessage(`Remove duplicate player name${duplicateNames.length === 1 ? "" : "s"}: ${duplicateNames.join(", ")}.`);
      return;
    }
    if (!reviewedPlayers.length) {
      setMessage("There are no missing players to create.");
      return;
    }
    setCreatingMissingPlayers(true);
    setMessage(null);
    try {
      const reviewedFingerprint = await reviewedPlayerBatchFingerprint(reviewedPlayers);
      const requestFingerprint = canonicalJson({ players: reviewedPlayers, reviewed_fingerprint: reviewedFingerprint });
      const idempotencyKey = createPlayersOperationRef.current?.fingerprint === requestFingerprint
        ? createPlayersOperationRef.current.key
        : `league-live-players:${globalThis.crypto.randomUUID()}`;
      createPlayersOperationRef.current = { fingerprint: requestFingerprint, key: idempotencyKey };
      const payload = await requestJson<AdminMatchUploaderCreatePlayersResult>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/players`, {
        method: "POST",
        body: JSON.stringify({
          players: reviewedPlayers,
          reviewed_fingerprint: reviewedFingerprint,
          idempotency_key: idempotencyKey,
          confirmation_text: "CREATE PLAYERS",
          source: "next_league_live_pasted_players"
        })
      });
      createPlayersOperationRef.current = null;
      const created = publicPlayersFromBatch(payload);
      setKnownPlayers((current) => {
        const byId = new Map(current.map((player) => [Number(player.id), player]));
        for (const player of created) byId.set(Number(player.id), player);
        return [...byId.values()].sort((left, right) => left.name.localeCompare(right.name));
      });
      const createdByName = new Map(created.map((player) => [playerNameKey(player.name), player]));
      const createdIds = missingPlayerDrafts.flatMap((draft) => {
        const player = createdByName.get(playerNameKey(draft.name));
        return player ? [Number(player.id)] : [];
      });
      appendPastedPlayerIds(createdIds);
      setPastedPlayerResolutions((current) => current.map((row) => {
        const player = createdByName.get(playerNameKey(row.inputName));
        return player ? { ...row, status: "existing", playerId: Number(player.id), playerName: player.name } : row;
      }));
      setMissingPlayerDrafts([]);
      invalidateFlexRosterPlan(`Created or confirmed ${payload.accepted_count ?? createdIds.length} player profile${(payload.accepted_count ?? createdIds.length) === 1 ? "" : "s"} and added them to this session.`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unable to create the missing players.";
      if (leagueLiveWriteIsUncertain(error)) {
        const operationKey = error instanceof LeagueLiveRequestError && error.operationKey
          ? error.operationKey
          : createPlayersOperationRef.current?.key;
        if (operationKey) {
          retainPlayerCreateRecovery({ operationKey, status: "uncertain", message: errorMessage, leagueName });
          setMessage(`${errorMessage} Exact player operation ${operationKey} is retained; inspect it before creating another batch.`);
        } else {
          setMessage(`${errorMessage} The outcome is uncertain; inspect Player Editor before retrying.`);
        }
      } else {
        createPlayersOperationRef.current = null;
        setMessage(errorMessage);
      }
    } finally {
      setCreatingMissingPlayers(false);
    }
  }

  function applyRecoveredPlayers(result: AdminMatchUploaderCreatePlayersResult | null | undefined, addToCurrentSession = true) {
    const recoveredPlayers = publicPlayersFromBatch(result);
    if (!recoveredPlayers.length) return;
    setKnownPlayers((current) => {
      const byId = new Map(current.map((player) => [Number(player.id), player]));
      for (const player of recoveredPlayers) byId.set(Number(player.id), player);
      return [...byId.values()].sort((left, right) => left.name.localeCompare(right.name));
    });
    if (!addToCurrentSession) return;
    appendPastedPlayerIds(recoveredPlayers.map((player) => Number(player.id)));
    const recoveredByName = new Map(recoveredPlayers.map((player) => [playerNameKey(player.name), player]));
    setPastedPlayerResolutions((current) => current.map((row) => {
      const player = recoveredByName.get(playerNameKey(row.inputName));
      return player ? { ...row, status: "existing", playerId: Number(player.id), playerName: player.name } : row;
    }));
    setMissingPlayerDrafts((current) => current.filter((draft) => !recoveredByName.has(playerNameKey(draft.name))));
  }

  async function inspectPlayerCreateOperation() {
    const recovery = playerCreateRecovery;
    if (!recovery) return;
    if (recovery.leagueName && recovery.leagueName !== leagueName) {
      setMessage(`This retained player batch belongs to ${recovery.leagueName}. Select that league before reconciling it; no players were added to ${leagueName || "the current workspace"}.`);
      return;
    }
    const addRecoveredPlayersToSession = Boolean(recovery.leagueName && recovery.leagueName === leagueName);
    const operationPath = `/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/player-operations/${encodeURIComponent(recovery.operationKey)}`;
    setCheckingPlayerCreateRecovery(true);
    setMessage(null);
    try {
      const operation = await requestJson<LeagueLivePlayerOperationResponse>(operationPath);
      const operationStatus = String(operation.status || "unknown");
      const recoveredResult = operation.result || operation.result_json;
      if (operationStatus === "completed") {
        applyRecoveredPlayers(recoveredResult, addRecoveredPlayersToSession);
        clearPlayerCreateRecovery();
        setMessage(addRecoveredPlayersToSession
          ? `Exact player operation ${recovery.operationKey} completed and its players were added to this session.`
          : `Exact player operation ${recovery.operationKey} completed. Its club players were recovered but not added to a session because the original league context was unavailable.`);
        return;
      }
      if (operationStatus === "failed" && operation.recovery_required !== true) {
        clearPlayerCreateRecovery();
        setMessage(`Exact player operation ${recovery.operationKey} is proven failed. Review the player list before submitting a new batch.`);
        return;
      }
      const reconciled = await requestJson<AdminMatchUploaderCreatePlayersResult>(`${operationPath}/reconcile`, {
        method: "POST",
        body: JSON.stringify({
          confirmation_text: "RECONCILE PLAYER BATCH",
          source: "next_league_live_player_operation_reconcile"
        })
      });
      if (reconciled.ok && reconciled.recovery_required !== true) {
        applyRecoveredPlayers(reconciled, addRecoveredPlayersToSession);
        clearPlayerCreateRecovery();
        setMessage(addRecoveredPlayersToSession
          ? `Exact player operation ${recovery.operationKey} was reconciled and its players were added to this session.`
          : `Exact player operation ${recovery.operationKey} was reconciled. Its club players were recovered but not added to a session because the original league context was unavailable.`);
        return;
      }
      if (String(reconciled.status || "") === "failed" && reconciled.recovery_required !== true) {
        clearPlayerCreateRecovery();
        setMessage(`Exact player operation ${recovery.operationKey} is proven failed and created no players. Review the list, then submit a new batch if needed.`);
        return;
      }
      retainPlayerCreateRecovery({
        ...recovery,
        status: String(reconciled.status || operationStatus),
        message: operation.error || operation.error_text || recovery.message
      });
      setMessage(`Exact player operation ${recovery.operationKey} still needs recovery. Do not create another player batch.`);
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Unable to inspect the player operation."} Operation ${recovery.operationKey} remains retained; do not retry with a new key.`);
    } finally {
      setCheckingPlayerCreateRecovery(false);
    }
  }

  async function fetchRosterSuggestion(
    leagueDetail: AdminLeagueManagerDetailResponse,
    requestedBenchIds: number[] = [],
    requestedBenchReason = "",
    requestedAttendeeIds: number[] = attendeePlayerIds
  ): Promise<LeagueLiveRosterSuggestion> {
    const fallbackIds = (leagueDetail.roster || []).filter((row) => row.in_league).map((row) => Number(row.player_id));
    const selectedIds = requestedAttendeeIds.length ? requestedAttendeeIds.map(Number) : fallbackIds;
    return requestJson<LeagueLiveRosterSuggestion>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live/roster-suggestion`, {
      method: "POST",
      body: JSON.stringify({
        roster: selectedSessionRosterPayload(leagueDetail, knownPlayers, selectedIds),
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

  async function loadLeagues(): Promise<string> {
    const generation = leagueListRequest.begin();
    const selectedLeagueBeforeRefresh = leagueName;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues`);
      if (!leagueListRequest.isCurrent(generation)) return "";
      const names = (payload.leagues || []).map((league) => league.league_name).filter(Boolean);
      setLeagues(names);
      const selectedLeague = names.includes(selectedLeagueBeforeRefresh)
        ? selectedLeagueBeforeRefresh
        : names.includes(selectedLeagueName) ? selectedLeagueName : "";
      setLeagueName(selectedLeague);
      if (selectedLeague) await loadLeagueDetail(selectedLeague);
      else {
        clearPersistedSessionBinding();
        setMessage("No leagues are available.");
      }
      return selectedLeague;
    } catch (error) {
      if (leagueListRequest.isCurrent(generation)) {
        setLeagues([]);
        setLeagueName("");
        clearPersistedSessionBinding();
        setMessage(error instanceof Error ? error.message : "Unable to load leagues.");
      }
      return "";
    } finally {
      if (leagueListRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function loadSessions(selectedLeague = leagueName) {
    const targetLeague = selectedLeague.trim();
    if (!targetLeague) {
      sessionListRequest.invalidate();
      setLiveSessions([]);
      setMessage("Select a league before loading unfinished live sessions.");
      return;
    }
    const generation = sessionListRequest.begin();
    const selectedSessionBeforeRefresh = sessionId;
    if (selectedSessionBeforeRefresh) {
      sessionDetailRequest.invalidate();
      clearPersistedSessionBinding(selectedSessionBeforeRefresh);
    }
    setBusy(true);
    setMessage(null);
    try {
      const query = new URLSearchParams({
        league_name: targetLeague,
        resumable_only: "true",
        limit: "100",
      });
      const payload = await requestJson<LeagueLiveListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions?${query.toString()}`);
      if (!sessionListRequest.isCurrent(generation)) return;
      const nextSessions = resumableSessionsForLeague(payload.sessions || [], targetLeague);
      setLiveSessions(nextSessions);
      if (selectedSessionBeforeRefresh && nextSessions.some((row) => row.id === selectedSessionBeforeRefresh)) {
        await loadSessionDetail(selectedSessionBeforeRefresh);
      } else if (selectedSessionBeforeRefresh) {
        clearPersistedSessionBinding();
        setMessage("The previously selected live session is complete or no longer available for this league.");
      } else {
        setMessage(nextSessions.length
          ? `Loaded ${nextSessions.length} unfinished live session(s) for ${targetLeague}.`
          : `No unfinished live sessions are available for ${targetLeague}.`);
      }
    } catch (error) {
      if (sessionListRequest.isCurrent(generation)) {
        setLiveSessions([]);
        if (selectedSessionBeforeRefresh) clearPersistedSessionBinding(selectedSessionBeforeRefresh);
        setMessage(error instanceof Error ? error.message : `Unable to load unfinished live sessions for ${targetLeague}.`);
      }
    } finally {
      if (sessionListRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function loadLeagueDetail(selectedLeague = leagueName) {
    const generation = leagueDetailRequest.begin();
    sessionDetailRequest.invalidate();
    setLeagueName(selectedLeague);
    clearPersistedSessionBinding();
    if (!selectedLeague) {
      setMessage("Select a league first.");
      return;
    }
    setBusy(true);
    setLoadingLeagueName(selectedLeague);
    setMessage(null);
    try {
      const payload = await requestJson<AdminLeagueManagerDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(selectedLeague)}`);
      if (!leagueDetailRequest.isCurrent(generation)) return;
      setDetail(payload);
      setMatchStructure(matchStructureFromDetail(payload));
      setLoadedLeagueName(selectedLeague);
      if (participationModeFromDetail(payload) === "flex") {
        setAttendeePlayerIds([]);
        setRosterSuggestion(null);
        setSessionRoster([]);
        setMessage("Flex participation: select today's attendees, then build fresh rating-seeded courts for this session.");
        return;
      }
      const suggestion = await fetchRosterSuggestion(payload, [], "", []);
      if (!leagueDetailRequest.isCurrent(generation)) return;
      setAttendeePlayerIds((payload.roster || []).filter((row) => row.in_league).map((row) => Number(row.player_id)));
      applyRosterSuggestion(suggestion);
      setMessage(`Set participation: loaded the persistent league roster and suggested ${suggestion.courts.length} court(s). Resume an existing session to keep its saved pod positions.`);
    } catch (error) {
      if (leagueDetailRequest.isCurrent(generation)) {
        clearPersistedSessionBinding();
        setLeagueName(selectedLeague);
        setMessage(error instanceof Error ? error.message : "Unable to load league detail.");
      }
    } finally {
      if (leagueDetailRequest.isCurrent(generation)) {
        setLoadingLeagueName(null);
        setBusy(false);
      }
    }
  }

  function selectLeague(selectedLeague: string) {
    setWorkflowStep(1);
    setLeagueName(selectedLeague);
    sessionListRequest.invalidate();
    setLiveSessions([]);
    void loadLeagueDetail(selectedLeague).then(() => loadSessions(selectedLeague));
  }

  async function loadInitialWorkspace() {
    const selectedLeague = await loadLeagues();
    if (selectedLeague) await loadSessions(selectedLeague);
  }

  useAuthenticatedAutoLoad(
    leagueStatus.enabled && uploaderStatus.enabled && liveDomainStatus.enabled ? accessToken : "",
    loadInitialWorkspace
  );

  async function refreshRosterSuggestion() {
    if (!detail || detail.league.league_name !== leagueName) {
      setMessage("Load a league roster before requesting another roster suggestion.");
      return;
    }
    if (participationModeFromDetail(detail) === "flex" && !attendeePlayerIds.length) {
      setMessage("Select at least one attendee before building Flex courts.");
      return;
    }
    const generation = leagueDetailRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const suggestion = await fetchRosterSuggestion(detail, benchOverrideIds, benchOverrideReason, attendeePlayerIds);
      if (!leagueDetailRequest.isCurrent(generation)) return;
      applyRosterSuggestion(suggestion);
      setMessage(`${participationModeFromDetail(detail) === "flex" ? "Fresh attendee" : "Persistent roster"} plan built ${suggestion.courts.length} court(s); ${suggestion.bench.length} player(s) are benched.`);
    } catch (error) {
      if (leagueDetailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to refresh the roster suggestion.");
    } finally {
      if (leagueDetailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function buildPlayerPlanAndContinue() {
    const setupError = setupContextError();
    if (setupError) {
      setWorkflowStep(1);
      setMessage(setupError);
      return;
    }
    if (!detail || detail.league.league_name !== leagueName) {
      setMessage("Load the selected league before continuing from Players.");
      return;
    }
    if (String(detail.league.league_type || "Individual").toLowerCase() !== "individual" || String(detail.league.match_format || "doubles").toLowerCase() !== "doubles") {
      setMessage("This guided League Live workflow currently supports Individual Doubles leagues only. It will not publish a Singles or Team round through a doubles court contract.");
      return;
    }
    if (pasteDuplicateNames.length) {
      setMessage("Remove duplicate pasted names before continuing.");
      return;
    }
    if (!pasteResolutionCurrent) {
      setMessage("Resolve the current pasted player list before continuing.");
      return;
    }
    if (playerCreateRecovery) {
      setMessage(`Check exact player operation ${playerCreateRecovery.operationKey} before continuing.`);
      return;
    }
    if (missingPlayerDrafts.length || pastedPlayerResolutions.some((row) => row.status !== "existing")) {
      setMessage("Create every missing pasted player and make ambiguous club player names unique before continuing.");
      return;
    }
    if (attendeePlayerIds.length < 4) {
      setMessage("Select at least four session players before building courts.");
      return;
    }
    if (sessionIsCurrentLeague && sessionRoster.length >= 4 && courts.some((court) => splitNames(court.playerNames).length)) {
      setWorkflowStep(3);
      setMessage("Saved session players and courts retained. Validate the assignments and generate this round's exact preview.");
      return;
    }
    const generation = leagueDetailRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const suggestion = await fetchRosterSuggestion(detail, [], "", attendeePlayerIds);
      if (!leagueDetailRequest.isCurrent(generation)) return;
      applyRosterSuggestion(suggestion);
      setWorkflowStep(3);
      setMessage(`${participationMode === "flex" ? "Fresh rating-seeded Flex" : "Set roster"} plan built ${suggestion.courts.length} court${suggestion.courts.length === 1 ? "" : "s"}. Review every assignment and generate the match preview.`);
    } catch (error) {
      if (leagueDetailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to build the session player plan.");
    } finally {
      if (leagueDetailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function applyRecoveredLiveSession(recovered: LeagueLiveCreateReconcileResponse) {
    if (!recovered.session) throw new Error("The completed operation did not include its League Live session.");
    const leagueDetail = await requestJson<AdminLeagueManagerDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(recovered.session.league_name)}`);
    applySession(recovered.session, recovered.courts || [], recovered.rounds || [], [], leagueDetail);
    setMatchDate("");
    setWorkflowStep(1);
    setLiveSessions((current) => resumableSessionsForLeague([
      ...current.filter((row) => row.id !== recovered.session?.id),
      recovered.session as LeagueLiveSession,
    ], selectedLeagueName).sort((left, right) => String(right.updated_at || "").localeCompare(String(left.updated_at || ""))));
  }

  async function reconcileCreateSession(
    retainedRecovery: LeagueLiveCreateRecovery | null = createRecovery,
  ): Promise<ActionCompletion> {
    if (!retainedRecovery) throw new Error("No League Live create operation is waiting for recovery.");
    const path = `/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-operations/${encodeURIComponent(retainedRecovery.operationKey)}`;
    setCheckingCreateRecovery(true);
    setMessage(null);
    try {
      const operation = await requestJson<LeagueLiveCreateOperationResponse>(path);
      let operationStatus = String(operation.status || "unknown");
      let recoveryRequired = operation.recovery_required === true;
      let recovered: LeagueLiveCreateReconcileResponse | null = operation.result || null;

      if (operationStatus !== "completed" && operationStatus !== "failed") {
        recovered = await requestJson<LeagueLiveCreateReconcileResponse>(`${path}/reconcile`, {
          method: "POST",
          body: JSON.stringify({
            confirmation_text: "RECONCILE LIVE SESSION",
            source: "next_league_live_session_create_reconcile",
          }),
        });
        operationStatus = recovered.ok === false ? String(recovered.status || "failed") : "completed";
        recoveryRequired = recovered.recovery_required === true;
      }

      if (operationStatus === "completed" && recovered?.ok !== false) {
        await applyRecoveredLiveSession(recovered || { ok: true });
        clearCreateRecovery();
        const successMessage = `League Live recovery ${retainedRecovery.operationKey} completed. Its session is now loaded.`;
        setMessage(successMessage);
        return actionSuccess("League Live session reconciled", successMessage);
      }

      if (operationStatus === "failed" && !recoveryRequired) {
        clearCreateRecovery();
        const failedMessage = `Exact League Live operation ${retainedRecovery.operationKey} is proven failed. Review current sessions before starting a new create request.`;
        setMessage(failedMessage);
        return actionSuccess("League Live operation checked", failedMessage);
      }

      const pending = {
        ...retainedRecovery,
        status: operationStatus,
        message: leagueLiveOperatorMessage(operation.error || retainedRecovery.message, "Unable to verify the League Live session.")
      };
      retainCreateRecovery(pending);
      return actionUncertain(
        "League Live session still needs verification",
        `Operation ${retainedRecovery.operationKey} is ${operationStatus.replace(/_/g, " ")}. Creating another session remains blocked.`,
        retainedRecovery.operationKey,
        "Check and reconcile exact operation",
        () => reconcileCreateSession(pending),
      );
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unable to reconcile the League Live session create.";
      const pending = { ...retainedRecovery, status: "recovery_required", message: errorMessage };
      retainCreateRecovery(pending);
      setMessage(`${errorMessage} Operation ${retainedRecovery.operationKey} remains retained; do not create another session.`);
      return actionUncertain(
        "League Live session still needs verification",
        `${errorMessage} The exact operation reference remains retained.`,
        retainedRecovery.operationKey,
        "Check and reconcile exact operation",
        () => reconcileCreateSession(pending),
      );
    } finally {
      setCheckingCreateRecovery(false);
    }
  }

  async function createSession(confirmationText: string): Promise<ActionCompletion> {
    const setupError = setupContextError();
    if (setupError) throw new Error(setupError);
    if (createRecovery) throw new Error(`Resolve exact operation ${createRecovery.operationKey} before creating another League Live session.`);
    if (!leagueName || !detail || loadedLeagueName !== leagueName || !rosterSuggestion || !sessionRoster.length) {
      const error = new Error("Load the selected league roster and court suggestion before creating a saved session.");
      setMessage(error.message);
      throw error;
    }
    const generation = actionRequest.begin();
    const retainedPreview = preview;
    const retainedScores = scores;
    const request = {
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
    };
    const requestFingerprint = JSON.stringify(request);
    const idempotencyKey = createSessionOperationRef.current?.fingerprint === requestFingerprint
      ? createSessionOperationRef.current.key
      : `league-live:${globalThis.crypto.randomUUID()}`;
    createSessionOperationRef.current = { fingerprint: requestFingerprint, key: idempotencyKey };
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions`, {
        method: "POST",
        body: JSON.stringify({ ...request, idempotency_key: idempotencyKey })
      });
      createSessionOperationRef.current = null;
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the League Live creation response was applied.");
      applySession(payload.session, payload.courts || [], [], [], detail);
      setPreview(retainedPreview);
      setScores(retainedScores);
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before League Live sessions could be refreshed.");
      await loadSessions();
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the created session was confirmed.");
      setMessage("Persisted League Live session created. You can now resume it later from this page.");
      return actionSuccess("League Live session created", "The persisted session was created and can be resumed later from this page.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to create persisted session.");
      if (leagueLiveWriteIsUncertain(error)) {
        const operationKey = error instanceof LeagueLiveRequestError && error.operationKey ? error.operationKey : idempotencyKey;
        const recovery: LeagueLiveCreateRecovery = {
          operationKey,
          status: "uncertain",
          message: error instanceof Error ? error.message : "Unable to confirm League Live session creation.",
        };
        retainCreateRecovery(recovery);
        return actionUncertain(
          "League Live session outcome needs checking",
          "The request was sent, but its durable result could not be confirmed. Check and reconcile the retained operation before creating another session.",
          operationKey,
          "Check and reconcile exact operation",
          () => reconcileCreateSession(recovery)
        );
      }
      createSessionOperationRef.current = null;
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function loadSessionDetail(selectedSessionId = sessionId) {
    const generation = sessionDetailRequest.begin();
    leagueDetailRequest.invalidate();
    clearPersistedSessionBinding(selectedSessionId);
    if (!selectedSessionId) {
      setMessage("Select a persisted session first.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(selectedSessionId)}`);
      if (!sessionDetailRequest.isCurrent(generation)) return;
      const leagueDetail = await requestJson<AdminLeagueManagerDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/leagues/${encodeURIComponent(payload.session.league_name)}`);
      if (!sessionDetailRequest.isCurrent(generation)) return;
      const publishedRoundNumber = applySession(payload.session, payload.courts || [], payload.rounds || [], payload.publish_operations || [], leagueDetail);
      const restoredDraftStep = publishedRoundNumber == null ? restoreStoredRoundDraft(payload.session, payload.courts || []) : null;
      if (publishedRoundNumber != null) {
        setMatchDate("");
        setWorkflowStep(6);
        setMessage(payload.session.status === "complete"
          ? "Completed League Live session loaded."
          : `Round ${publishedRoundNumber} is already published. Start the next round or finish this session.`);
      } else if (restoredDraftStep != null) {
        setWorkflowStep(payload.session.status === "active" ? restoredDraftStep : 1);
        setMessage(payload.session.status === "active"
          ? restoredDraftStep === 1
            ? `Recovered the saved browser draft for Round ${payload.session.current_round}. Confirm its date before continuing.`
            : `Recovered the saved browser draft for Round ${payload.session.current_round}. Review it before continuing.`
          : `Recovered the saved browser draft for Round ${payload.session.current_round}. Resume this ${payload.session.status} session before editing or publishing it.`);
      } else {
        setMatchDate("");
        setWorkflowStep(1);
        setMessage("Persisted League Live session loaded. Confirm this round's date before reviewing its saved players and courts.");
      }
    } catch (error) {
      if (sessionDetailRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load persisted session.");
    } finally {
      if (sessionDetailRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function selectSession(selectedSessionId: string) {
    if (selectedSessionId) void loadSessionDetail(selectedSessionId);
    else {
      sessionDetailRequest.invalidate();
      clearPersistedSessionBinding();
    }
  }

  async function saveSessionSnapshot(confirmationText: string, statusOverride = sessionStatus): Promise<ActionCompletion> {
    if (!requireCurrentSession("saving a snapshot")) throw new Error("Resume the current persisted session before saving a snapshot.");
    const generation = actionRequest.begin();
    const requestedSessionId = loadedSessionId;
    const retainedPreview = preview;
    const retainedScores = scores;
    const retainedReviewMode = scoreReviewMode;
    const retainedScoresReviewed = scoresReviewed;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(requestedSessionId)}/snapshot`, {
        method: "PATCH",
        body: JSON.stringify({
          status: statusOverride,
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
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the saved snapshot response was applied.");
      applySession(payload.session, payload.courts || [], roundHistory, publishOperations, detail);
      setPreview(retainedPreview);
      setScores(retainedScores);
      setScoreReviewMode(retainedReviewMode);
      setScoresReviewed(retainedScoresReviewed);
      setMessage("League Live session snapshot saved.");
      return actionSuccess("Session snapshot saved", "The League Live session snapshot was saved.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save session snapshot.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function createSessionAndContinue(confirmationText: string): Promise<ActionCompletion> {
    const completion = await createSession(confirmationText);
    if (completion.status !== "success") return completion;
    setWorkflowStep(4);
    setMessage("Session saved. Enter every score, then review the exact teams and results before movement.");
    return completion;
  }

  async function saveSnapshotAndContinue(confirmationText: string): Promise<ActionCompletion> {
    const setupError = setupContextError();
    if (setupError) throw new Error(setupError);
    const completion = await saveSessionSnapshot(confirmationText);
    if (completion.status !== "success") return completion;
    setWorkflowStep(4);
    setMessage("Session snapshot saved. Enter every score, then review the exact teams and results before movement.");
    return completion;
  }

  async function resumePausedSession(confirmationText: string): Promise<ActionCompletion> {
    if (!sessionIsCurrentLeague || !["paused", "setup"].includes(sessionStatus)) {
      throw new Error("Only a saved setup or paused League Live session can be activated here.");
    }
    const completion = await saveSessionSnapshot(confirmationText, "active");
    if (completion.status !== "success") return completion;
    setSessionStatus("active");
    setWorkflowStep(1);
    setMessage("League Live session is active. Confirm this round's date, then continue through Players and the saved court preview.");
    return actionSuccess("League Live session resumed", "The saved session is active and ready for its round date and operator review.");
  }

  async function pauseActiveSession(confirmationText: string): Promise<ActionCompletion> {
    if (!sessionIsCurrentLeague || sessionStatus !== "active" || roundPublished) {
      throw new Error("Only an active, unpublished League Live round can be paused here.");
    }
    const completion = await saveSessionSnapshot(confirmationText, "paused");
    if (completion.status !== "success") return completion;
    setSessionStatus("paused");
    setWorkflowStep(1);
    setMessage("League Live session paused. Its saved players and courts remain intact; resume it before editing or publishing.");
    return actionSuccess("League Live session paused", "The persisted session is paused and its saved players and courts remain available.");
  }

  async function finishSession(confirmationText: string): Promise<ActionCompletion> {
    if (!roundContextValid) throw new Error("Use a valid whole-number round range through 50 before finishing this session.");
    if (!requireCurrentSession("finishing the session")) throw new Error("Resume the current persisted session before finishing it.");
    const generation = actionRequest.begin();
    const requestedSessionId = loadedSessionId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(requestedSessionId)}/snapshot`, {
        method: "PATCH",
        body: JSON.stringify({
          status: "complete",
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
          source: "next_league_live_session_finish"
        })
      });
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before completion was applied.");
      applySession(payload.session, payload.courts || [], roundHistory, publishOperations, detail);
      setLiveSessions((current) => current.filter((row) => row.id !== requestedSessionId));
      setSessionStatus("complete");
      setWorkflowStep(6);
      setRoundPublished(true);
      setMessage("League Live session completed. Its rounds, ratings, and recovery history remain available below.");
      return actionSuccess("League session complete", "The League Live session was marked complete.");
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to finish the League Live session.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function reviewEnteredScores() {
    const setupError = setupContextError();
    if (setupError) {
      setWorkflowStep(1);
      setMessage(setupError);
      return;
    }
    if (!allSeriesComplete) {
      setMessage(`Complete every matchup before review (${completeSeriesCount} of ${allPreviewMatches.length} complete).`);
      return;
    }
    setScoresReviewed(false);
    setScoreReviewMode(true);
    setMessage("Nothing is official yet. Verify every team and score below.");
  }

  function confirmScoresAndContinue() {
    if (!allSeriesComplete || !scoreReviewMode) {
      setMessage("Open the score review and verify every matchup before continuing.");
      return;
    }
    setScoresReviewed(true);
    setScoreReviewMode(false);
    setWorkflowStep(5);
    setMessage("Scores reviewed. Preview the authoritative movement plan; any score edit will require another review.");
  }

  function editReviewedScores() {
    setScoreReviewMode(false);
    setScoresReviewed(false);
    setWorkflowStep(4);
    setMessage("Score review cleared. Edit the scores, then review them again.");
  }

  function startNextRound() {
    if (!roundPublished || lastPublishedRound == null) {
      setMessage("Publish the current round before starting the next round workflow.");
      return;
    }
    if (!sessionIsCurrentLeague || sessionStatus !== "active") {
      setMessage(`This session is ${sessionStatus.replace(/_/g, " ")} and cannot start another round.`);
      return;
    }
    const nextRound = lastPublishedRound + 1;
    if (!Number.isInteger(parsedTotalRounds) || nextRound > safeTotalRounds || safeTotalRounds > 50) {
      setWorkflowStep(1);
      setMessage(`Round ${lastPublishedRound} is the configured final round. Increase Total rounds to a whole number through 50, or finish the session.`);
      return;
    }
    setRoundPublished(false);
    setLastPublishedRound(null);
    setRoundNumber(String(nextRound));
    setRoundLabel(`Round ${nextRound}`);
    setMatchDate("");
    setScoreReviewMode(false);
    setScoresReviewed(false);
    setPreview(null);
    setScores({});
    setMovementPlan(null);
    setMovementPlanStale(false);
    setMovementOverrides({});
    setOverrideReason("");
    setRosterAction("none");
    setIncomingPlayerId("");
    setReplacedPlayerId("");
    setWorkflowStep(1);
    setMessage(`Round ${nextRound} is ready with the saved movement and roster order. Confirm its date before continuing.`);
  }

  function updateCourt(index: number, patch: Partial<CourtDraft>) {
    setCourts((current) => current.map((row, idx) => idx === index ? { ...row, ...patch } : row));
    setPreview(null);
    setScores({});
    invalidateScoreReview();
    markPlanStale();
  }

  function addCourt() {
    setCourts((current) => [...current, { court: String(current.length + 1), formatType: "4-player", playerNames: "" }]);
    setPreview(null);
    setScores({});
    invalidateScoreReview();
    markPlanStale();
  }

  function removeCourt(index: number) {
    setCourts((current) => current.filter((_, idx) => idx !== index).map((row, idx) => ({ ...row, court: String(idx + 1) })));
    setPreview(null);
    setScores({});
    invalidateScoreReview();
    markPlanStale();
  }

  function courtValidationErrors(): string[] {
    const errors: string[] = [];
    const seenNames = new Set<string>();
    const seenCourtNumbers = new Set<number>();
    const rosterByName = new Map(sessionRoster.map((row) => [playerNameKey(row.player_name), row]));
    const clubPlayerNameCounts = knownPlayers.reduce((counts, player) => {
      const key = playerNameKey(player.name);
      counts.set(key, (counts.get(key) || 0) + 1);
      return counts;
    }, new Map<string, number>());
    const benchedPlayerIds = new Set(benchOverrideIds.map(Number));
    const expectedActiveNames = new Map(
      sessionRoster
        .filter((row) => !benchedPlayerIds.has(Number(row.player_id)))
        .map((row) => [playerNameKey(row.player_name), row.player_name])
    );
    if (rosterSuggestion && !sameNumberSet(benchOverrideIds, rosterSuggestion.bench_player_ids || [])) {
      errors.push("Bench selection changed. Refresh the roster suggestion before previewing courts.");
    }
    courts.forEach((court, index) => {
      const label = `Court ${court.court || index + 1}`;
      const rawCourtNumber = court.court.trim();
      const courtNumber = Number(rawCourtNumber);
      if (!/^[1-9]\d*$/.test(rawCourtNumber) || !Number.isSafeInteger(courtNumber)) {
        errors.push(`${label}: court number must be a positive whole number.`);
      } else if (seenCourtNumbers.has(courtNumber)) {
        errors.push(`Court ${courtNumber} is listed more than once.`);
      } else {
        seenCourtNumbers.add(courtNumber);
      }
      const names = splitNames(court.playerNames);
      const expectedSize = Number.parseInt(court.formatType, 10);
      if (![4, 5].includes(expectedSize)) errors.push(`${label}: League Live supports four- or five-player courts.`);
      else if (names.length !== expectedSize) errors.push(`${label}: ${court.formatType} requires exactly ${expectedSize} players.`);
      for (const name of names) {
        const key = playerNameKey(name);
        if (seenNames.has(key)) errors.push(`${name} appears on more than one court.`);
        seenNames.add(key);
        const rosterRow = rosterByName.get(key);
        if (!rosterRow) errors.push(`${name} is not in this session player roster.`);
        else if (benchedPlayerIds.has(Number(rosterRow.player_id))) errors.push(`${name} is selected for the bench and cannot also be assigned to a court.`);
        if ((clubPlayerNameCounts.get(key) || 0) > 1) errors.push(`${name} matches multiple club player IDs. Make those player names unique in Players before generating a League Live preview.`);
      }
    });
    for (const [key, playerName] of expectedActiveNames) {
      if (!seenNames.has(key)) errors.push(`${playerName} is active for this round but is not assigned to a court.`);
    }
    const orderedCourtNumbers = [...seenCourtNumbers].sort((left, right) => left - right);
    if (orderedCourtNumbers.length === courts.length && orderedCourtNumbers.some((courtNumber, index) => courtNumber !== index + 1)) {
      errors.push("Court numbers must be contiguous starting at 1.");
    }
    if (!courts.length) errors.push("Add at least one court.");
    return [...new Set(errors)];
  }

  async function generatePreview() {
    const setupError = setupContextError();
    if (setupError) {
      setWorkflowStep(1);
      setMessage(setupError);
      return;
    }
    const validationErrors = courtValidationErrors();
    if (validationErrors.length) {
      setPreview(null);
      setMessage(validationErrors.join(" "));
      return;
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const courtPayload = courtsToPayload(courts, currentRound).map((court) => ({ court: court.court, format_type: court.format_type, player_names: court.player_names }));
      const payload = await requestJson<AdminMatchUploaderRoundRobinPreview>(`/admin/clubs/${encodeURIComponent(clubId)}/match-uploader/round-robin/preview`, {
        method: "POST",
        body: JSON.stringify({ courts: courtPayload, schedule_mode: "full", source: "next_league_manager_live_preview" })
      });
      if (!actionRequest.isCurrent(generation)) return;
      if (payload.missing_players?.length) {
        setPreview(null);
        setMessage(`Missing players: ${payload.missing_players.join(", ")}. Return to Players and resolve them before continuing.`);
        return;
      }
      setPreview(payload);
      const nextScores: Record<string, ScoreDraft> = {};
      for (const match of (payload.courts || []).flatMap((court) => court.matches || [])) {
        for (let gameNumber = 1; gameNumber <= matchStructure.games; gameNumber += 1) {
          nextScores[scoreKey(match.row_id, gameNumber)] = { scoreT1: "", scoreT2: "" };
        }
      }
      setScores(nextScores);
      setMovementPlan(null);
      setMovementPlanStale(false);
      setMovementOverrides({});
      setScoreReviewMode(false);
      setScoresReviewed(false);
      setMessage(`Generated ${payload.match_count || 0} matchup slot(s) using ${matchStructureLabel(matchStructure)}. Review the court preview, then save the resumable session.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to generate round preview.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function buildScoredMatches(): Array<Record<string, unknown>> {
    return allPreviewMatches.flatMap((match) => {
      const state = seriesScoreState(match.row_id, scores, matchStructure);
      if (!state.complete) return [];
      return Array.from({ length: state.playedGames }, (_, index) => {
        const gameNumber = index + 1;
        const score = scores[scoreKey(match.row_id, gameNumber)] || { scoreT1: "", scoreT2: "" };
        return {
          date: matchDate,
          league: leagueName,
          week_tag: weekTag,
          match_type: "League Manager Live",
          court: match.court,
          t1_p1: match.t1_p1,
          t1_p2: match.t1_p2,
          t2_p1: match.t2_p1,
          t2_p2: match.t2_p2,
          score_t1: Number(score.scoreT1),
          score_t2: Number(score.scoreT2),
          series_key: match.row_id,
          series_kind: matchStructure.kind,
          series_games: matchStructure.games,
          game_number: gameNumber
        };
      });
    });
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
    const setupError = setupContextError();
    if (setupError) {
      setWorkflowStep(1);
      setMessage(setupError);
      return;
    }
    if (!requireCurrentSession("previewing next-round movement")) return;
    if (!scoresReviewed) {
      setMessage("Review and confirm the entered scores before previewing movement.");
      return;
    }
    const matches = buildScoredMatches();
    if (!allSeriesComplete || !matches.length) {
      setMessage(`Complete every ${matchStructureLabel(matchStructure)} matchup before previewing next-round movement (${completeSeriesCount} of ${allPreviewMatches.length} complete).`);
      return;
    }
    const generation = actionRequest.begin();
    const requestedSessionId = loadedSessionId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveRoundPlan>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(requestedSessionId)}/rounds/${encodeURIComponent(String(currentRound))}/plan`, {
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
      if (!actionRequest.isCurrent(generation)) return;
      setMovementPlan(payload);
      setMovementPlanStale(false);
      setMessage(`Next-round plan ${payload.operation_key.slice(0, 12)}… is ready for Round ${payload.next_round}.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMovementPlanStale(true);
        setMessage(error instanceof Error ? error.message : "Unable to preview court movement.");
      }
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function submitRound(confirmationText: string): Promise<ActionCompletion> {
    const setupError = setupContextError();
    if (setupError) throw new Error(setupError);
    if (!requireCurrentSession("submitting official scores")) throw new Error("Resume the current persisted session before submitting official scores.");
    if (blockingCurrentRoundPublishOperation) {
      const error = new Error(
        `Round ${currentRound} already has a durable ${blockingCurrentRoundPublishOperation.status.replace(/_/g, " ")} publish. Use its Retry or Reconcile action below instead of starting a new publish.`
      );
      setMessage(error.message);
      throw error;
    }
    if (!scoresReviewed) {
      const error = new Error("Review and confirm every entered score before publishing the round.");
      setMessage(error.message);
      throw error;
    }
    const matches = buildScoredMatches();
    if (!allSeriesComplete || !matches.length) {
      const error = new Error(`Complete every generated matchup before publishing (${completeSeriesCount} of ${allPreviewMatches.length} complete; ${matches.length} game result(s) ready).`);
      setMessage(error.message);
      throw error;
    }
    if (!movementPlan || movementPlanStale) {
      const error = new Error("Preview the current movement plan before submitting. Any score, roster, bench, or override change makes the previous plan stale.");
      setMessage(error.message);
      throw error;
    }
    const publishedRoundNumber = currentRound;
    const generation = actionRequest.begin();
    const requestedSessionId = loadedSessionId;
    const roundOperationKey = movementPlan.operation_key;
    setBusy(true);
    setMessage(null);
    try {
      const plannedMovement = movementPlan.movement;
      const payload = await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(requestedSessionId)}/rounds/${encodeURIComponent(String(currentRound))}/submit`, {
        method: "POST",
        body: JSON.stringify({
          round_label: roundLabel,
          match_date: matchDate,
          preview: preview || {},
          matches,
          expected_match_count: matches.length,
          movement_overrides: movementOverridePayload(),
          override_reason: overrideReason || null,
          roster_change: rosterChangePayload(),
          bench_player_ids: benchOverrideIds,
          bench_override_reason: benchOverrideReason || null,
          expected_updated_at: sessionUpdatedAt,
          expected_operation_key: roundOperationKey,
          idempotency_key: roundOperationKey,
          courts: courtsToPayload(courts, currentRound),
          confirmation_text: confirmationText,
          source: "next_league_live_round_submit"
        })
      });
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the submitted round response was applied.");
      if (payload.session) {
        applySession(payload.session, payload.courts || [], payload.rounds || [...roundHistory, ...(payload.round ? [payload.round] : [])], publishOperations, detail);
        setRoundNumber(String(payload.session.current_round || currentRound));
        setRoundLabel(`Round ${payload.session.current_round || currentRound}`);
      }
      setPreview(null);
      setScores({});
      if (requestedSessionId) {
        await loadSessionDetail(requestedSessionId);
        if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the published round could be refreshed.");
      }
      setRatingReview(payload.rating_review || null);
      const movementText = plannedMovement?.applied ? ` Applied ${plannedMovement.rows.filter((row) => row.direction !== "stay").length} court movement(s) for the next round.` : " No court movement was required.";
      const successMessage = `${payload.idempotent_replay ? "Reconciled" : "Published"} ${payload.published_match_ids?.length ?? matches.length} league match(es) through one durable publish operation.${movementText}`;
      clearStoredRoundDraft(requestedSessionId, publishedRoundNumber);
      setLastPublishedRound(publishedRoundNumber);
      setRoundPublished(true);
      setWorkflowStep(6);
      setMessage(successMessage);
      return actionSuccess(payload.idempotent_replay ? "League round reconciled" : "League round published", successMessage);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to submit league round.");
      if (leagueLiveWriteIsUncertain(error)) {
        const operationReference = error instanceof LeagueLiveRequestError && error.operationKey
          ? error.operationKey
          : roundOperationKey;
        const failureDetail = error instanceof Error
          ? error.message
          : "The first request did not finish cleanly.";
        return actionUncertain(
          "League round publish needs verification",
          `${failureDetail} Retry these exact scores; League Live verifies any existing matches before it writes anything, so this cannot duplicate the round.`,
          operationReference,
          "Retry exact league-round publish",
          () => submitRound(confirmationText)
        );
      }
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function publishRoundAndStay(confirmationText: string): Promise<ActionCompletion> {
    const publishedRoundNumber = currentRound;
    const completion = await submitRound(confirmationText);
    if (completion.status !== "success") return completion;
    setLastPublishedRound(publishedRoundNumber);
    setRoundPublished(true);
    setWorkflowStep(6);
    return completion;
  }

  async function createGuest(confirmationText: string): Promise<ActionCompletion> {
    if (!requireCurrentSession("creating a guest")) throw new Error("Resume the current persisted session before creating a guest.");
    const generation = actionRequest.begin();
    const requestedSessionId = loadedSessionId;
    const idempotencyKey = `guest:${requestedSessionId}:${guestName.trim().toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 60)}`;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveGuestResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(requestedSessionId)}/guests`, {
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
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the guest response was applied.");
      const guest = { id: Number(payload.player.id), name: payload.player.name, rating: payload.player.rating };
      setGuestPlayers((current) => current.some((row) => row.id === guest.id) ? current : [...current, guest]);
      setIncomingPlayerId(String(guest.id));
      setGuestName("");
      setGuestReason("");
      markPlanStale();
      const successMessage = `${payload.idempotent_replay ? "Recovered" : "Created"} guest ${guest.name}. Select add or substitute, then preview movement again.`;
      setMessage(successMessage);
      return actionSuccess(payload.idempotent_replay ? "Guest recovered" : "Guest created", successMessage);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to create League Live guest.");
      if (leagueLiveWriteIsUncertain(error)) {
        return actionUncertain(
          "Guest creation needs verification",
          "The guest request may have completed. Retry the exact retained request to reconcile before adding the guest again.",
          idempotencyKey,
          "Retry exact guest creation",
          () => createGuest(confirmationText)
        );
      }
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function reconcileRound(round: number, confirmationText: string): Promise<ActionCompletion> {
    if (!requireCurrentSession("reconciling a round")) throw new Error("Resume the current persisted session before reconciling a round.");
    const generation = actionRequest.begin();
    const requestedSessionId = loadedSessionId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(requestedSessionId)}/rounds/${encodeURIComponent(String(round))}/reconcile`, {
        method: "POST",
        body: JSON.stringify({ confirmation_text: confirmationText, source: "next_league_live_round_reconcile" })
      });
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the round reconciliation response was applied.");
      await loadSessionDetail(requestedSessionId);
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the reconciled round could be refreshed.");
      setRatingReview(payload.rating_review || null);
      setMessage(`Round ${round} publish and League Live snapshot are reconciled. No match was republished.`);
      return actionSuccess("League round reconciled", `Round ${round} and its League Live snapshot were reconciled without republishing a match.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to reconcile League Live round.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function retryRetainedRound(round: number, confirmationText: string): Promise<ActionCompletion> {
    if (!requireCurrentSession("retrying a retained round publish")) throw new Error("Resume the current persisted session before retrying the retained publish.");
    const generation = actionRequest.begin();
    const requestedSessionId = loadedSessionId;
    const operationReference = publishOperations.find((operation) => operation.round_number === round)?.id || requestedSessionId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(requestedSessionId)}/rounds/${encodeURIComponent(String(round))}/retry`, {
        method: "POST",
        body: JSON.stringify({ confirmation_text: confirmationText, source: "next_league_live_round_retry" })
      });
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the retained round retry response was applied.");
      await loadSessionDetail(requestedSessionId);
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the recovered round could be refreshed.");
      setRatingReview(payload.rating_review || null);
      clearStoredRoundDraft(requestedSessionId, round);
      setLastPublishedRound(round);
      setRoundPublished(true);
      setWorkflowStep(6);
      const matchCount = payload.published_match_ids?.length || 0;
      const successMessage = `Retried the retained Round ${round} publish with its original key and verified ${matchCount} official match${matchCount === 1 ? "" : "es"}.`;
      setMessage(successMessage);
      return actionSuccess("League round publish recovered", successMessage);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to retry the retained League Live round.");
      if (leagueLiveWriteIsUncertain(error)) {
        const failureDetail = error instanceof Error ? error.message : "The retained publish retry did not finish cleanly.";
        return actionUncertain(
          "League round retry needs verification",
          `${failureDetail} Retry the retained operation again; the server reuses its original request and verifies existing matches first.`,
          operationReference,
          "Retry retained league-round publish",
          () => retryRetainedRound(round, confirmationText)
        );
      }
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function verifyCompensation(round: number, confirmationText: string): Promise<ActionCompletion> {
    if (!requireCurrentSession("verifying compensation")) throw new Error("Resume the current persisted session before verifying compensation.");
    const generation = actionRequest.begin();
    const requestedSessionId = loadedSessionId;
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<LeagueLiveWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(requestedSessionId)}/rounds/${encodeURIComponent(String(round))}/compensate`, {
        method: "POST",
        body: JSON.stringify({
          recovery_reference: compensationReference,
          reason: compensationReason,
          confirmation_text: confirmationText,
          source: "next_league_live_round_compensate"
        })
      });
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the compensation response was applied.");
      setCompensationReference("");
      setCompensationReason("");
      await loadSessionDetail(requestedSessionId);
      if (!actionRequest.isCurrent(generation)) throw new Error("The admin session changed before the compensated round could be refreshed.");
      setMessage(`Round ${round} recovery is recorded as compensated. No active deterministic match context remained.`);
      return actionSuccess("Round compensation verified", `Round ${round} recovery was recorded as compensated.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to verify League Live compensation.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function downloadExport(kind: "matches" | "ratings" | "roster" | "rounds") {
    if (!requireCurrentSession("exporting session data")) return;
    const generation = actionRequest.begin();
    const requestedSessionId = loadedSessionId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<LeagueLiveExportResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/league-manager/live-sessions/${encodeURIComponent(requestedSessionId)}/export?kind=${encodeURIComponent(kind)}`);
      if (!actionRequest.isCurrent(generation)) return;
      const blob = new Blob([payload.csv_text], { type: payload.content_type });
      const href = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = payload.filename;
      anchor.click();
      URL.revokeObjectURL(href);
      setMessage(`Exported ${payload.row_count} ${kind} row(s).`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : `Unable to export ${kind}.`);
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  const liveCourtFormats = (uploaderStatus.round_robin_format_options || ["4-Player", "5-Player"])
    .filter((option) => /^[45]-player$/i.test(option));
  if (!liveCourtFormats.length) liveCourtFormats.push("4-Player", "5-Player");
  const unresolvedPastedPlayers = pastedPlayerResolutions.filter((row) => row.status !== "existing").length;
  const persistedSessionPlayable = !sessionIsCurrentLeague || sessionStatus === "active";
  const setupReady = Boolean(detail && loadedLeagueName === leagueName && roundContextValid && matchDateValid && persistedSessionPlayable);
  const playersReady = setupReady && attendeePlayerIds.length >= 4 && pasteResolutionCurrent && !pasteDuplicateNames.length && !unresolvedPastedPlayers && !playerCreateRecovery && Boolean(rosterSuggestion || (sessionIsCurrentLeague && sessionRoster.length >= 4));
  const courtsReady = playersReady && Boolean(preview?.courts?.length) && sessionIsCurrentLeague;
  const scoreEntryReady = courtsReady && scoresReviewed;
  const movementReady = scoreEntryReady && Boolean(movementPlan) && !movementPlanStale;
  const maxReachableStep: WorkflowStep = roundPublished
    ? 6
    : movementReady
      ? 6
      : scoreEntryReady
        ? 5
        : courtsReady
          ? 4
          : playersReady
            ? 3
            : setupReady
              ? 2
              : 1;

  function navigateWorkflow(step: WorkflowStep) {
    if (step >= 2 && step <= 5 && sessionIsCurrentLeague && (sessionStatus !== "active" || roundPublished)) {
      setMessage(roundPublished
        ? "This round is already published. Use Repeat or Finish to start the next round or close the session."
        : `This session is ${sessionStatus.replace(/_/g, " ")}. Resume it from Setup before editing players, courts, or scores.`);
      return;
    }
    if (step > maxReachableStep) {
      setMessage(`Complete ${WORKFLOW_STEPS[maxReachableStep - 1].label} before opening ${WORKFLOW_STEPS[step - 1].label}.`);
      return;
    }
    setWorkflowStep(step);
  }

  if (!leagueStatus.enabled || !uploaderStatus.enabled || !liveDomainStatus.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>League Live is not available yet</h2>
        <p style={{ color: "#475569" }}>Live round scoring remains unavailable in this build. Continue using the existing League Manager scoring workflow for league night.</p>
      </article>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2>
        <p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>
        <p style={{ color: "#166534" }}><strong>Movement planning:</strong> {liveDomainStatus.movement_authority === "python_fastapi" ? "Ready" : "Unavailable"}. League Live calculates the plan; this page only displays it.</p>
        <p style={{ color: liveDomainStatus.submit_enabled ? "#166534" : "#92400e" }}><strong>Result publishing:</strong> {liveDomainStatus.submit_enabled ? "Available" : "Unavailable in this build"}.</p>
        <p style={{ color: "#475569" }}><strong>Recovery:</strong> Interrupted operations can be reviewed below before play resumes.</p>
        {sessionLoading ? <p>Checking session…</p> : null}
        {sessionMessage ? <p style={{ color: "#64748b" }}>{sessionMessage}</p> : null}
      </article>

      {createRecovery ? (
        <article aria-live="polite" style={{ ...cardStyle, background: "#fffbeb", borderColor: "#f59e0b" }}>
          <h2 style={{ marginTop: 0 }}>Session create needs exact-operation recovery</h2>
          <p style={{ color: "#92400e" }}>Do not create another persisted League Live session until this exact operation is reconciled.</p>
          <p><strong>Operation key:</strong> <code style={{ overflowWrap: "anywhere" }}>{createRecovery.operationKey}</code><br /><strong>Last known status:</strong> {createRecovery.status.replace(/_/g, " ")}</p>
          <p>{leagueLiveOperatorMessage(createRecovery.message, "Unable to verify the League Live session.")}</p>
          <button type="button" onClick={() => void reconcileCreateSession()} disabled={checkingCreateRecovery || !accessToken} style={ghostButtonStyle}>
            {checkingCreateRecovery ? "Checking and reconciling…" : "Check and reconcile exact operation"}
          </button>
        </article>
      ) : null}

      <nav aria-label="League Live workflow" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 150px), 1fr))", gap: "0.5rem" }}>
        {WORKFLOW_STEPS.map((step) => {
          const active = workflowStep === step.id;
          const sessionLocksEditing = step.id >= 2 && step.id <= 5 && sessionIsCurrentLeague && (sessionStatus !== "active" || roundPublished);
          const available = step.id <= maxReachableStep && !sessionLocksEditing;
          return (
            <button
              key={step.id}
              type="button"
              aria-current={active ? "step" : undefined}
              onClick={() => navigateWorkflow(step.id)}
              disabled={busy || !available}
              style={{
                ...ghostButtonStyle,
                borderColor: active ? "#2563eb" : available ? "#94a3b8" : "#e2e8f0",
                background: active ? "#dbeafe" : "white",
                color: active ? "#1d4ed8" : available ? "#0f172a" : "#94a3b8",
                textAlign: "left"
              }}
            >
              <span style={{ display: "block", fontSize: "0.75rem" }}>Step {step.id}</span>
              {step.label}
            </button>
          );
        })}
      </nav>

      {message ? (
        <p role="status" aria-live="polite" style={{ margin: 0, padding: "0.75rem 1rem", borderRadius: "10px", background: "#f8fafc", color: /unable|missing|duplicate|required|error|failed|stop/i.test(message) ? "#b91c1c" : "#166534" }}>
          {message}
        </p>
      ) : null}

      {workflowStep === 1 ? (
        <article style={cardStyle} aria-labelledby="league-live-setup-heading">
          <h2 id="league-live-setup-heading" style={{ marginTop: 0 }}>1. Setup</h2>
          <p style={{ color: "#475569" }}>Choose the Individual Doubles league and round context, or resume a saved session. Singles and Team live rounds stay closed because this court workflow requires four distinct doubles players per match.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            <label>League<br /><select value={leagueName} onChange={(event) => selectLeague(event.target.value)} disabled={busy || !accessToken} aria-busy={busy && !leagues.length} style={inputStyle}><option value="" disabled>{busy && !leagues.length ? "Loading leagues…" : "Choose a league"}</option>{leagues.map((name) => <option key={name} value={name}>{name}</option>)}</select></label>
            <label>Week<br /><input value={weekTag} onChange={(event) => { setWeekTag(event.target.value); invalidateRoundDraft("Week changed. Generate and review a fresh preview before publishing."); }} disabled={busy || roundPublished || sessionStatus === "complete" || sessionStatus === "archived"} style={inputStyle} /></label>
            <label>Round #<br /><input type="number" min={1} max={50} step={1} value={roundNumber} onChange={(event) => { setRoundNumber(event.target.value); setRoundLabel(`Round ${event.target.value || 1}`); invalidateRoundDraft("Round number changed. Generate and review a fresh preview before publishing."); }} disabled={busy || sessionIsCurrentLeague || roundPublished} style={inputStyle} /></label>
            <label>Total rounds<br /><input type="number" min={1} max={50} step={1} value={totalRounds} onChange={(event) => { setTotalRounds(event.target.value); if (roundPublished) setMessage("Total rounds changed. Start the next round only after the new total is a valid whole number through 50."); else invalidateRoundDraft("Total rounds changed. Generate and review a fresh preview before publishing."); }} disabled={busy || sessionStatus === "complete" || sessionStatus === "archived"} style={inputStyle} /></label>
            <label>Date *<br /><input required type="date" value={matchDate} onChange={(event) => { setMatchDate(event.target.value); invalidateRoundDraft("Round date changed. Generate and review a fresh preview before publishing."); }} disabled={busy || roundPublished || sessionStatus === "complete" || sessionStatus === "archived"} style={inputStyle} /></label>
            <label>Round label<br /><input value={roundLabel} onChange={(event) => { setRoundLabel(event.target.value); invalidateRoundDraft("Round label changed. Review a fresh preview before publishing."); }} disabled={busy || roundPublished || sessionStatus === "complete" || sessionStatus === "archived"} style={inputStyle} /></label>
            <label>Notes<br /><input value={sessionNotes} onChange={(event) => setSessionNotes(event.target.value)} disabled={busy} style={inputStyle} /></label>
            <button type="button" onClick={() => void loadLeagueDetail()} disabled={busy || !leagueName || sessionIsCurrentLeague} style={ghostButtonStyle}>{loadingLeagueName ? "Loading roster…" : "Reload roster"}</button>
          </div>
          {loadingLeagueName ? <p role="status" style={{ color: "#475569" }}>Loading {loadingLeagueName}. Session writes remain unavailable until the replacement roster is ready.</p> : null}
          {!roundContextValid ? <p role="alert" style={{ color: "#b91c1c" }}><strong>Round setup required:</strong> use whole numbers with 1 ≤ Round # ≤ Total rounds ≤ 50.</p> : null}
          {!matchDateValid && !roundPublished ? <p role="alert" style={{ color: "#b91c1c" }}><strong>Date required:</strong> confirm the date for this round. A resumed session never assumes today&apos;s date.</p> : null}
          {sessionIsCurrentLeague && ["paused", "setup"].includes(sessionStatus) ? (
            <section style={{ marginTop: "0.75rem", padding: "0.75rem", border: "1px solid #f59e0b", borderRadius: "12px", background: "#fffbeb" }}>
              <strong>This saved session is {sessionStatus}.</strong>
              <p>Activate it before editing players, courts, or scores.</p>
              <ConfirmAction triggerLabel={sessionStatus === "paused" ? "Resume session" : "Activate session"} title={sessionStatus === "paused" ? "Resume this League Live session?" : "Activate this League Live session?"} description="This changes only the persisted session status to active; saved players and courts remain intact." confirmLabel="Yes, make active" confirmationText="SAVE SESSION" disabled={busy} busy={busy} onConfirm={resumePausedSession} />
            </section>
          ) : null}
          {sessionIsCurrentLeague && sessionStatus === "active" && !roundPublished ? (
            <section style={{ marginTop: "0.75rem", padding: "0.75rem", border: "1px solid #cbd5e1", borderRadius: "12px", background: "#f8fafc" }}>
              <strong>This saved session is active.</strong>
              <p>Pause it when league-night work must stop; saved players and courts remain available for a guarded resume.</p>
              <ConfirmAction triggerLabel="Pause session" title="Pause this League Live session?" description="This persists paused status without changing official matches, saved players, or saved courts." confirmLabel="Yes, pause session" confirmationText="SAVE SESSION" disabled={busy} busy={busy} onConfirm={pauseActiveSession} />
            </section>
          ) : null}
          {sessionIsCurrentLeague && !["active", "paused", "setup"].includes(sessionStatus) ? <p role="alert" style={{ color: "#92400e" }}>This session is {sessionStatus}. Its players, courts, and scores are read-only; use Repeat or Finish and recovery/history below.</p> : null}
          <section style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid #e2e8f0" }}>
            <h3 style={{ marginTop: 0 }}>Resume an unfinished session</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 220px), 1fr))", gap: "0.75rem", alignItems: "end" }}>
              <label>Unfinished sessions for this league<br /><select value={sessionId} onChange={(event) => selectSession(event.target.value)} disabled={busy} style={inputStyle}><option value="">Select session…</option>{liveSessions.map((row) => <option key={row.id} value={row.id}>{row.week_tag} · R{row.current_round}/{row.total_rounds} · {row.status}</option>)}</select></label>
              <button type="button" onClick={() => void loadSessions()} disabled={busy || !accessToken || !leagueName} style={ghostButtonStyle}>Refresh sessions</button>
              <button type="button" onClick={() => void loadSessionDetail()} disabled={busy || !sessionId} style={ghostButtonStyle}>Retry selected session</button>
            </div>
          </section>
          <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center" }}>
            <button type="button" onClick={() => void loadInitialWorkspace()} disabled={busy || !accessToken} style={ghostButtonStyle}>{busy ? "Refreshing…" : "Refresh leagues"}</button>
            <button type="button" onClick={() => setWorkflowStep(2)} disabled={busy || !setupReady} style={buttonStyle}>Continue to Players</button>
          </p>
        </article>
      ) : null}

      {workflowStep === 2 ? (
        <article style={cardStyle} aria-labelledby="league-live-players-heading">
          <h2 id="league-live-players-heading" style={{ marginTop: 0 }}>2. Players</h2>
          {participationMode === "flex" ? (
            <section style={{ padding: "0.75rem", border: "1px solid #bfdbfe", borderRadius: "12px", background: "#eff6ff" }}>
              <h3 style={{ marginTop: 0 }}>Flex attendance for this session</h3>
              <p style={{ color: "#475569" }}>Select today&apos;s attendees. Every new Flex session resets attendance and rebuilds rating-seeded pods/courts from this list.</p>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                <button type="button" onClick={() => replaceFlexAttendance(activeLeagueMembers.map((row) => Number(row.player_id)), "All active league members selected. Paste or add any non-roster attendees next.")} disabled={busy || !activeLeagueMembers.length} style={ghostButtonStyle}>Select all rostered players</button>
                <button type="button" onClick={() => replaceFlexAttendance([], "Flex attendance cleared.")} disabled={busy || !attendeePlayerIds.length} style={ghostButtonStyle}>Clear attendance</button>
              </div>
            </section>
          ) : null}
          {participationMode === "set" ? <p style={{ color: "#475569" }}><strong>Set participation:</strong> the season roster starts selected. Resume a persisted session to retain its saved pods and positions; existing or newly created non-roster players can still be added for this night.</p> : null}

          <section style={{ marginTop: "1rem" }}>
            <h3>League roster</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.35rem" }}>
              {activeLeagueMembers.map((row) => (
                <label key={row.player_id} style={{ display: "flex", gap: "0.45rem", alignItems: "center" }}>
                  <input type="checkbox" checked={attendeePlayerIds.includes(Number(row.player_id))} onChange={(event) => changeFlexAttendance(Number(row.player_id), event.target.checked)} disabled={busy} />
                  {row.player_name}{row.overall_rating_jupr == null ? "" : ` · ${Number(row.overall_rating_jupr).toFixed(2)}`}
                </label>
              ))}
            </div>
            <label style={{ display: "block", marginTop: "0.75rem" }}>Add an existing club player, including a non-roster player<br />
              <select value="" onChange={(event) => { if (event.target.value) { appendSelectedPlayerIds([Number(event.target.value)]); invalidateFlexRosterPlan("Existing club player added. Build fresh courts when the player list is final."); } }} disabled={busy} style={inputStyle}>
                <option value="">Choose an existing player…</option>
                {knownPlayers.filter((player) => !attendeePlayerIds.includes(Number(player.id))).map((player) => <option key={String(player.id)} value={String(player.id)}>{player.name} · {playerJuprLabel(player)} · ID {player.id}</option>)}
              </select>
            </label>
          </section>

          <section style={{ marginTop: "1rem", padding: "0.75rem", border: "1px solid #e2e8f0", borderRadius: "12px" }}>
            <h3 style={{ marginTop: 0 }}>Paste players</h3>
            <p style={{ color: "#475569" }}>Paste names separated by commas or new lines. Existing club players are matched case-insensitively. Missing names must be created with an explicit Starting JUPR before continuing.</p>
            <textarea value={pastedPlayerText} onChange={(event) => changePastedPlayerText(event.target.value)} disabled={busy || creatingMissingPlayers} rows={5} placeholder={"Alex Rivera, Casey Lee\nMorgan Chen"} style={inputStyle} />
            <p><button type="button" onClick={resolvePlayerNames} disabled={busy || creatingMissingPlayers || !pastedPlayerText.trim() || Boolean(playerCreateRecovery)} style={ghostButtonStyle}>Resolve and add pasted players</button></p>
            {!pasteResolutionCurrent && pastedPlayerText.trim() ? <p role="alert" style={{ color: "#92400e" }}><strong>Resolve required:</strong> the pasted list changed. Resolve the complete list before building courts; players added by the previous paste were removed from the session order.</p> : null}
            {pasteDuplicateNames.length ? <p role="alert" style={{ color: "#b91c1c" }}><strong>Duplicate names rejected:</strong> {pasteDuplicateNames.join(", ")}. Remove duplicates and resolve the list again.</p> : null}
            {pastedPlayerResolutions.length ? (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Pasted name</th><th align="left">Resolution</th><th align="left">Matched player</th></tr></thead><tbody>
                  {pastedPlayerResolutions.map((row) => <tr key={playerNameKey(row.inputName)}><td>{row.inputName}</td><td>{row.status === "existing" ? "Existing player" : row.status === "ambiguous" ? "Ambiguous — make the duplicate club player names unique in Players, then reload" : "Missing — create new player"}</td><td>{row.playerName || "Not found"}</td></tr>)}
                </tbody></table>
              </div>
            ) : null}
            {missingPlayerDrafts.length ? (
              <section style={{ marginTop: "1rem", padding: "0.75rem", borderRadius: "10px", background: "#fffbeb", border: "1px solid #f59e0b" }}>
                <h4 style={{ marginTop: 0 }}>Create missing players</h4>
                <p style={{ color: "#92400e" }}>Every new player must receive an explicit Starting JUPR from 1.0 through 7.0. No value is assumed.</p>
                <div style={{ display: "grid", gap: "0.5rem" }}>
                  {missingPlayerDrafts.map((draft, index) => (
                    <div key={playerNameKey(draft.name)} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 180px), 1fr))", gap: "0.5rem" }}>
                      <label>Name<br /><input value={draft.name} readOnly aria-readonly="true" style={{ ...inputStyle, background: "#f8fafc" }} /></label>
                      <label>Starting JUPR *<br /><input required type="number" min={1} max={7} step={0.01} value={draft.startingJupr} onChange={(event) => setMissingPlayerDrafts((current) => current.map((row, rowIndex) => rowIndex === index ? { ...row, startingJupr: event.target.value } : row))} disabled={creatingMissingPlayers} placeholder="Required" style={inputStyle} /></label>
                    </div>
                  ))}
                </div>
                <p><button type="button" onClick={() => void createMissingPlayers()} disabled={creatingMissingPlayers || busy || Boolean(playerCreateRecovery) || missingPlayerDrafts.some((draft) => !Number.isFinite(Number(draft.startingJupr)) || Number(draft.startingJupr) < 1 || Number(draft.startingJupr) > 7)} style={buttonStyle}>{creatingMissingPlayers ? "Creating players…" : "Create missing players"}</button></p>
              </section>
            ) : null}
            {playerCreateRecovery ? (
              <section aria-live="polite" style={{ marginTop: "1rem", padding: "0.75rem", borderRadius: "10px", background: "#fffbeb", border: "1px solid #f59e0b" }}>
                <strong>Player creation needs exact-operation recovery.</strong>
                <p style={{ overflowWrap: "anywhere" }}>Operation: <code>{playerCreateRecovery.operationKey}</code>. Do not create another batch.</p>
                <button type="button" onClick={() => void inspectPlayerCreateOperation()} disabled={checkingPlayerCreateRecovery || !accessToken} style={ghostButtonStyle}>{checkingPlayerCreateRecovery ? "Checking and reconciling…" : "Check and reconcile exact player operation"}</button>
              </section>
            ) : null}
          </section>

          <section style={{ marginTop: "1rem" }}>
            <h3>Session player order ({selectedSessionPlayers.length})</h3>
            <p style={{ color: "#475569" }}>This is the live-order source. Late arrivals added between rounds append to the order; substitutions replace the selected ordered player.</p>
            {selectedSessionPlayers.length ? <ol>{selectedSessionPlayers.map((player) => <li key={String(player.id)} style={{ marginBottom: "0.35rem" }}>{player.name} · {playerJuprLabel(player)} <button type="button" onClick={() => removeSelectedPlayer(Number(player.id))} disabled={busy} style={{ ...ghostButtonStyle, padding: "0.2rem 0.55rem", marginLeft: "0.5rem" }}>Remove</button></li>)}</ol> : <p>No players selected.</p>}
          </section>
          <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", justifyContent: "space-between" }}>
            <button type="button" onClick={() => setWorkflowStep(1)} disabled={busy} style={ghostButtonStyle}>Back to Setup</button>
            <button type="button" onClick={() => void buildPlayerPlanAndContinue()} disabled={busy || creatingMissingPlayers || Boolean(playerCreateRecovery) || attendeePlayerIds.length < 4 || !pasteResolutionCurrent || Boolean(pasteDuplicateNames.length) || Boolean(unresolvedPastedPlayers)} style={buttonStyle}>{sessionIsCurrentLeague && sessionRoster.length ? "Continue with saved courts" : `Build courts from ${attendeePlayerIds.length} attendee${attendeePlayerIds.length === 1 ? "" : "s"}`}</button>
          </p>
        </article>
      ) : null}

      {workflowStep === 3 ? (
        <article style={cardStyle} aria-labelledby="league-live-courts-heading">
          <h2 id="league-live-courts-heading" style={{ marginTop: 0 }}>3. Courts and Preview</h2>
          <p style={{ color: "#475569" }}>Review the rating-seeded roster and bench, validate each court, then preview the exact teams and match slots before creating or updating the persisted session.</p>
          {rosterSuggestion ? (
            <section style={{ padding: "0.75rem", border: "1px solid #dbeafe", borderRadius: "12px", background: "#eff6ff" }}>
              <h3 style={{ marginTop: 0 }}>Roster and bench suggestion</h3>
              <p style={{ color: "#475569" }}>{participationMode === "flex" ? "Fresh Flex courts are rating-seeded from this session's attendees." : "Set participation keeps saved pods and positions when a persisted session is resumed."} Select exactly the required bench count and explain any non-default choice.</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.35rem" }}>
                {rosterSuggestion.roster.map((row) => (
                  <label key={row.player_id} style={{ display: "flex", gap: "0.45rem", alignItems: "center" }}>
                    <input type="checkbox" checked={benchOverrideIds.includes(Number(row.player_id))} disabled={busy} onChange={(event) => { setBenchOverrideIds((current) => event.target.checked ? [...new Set([...current, Number(row.player_id)])] : current.filter((id) => id !== Number(row.player_id))); setPreview(null); markPlanStale(); }} />
                    Bench {row.player_name}{row.court_number ? ` (currently Court ${row.court_number})` : ""}
                  </label>
                ))}
              </div>
              <div data-responsive-bench-controls style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 240px), 1fr))", gap: "0.75rem", alignItems: "end", marginTop: "0.75rem" }}>
                <label>Bench override reason<br /><input value={benchOverrideReason} onChange={(event) => { setBenchOverrideReason(event.target.value); setPreview(null); markPlanStale(); }} disabled={busy} placeholder="Required when changing the default bench" style={inputStyle} /></label>
                <button type="button" onClick={refreshRosterSuggestion} disabled={busy} style={ghostButtonStyle}>Refresh roster suggestion</button>
              </div>
              <small style={{ color: "#64748b" }}>Roster fingerprint: {rosterSuggestion.fingerprint.slice(0, 16)}…</small>
            </section>
          ) : sessionIsCurrentLeague ? <p style={{ color: "#475569" }}><strong>Saved session roster loaded.</strong> Its persisted player order, bench, and courts are shown below; generate the current-round preview to continue.</p> : <p role="alert" style={{ color: "#b91c1c" }}>Return to Players and build a roster plan first.</p>}

          <datalist id="league-live-players">{playerOptions.map((name) => <option key={name} value={name} />)}</datalist>
          <section style={{ marginTop: "1rem" }}>
            <h3>Court assignments</h3>
            {courts.map((court, index) => (
              <div key={`${court.court}-${index}`} style={{ borderTop: index ? "1px solid #e2e8f0" : undefined, paddingTop: index ? "0.75rem" : 0, marginTop: index ? "0.75rem" : 0 }}>
                <div data-responsive-court-grid style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 180px), 1fr))", gap: "0.75rem", alignItems: "start" }}>
                  <label>Court<br /><input value={court.court} onChange={(event) => updateCourt(index, { court: event.target.value })} disabled={busy} style={inputStyle} /></label>
                  <label>Format<br /><select value={court.formatType} onChange={(event) => updateCourt(index, { formatType: event.target.value })} disabled={busy} style={inputStyle}>{liveCourtFormats.map((option) => <option key={option} value={option}>{option}</option>)}</select></label>
                  <label>Players, one per line<br /><textarea value={court.playerNames} onChange={(event) => updateCourt(index, { playerNames: event.target.value })} disabled={busy} rows={5} style={inputStyle} /></label>
                  <button type="button" onClick={() => removeCourt(index)} disabled={busy || courts.length === 1} style={ghostButtonStyle}>Remove court</button>
                </div>
              </div>
            ))}
            <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}><button type="button" onClick={addCourt} disabled={busy} style={ghostButtonStyle}>Add court</button><button type="button" onClick={() => void generatePreview()} disabled={busy || (!rosterSuggestion && !sessionIsCurrentLeague)} style={buttonStyle}>Validate courts and generate preview</button></p>
          </section>

          {preview?.courts?.length ? (
            <section style={{ marginTop: "1rem", padding: "0.75rem", border: "1px solid #86efac", borderRadius: "12px", background: "#f0fdf4" }}>
              <h3 style={{ marginTop: 0 }}>Match preview · {preview.match_count || 0} slots</h3>
              {(preview.courts as AdminMatchUploaderRoundRobinCourt[]).map((court) => (
                <section key={court.court} style={{ paddingTop: "0.5rem" }}>
                  <h4 style={{ margin: "0 0 0.35rem" }}>Court {court.court} · {court.format_type}</h4>
                  <ol>{(court.matches || []).map((match) => <li key={match.row_id}>{match.label}: {match.t1.map((player) => player.name).join(" / ")} vs {match.t2.map((player) => player.name).join(" / ")}</li>)}</ol>
                </section>
              ))}
              <p style={{ marginBottom: 0, color: "#166534" }}>Preview only — no scores are official yet.</p>
            </section>
          ) : null}

          {sessionIsCurrentLeague ? <p style={{ color: "#475569" }}><strong>Active persisted session:</strong> {loadedSessionId}</p> : <p style={{ color: "#92400e" }}>This session will be persisted after the court preview is ready.</p>}
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center" }}>
            <button type="button" onClick={() => setWorkflowStep(2)} disabled={busy} style={ghostButtonStyle}>Back to Players</button>
            {sessionIsCurrentLeague ? (
              <ConfirmAction triggerLabel="Save preview and continue" title="Save this session snapshot?" description="This persists the current player order, bench, courts, round, and notes before score entry." confirmLabel="Yes, save and continue" confirmationText="SAVE SESSION" disabled={busy || !preview?.courts?.length || Boolean(courtValidationErrors().length)} busy={busy} onConfirm={saveSnapshotAndContinue} />
            ) : (
              <ConfirmAction triggerLabel="Create session and continue" title="Create this persisted League Live session?" description="This saves the reviewed player order, bench, courts, and round so the night can be resumed." confirmLabel="Yes, create and continue" confirmationText="CREATE LIVE SESSION" disabled={busy || Boolean(createRecovery) || !preview?.courts?.length || Boolean(courtValidationErrors().length) || !rosterSuggestion || !sessionRoster.length} busy={busy} onConfirm={createSessionAndContinue} />
            )}
          </div>
        </article>
      ) : null}

      {workflowStep === 4 ? (
        <article style={cardStyle} aria-labelledby="league-live-scores-heading">
          <h2 id="league-live-scores-heading" style={{ marginTop: 0 }}>4. Score Entry with Review</h2>
          <p style={{ color: "#475569" }}><strong>Configured format: {matchStructureLabel(matchStructure)}.</strong> League Live publishes the whole round only after every matchup is complete, the exact teams and scores are reviewed, and movement is approved.</p>
          {!scoreReviewMode ? (
            <>
              <div style={{ display: "grid", gap: "0.75rem" }}>
                {(preview?.courts || []).map((court) => (
                  <section key={court.court} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
                    <h3 style={{ marginTop: 0 }}>Court {court.court} · {court.format_type}</h3>
                    {(court.matches || []).map((match) => {
                      const state = seriesScoreState(match.row_id, scores, matchStructure);
                      return (
                        <section key={match.row_id} style={{ borderTop: "1px solid #f1f5f9", padding: "0.75rem 0" }}>
                          <div><strong>{match.label}</strong><br />{match.t1.map((player) => player.name).join(" / ")} vs {match.t2.map((player) => player.name).join(" / ")}</div>
                          <div style={{ display: "grid", gap: "0.45rem", marginTop: "0.5rem" }}>
                            {Array.from({ length: matchStructure.games }, (_, gameIndex) => {
                              const gameNumber = gameIndex + 1;
                              const key = scoreKey(match.row_id, gameNumber);
                              const score = scores[key] || { scoreT1: "", scoreT2: "" };
                              const afterClinch = state.clinchedAt != null && gameNumber > state.clinchedAt;
                              const canClearUnexpectedScore = afterClinch && scoreHasValue(score);
                              return (
                                <div key={key} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 140px), 1fr))", gap: "0.6rem", alignItems: "center" }}>
                                  <strong>Game {gameNumber}</strong>
                                  <input aria-label={`${match.label} Game ${gameNumber} Team 1 score`} value={score.scoreT1} onChange={(event) => { setScores((current) => ({ ...current, [key]: { ...(current[key] || { scoreT1: "", scoreT2: "" }), scoreT1: event.target.value } })); setScoresReviewed(false); setScoreReviewMode(false); setMovementPlan(null); }} disabled={busy || (afterClinch && !canClearUnexpectedScore)} inputMode="numeric" placeholder="Team 1" style={inputStyle} />
                                  <input aria-label={`${match.label} Game ${gameNumber} Team 2 score`} value={score.scoreT2} onChange={(event) => { setScores((current) => ({ ...current, [key]: { ...(current[key] || { scoreT1: "", scoreT2: "" }), scoreT2: event.target.value } })); setScoresReviewed(false); setScoreReviewMode(false); setMovementPlan(null); }} disabled={busy || (afterClinch && !canClearUnexpectedScore)} inputMode="numeric" placeholder="Team 2" style={inputStyle} />
                                  <small style={{ color: afterClinch ? "#64748b" : scoreIsValid(score) ? "#166534" : "#64748b" }}>{afterClinch && !canClearUnexpectedScore ? "Not played — series already clinched" : scoreIsValid(score) ? "Complete" : "Enter a non-tied final score"}</small>
                                </div>
                              );
                            })}
                          </div>
                          <p style={{ marginBottom: 0, color: state.complete ? "#166534" : "#92400e" }}>{state.complete ? `Matchup complete · ${state.playedGames} official game${state.playedGames === 1 ? "" : "s"}` : state.hasScoreAfterClinch ? "Clear any score entered after the series was clinched." : state.hasGap ? "Complete games in order without leaving a gap." : `Matchup incomplete · ${state.playedGames} game${state.playedGames === 1 ? "" : "s"} entered`}</p>
                        </section>
                      );
                    })}
                  </section>
                ))}
              </div>
              <p style={{ color: "#475569" }}>Completed matchups: {completeSeriesCount} / {allPreviewMatches.length} · Official game results ready: {validScoreCount}</p>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", justifyContent: "space-between" }}>
                <button type="button" onClick={() => setWorkflowStep(3)} disabled={busy} style={ghostButtonStyle}>Back to Courts and Preview</button>
                <button type="button" onClick={reviewEnteredScores} disabled={busy || !allSeriesComplete || !roundContextValid || !matchDateValid || sessionStatus !== "active"} style={buttonStyle}>Review scores</button>
              </div>
            </>
          ) : (
            <section style={{ padding: "0.75rem", border: "1px solid #f59e0b", borderRadius: "12px", background: "#fffbeb" }}>
              <h3 style={{ marginTop: 0 }}>Review entered scores</h3>
              <p style={{ color: "#92400e" }}><strong>Nothing has been submitted.</strong> Verify court, teams, and score for every game.</p>
              <p><strong>{roundLabel || `Round ${currentRound}`}</strong> · {weekTag || "No week label"} · {matchDate}</p>
              <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Court</th><th align="left">Team 1</th><th align="center">Score</th><th align="left">Team 2</th></tr></thead><tbody>{scoreReviewRows.map((row) => <tr key={row.key}><td>{row.court}</td><td>{row.teamOne}</td><td align="center"><strong>{row.score}</strong></td><td>{row.teamTwo}</td></tr>)}</tbody></table></div>
              <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", justifyContent: "space-between" }}>
                <button type="button" onClick={editReviewedScores} disabled={busy} style={ghostButtonStyle}>Edit scores</button>
                <button type="button" onClick={confirmScoresAndContinue} disabled={busy || !allSeriesComplete || !scoreReviewMode} style={buttonStyle}>Confirm scores and continue</button>
              </p>
            </section>
          )}
        </article>
      ) : null}

      {workflowStep === 5 ? (
        <article style={cardStyle} aria-labelledby="league-live-movement-heading">
          <h2 id="league-live-movement-heading" style={{ marginTop: 0 }}>5. Movement</h2>
          <p style={{ color: "#475569" }}>Scores are reviewed but not official. Configure any between-round roster change, then preview the server-authoritative movement plan. League Live calculates the plan; this page only displays it.</p>
          <section style={{ padding: "0.75rem", border: "1px solid #e2e8f0", borderRadius: "12px", marginBottom: "0.75rem" }}>
            <h3 style={{ marginTop: 0 }}>Next-round player change</h3>
            <p style={{ color: "#475569" }}>A late arrival appends to the live order. A substitution replaces the selected ordered player. Completed rounds never change.</p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem" }}>
              <label>Action<br /><select value={rosterAction} onChange={(event) => { setRosterAction(event.target.value as "none" | "add" | "substitute"); markPlanStale(); }} disabled={busy} style={inputStyle}><option value="none">No roster change</option><option value="add">Add late arrival</option><option value="substitute">Substitute player</option></select></label>
              {rosterAction !== "none" ? <label>Incoming player<br /><select value={incomingPlayerId} onChange={(event) => { setIncomingPlayerId(event.target.value); markPlanStale(); }} disabled={busy} style={inputStyle}><option value="">Select player…</option>{incomingPlayerOptions.map((player) => <option key={String(player.id)} value={String(player.id)}>{player.name}</option>)}</select></label> : null}
              {rosterAction === "substitute" ? <label>Replace active player<br /><select value={replacedPlayerId} onChange={(event) => { setReplacedPlayerId(event.target.value); markPlanStale(); }} disabled={busy} style={inputStyle}><option value="">Select player…</option>{activeSessionRoster.map((player) => <option key={player.player_id} value={String(player.player_id)}>{player.player_name}</option>)}</select></label> : null}
            </div>
            {liveDomainStatus.submit_enabled ? (
              <details style={{ marginTop: "0.75rem" }}>
                <summary>Create a new non-roster player for a late arrival</summary>
                <p style={{ color: "#475569" }}>This creates a real club player linked to ratings and recovery. Starting JUPR and an operator reason are required.</p>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
                  <label>Player name<br /><input value={guestName} onChange={(event) => setGuestName(event.target.value)} disabled={busy} style={inputStyle} /></label>
                  <label>Starting JUPR *<br /><input required value={guestJupr} onChange={(event) => setGuestJupr(event.target.value)} disabled={busy} type="number" min="1" max="7" step="0.1" placeholder="Required" style={inputStyle} /></label>
                  <label>Operator reason<br /><input value={guestReason} onChange={(event) => setGuestReason(event.target.value)} disabled={busy} placeholder="At least 10 characters" style={inputStyle} /></label>
                  <ConfirmAction triggerLabel="Create late-arrival player" title="Create this club player?" description="This creates a real club player linked to ratings and Match Log recovery." confirmLabel="Yes, create player" confirmationText="CREATE LIVE GUEST" disabled={busy || !sessionIsCurrentLeague || !guestName.trim() || !Number.isFinite(Number(guestJupr)) || Number(guestJupr) < 1 || Number(guestJupr) > 7 || guestReason.trim().length < 10} busy={busy} onConfirm={createGuest} />
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
            <section style={{ padding: "0.75rem", border: `1px solid ${movementPlanStale ? "#fca5a5" : "#86efac"}`, borderRadius: "12px", background: movementPlanStale ? "#fef2f2" : "#f0fdf4" }}>
              <h3 style={{ marginTop: 0 }}>Next-round movement plan</h3>
              <p>{movementPlan.movement.applied ? `${movementPlan.movement.rows.filter((row) => row.direction !== "stay").length} player movement(s)` : "No court movement required"} for Round {movementPlan.movement.next_round}.</p>
              <p style={{ color: movementPlanStale ? "#b91c1c" : "#166534" }}>{movementPlanStale ? "This plan is stale. Preview movement again before continuing." : `Verified operation key ${movementPlan.operation_key.slice(0, 16)}…`}</p>
              <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Player</th><th align="right">Wins</th><th align="right">Diff</th><th align="left">From</th><th align="left">Planned</th><th align="left">Final target</th></tr></thead><tbody>{movementPlan.movement.rows.map((row) => <tr key={row.player_id}><td>{row.player_name}</td><td align="right">{row.wins}</td><td align="right">{row.differential}</td><td>Court {row.from_court}</td><td>Court {row.suggested_court}</td><td><select aria-label={`Final court for ${row.player_name}`} value={movementOverrides[row.player_id] || String(row.to_court)} onChange={(event) => { setMovementOverrides((current) => ({ ...current, [row.player_id]: event.target.value })); setMovementPlanStale(true); }} disabled={busy} style={inputStyle}>{courts.map((court) => <option key={court.court} value={court.court}>Court {court.court}</option>)}</select></td></tr>)}</tbody></table></div>
              <h4>Authoritative next-round courts</h4>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 220px), 1fr))", gap: "0.75rem" }}>
                {movementPlan.next_courts.map((court) => <section key={court.court_number} style={{ padding: "0.75rem", border: "1px solid #bbf7d0", borderRadius: "10px", background: "white" }}><strong>Court {court.court_number}</strong><ol>{court.player_names.map((playerName, index) => <li key={`${court.court_number}-${index}-${playerName}`}>{playerName}</li>)}</ol></section>)}
              </div>
              <p><strong>Next-round bench:</strong> {movementPlan.bench.length ? movementPlan.bench.map((player) => player.player_name).join(", ") : "None"}.</p>
              <label style={{ display: "block", marginTop: "0.75rem" }}>Manual movement override reason<br /><input value={overrideReason} onChange={(event) => { setOverrideReason(event.target.value); markPlanStale(); }} disabled={busy} placeholder="At least 10 characters when changing a planned target" style={inputStyle} /></label>
            </section>
          ) : null}
          <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", justifyContent: "space-between" }}>
            <button type="button" onClick={editReviewedScores} disabled={busy} style={ghostButtonStyle}>Back to score entry</button>
            <span style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              <button type="button" onClick={() => void previewPythonMovement()} disabled={busy || !scoresReviewed || !allSeriesComplete || !sessionIsCurrentLeague || !roundContextValid || !matchDateValid || sessionStatus !== "active"} style={ghostButtonStyle}>{busy ? "Planning…" : "Preview movement"}</button>
              <button type="button" onClick={() => setWorkflowStep(6)} disabled={busy || !scoresReviewed || !movementPlan || movementPlanStale} style={buttonStyle}>Continue to Repeat or Finish</button>
            </span>
          </p>
        </article>
      ) : null}

      {workflowStep === 6 ? (
        <article style={cardStyle} aria-labelledby="league-live-finish-heading">
          <h2 id="league-live-finish-heading" style={{ marginTop: 0 }}>6. Repeat or Finish</h2>
          {!roundPublished ? (
            <>
              <p style={{ color: "#475569" }}>Final check for {roundLabel || `Round ${currentRound}`} · {weekTag || "No week label"} · {matchDate}: {scoreReviewRows.length} reviewed official game result{scoreReviewRows.length === 1 ? "" : "s"}; {movementPlan?.movement.rows.filter((row) => row.direction !== "stay").length || 0} planned court movement{(movementPlan?.movement.rows.filter((row) => row.direction !== "stay").length || 0) === 1 ? "" : "s"}.</p>
              <section style={{ padding: "0.75rem", border: "1px solid #e2e8f0", borderRadius: "12px" }}>
                <h3 style={{ marginTop: 0 }}>Reviewed round</h3>
                <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th align="left">Court</th><th align="left">Team 1</th><th align="center">Score</th><th align="left">Team 2</th></tr></thead><tbody>{scoreReviewRows.map((row) => <tr key={row.key}><td>{row.court}</td><td>{row.teamOne}</td><td align="center"><strong>{row.score}</strong></td><td>{row.teamTwo}</td></tr>)}</tbody></table></div>
              </section>
              {movementPlan ? <section style={{ marginTop: "0.75rem", padding: "0.75rem", border: "1px solid #86efac", borderRadius: "12px", background: "#f0fdf4" }}><h3 style={{ marginTop: 0 }}>Approved next-round roster and courts</h3><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 220px), 1fr))", gap: "0.75rem" }}>{movementPlan.next_courts.map((court) => <div key={court.court_number}><strong>Court {court.court_number}</strong><ol>{court.player_names.map((playerName, index) => <li key={`${court.court_number}-${index}-${playerName}`}>{playerName}</li>)}</ol></div>)}</div><p><strong>Bench:</strong> {movementPlan.bench.length ? movementPlan.bench.map((player) => player.player_name).join(", ") : "None"}.</p></section> : null}
              <p style={{ color: "#92400e" }}>Publishing writes every reviewed score as official, recalculates ratings, saves movement and the next roster/courts, and advances the persisted session in one durable operation.</p>
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", justifyContent: "space-between" }}>
                <button type="button" onClick={() => setWorkflowStep(5)} disabled={busy} style={ghostButtonStyle}>Back to Movement</button>
                {blockingCurrentRoundPublishOperation ? (
                  <p role="alert" style={{ margin: 0, maxWidth: "44rem", color: "#9a3412", fontWeight: 700 }}>
                    Round {currentRound} already has a durable {blockingCurrentRoundPublishOperation.status.replace(/_/g, " ")} publish. Use its Retry or Reconcile action in Publish recovery below; a new publish is intentionally blocked.
                  </p>
                ) : (
                  <ConfirmAction triggerLabel="Publish reviewed round" title={`Publish Round ${currentRound}?`} description="This makes every reviewed result official and applies the approved next-round movement." confirmLabel="Yes, publish the round" confirmationText="SUBMIT LEAGUE ROUND" tone="danger" disabled={busy || !scoresReviewed || !liveDomainStatus.submit_enabled || !allSeriesComplete || !sessionIsCurrentLeague || !movementPlan || movementPlanStale || !roundContextValid || !matchDateValid || sessionStatus !== "active"} busy={busy} onConfirm={publishRoundAndStay} />
                )}
              </div>
            </>
          ) : (
            <section style={{ padding: "1rem", border: "1px solid #86efac", borderRadius: "12px", background: "#f0fdf4" }}>
              <h3 style={{ marginTop: 0 }}>{sessionStatus === "complete" ? "Session complete" : `Round ${lastPublishedRound ?? currentRound} published`}</h3>
              {sessionStatus === "complete" ? (
                <p>The session is closed. Round history, rating readback, recovery controls, and exports remain available below.</p>
              ) : (
                <>
                  <p>Choose the next explicit operator action. Starting the next round uses the saved moved courts and roster; finishing marks this persisted session complete.</p>
                  <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
                    <button type="button" onClick={startNextRound} disabled={busy || sessionStatus !== "active" || lastPublishedRound == null || !Number.isInteger(parsedTotalRounds) || lastPublishedRound + 1 > safeTotalRounds} style={buttonStyle}>Start next round</button>
                    <ConfirmAction triggerLabel="Finish session" title="Complete this League Live session?" description="This marks the persisted session complete. Published rounds and ratings remain unchanged." confirmLabel="Yes, complete session" confirmationText="SAVE SESSION" disabled={busy || !sessionIsCurrentLeague || sessionStatus !== "active" || !roundContextValid} busy={busy} onConfirm={finishSession} />
                  </div>
                  {lastPublishedRound != null && lastPublishedRound >= safeTotalRounds ? <p style={{ color: "#475569" }}>All {safeTotalRounds} configured rounds are complete. Finish the session, or return to Setup and deliberately extend the round count before publishing another round.</p> : null}
                </>
              )}
            </section>
          )}
        </article>
      ) : null}

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
          <p style={{ color: "#475569" }}>A publish operation records intent before scores, verifies every deterministic match context, then reconciles the League Live snapshot. A retained retry reuses the original request and key; reconciliation is reserved for operations with official match evidence.</p>
          {publishOperations.length ? (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
                <thead><tr><th align="left">Round</th><th align="left">State</th><th align="right">Attempts</th><th align="right">Matches</th><th align="left">Recovery</th></tr></thead>
                <tbody>{publishOperations.map((operation) => (
                  <tr key={operation.id}>
                    <td>{operation.round_number}</td><td>{operation.status}</td><td align="right">{operation.attempt_count}</td><td align="right">{operation.published_match_ids?.length || 0}</td>
                    <td>{operation.status === "completed"
                      ? "Verified"
                      : RETRYABLE_PUBLISH_STATUSES.has(operation.status)
                        ? leagueLiveOperatorMessage(operation.error_text, "Retry the retained original publish; its request and key are stored on the server.")
                        : leagueLiveOperatorMessage(operation.error_text, "Reconcile the verified official matches with the League Live snapshot.")}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ) : <p>No publish operations recorded for this session.</p>}
          {publishOperations.some((operation) => RETRYABLE_PUBLISH_STATUSES.has(operation.status)) ? (
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.75rem" }}>
              {publishOperations.filter((operation) => RETRYABLE_PUBLISH_STATUSES.has(operation.status)).map((operation) => <ConfirmAction
                key={operation.id}
                triggerLabel={`Retry R${operation.round_number}`}
                title={`Retry the retained League Live round ${operation.round_number} publish?`}
                description="This reuses the original server-retained request and idempotency key. It verifies deterministic match contexts before writing, so an interrupted response cannot duplicate the round."
                confirmLabel="Yes, retry original publish"
                confirmationText="RETRY LEAGUE ROUND"
                tone="danger"
                disabled={busy}
                busy={busy}
                onConfirm={(confirmationText) => retryRetainedRound(operation.round_number, confirmationText)}
              />)}
            </div>
          ) : null}
          {publishOperations.some((operation) => RECONCILABLE_PUBLISH_STATUSES.has(operation.status)) ? (
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "0.75rem" }}>
              {publishOperations.filter((operation) => RECONCILABLE_PUBLISH_STATUSES.has(operation.status)).map((operation) => <ConfirmAction
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
          {publishOperations.some((operation) => RECONCILABLE_PUBLISH_STATUSES.has(operation.status)) ? (
            <details style={{ marginTop: "0.75rem" }}>
              <summary>Record completed Match Log / Replay History compensation</summary>
              <p style={{ color: "#92400e" }}>Use this only after recovery removed or excluded every related match and the ratings rebuild is complete. The recovery check confirms that no active match remains.</p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
                <label>Recovery reference<br /><input value={compensationReference} onChange={(event) => setCompensationReference(event.target.value)} disabled={busy} placeholder="Match Log / replay operation ID" style={inputStyle} /></label>
                <label>Reason<br /><input value={compensationReason} onChange={(event) => setCompensationReason(event.target.value)} disabled={busy} placeholder="At least 10 characters" style={inputStyle} /></label>
                <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>{publishOperations.filter((operation) => RECONCILABLE_PUBLISH_STATUSES.has(operation.status)).map((operation) => <ConfirmAction
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

    </div>
  );
}
