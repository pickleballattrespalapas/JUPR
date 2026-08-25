"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, actionUncertain, type ActionCompletion } from "@/components/interaction";
import type {
  AdminTournamentOpsSnapshotResponse,
  AdminTournamentOpsPlayer,
  AdminTournamentOpsTeam,
  AdminTournamentResultsImportPreviewResponse,
  AdminTournamentStatusResponse,
  AdminTournamentWriteResponse
} from "@/lib/adminTournamentApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";
import { tournamentRouteHref } from "@/lib/tournamentRouteContext";

export type OpsWorkflow = "all" | "draws" | "import" | "results" | "publish";
type Props = { apiBase: string | null; clubId: string; status: AdminTournamentStatusResponse; workflow?: OpsWorkflow; initialTournamentId: string; initialDrawId?: string | null };
type TeamEditorRow = { editor_key: string; team_number: string; player1_id: string; player2_id: string; seed: string; notes: string };
type RegistrationImportBody = {
  import_mode: string;
  idempotency_key: string;
  expected_state_fingerprint: string;
  expected_draw_updated_at: string;
  confirmation_text: string;
  source: string;
};
type RegistrationImportRecovery = {
  version: 1;
  clubId: string;
  tournamentId: string;
  drawId: string;
  createdAt: string;
  operationReference: string;
  message: string;
  body: RegistrationImportBody;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const registrationImportReconcileConfirmation = "RECONCILE REGISTRATION IMPORT";
const NON_PLAYED_RESULT_TYPES: ReadonlySet<string> = new Set(["FORFEIT", "NO_SHOW", "RETIREMENT"]);
const DISABLED_EVENT_STATUSES: ReadonlySet<string> = new Set(["disabled", "cancelled", "canceled", "inactive", "archived", "deleted", "void", "voided"]);

class TournamentOpsRequestError extends Error {
  readonly status: number;
  readonly uncertain: boolean;
  readonly operationReference: string | null;

  constructor(message: string, status: number, uncertain = false, operationReference: string | null = null) {
    super(message);
    this.name = "TournamentOpsRequestError";
    this.status = status;
    this.uncertain = uncertain;
    this.operationReference = operationReference;
  }
}

function registrationImportStorageKey(clubId: string): string {
  return `jupr_tournament_ops_registration_import_pending_v1:${clubId}`;
}

function registrationImportErrorIsUncertain(error: unknown): boolean {
  return !(error instanceof TournamentOpsRequestError) || error.uncertain;
}

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function shortValue(value: unknown): string {
  if (value == null || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value).slice(0, 160);
  return String(value);
}

function gameResultType(game: Record<string, unknown>): string {
  return String(game.result_type || "PLAYED").toUpperCase();
}

function isNonPlayedGame(game: Record<string, unknown>): boolean {
  return NON_PLAYED_RESULT_TYPES.has(gameResultType(game));
}

function resultTypeLabel(game: Record<string, unknown>): string {
  const resultType = gameResultType(game);
  if (resultType === "NO_SHOW") return "No-show";
  if (resultType === "RETIREMENT") return "Retirement";
  if (resultType === "FORFEIT") return "Forfeit";
  return resultType === "PLAYED" ? "Played" : resultType.replace(/_/g, " ").toLowerCase();
}

function eventOptionLabel(row: Record<string, unknown>): string {
  const family = String(row.event_family_label || "").trim();
  const division = String(row.division_name || row.label || "").trim();
  if (family && division && family !== division) return `${family} / ${division}`;
  return division || family || String(row.id || "Event");
}

function eventOptionEnabled(row: Record<string, unknown>): boolean {
  const enabledValue = row.enabled;
  const enabled = typeof enabledValue === "boolean"
    ? enabledValue
    : !["0", "false", "no", "off", "disabled", "cancelled", "canceled"].includes(String(enabledValue ?? "true").trim().toLowerCase());
  const status = String(row.status || "").trim().toLowerCase();
  return enabled && !DISABLED_EVENT_STATUSES.has(status);
}

function playerLabel(players: AdminTournamentOpsPlayer[], playerId: number | null | undefined): string {
  if (playerId == null) return "Player unavailable";
  return players.find((player) => Number(player.id) === Number(playerId))?.name || "Player unavailable";
}

function teamLabel(team: AdminTournamentOpsTeam | undefined, players: AdminTournamentOpsPlayer[]): string {
  if (!team) return "Team unavailable";
  const first = playerLabel(players, team.player1_id);
  const second = team.player2_id == null ? "" : playerLabel(players, team.player2_id);
  return second ? `${first} / ${second}` : first;
}

function gameLabel(
  row: Record<string, unknown>,
  teamsById: Map<string, AdminTournamentOpsTeam>,
  players: AdminTournamentOpsPlayer[]
): string {
  const stage = String(row.stage || "Game");
  const round = row.rr_round_number ? `R${row.rr_round_number}` : String(row.playoff_round || "");
  const slot = row.rr_slot_number ? `S${row.rr_slot_number}` : String(row.playoff_game_code || "");
  const teams = `${teamLabel(teamsById.get(String(row.team_a_id || "")), players)} vs ${teamLabel(teamsById.get(String(row.team_b_id || "")), players)}`;
  return [stage, round, slot, teams].filter(Boolean).join(" · ");
}

function teamRowsFromTeams(teams: AdminTournamentOpsTeam[], drawId: string): TeamEditorRow[] {
  const scoped = teams
    .filter((row) => !drawId || String(row.draw_id || "") === drawId)
    .sort((left, right) => Number(left.team_number || 0) - Number(right.team_number || 0));
  if (!scoped.length) {
    return [1, 2, 3, 4].map((teamNumber) => ({ editor_key: `empty-team-${teamNumber}`, team_number: String(teamNumber), player1_id: "", player2_id: "", seed: String(teamNumber), notes: "" }));
  }
  return scoped.map((row, index) => ({
    editor_key: String(row.id || `loaded-team-${row.team_number || index + 1}`),
    team_number: String(row.team_number || index + 1),
    player1_id: row.player1_id == null ? "" : String(row.player1_id),
    player2_id: row.player2_id == null ? "" : String(row.player2_id),
    seed: row.seed == null ? "" : String(row.seed),
    notes: row.notes || ""
  }));
}

export default function TournamentOpsPanel({
  apiBase,
  clubId,
  status,
  workflow = "all",
  initialTournamentId,
  initialDrawId = ""
}: Props) {
  const { accessToken } = useAdminSession();
  const [selectedTournamentId, setSelectedTournamentId] = useState(initialTournamentId);
  const [selectedDrawId, setSelectedDrawId] = useState(initialDrawId || "");
  const [snapshot, setSnapshot] = useState<AdminTournamentOpsSnapshotResponse | null>(null);
  const [drawEventOptionId, setDrawEventOptionId] = useState("");
  const [emptyEventOptionId, setEmptyEventOptionId] = useState("");
  const [drawName, setDrawName] = useState("");
  const [teamRows, setTeamRows] = useState<TeamEditorRow[]>(() => teamRowsFromTeams([], ""));
  const [registrationImportMode, setRegistrationImportMode] = useState("REPLACE");
  const [bulkTeamMode, setBulkTeamMode] = useState("REPLACE");
  const [bulkTeamText, setBulkTeamText] = useState("Player 1,Player 2,Seed,Notes\n");
  const [scoreGameId, setScoreGameId] = useState("");
  const [scoreA, setScoreA] = useState("");
  const [scoreB, setScoreB] = useState("");
  const [unusualScoreAcknowledged, setUnusualScoreAcknowledged] = useState(false);
  const [playoffAdvanceCount, setPlayoffAdvanceCount] = useState("4");
  const [resultsImportMode, setResultsImportMode] = useState("REPLACE");
  const [resultsRawText, setResultsRawText] = useState("playerA1,playerB1,teamAGame1,teamBGame1\n");
  const [resultsPreview, setResultsPreview] = useState<AdminTournamentResultsImportPreviewResponse | null>(null);
  const [resultsMappings, setResultsMappings] = useState<Record<string, { action?: string; player_id?: string | number | null }>>({});
  const [resultsMatchReviews, setResultsMatchReviews] = useState<Record<string, { include?: boolean; stage?: string }>>({});
  const [resultsPodiumRefs, setResultsPodiumRefs] = useState<Record<string, string | null>>({});
  const [allowDuplicateMapping, setAllowDuplicateMapping] = useState(false);
  const [unusualImportScoresAcknowledged, setUnusualImportScoresAcknowledged] = useState(false);
  const [resultsReviewDirty, setResultsReviewDirty] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [registrationImportRecovery, setRegistrationImportRecovery] = useState<RegistrationImportRecovery | null>(null);
  const [registrationImportRecoveryLoaded, setRegistrationImportRecoveryLoaded] = useState(false);
  const snapshotRequest = useLatestRequestGuard(accessToken);
  const actionRequest = useLatestRequestGuard(accessToken);
  const importRecoveryStorageKey = registrationImportStorageKey(clubId);

  useEffect(() => {
    setRegistrationImportRecovery(null);
    setRegistrationImportRecoveryLoaded(false);
    try {
      const raw = globalThis.sessionStorage?.getItem(importRecoveryStorageKey);
      if (!raw) return;
      const stored = JSON.parse(raw) as Partial<RegistrationImportRecovery>;
      if (
        stored.version === 1
        && stored.clubId === clubId
        && typeof stored.tournamentId === "string"
        && typeof stored.drawId === "string"
        && typeof stored.createdAt === "string"
        && typeof stored.operationReference === "string"
        && typeof stored.message === "string"
        && typeof stored.body?.idempotency_key === "string"
      ) {
        setRegistrationImportRecovery(stored as RegistrationImportRecovery);
      }
    } catch {
      // The in-memory guard remains available when browser storage is blocked.
    } finally {
      setRegistrationImportRecoveryLoaded(true);
    }
  }, [clubId, importRecoveryStorageKey]);

  const operationsWriteReady = Boolean(
    status.mutation_runtime?.service_role_ready
    && status.mutation_runtime?.surface_flags?.operations?.enabled
    && status.operations_runtime?.operations_mutations_enabled
  );
  const reviewedState = snapshot?.state_fingerprint || "";
  const selectedDraw = snapshot?.draws?.find((row) => String(row.id || "") === selectedDrawId) || null;
  const players = snapshot?.players || [];
  const teamsById = new Map((snapshot?.teams || []).map((team) => [String(team.id || ""), team]));
  const scoreableGames = (snapshot?.games || []).filter((game) => !isNonPlayedGame(game));
  const nonPlayedGames = (snapshot?.games || []).filter(isNonPlayedGame);
  const selectedScoreGame = scoreableGames.find((game) => String(game.id || "") === scoreGameId) || null;
  const teamEditorPlayerChoicesReady = players.length > 0;
  const reviewedDrawUpdatedAt = String(selectedDraw?.updated_at || "").trim();
  const reviewedTeamVersions = (snapshot?.teams || [])
    .filter((row) => String(row.draw_id || "") === selectedDrawId)
    .map((row) => ({ id: String(row.id || ""), updated_at: String(row.updated_at || "") }));
  const reviewedSourceGameVersions = (snapshot?.games || [])
    .filter((row) => String(row.draw_id || "") === selectedDrawId)
    .map((row) => ({ id: String(row.id || ""), updated_at: String(row.updated_at || "") }));
  const officialPublishReady = Boolean(snapshot?.operation_runtime?.official_publish_enabled);
  // One tab retains one exact registration-import request for this club. Block
  // every guarded Ops write until it is reconciled so navigating to another
  // tournament cannot overwrite that sole recovery slot.
  const registrationImportBlocksWrites = registrationImportRecovery !== null;
  const guardedWriteDisabled = busy || !accessToken || !operationsWriteReady || !reviewedState || !registrationImportRecoveryLoaded || registrationImportBlocksWrites;
  const drawCasWriteDisabled = guardedWriteDisabled || !reviewedDrawUpdatedAt;
  const teamSnapshotCasDisabled = drawCasWriteDisabled || !reviewedTeamVersions.length || reviewedTeamVersions.some((row) => !row.id || !row.updated_at);
  const gameSnapshotCasDisabled = teamSnapshotCasDisabled || !reviewedSourceGameVersions.length || reviewedSourceGameVersions.some((row) => !row.id || !row.updated_at);
  const shows = (name: Exclude<OpsWorkflow, "all">) => workflow === "all" || workflow === name;
  const showsLegacyDrawRuntime = workflow === "all";

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Tournament Ops.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = payload?.detail;
      const detailRecord = detail && typeof detail === "object" ? detail as Record<string, unknown> : null;
      const detailMessage = typeof detail === "string"
        ? detail
        : String(detailRecord?.message || `API error (${response.status})`);
      const recoveryRequired = detailRecord?.recovery_required === true
        || detailRecord?.kind === "uncertain"
        || (response.status === 409 && /recovery|required|may already|identical request|response.?lost/i.test(detailMessage));
      const explicitlyFailed = detailRecord?.kind === "failed"
        && detailRecord?.recovery_required !== true;
      const operationReference = typeof detailRecord?.operation_reference === "string"
        ? detailRecord.operation_reference
        : null;
      const message = response.status === 409 && !recoveryRequired
        ? `${detailMessage} Reload the authoritative Ops snapshot before submitting a new request.`
        : detailMessage;
      throw new TournamentOpsRequestError(
        message,
        response.status,
        recoveryRequired || (!explicitlyFailed && (response.status >= 500 || [408, 425, 429].includes(response.status))),
        operationReference,
      );
    }
    return payload as T;
  }

  function clearProtectedOpsState() {
    snapshotRequest.invalidate();
    setBusy(false); setMessage(null);
    setSelectedTournamentId(initialTournamentId); setSelectedDrawId(""); setSnapshot(null);
    setDrawEventOptionId(""); setEmptyEventOptionId(""); setTeamRows(teamRowsFromTeams([], "")); setScoreGameId(""); setScoreA(""); setScoreB(""); setUnusualScoreAcknowledged(false);
    setResultsPreview(null); setResultsMappings({}); setResultsMatchReviews({}); setResultsPodiumRefs({}); setUnusualImportScoresAcknowledged(false); setResultsReviewDirty(true);
  }

  function operationSuffix(payload: AdminTournamentWriteResponse): string {
    if (payload.reconciled) return ` Reconciled without repeating the domain write (${payload.operation_key?.slice(0, 12) || "operation"}).`;
    if (payload.idempotent_replay) return ` Idempotent replay; no second domain write (${payload.operation_key?.slice(0, 12) || "operation"}).`;
    return payload.operation_key ? ` Operation ${payload.operation_key.slice(0, 12)} recorded.` : "";
  }

  function warningSuffix(payload: AdminTournamentWriteResponse): string {
    const warnings = (payload.warnings || []).map((warning) => String(warning || "").trim()).filter(Boolean);
    return warnings.length ? ` ${warnings.join(" ")}` : "";
  }

  function persistRegistrationImportRecovery(recovery: RegistrationImportRecovery | null) {
    try {
      if (recovery) globalThis.sessionStorage?.setItem(importRecoveryStorageKey, JSON.stringify(recovery));
      else globalThis.sessionStorage?.removeItem(importRecoveryStorageKey);
    } catch {
      // State still blocks replacement writes for the lifetime of this page.
    }
    setRegistrationImportRecovery(recovery);
  }

  function resetTeamEditor(nextSnapshot: AdminTournamentOpsSnapshotResponse | null, drawId: string) {
    setTeamRows(teamRowsFromTeams(nextSnapshot?.teams || [], drawId));
  }

  function resetScoreEditor(nextSnapshot: AdminTournamentOpsSnapshotResponse | null) {
    const firstGame = (nextSnapshot?.games || []).find((game) => !isNonPlayedGame(game)) || null;
    setScoreGameId(firstGame ? String(firstGame.id || "") : "");
    setScoreA(firstGame?.score_a == null ? "" : String(firstGame.score_a));
    setScoreB(firstGame?.score_b == null ? "" : String(firstGame.score_b));
    setUnusualScoreAcknowledged(false);
  }

  async function loadOps(
    tournamentId = selectedTournamentId,
    drawId = selectedDrawId
  ): Promise<AdminTournamentOpsSnapshotResponse | null> {
    const generation = snapshotRequest.begin();
    if (!tournamentId) {
      setMessage("Select a tournament first.");
      return null;
    }
    setBusy(true);
    setMessage(null);
    try {
      const params = new URLSearchParams();
      if (drawId) params.set("draw_id", drawId);
      const suffix = params.toString() ? `?${params.toString()}` : "";
      const payload = await requestJson<AdminTournamentOpsSnapshotResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/ops${suffix}`);
      if (!snapshotRequest.isCurrent(generation)) return null;
      setSnapshot(payload);
      const eligibleEventOptions = (payload.event_options || []).filter(eventOptionEnabled);
      setDrawEventOptionId((current) => eligibleEventOptions.some((row) => String(row.id || "") === current)
        ? current
        : String(eligibleEventOptions[0]?.id || ""));
      setEmptyEventOptionId((current) => eligibleEventOptions.some((row) => String(row.id || "") === current)
        ? current
        : String(eligibleEventOptions[0]?.id || ""));
      resetTeamEditor(payload, drawId);
      resetScoreEditor(payload);
      setResultsPreview(null);
      setResultsReviewDirty(true);
      setMessage(null);
      return payload;
    } catch (error) {
      if (snapshotRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load tournament operations.");
      return null;
    } finally {
      if (snapshotRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function selectDraw(drawId: string) {
    setSelectedDrawId(drawId);
    setSnapshot(null);
    setResultsPreview(null);
    setResultsReviewDirty(true);
    if (selectedTournamentId) void loadOps(selectedTournamentId, drawId);
    else snapshotRequest.invalidate();
  }

  async function createDraw(confirmationText: string) {
    if (!selectedTournamentId) {
      setMessage("Select a tournament before creating a draw.");
      throw new Error("Select a tournament before creating a draw.");
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    setBusy(true);
    setMessage(null);
    try {
      const selectedEvent = snapshot?.event_options?.find((row) => String(row.id || "") === drawEventOptionId) || null;
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws`, {
        method: "POST",
        body: JSON.stringify({ event_option_id: drawEventOptionId || null, registration_day_id: String(selectedEvent?.registration_day_id || "") || null, name: drawName, expected_state_fingerprint: reviewedState, confirmation_text: confirmationText, source: "next_tournament_ops_create_draw" })
      });
      const nextDrawId = payload.draw?.id || "";
      const completion = actionSuccess("Draw created", `The draft draw${payload.draw?.name ? ` ${payload.draw.name}` : ""} was created.`);
      if (!actionRequest.isCurrent(generation)) return completion;
      setSelectedDrawId(nextDrawId);
      await loadOps(tournamentId, nextDrawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage(`Draw created${payload.draw?.name ? `: ${payload.draw.name}` : ""}.${operationSuffix(payload)}`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to create draw.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function executeRegistrationImport(
    recovery: RegistrationImportRecovery,
    generation: number,
  ): Promise<ActionCompletion> {
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(recovery.tournamentId)}/draws/${encodeURIComponent(recovery.drawId)}/teams/import-registrations`, {
        method: "POST",
        body: JSON.stringify(recovery.body),
      });
      const importedCount = payload.updated_count ?? payload.teams?.length ?? 0;
      const importWarning = warningSuffix(payload);
      const completion = actionSuccess("Registration teams imported", `${importedCount} registration team${importedCount === 1 ? " was" : "s were"} imported.${importWarning}`);
      persistRegistrationImportRecovery(null);
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadOps(recovery.tournamentId, recovery.drawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage(`Imported ${importedCount} registration team(s) with ${payload.import_mode || recovery.body.import_mode} mode.${importWarning}${operationSuffix(payload)}`);
      return completion;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unable to confirm the registration import.";
      if (!registrationImportErrorIsUncertain(error)) {
        persistRegistrationImportRecovery(null);
        if (actionRequest.isCurrent(generation)) setMessage(errorMessage);
        throw error;
      }
      const operationReference = error instanceof TournamentOpsRequestError && error.operationReference
        ? error.operationReference
        : recovery.operationReference;
      const pending = { ...recovery, operationReference, message: errorMessage };
      persistRegistrationImportRecovery(pending);
      if (actionRequest.isCurrent(generation)) {
        setMessage(`${errorMessage} The exact browser request is retained; reconcile it before another guarded Ops write in this club tab.`);
      }
      return actionUncertain(
        "Registration import outcome needs checking",
        "The request reached an uncertain outcome. Reconcile the protected operation; the recovery action will never repeat the import or guess from a momentary readback.",
        operationReference,
        "Reconcile protected operation",
        () => reconcileRegistrationImport(pending),
      );
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function reconcileRegistrationImport(
    recovery: RegistrationImportRecovery,
  ): Promise<ActionCompletion> {
    const generation = actionRequest.begin();
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(recovery.tournamentId)}/draws/${encodeURIComponent(recovery.drawId)}/teams/import-registrations/operations/${encodeURIComponent(recovery.operationReference)}/reconcile`,
        {
          method: "POST",
          body: JSON.stringify({
            retained_request: recovery.body,
            confirmation_text: registrationImportReconcileConfirmation,
            source: "next_tournament_ops_registration_import_reconcile",
          }),
        },
      );
      const notApplied = payload.recovery_disposition === "not_applied";
      const completion = actionSuccess(
        notApplied ? "Registration import did not run" : "Registration import reconciled",
        notApplied
          ? "The exact recovery reservation prevented this request from beginning, so no teams were changed. The prior lock is closed; reload and review before starting a new import."
          : "Authoritative evidence proves the exact import completed. Its stored result was recovered without repeating the write.",
      );
      persistRegistrationImportRecovery(null);
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadOps(recovery.tournamentId, recovery.drawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage(notApplied
        ? "The interrupted registration import is proven not applied. Review the reloaded draw before starting a new import."
        : `The interrupted registration import was reconciled.${operationSuffix(payload)}`);
      return completion;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : "Unable to reconcile the registration import.";
      const pending = { ...recovery, message: errorMessage };
      persistRegistrationImportRecovery(pending);
      if (actionRequest.isCurrent(generation)) {
        setMessage(`${errorMessage} The retained operation remains blocked; do not submit a replacement import.`);
      }
      return actionUncertain(
        "Registration import still needs verification",
        "The protected operation has no commit-safe completion evidence yet. The original request remains retained and no import was repeated.",
        recovery.operationReference,
        "Reconcile protected operation again",
        () => reconcileRegistrationImport(pending),
      );
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function importRegistrations(confirmationText: string): Promise<ActionCompletion> {
    const generation = actionRequest.begin();
    if (!selectedTournamentId || !selectedDrawId || !reviewedState || !reviewedDrawUpdatedAt) {
      setMessage("Reload a tournament draw before importing registrations.");
      throw new Error("Reload a tournament draw before importing registrations.");
    }
    if (registrationImportRecovery) {
      const blocked = `Reconcile retained operation ${registrationImportRecovery.operationReference} before another guarded Ops write in this club tab.`;
      setMessage(blocked);
      throw new Error(blocked);
    }
    if (!actionRequest.isCurrent(generation)) {
      throw new Error("The admin session changed before this registration import could start.");
    }
    const idempotencyKey = globalThis.crypto.randomUUID();
    const recovery: RegistrationImportRecovery = {
      version: 1,
      clubId,
      tournamentId: selectedTournamentId,
      drawId: selectedDrawId,
      createdAt: new Date().toISOString(),
      operationReference: idempotencyKey,
      message: "Registration import request is being submitted.",
      body: {
        import_mode: registrationImportMode,
        idempotency_key: idempotencyKey,
        expected_state_fingerprint: reviewedState,
        expected_draw_updated_at: reviewedDrawUpdatedAt,
        confirmation_text: confirmationText,
        source: "next_tournament_ops_import_registrations",
      },
    };
    persistRegistrationImportRecovery(recovery);
    return executeRegistrationImport(recovery, generation);
  }

  async function importBulkTeams(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before importing teams.");
      throw new Error("Select a tournament and draw before importing teams.");
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/teams/import-bulk`, {
        method: "POST",
        body: JSON.stringify({ raw_text: bulkTeamText, import_mode: bulkTeamMode, expected_state_fingerprint: reviewedState, expected_draw_updated_at: reviewedDrawUpdatedAt, confirmation_text: confirmationText, source: "next_tournament_ops_import_bulk_teams" })
      });
      const importedCount = payload.updated_count ?? payload.teams?.length ?? 0;
      const completion = actionSuccess("Teams imported", `${importedCount} team${importedCount === 1 ? " was" : "s were"} imported from the reviewed file.`);
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage(`Imported ${payload.updated_count ?? payload.teams?.length ?? 0} bulk team(s) with ${payload.import_mode || bulkTeamMode} mode.${operationSuffix(payload)}`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to import bulk teams.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveTeams(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before saving teams.");
      throw new Error("Select a tournament and draw before saving teams.");
    }
    const teams = teamRows
      .filter((row) => row.player1_id.trim())
      .map((row, index) => ({ team_number: Number(row.team_number || index + 1), player1_id: Number(row.player1_id), player2_id: row.player2_id.trim() ? Number(row.player2_id) : null, seed: row.seed.trim() ? Number(row.seed) : null, source: "MANUAL", notes: row.notes }));
    if (!teams.length) {
      setMessage("Add at least one team with Player 1 before saving.");
      throw new Error("Add at least one team with Player 1 before saving.");
    }
    if (teams.some((team) => !Number.isFinite(team.team_number) || !Number.isFinite(team.player1_id) || (team.player2_id !== null && !Number.isFinite(team.player2_id)))) {
      setMessage("One or more team entries are invalid. Reload the player choices and review every row.");
      throw new Error("One or more team entries are invalid. Reload the player choices and review every row.");
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/teams`, {
        method: "PUT",
        body: JSON.stringify({ teams, expected_state_fingerprint: reviewedState, expected_draw_updated_at: reviewedDrawUpdatedAt, confirmation_text: confirmationText, source: "next_tournament_ops_team_editor" })
      });
      const savedCount = payload.updated_count ?? payload.teams?.length ?? teams.length;
      const completion = actionSuccess("Teams saved", `${savedCount} team${savedCount === 1 ? " was" : "s were"} saved to the draw.`);
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage(`Saved ${payload.updated_count ?? payload.teams?.length ?? teams.length} team(s).${operationSuffix(payload)}`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save teams.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function generateGames(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before generating games.");
      throw new Error("Select a tournament and draw before generating games.");
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/games/round-robin`, {
        method: "POST",
        body: JSON.stringify({ expected_state_fingerprint: reviewedState, expected_draw_updated_at: reviewedDrawUpdatedAt, expected_team_versions: reviewedTeamVersions, confirmation_text: confirmationText, source: "next_tournament_ops_generate_round_robin" })
      });
      const gameCount = payload.game_count ?? payload.games?.length ?? 0;
      const completion = actionSuccess("Round-robin games generated", `${gameCount} round-robin game${gameCount === 1 ? " was" : "s were"} generated.`);
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage(`Generated ${payload.game_count ?? payload.games?.length ?? 0} round-robin game(s).${operationSuffix(payload)}`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to generate games.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function recoverRoundRobin(
    mode: "reconcile" | "rebuild",
    confirmationText: string
  ) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before repairing games.");
      throw new Error("Select a tournament and draw before repairing games.");
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/games/round-robin/${mode}`,
        {
          method: "POST",
          body: JSON.stringify({
            expected_state_fingerprint: reviewedState,
            expected_draw_updated_at: reviewedDrawUpdatedAt,
            expected_team_versions: reviewedTeamVersions,
            confirmation_text: confirmationText,
            source: `next_tournament_ops_round_robin_${mode}`
          })
        }
      );
      const gameCount = payload.game_count ?? payload.games?.length ?? 0;
      const completion = actionSuccess(
        mode === "reconcile" ? "Round robin reconciled" : "Round robin rebuilt",
        mode === "reconcile"
          ? `${gameCount} total round-robin games are now present; existing valid games were preserved.`
          : `${gameCount} unstarted round-robin games were rebuilt from the reviewed team list.`
      );
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage(
        `${mode === "reconcile" ? "Reconciled" : "Rebuilt"} the round-robin schedule.${operationSuffix(payload)}`
      );
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(
          error instanceof Error
            ? error.message
            : `Unable to ${mode} the round-robin schedule.`
        );
      }
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function cancelEmptyDraw(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      throw new Error("Select an empty draw before cancelling it.");
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/cancel-empty`,
        {
          method: "POST",
          body: JSON.stringify({
            expected_state_fingerprint: reviewedState,
            expected_draw_updated_at: reviewedDrawUpdatedAt,
            confirmation_text: confirmationText,
            source: "next_tournament_ops_cancel_empty_draw"
          })
        }
      );
      const completion = actionSuccess(
        "Empty draw cancelled",
        "The empty draw was disabled without changing teams, games, results, or publication evidence."
      );
      if (!actionRequest.isCurrent(generation)) return completion;
      setSelectedDrawId("");
      await loadOps(tournamentId, "");
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage(`Empty draw cancelled.${operationSuffix(payload)}`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to cancel the empty draw.");
      }
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function cancelEmptyEvent(confirmationText: string) {
    if (!selectedTournamentId || !emptyEventOptionId) {
      throw new Error("Choose an empty event before cancelling it.");
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const eventOptionId = emptyEventOptionId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/events/${encodeURIComponent(eventOptionId)}/cancel-empty`,
        {
          method: "POST",
          body: JSON.stringify({
            expected_state_fingerprint: reviewedState,
            confirmation_text: confirmationText,
            source: "next_tournament_ops_cancel_empty_event"
          })
        }
      );
      const completion = actionSuccess(
        "Empty event cancelled",
        "The zero-entry event was disabled and no registrations, teams, games, or results were changed."
      );
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadOps(tournamentId, selectedDrawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage(`Empty event cancelled.${operationSuffix(payload)}`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) {
        setMessage(error instanceof Error ? error.message : "Unable to cancel the empty event.");
      }
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function generatePlayoffs(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before generating playoffs.");
      throw new Error("Select a tournament and draw before generating playoffs.");
    }
    const advanceCount = Number(playoffAdvanceCount);
    if (!Number.isFinite(advanceCount)) {
      setMessage("Advance count must be numeric.");
      throw new Error("Advance count must be numeric.");
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/games/playoffs`, {
        method: "POST",
        body: JSON.stringify({ advance_count: advanceCount, expected_state_fingerprint: reviewedState, expected_draw_updated_at: reviewedDrawUpdatedAt, expected_team_versions: reviewedTeamVersions, expected_source_game_versions: reviewedSourceGameVersions, confirmation_text: confirmationText, source: "next_tournament_ops_generate_playoffs" })
      });
      const gameCount = payload.game_count ?? payload.games?.length ?? 0;
      const completion = actionSuccess("Playoff games generated", `${gameCount} playoff game${gameCount === 1 ? " was" : "s were"} generated.`);
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage(`Generated ${payload.game_count ?? payload.games?.length ?? 0} playoff game(s).${operationSuffix(payload)}`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to generate playoffs.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveScore(confirmationText: string) {
    if (!selectedTournamentId || !scoreGameId) {
      setMessage("Select a game before saving a score.");
      throw new Error("Select a game before saving a score.");
    }
    const selectedGame = (snapshot?.games || []).find((row) => String(row.id || "") === scoreGameId) || null;
    if (!selectedGame || isNonPlayedGame(selectedGame)) {
      setMessage("This game has a non-played outcome and cannot be changed through ordinary score entry.");
      throw new Error("This game has a non-played outcome and cannot be changed through ordinary score entry.");
    }
    if (!scoreA.trim() || !scoreB.trim()) {
      setMessage("Enter both team scores before saving.");
      throw new Error("Enter both team scores before saving.");
    }
    const nextA = Number(scoreA);
    const nextB = Number(scoreB);
    if (!Number.isInteger(nextA) || !Number.isInteger(nextB) || nextA < 0 || nextB < 0 || nextA === nextB) {
      setMessage("Enter two non-tied, non-negative whole-number scores.");
      throw new Error("Enter two non-tied, non-negative whole-number scores.");
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    const gameId = scoreGameId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/games/${encodeURIComponent(gameId)}/score`, {
        method: "PATCH",
        body: JSON.stringify({ score_a: nextA, score_b: nextB, unusual_score_acknowledged: unusualScoreAcknowledged, expected_state_fingerprint: reviewedState, expected_game_updated_at: String(selectedGame?.updated_at || "") || null, expected_draw_updated_at: reviewedDrawUpdatedAt, confirmation_text: confirmationText, source: "next_tournament_ops_score_game" })
      });
      const selectedMatchup = selectedGame ? gameLabel(selectedGame, teamsById, players) : "the selected matchup";
      const completion = actionSuccess("Score saved", `The score for ${selectedMatchup} was saved.`);
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage(`Saved score for ${selectedMatchup}.${operationSuffix(payload)}`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save score.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function generatePodium(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before generating a podium.");
      throw new Error("Select a tournament and draw before generating a podium.");
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/podium`, {
        method: "POST",
        body: JSON.stringify({ expected_state_fingerprint: reviewedState, expected_draw_updated_at: reviewedDrawUpdatedAt, expected_team_versions: reviewedTeamVersions, expected_source_game_versions: reviewedSourceGameVersions, confirmation_text: confirmationText, source: "next_tournament_ops_generate_podium" })
      });
      const podiumCount = payload.podium?.length ?? 0;
      const completion = actionSuccess("Podium generated", `${podiumCount} podium placement${podiumCount === 1 ? " was" : "s were"} generated.`);
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      setMessage(`Generated ${payload.podium?.length ?? 0} ${payload.podium_source || "draw"} podium placement(s).${operationSuffix(payload)}`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to generate podium.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function previewResultsImport() {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before previewing results.");
      return;
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentResultsImportPreviewResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/results-import/preview`, {
        method: "POST",
        body: JSON.stringify({
          raw_text: resultsRawText,
          import_mode: resultsImportMode,
          mapping_decisions: Object.keys(resultsMappings).length ? resultsMappings : null,
          match_reviews: Object.keys(resultsMatchReviews).length ? resultsMatchReviews : null,
          podium_refs: Object.keys(resultsPodiumRefs).length ? resultsPodiumRefs : null,
          allow_duplicate_mapping: allowDuplicateMapping
        })
      });
      if (!actionRequest.isCurrent(generation)) return;
      setResultsPreview(payload);
      setResultsMappings(payload.mapping_decisions || {});
      setResultsMatchReviews(payload.match_reviews || {});
      setResultsPodiumRefs(payload.podium_refs || {});
      setUnusualImportScoresAcknowledged(false);
      setResultsReviewDirty(false);
      setMessage(payload.ok
        ? `Reviewed ${payload.summary.matches} match(es) across ${payload.summary.teams} team(s). No data was written.`
        : `Preview found ${payload.errors.length} blocking issue(s). Resolve mappings or match review choices, then preview again.`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to preview tournament results.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function commitResultsImport(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId || !resultsPreview) {
      setMessage("Create and review a results preview before committing.");
      throw new Error("Create and review a results preview before committing.");
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/results-import/commit`, {
        method: "POST",
        body: JSON.stringify({
          raw_text: resultsRawText,
          import_mode: resultsImportMode,
          mapping_decisions: resultsMappings,
          match_reviews: resultsMatchReviews,
          podium_refs: resultsPodiumRefs,
          allow_duplicate_mapping: allowDuplicateMapping,
          unusual_scores_acknowledged: unusualImportScoresAcknowledged,
          expected_review_fingerprint: resultsPreview.review_fingerprint,
          expected_state_fingerprint: reviewedState,
          expected_draw_updated_at: reviewedDrawUpdatedAt,
          confirmation_text: confirmationText,
          source: "next_tournament_ops_results_import"
        })
      });
      const gameCount = payload.game_count ?? 0;
      const completion = actionSuccess("Tournament results imported", `${gameCount} reviewed result${gameCount === 1 ? " was" : "s were"} committed to the draw.`);
      if (!actionRequest.isCurrent(generation)) return completion;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      setResultsReviewDirty(true);
      setMessage(`Imported ${payload.game_count ?? 0} reviewed result(s), ${payload.team_count ?? 0} team(s), and ${payload.podium_count ?? 0} podium row(s).${operationSuffix(payload)}`);
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to commit tournament results.");
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function loadResultsFile(file: File | null) {
    if (!file) return;
    if (file.size > 1_000_000) {
      setMessage("Results CSV files are capped at 1 MB.");
      return;
    }
    try {
      setResultsRawText(await file.text());
      setResultsPreview(null);
      setResultsReviewDirty(true);
      setMessage(`Loaded ${file.name}. Preview it before committing.`);
    } catch {
      setMessage("Unable to read the selected results CSV.");
    }
  }

  function selectScoreGame(gameId: string) {
    const game = (snapshot?.games || []).find((row) => String(row.id || "") === gameId) || null;
    if (game && isNonPlayedGame(game)) {
      setMessage(`${resultTypeLabel(game)} results are locked as non-played outcomes. Use the guarded Day Workspace to review or change that outcome.`);
      return;
    }
    setScoreGameId(gameId);
    setScoreA(game?.score_a == null ? "" : String(game.score_a));
    setScoreB(game?.score_b == null ? "" : String(game.score_b));
    setUnusualScoreAcknowledged(false);
  }

  function updateTeamRow(index: number, patch: Partial<TeamEditorRow>) {
    setTeamRows((current) => current.map((row, rowIndex) => rowIndex === index ? { ...row, ...patch } : row));
  }

  function playerSelectValue(id: string) {
    return id || "";
  }

  function importedPlayerLabel(playerRef: unknown): string {
    const ref = String(playerRef || "");
    if (ref.startsWith("existing:")) {
      const playerId = ref.slice("existing:".length);
      return resultsPreview?.player_options.find((player) => String(player.id) === playerId)?.name || "Player unavailable";
    }
    if (ref.startsWith("create:")) {
      const importKey = ref.slice("create:".length);
      const imported = resultsPreview?.players.find((player) => String(player.import_key || "") === importKey);
      return String(imported?.display_name || imported?.name || "Player unavailable");
    }
    return "Player unavailable";
  }

  function importedTeamLabel(teamRef: unknown): string {
    const team = resultsPreview?.teams.find((row) => String(row.team_ref || "") === String(teamRef || ""));
    if (!team) return "Team unavailable";
    const first = importedPlayerLabel(team.p1_ref);
    const second = team.p2_ref ? importedPlayerLabel(team.p2_ref) : "";
    return second ? `${first} / ${second}` : first;
  }

  useAuthenticatedAutoLoad(
    status.enabled ? `${accessToken}\u0000${initialTournamentId}` : "",
    () => loadOps(initialTournamentId, initialDrawId || "")
  );


  if (!accessToken) {
    return (
      <article style={{ ...cardStyle, background: "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Admin sign-in required</h2>
        <p><Link href="/admin/login">Open admin login</Link></p>
      </article>
    );
  }
  if (!status.enabled) {
    return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Next Tournament Admin is disabled</h2><p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the Tournament Admin pilot flag on FastAPI."}</p></article>;
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      {!operationsWriteReady ? (
        <article data-testid="tournament-ops-read-only-banner" style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
          <h2 style={{ marginTop: 0 }}>Tournament Ops is read-only</h2>
          <p style={{ color: "#7c2d12", marginBottom: 0 }}>
            Tournament and draw snapshots remain available. POST-backed previews and mutation controls are hidden until the service role, dedicated Tournament Ops mutation flag, and operations runtime are all enabled in staging.
          </p>
        </article>
      ) : null}

      {registrationImportRecovery ? (
        <article data-testid="registration-import-recovery" style={{ ...cardStyle, background: "#fff7ed", borderColor: "#f59e0b" }}>
          <h2 style={{ marginTop: 0 }}>Interrupted registration import retained</h2>
          <p>The exact registration import request is retained in this browser tab. Its technical recovery reference remains protected without exposing tournament or draw identifiers in the operator workflow.</p>
          <p style={{ color: "#92400e" }}>{registrationImportRecovery.message}</p>
          <p>All guarded Ops writes in this club tab stay blocked until commit-safe operation evidence proves completion or the exact recovery reservation proves the request could not begin.</p>
          <ConfirmAction
            triggerLabel="Reconcile protected operation"
            title="Reconcile this registration import?"
            description="This reads the retained operation and never repeats the import. A normal empty recovery remains blocked unless commit-safe evidence exists."
            confirmLabel="Yes, reconcile operation"
            confirmationText={registrationImportReconcileConfirmation}
            disabled={!accessToken || !operationsWriteReady}
            busy={busy}
            onConfirm={() => reconcileRegistrationImport(registrationImportRecovery)}
          />
        </article>
      ) : null}

      {operationsWriteReady && snapshot && shows("draws") ? (
        <article style={{ ...cardStyle, background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Create empty division draw</h2>
          <p style={{ color: "#475569" }}>This creates a DRAFT draw shell scoped to the selected registration division.</p>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(240px, 1fr) minmax(180px, 1fr)", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>Registration division</strong><br /><select value={drawEventOptionId} onChange={(event) => setDrawEventOptionId(event.target.value)} style={inputStyle}><option value="">Legacy / tournament-wide draw</option>{(snapshot.event_options || []).filter(eventOptionEnabled).map((row) => <option key={String(row.id)} value={String(row.id)}>{eventOptionLabel(row)}</option>)}</select></label>
            <label><strong>Draw name</strong><br /><input value={drawName} onChange={(event) => setDrawName(event.target.value)} placeholder="optional" style={inputStyle} /></label>
          </div>
          <p><ConfirmAction triggerLabel="Create draw" title="Create this tournament draw?" description={`This creates a new draft draw${drawName.trim() ? ` named ${drawName.trim()}` : ""}${drawEventOptionId ? " for the selected registration division" : " for the tournament-wide legacy scope"}.`} confirmLabel="Yes, create draw" confirmationText="CREATE DRAW" disabled={!accessToken || !operationsWriteReady || !reviewedState || !registrationImportRecoveryLoaded || registrationImportBlocksWrites} busy={busy} onConfirm={createDraw} /></p>
        </article>
      ) : null}

      {snapshot ? (
        <>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>{snapshot.tournament.name}</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem" }}>
              <div><strong>Draws</strong><br />{snapshot.summary.draws}</div>
              <div><strong>Teams</strong><br />{snapshot.summary.teams}</div>
              <div><strong>Games</strong><br />{snapshot.summary.games}</div>
              <div><strong>Team rating children</strong><br />{snapshot.summary.rating_children ?? 0}</div>
              <div><strong>Completed games</strong><br />{snapshot.summary.completed_games ?? 0}</div>
              <div><strong>Podium rows</strong><br />{snapshot.summary.podium}</div>
            </div>
            {snapshot.warnings?.length ? <ul style={{ color: "#92400e" }}>{snapshot.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
            <div style={{ marginTop: "1rem", padding: "0.75rem", borderRadius: "10px", background: operationsWriteReady && reviewedState ? "#f0fdf4" : "#fff7ed", color: operationsWriteReady && reviewedState ? "#166534" : "#9a3412" }}>
              <strong>{operationsWriteReady && reviewedState ? "Guarded staging writes ready" : "Writes remain closed"}</strong>
              <div>Reviewed state: <code>{reviewedState ? reviewedState.slice(0, 16) : "unavailable"}</code></div>
              <div>Official publish: {officialPublishReady ? "gate open" : "gate closed"} · email handoff: {snapshot.operation_runtime?.email_handoff_enabled ? "gate open" : "gate closed"} · email mode: {snapshot.operation_runtime?.email_mode || "unknown"}</div>
              {!operationsWriteReady ? <p style={{ marginBottom: 0 }}>The FastAPI service role, dedicated Tournament Ops mutation flag, and operations runtime must all be enabled in staging.</p> : null}
              {!reviewedState ? <p style={{ marginBottom: 0 }}>The authoritative state could not be fingerprinted, so mutations fail closed.</p> : null}
              <p style={{ marginBottom: 0 }}>For uncertain outcomes, preserve the exact request and operation key. <a href={snapshot.streamlit_fallback_url || status.streamlit_fallback_url || "https://juprtrespalapas.streamlit.app"} target="_blank" rel="noreferrer">Open the Streamlit fallback</a>.</p>
            </div>
          </article>

          <article style={{ ...cardStyle, background: "#f8fafc" }}>
            <h2 style={{ marginTop: 0 }}>Draw selection</h2>
            <div style={{ display: "grid", gridTemplateColumns: "minmax(240px, 1fr) auto", gap: "0.75rem", alignItems: "end" }}>
              <label><strong>Draw</strong><br /><select value={selectedDrawId} onChange={(event) => selectDraw(event.target.value)} disabled={busy} style={inputStyle}><option value="">Choose a draw…</option>{snapshot.draws.map((draw, index) => <option key={draw.id} value={draw.id}>{draw.name || `Unnamed draw ${index + 1}`}</option>)}</select></label>
              <button type="button" onClick={() => selectedTournamentId && selectedDrawId ? loadOps(selectedTournamentId, selectedDrawId) : undefined} disabled={!selectedTournamentId || !selectedDrawId || busy} style={ghostButtonStyle}>Reload selected draw</button>
            </div>
          </article>

          {operationsWriteReady && shows("import") ? <>
          <article style={{ ...cardStyle, background: "#f8fafc" }}>
            <h2 style={{ marginTop: 0 }}>Import confirmed registrations</h2>
            <p style={{ color: "#475569" }}>Imports confirmed registration entries for the selected draw’s registration day/division. Each registration must already be linked to a JUPR player.</p>
            <div style={{ display: "grid", gridTemplateColumns: "minmax(160px, 220px) auto", gap: "0.75rem", alignItems: "end" }}>
              <label><strong>Mode</strong><br /><select value={registrationImportMode} onChange={(event) => setRegistrationImportMode(event.target.value)} style={inputStyle}><option value="REPLACE">Replace current teams</option><option value="APPEND">Append after current teams</option></select></label>
              <ConfirmAction triggerLabel="Import registrations" title={`${registrationImportMode === "REPLACE" ? "Replace teams from" : "Append teams from"} confirmed registrations?`} description={registrationImportMode === "REPLACE" ? "This replaces the draw's current team list with teams built from confirmed, player-linked registrations." : "This appends teams built from confirmed, player-linked registrations after the current teams."} confirmLabel={registrationImportMode === "REPLACE" ? "Yes, replace teams" : "Yes, append teams"} confirmationText="IMPORT REGISTRATIONS" tone={registrationImportMode === "REPLACE" ? "danger" : "default"} disabled={drawCasWriteDisabled || !selectedDrawId} busy={busy} onConfirm={importRegistrations} />
            </div>
          </article>

          <article style={{ ...cardStyle, background: "#f8fafc" }}>
            <h2 style={{ marginTop: 0 }}>Bulk import teams</h2>
            <p style={{ color: "#475569" }}>Paste CSV or TSV with headers like <code>Player 1, Player 2, Seed, Notes</code>. Player names must match the club roster. Import is blocked after games exist.</p>
            <textarea value={bulkTeamText} onChange={(event) => setBulkTeamText(event.target.value)} style={{ ...inputStyle, minHeight: "8rem", fontFamily: "monospace" }} />
            <div style={{ display: "grid", gridTemplateColumns: "minmax(160px, 220px) auto", gap: "0.75rem", alignItems: "end", marginTop: "0.75rem" }}>
              <label><strong>Mode</strong><br /><select value={bulkTeamMode} onChange={(event) => setBulkTeamMode(event.target.value)} style={inputStyle}><option value="REPLACE">Replace current teams</option><option value="APPEND">Append after current teams</option></select></label>
              <ConfirmAction triggerLabel="Import teams" title={`${bulkTeamMode === "REPLACE" ? "Replace" : "Append"} teams from this file?`} description={bulkTeamMode === "REPLACE" ? "This replaces the draw's current teams with the reviewed CSV or TSV contents." : "This appends teams from the reviewed CSV or TSV contents after the current teams."} confirmLabel={bulkTeamMode === "REPLACE" ? "Yes, replace teams" : "Yes, append teams"} confirmationText="IMPORT TEAMS" tone={bulkTeamMode === "REPLACE" ? "danger" : "default"} disabled={drawCasWriteDisabled || !selectedDrawId} busy={busy} onConfirm={importBulkTeams} />
            </div>
          </article>
          </> : null}

          {operationsWriteReady && shows("draws") ? <>
          <article style={{ ...cardStyle, background: "#f8fafc" }}>
            <h2 style={{ marginTop: 0 }}>Team editor</h2>
            <p style={{ color: "#475569" }}>Assign players manually, then review the full team list before saving.</p>
            {selectedDrawId && teamEditorPlayerChoicesReady ? (
              <>
                <ConfirmAction triggerLabel="Save teams" title="Replace the draw's saved teams?" description="This saves the currently reviewed team rows as the authoritative team list for the selected draw." confirmLabel="Yes, save teams" confirmationText="SAVE TEAMS" tone="danger" disabled={drawCasWriteDisabled} busy={busy} onConfirm={saveTeams} />
                <div style={{ overflowX: "auto", marginTop: "1rem" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "920px" }}>
                    <thead><tr>{["Team #", "Player 1", "Player 2", "Seed", "Notes", "Action"].map((header) => <th key={header} style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>{header}</th>)}</tr></thead>
                    <tbody>{teamRows.map((row, index) => <tr key={row.editor_key}>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><input value={row.team_number} onChange={(event) => updateTeamRow(index, { team_number: event.target.value })} style={inputStyle} /></td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><select value={playerSelectValue(row.player1_id)} onChange={(event) => updateTeamRow(index, { player1_id: event.target.value })} style={inputStyle}><option value="">Choose player…</option>{players.map((player) => <option key={player.id} value={String(player.id)}>{player.name}</option>)}</select></td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><select value={playerSelectValue(row.player2_id)} onChange={(event) => updateTeamRow(index, { player2_id: event.target.value })} style={inputStyle}><option value="">Singles / no partner</option>{players.map((player) => <option key={player.id} value={String(player.id)}>{player.name}</option>)}</select></td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><input value={row.seed} onChange={(event) => updateTeamRow(index, { seed: event.target.value })} style={inputStyle} /></td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><input value={row.notes} onChange={(event) => updateTeamRow(index, { notes: event.target.value })} style={inputStyle} /></td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><button type="button" onClick={() => setTeamRows((current) => current.filter((_, rowIndex) => rowIndex !== index))} style={ghostButtonStyle}>Remove</button></td>
                    </tr>)}</tbody>
                  </table>
                </div>
                <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}><button type="button" onClick={() => setTeamRows((current) => [...current, { editor_key: `new-team-${Date.now()}`, team_number: String(current.length + 1), player1_id: "", player2_id: "", seed: String(current.length + 1), notes: "" }])} style={ghostButtonStyle}>Add team row</button><button type="button" onClick={() => resetTeamEditor(snapshot, selectedDrawId)} style={ghostButtonStyle}>Reset from snapshot</button></p>
              </>
            ) : <p style={{ color: "#64748b" }}>{selectedDrawId ? "Club player names are unavailable, so manual team setup is disabled. Reload the authoritative snapshot or use the guarded registration import." : "Create or select a draw before editing teams."}</p>}
          </article>

          <article style={{ ...cardStyle, background: "#f8fafc" }}>
            <h2 style={{ marginTop: 0 }}>Round-robin schedule</h2>
            <p style={{ color: "#475569" }}>
              Generate a new schedule, add only missing pairings to a partial schedule,
              or rebuild an unstarted schedule. Rebuild is refused after any score,
              publication, award, podium, or day-live evidence exists.
            </p>
            <div style={{ display: "flex", gap: "0.65rem", flexWrap: "wrap" }}>
              <ConfirmAction triggerLabel="Generate games" title="Generate round-robin games?" description="This creates the schedule from the currently reviewed teams and draw version." confirmLabel="Yes, generate games" confirmationText="GENERATE GAMES" disabled={teamSnapshotCasDisabled || !selectedDrawId} busy={busy} onConfirm={generateGames} />
              <ConfirmAction triggerLabel="Reconcile missing games" title="Reconcile this partial round robin?" description="This preserves every valid existing game, including finalized games, and inserts only missing current-roster pairings." confirmLabel="Yes, reconcile games" confirmationText="RECONCILE GAMES" disabled={teamSnapshotCasDisabled || !selectedDrawId || !reviewedSourceGameVersions.length} busy={busy} onConfirm={(text) => recoverRoundRobin("reconcile", text)} />
              <ConfirmAction triggerLabel="Rebuild unstarted games" title="Rebuild this unstarted round robin?" description="This removes the reviewed unstarted round-robin games and replaces them with one complete schedule. Any result or downstream evidence blocks the action." confirmLabel="Yes, rebuild games" confirmationText="REBUILD GAMES" tone="danger" disabled={teamSnapshotCasDisabled || !selectedDrawId || !reviewedSourceGameVersions.length} busy={busy} onConfirm={(text) => recoverRoundRobin("rebuild", text)} />
            </div>
          </article>
          <article style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
            <h2 style={{ marginTop: 0 }}>Cancel empty setup</h2>
            <p style={{ color: "#7c2d12" }}>
              Use these guarded actions only for a draw or event that received no
              participation. The server refuses either action if registrations,
              teams, games, podiums, awards, official matches, or day-live evidence exists.
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "minmax(240px, 1fr) auto auto", gap: "0.75rem", alignItems: "end" }}>
              <label><strong>Event</strong><br />
                <select value={emptyEventOptionId} onChange={(event) => setEmptyEventOptionId(event.target.value)} style={inputStyle}>
                  <option value="">Choose an event…</option>
                  {(snapshot.event_options || []).filter(eventOptionEnabled).map((row) => <option key={String(row.id)} value={String(row.id)}>{eventOptionLabel(row)}</option>)}
                </select>
              </label>
              <ConfirmAction triggerLabel="Cancel selected empty event" title="Cancel this empty event?" description="This disables only the selected zero-entry event after the server verifies it has no registration or draw evidence." confirmLabel="Yes, cancel empty event" confirmationText="CANCEL EMPTY EVENT" tone="danger" disabled={guardedWriteDisabled || !emptyEventOptionId} busy={busy} onConfirm={cancelEmptyEvent} />
              <ConfirmAction triggerLabel="Cancel selected empty draw" title="Cancel this empty draw?" description="This disables only the working draw after the server verifies it has no team, game, result, award, publication, or day-live evidence." confirmLabel="Yes, cancel empty draw" confirmationText="CANCEL EMPTY DRAW" tone="danger" disabled={drawCasWriteDisabled || !selectedDrawId} busy={busy} onConfirm={cancelEmptyDraw} />
            </div>
          </article>
          {showsLegacyDrawRuntime ? <>
          <article style={{ ...cardStyle, background: "#f8fafc" }}>
            <h2 style={{ marginTop: 0 }}>Score game</h2>
            <p style={{ color: "#475569" }}>Select a matchup and enter the score. The configured scoring format is enforced; unusual but possible scores require an explicit review acknowledgement.</p>
            {scoreableGames.length ? <div style={{ display: "grid", gap: "0.75rem" }}>
              <div style={{ display: "grid", gridTemplateColumns: "minmax(260px, 1fr) minmax(100px, 140px) minmax(100px, 140px) auto", gap: "0.75rem", alignItems: "end" }}>
                <label><strong>Matchup</strong><br /><select value={scoreGameId} onChange={(event) => selectScoreGame(event.target.value)} style={inputStyle}><option value="">Choose a matchup…</option>{scoreableGames.map((game) => <option key={String(game.id)} value={String(game.id)}>{gameLabel(game, teamsById, players)}</option>)}</select></label>
                <label><strong>Score A</strong><br /><input type="number" min="0" step="1" value={scoreA} onChange={(event) => { setScoreA(event.target.value); setUnusualScoreAcknowledged(false); }} style={inputStyle} /></label>
                <label><strong>Score B</strong><br /><input type="number" min="0" step="1" value={scoreB} onChange={(event) => { setScoreB(event.target.value); setUnusualScoreAcknowledged(false); }} style={inputStyle} /></label>
                <ConfirmAction triggerLabel="Save score" title="Save this matchup score?" description={`This records ${scoreA || "—"}–${scoreB || "—"} for the selected matchup${unusualScoreAcknowledged ? " with an explicit unusual-score acknowledgement" : ""}.`} confirmLabel="Yes, save score" confirmationText="SAVE SCORE" disabled={drawCasWriteDisabled || !selectedScoreGame || !scoreA.trim() || !scoreB.trim()} busy={busy} onConfirm={saveScore} />
              </div>
              <label style={{ display: "flex", gap: "0.45rem", alignItems: "flex-start", color: "#475569" }}>
                <input type="checkbox" checked={unusualScoreAcknowledged} onChange={(event) => setUnusualScoreAcknowledged(event.target.checked)} />
                I reviewed this score and confirm it is intentional if the server classifies it as unusual.
              </label>
            </div> : <p style={{ color: "#64748b" }}>{snapshot.games.length ? "No ordinary played matchup is available for score entry." : "Generate games before scoring."}</p>}
            {nonPlayedGames.length ? (
              <section aria-label="Non-played tournament outcomes" style={{ marginTop: "1rem", padding: "0.85rem", border: "1px solid #fed7aa", borderRadius: "10px", background: "#fff7ed" }}>
                <h3 style={{ marginTop: 0 }}>Non-played outcomes</h3>
                <p style={{ color: "#7c2d12" }}>These results stay visible but are locked in this ordinary score editor. Review or change them only in the guarded Day Workspace.</p>
                <ul>
                  {nonPlayedGames.map((game) => (
                    <li key={String(game.id || gameLabel(game, teamsById, players))}>
                      <strong>{resultTypeLabel(game)} — not played:</strong> {gameLabel(game, teamsById, players)}. Winner: {teamLabel(teamsById.get(String(game.winner_team_id || "")), players)}. {shortValue(game.result_note)}
                    </li>
                  ))}
                </ul>
              </section>
            ) : null}
          </article>
          <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Generate playoffs</h2><p style={{ color: "#475569" }}>After all round-robin games are scored, choose how many teams advance.</p><div style={{ display: "grid", gridTemplateColumns: "minmax(140px, 180px) auto", gap: "0.75rem", alignItems: "end" }}><label><strong>Advance count</strong><br /><select value={playoffAdvanceCount} onChange={(event) => setPlayoffAdvanceCount(event.target.value)} style={inputStyle}><option value="4">4 teams</option><option value="5">5 teams</option><option value="6">6 teams</option></select></label><ConfirmAction triggerLabel="Generate playoffs" title="Generate the playoff bracket?" description={`This advances ${playoffAdvanceCount} teams from the reviewed round-robin results into the playoff bracket.`} confirmLabel="Yes, generate playoffs" confirmationText="GENERATE PLAYOFFS" disabled={gameSnapshotCasDisabled || !selectedDrawId} busy={busy} onConfirm={generatePlayoffs} /></div></article>
          <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Generate podium</h2><p style={{ color: "#475569" }}>Creates draw-scoped podium rows from finalized playoffs, or from completed round-robin standings when no playoffs exist.</p><ConfirmAction triggerLabel="Generate podium" title="Generate podium placements?" description="This calculates and stores podium rows from the currently reviewed final results." confirmLabel="Yes, generate podium" confirmationText="GENERATE PODIUM" disabled={gameSnapshotCasDisabled || !selectedDrawId} busy={busy} onConfirm={generatePodium} /></article>
          <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Review and award podium</h2><p style={{ color: "#475569" }}>Podium awards require current explicit review evidence and exact award versions. This legacy editor cannot mint awards.</p><Link href={tournamentRouteHref("/admin/tournaments/live-operations/podium", { tournamentId: selectedTournamentId, tournamentName: snapshot.tournament.name || "", drawId: selectedDrawId })}>Open guarded Podium review</Link></article>
          </> : null}
          </> : null}

          {operationsWriteReady && shows("results") ? (
            <article style={{ ...cardStyle, background: "#eff6ff", borderColor: "#bfdbfe" }}>
              <h2 style={{ marginTop: 0 }}>Reviewed DUPR results CSV</h2>
              <p style={{ color: "#1e3a8a" }}>Upload or paste a DUPR-style CSV, preview every player mapping and included match without writing, then commit only the exact reviewed fingerprint. The raw CSV stays out of the durable operation ledger.</p>
              <div style={{ display: "grid", gridTemplateColumns: "minmax(160px, 220px) minmax(240px, 1fr)", gap: "0.75rem", alignItems: "end" }}>
                <label><strong>Import mode</strong><br /><select value={resultsImportMode} onChange={(event) => { setResultsImportMode(event.target.value); setResultsPreview(null); setResultsReviewDirty(true); }} style={inputStyle}><option value="REPLACE">Replace draw results</option><option value="APPEND">Append imported results</option></select></label>
                <label><strong>CSV file (1 MB maximum)</strong><br /><input type="file" accept=".csv,text/csv" onChange={(event) => void loadResultsFile(event.target.files?.[0] || null)} style={inputStyle} /></label>
              </div>
              <label style={{ display: "block", marginTop: "0.75rem" }}><strong>CSV contents</strong><br /><textarea value={resultsRawText} onChange={(event) => { setResultsRawText(event.target.value); setResultsPreview(null); setResultsReviewDirty(true); }} style={{ ...inputStyle, minHeight: "10rem", fontFamily: "monospace" }} /></label>
              <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}>
                <button type="button" onClick={previewResultsImport} disabled={busy || !accessToken || !selectedDrawId || !resultsRawText.trim()} style={ghostButtonStyle}>{busy ? "Reviewing…" : resultsPreview ? "Re-preview exact choices" : "Preview without writing"}</button>
                <label style={{ display: "flex", gap: "0.4rem", alignItems: "center" }}><input type="checkbox" checked={allowDuplicateMapping} onChange={(event) => { setAllowDuplicateMapping(event.target.checked); setResultsReviewDirty(true); }} />Explicitly allow duplicate player mappings</label>
                <label style={{ display: "flex", gap: "0.4rem", alignItems: "center" }}><input type="checkbox" checked={unusualImportScoresAcknowledged} onChange={(event) => setUnusualImportScoresAcknowledged(event.target.checked)} />I reviewed every score marked unusual in the preview</label>
              </p>

              {resultsPreview ? (
                <div style={{ display: "grid", gap: "1rem" }}>
                  <div style={{ padding: "0.75rem", background: resultsPreview.ok && !resultsReviewDirty ? "#f0fdf4" : "#fff7ed", borderRadius: "10px" }}>
                    <strong>{resultsPreview.ok ? "Preview parsed" : "Preview needs review"}</strong> · {resultsPreview.summary.imported_players} players · {resultsPreview.summary.teams} teams · {resultsPreview.summary.matches} matches · {resultsPreview.summary.create_players} proposed new players · {resultsPreview.summary.unusual_scores || 0} unusual scores
                    <div>Review fingerprint: <code>{resultsPreview.review_fingerprint.slice(0, 16)}</code>{resultsReviewDirty ? " · choices changed; preview again" : " · exact review current"}</div>
                    {resultsPreview.errors.length ? <ul style={{ color: "#b91c1c" }}>{resultsPreview.errors.map((error) => <li key={error}>{error}</li>)}</ul> : null}
                    {resultsPreview.warnings.length ? <ul style={{ color: "#92400e" }}>{resultsPreview.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
                  </div>

                  <div style={{ overflowX: "auto" }}>
                    <h3>Player mappings</h3>
                    <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}>
                      <thead><tr>{["Imported player", "Decision", "Existing JUPR player", "Suggestion"].map((header) => <th key={header} style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #93c5fd" }}>{header}</th>)}</tr></thead>
                      <tbody>{resultsPreview.players.map((player, index) => {
                        const importKey = String(player.import_key || index);
                        const decision = resultsMappings[importKey] || {};
                        const suggestion = resultsPreview.suggestions[importKey] || {};
                        return <tr key={importKey}>
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}>{String(player.display_name || player.name || `Imported player ${index + 1}`)}</td>
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}><select value={decision.action || "unresolved"} onChange={(event) => { const action = event.target.value; setResultsMappings((current) => ({ ...current, [importKey]: { action, player_id: action === "use_existing" ? current[importKey]?.player_id ?? null : null } })); setResultsReviewDirty(true); }} style={inputStyle}><option value="unresolved">Resolve before commit</option><option value="use_existing">Use existing player</option><option value="create_new">Create new player</option></select></td>
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}>{decision.action === "use_existing" ? <select value={String(decision.player_id || "")} onChange={(event) => { setResultsMappings((current) => ({ ...current, [importKey]: { action: "use_existing", player_id: event.target.value || null } })); setResultsReviewDirty(true); }} style={inputStyle}><option value="">Choose player…</option>{resultsPreview.player_options.map((option) => <option key={String(option.id)} value={String(option.id)}>{option.name}</option>)}</select> : <span>—</span>}</td>
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}>{shortValue(suggestion.suggested_player_name || suggestion.suggested_name || suggestion.reason)}</td>
                        </tr>;
                      })}</tbody>
                    </table>
                  </div>

                  <div style={{ overflowX: "auto" }}>
                    <h3>Match review</h3>
                    <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "760px" }}>
                      <thead><tr>{["Source row", "Include", "Stage", "Team A", "Team B", "Score"].map((header) => <th key={header} style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #93c5fd" }}>{header}</th>)}</tr></thead>
                      <tbody>{resultsPreview.matches.map((match, index) => {
                        const rowKey = String(match.source_row || index + 1);
                        const review = resultsMatchReviews[rowKey] || { include: true, stage: String(match.stage || "PLAYOFF") };
                        return <tr key={rowKey}>
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}>{rowKey}</td>
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}><input type="checkbox" checked={review.include !== false} onChange={(event) => { setResultsMatchReviews((current) => ({ ...current, [rowKey]: { ...review, include: event.target.checked } })); setResultsReviewDirty(true); }} /></td>
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}><select value={review.stage || "PLAYOFF"} onChange={(event) => { setResultsMatchReviews((current) => ({ ...current, [rowKey]: { ...review, stage: event.target.value } })); setResultsReviewDirty(true); }} style={inputStyle}><option value="ROUND_ROBIN">Round robin</option><option value="PLAYOFF">Playoff</option><option value="BRONZE">Bronze</option><option value="FINAL">Final</option></select></td>
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}>{String(match.team_a_label || importedTeamLabel(match.team_a_ref))}</td>
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}>{String(match.team_b_label || importedTeamLabel(match.team_b_ref))}</td>
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}>{shortValue(match.score_a)}–{shortValue(match.score_b)}</td>
                        </tr>;
                      })}</tbody>
                    </table>
                  </div>

                  <div>
                    <h3>Podium review</h3>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>{["1", "2", "3"].map((placement) => <label key={placement}><strong>Place {placement}</strong><br /><select value={resultsPodiumRefs[placement] || ""} onChange={(event) => { setResultsPodiumRefs((current) => ({ ...current, [placement]: event.target.value || null })); setResultsReviewDirty(true); }} style={inputStyle}><option value="">No placement</option>{resultsPreview.podium_candidates.map((teamRef) => <option key={teamRef} value={teamRef}>{importedTeamLabel(teamRef)}</option>)}</select></label>)}</div>
                  </div>

                  {operationsWriteReady ? <div style={{ padding: "0.9rem", border: "1px solid #93c5fd", borderRadius: "10px", background: "white" }}>
                    <ConfirmAction triggerLabel="Commit reviewed results" title={`${resultsImportMode === "REPLACE" ? "Replace" : "Import"} the draw results?`} description={<>{resultsImportMode === "REPLACE" ? "This replaces the draw's teams, games, and podium with the exact reviewed CSV fingerprint." : "This appends the exact reviewed CSV results to the selected draw."}{resultsPreview.summary.create_players ? ` It also creates ${resultsPreview.summary.create_players} permanent player record${resultsPreview.summary.create_players === 1 ? "" : "s"} from the reviewed mappings.` : " It creates no new player records."}{resultsPreview.summary.unusual_scores ? ` You acknowledged ${resultsPreview.summary.unusual_scores} unusual score${resultsPreview.summary.unusual_scores === 1 ? "" : "s"}.` : ""}</>} confirmLabel={resultsImportMode === "REPLACE" ? "Yes, replace results" : "Yes, import results"} confirmationText={resultsImportMode === "REPLACE" ? "REPLACE RESULTS" : "IMPORT RESULTS"} tone={resultsImportMode === "REPLACE" ? "danger" : "default"} disabled={drawCasWriteDisabled || !selectedDrawId || !resultsPreview.ok || resultsReviewDirty || Boolean(resultsPreview.summary.unusual_scores && !unusualImportScoresAcknowledged)} busy={busy} onConfirm={commitResultsImport} />
                  </div> : null}
                </div>
              ) : null}
            </article>
          ) : null}

          {operationsWriteReady && shows("publish") ? (
            <article style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
              <h2 style={{ marginTop: 0 }}>Official publishing moved</h2>
              <p style={{ color: "#7c2d12" }}>This legacy operations editor cannot publish official matches or four-player rating children. The guarded Publish workspace requires tournament-wide score, podium-review, award, official-link, and recovery evidence before any write.</p>
              <Link href={tournamentRouteHref("/admin/tournaments/ops/publish", { tournamentId: selectedTournamentId, tournamentName: snapshot.tournament.name || "", drawId: selectedDrawId })}>Open guarded Publish workspace</Link>
            </article>
          ) : null}

          {showsLegacyDrawRuntime ? (
          <article data-testid="legacy-ops-human-summary" style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Human-readable live details</h2>
            <p style={{ color: "#475569" }}>Raw draw, player, team, game, and podium identifiers are intentionally hidden from this legacy workspace. Use the focused Live views for matchup cards, score corrections, and podium review.</p>
            <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
              <Link href={tournamentRouteHref("/admin/tournaments/live-operations/draws", { tournamentId: selectedTournamentId, tournamentName: snapshot.tournament.name || "", drawId: selectedDrawId })}>Open draw overview</Link>
              <Link href={tournamentRouteHref("/admin/tournaments/live-operations/scoring", { tournamentId: selectedTournamentId, tournamentName: snapshot.tournament.name || "", drawId: selectedDrawId })}>Open matchup scoring</Link>
              <Link href={tournamentRouteHref("/admin/tournaments/live-operations/podium", { tournamentId: selectedTournamentId, tournamentName: snapshot.tournament.name || "", drawId: selectedDrawId })}>Open podium review</Link>
            </p>
          </article>
          ) : null}
        </>
      ) : null}

      {message ? <p role="status" aria-live="polite" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("error") || message.toLowerCase().includes("sign in") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
