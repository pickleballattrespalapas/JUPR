"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, actionUncertain } from "@/components/interaction";
import type { ActionCompletion } from "@/components/interaction";
import type { AdminTournamentLiveStatusResponse } from "@/lib/adminTournamentApi";
import {
  AdminTournamentDayOpsApiError,
  executeAdminTournamentDayCommand,
  fetchAdminTournamentDayOptions,
  fetchAdminTournamentDayWorkspace,
  reconcileAdminTournamentDayOperation
} from "@/lib/adminTournamentDayOpsApi";
import type {
  AdminTournamentDayBlockerValue,
  AdminTournamentDayCommandAction,
  AdminTournamentDayCommandExpected,
  AdminTournamentDayCommandPayload,
  AdminTournamentDayCommandRequest,
  AdminTournamentDayDraw,
  AdminTournamentDayGame,
  AdminTournamentDayOperation,
  AdminTournamentDayOption,
  AdminTournamentDayReadiness,
  AdminTournamentDayWorkspaceSnapshot
} from "@/lib/adminTournamentDayOpsApi";
import {
  advanceCountSelection,
  dayActionConfirmation,
  dayRunAcceptsLiveCommands,
  dayRunHasStarted,
  retainedDayCommandStorageKey,
  validateDayCorrectionDraft,
  validateDayScoreDraft,
  validateNonPlayedOutcomeDraft,
  visibleServerQueue,
  workspaceScopeKey
} from "@/lib/tournamentDayWorkspaceState.mjs";
import { tournamentRouteHref } from "@/lib/tournamentRouteContext";
import { useAdminSession } from "@/lib/useAdminSession";
import { useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import styles from "./TournamentDayWorkspacePanel.module.css";

export type TournamentDayWorkspacePanelFocus = "board" | "queue" | "draws" | "corrections";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentLiveStatusResponse;
  tournamentId: string;
  tournamentName: string;
  initialDayId: string;
  initialDrawId: string;
  initialCourtId: string;
  initialGameId: string;
  initialPanel: TournamentDayWorkspacePanelFocus;
};

type PendingDayCommand = {
  clubId: string;
  tournamentId: string;
  dayId: string;
  createdAt: string;
  request: AdminTournamentDayCommandRequest;
};

type ReviewedGameTruth = {
  version: string;
  drawId: string;
  courtId: string;
  queueEntryVersion: string;
  scoreA: string;
  scoreB: string;
  resultType: string;
  resultNote: string;
  finalizedAt: string;
};

type ScoreEditor = {
  gameId: string;
  courtId: string;
  scoreA: string;
  scoreB: string;
  reviewing: boolean;
  error: string;
  unusualScoreAcknowledged: boolean;
  expected: AdminTournamentDayCommandExpected;
  reviewedGame: ReviewedGameTruth;
  reviewedAssignmentVersion: string;
};

type CorrectionEditor = {
  gameId: string;
  scoreA: string;
  scoreB: string;
  reviewing: boolean;
  error: string;
  unusualScoreAcknowledged: boolean;
  expected: AdminTournamentDayCommandExpected;
  reviewedGame: ReviewedGameTruth;
};

type OutcomeEditor = {
  gameId: string;
  resultType: "FORFEIT" | "NO_SHOW" | "RETIREMENT";
  winnerTeamId: string;
  resultNote: string;
  reviewing: boolean;
  error: string;
  expected: AdminTournamentDayCommandExpected;
  reviewedGame: ReviewedGameTruth;
};

const RECOVERY_STATUSES = new Set(["intent", "mutated", "recovery_required"]);

function assertWorkspaceSnapshotScope(
  payload: AdminTournamentDayWorkspaceSnapshot,
  clubId: string,
  tournamentId: string,
  dayId: string
): void {
  const returnedClubId = String(payload.scope?.club_id || "");
  const scopedTournamentId = String(payload.scope?.tournament_id || "");
  const scopedDayId = String(payload.scope?.registration_day_id || "");
  const returnedTournamentId = String(payload.tournament?.id || "");
  const returnedDayId = String(payload.day_scope?.selected_day_id || "");
  const dayRunScope = String(payload.day_run?.registration_day_id || "");
  if (
    returnedClubId !== clubId
    || scopedTournamentId !== tournamentId
    || scopedDayId !== dayId
    || returnedTournamentId !== tournamentId
    || returnedDayId !== dayId
    || (dayRunScope && dayRunScope !== dayId)
  ) {
    throw new Error(
      "The response belongs to a different tournament-day scope. Reload from Tournament Manager before taking another action."
    );
  }
}

function statusLabel(value: string | null | undefined): string {
  return String(value || "Unknown")
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/^./, (letter) => letter.toUpperCase());
}

function timestamp(value: string | null | undefined): string {
  if (!value) return "Not available";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function sideLabel(side: AdminTournamentDayGame["team_a"]): string {
  if (side.name) return side.name;
  return side.participant_names.filter(Boolean).join(" / ") || "Team unavailable";
}

function matchupLabel(game: AdminTournamentDayGame | undefined): string {
  if (!game) return "Matchup unavailable";
  return `${sideLabel(game.team_a)} vs ${sideLabel(game.team_b)}`;
}

function gameStageLabel(game: AdminTournamentDayGame | undefined): string {
  if (!game) return "Scheduled game";
  return [game.draw_name, game.stage, game.round_label, game.slot_label]
    .map((value) => String(value || "").trim())
    .filter(Boolean)
    .join(" · ");
}

function resultTypeLabel(game: AdminTournamentDayGame): string {
  return statusLabel(game.result_type || "PLAYED");
}

function reviewedGameTruth(
  game: AdminTournamentDayGame,
  context: Partial<Pick<ReviewedGameTruth, "courtId" | "queueEntryVersion">> = {}
): ReviewedGameTruth {
  return {
    version: String(game.version || ""),
    drawId: String(game.draw_id || ""),
    courtId: String(game.court_id || ""),
    queueEntryVersion: String(game.queue_entry_version || ""),
    scoreA: game.score_a == null ? "" : String(game.score_a),
    scoreB: game.score_b == null ? "" : String(game.score_b),
    resultType: String(game.result_type || "PLAYED").toUpperCase(),
    resultNote: String(game.result_note || ""),
    finalizedAt: String(game.finalized_at || ""),
    ...context
  };
}

function reviewedGameStillCurrent(game: AdminTournamentDayGame | undefined, reviewed: ReviewedGameTruth): boolean {
  if (!game) return false;
  const current = reviewedGameTruth(game);
  return current.version === reviewed.version
    && current.drawId === reviewed.drawId
    && current.scoreA === reviewed.scoreA
    && current.scoreB === reviewed.scoreB
    && current.resultType === reviewed.resultType
    && current.resultNote === reviewed.resultNote
    && current.finalizedAt === reviewed.finalizedAt;
}

function blockerText(blocker: AdminTournamentDayBlockerValue): string {
  if (typeof blocker === "string") return blocker;
  return blocker.message || blocker.detail || blocker.title || blocker.code;
}

function blockerCode(blocker: AdminTournamentDayBlockerValue): string {
  return typeof blocker === "string" ? "" : String(blocker.code || "");
}

function readinessHasCode(readiness: AdminTournamentDayReadiness, code: string): boolean {
  return readiness.blockers.some((blocker) => blockerCode(blocker) === code);
}

function readinessOrBlocked(
  readiness: AdminTournamentDayReadiness | null | undefined,
  fallback: string
): AdminTournamentDayReadiness {
  return readiness || {
    ready: false,
    confirmation: null,
    blockers: [{ code: "READINESS_UNAVAILABLE", message: fallback }]
  };
}

function ReadinessBlockers({ readiness }: { readiness: AdminTournamentDayReadiness }) {
  if (readiness.ready) return <p className={styles.ready}>Server readiness: ready.</p>;
  return (
    <ul className={styles.blockers}>
      {(readiness.blockers.length
        ? readiness.blockers
        : [{ code: "BLOCKED", message: "The server has not marked this action ready." }]
      ).map((blocker, index) => (
        <li key={`${blockerText(blocker)}:${index}`}>{blockerText(blocker)}</li>
      ))}
    </ul>
  );
}

function readPendingCommand(clubId: string, tournamentId: string, dayId: string): PendingDayCommand | null {
  try {
    const raw = window.localStorage.getItem(retainedDayCommandStorageKey(clubId, tournamentId, dayId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PendingDayCommand;
    if (
      parsed.clubId !== clubId
      || parsed.tournamentId !== tournamentId
      || parsed.dayId !== dayId
      || !parsed.request?.client_idempotency_key
    ) return null;
    return parsed;
  } catch {
    return null;
  }
}

function drawAction(draw: AdminTournamentDayDraw): "activate_draw" | "pause_draw" | "resume_draw" {
  const state = String(draw.activation_state || draw.state || "").toUpperCase();
  if (state === "ACTIVE") return "pause_draw";
  if (state === "PAUSED") return "resume_draw";
  return "activate_draw";
}

function drawActionReadiness(draw: AdminTournamentDayDraw, action: ReturnType<typeof drawAction>) {
  if (action === "pause_draw") return readinessOrBlocked(draw.readiness.pause, "Pause readiness is unavailable.");
  if (action === "resume_draw") return readinessOrBlocked(draw.readiness.resume, "Resume readiness is unavailable.");
  return readinessOrBlocked(draw.readiness.activate, "Activation readiness is unavailable.");
}

function drawActionLabel(action: ReturnType<typeof drawAction>): string {
  if (action === "pause_draw") return "Pause draw";
  if (action === "resume_draw") return "Resume draw";
  return "Activate draw";
}

export default function TournamentDayWorkspacePanel({
  apiBase,
  clubId,
  status,
  tournamentId,
  tournamentName,
  initialDayId,
  initialDrawId,
  initialCourtId,
  initialGameId,
  initialPanel
}: Props) {
  const { accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const pathname = usePathname();
  const router = useRouter();
  const [dayOptions, setDayOptions] = useState<AdminTournamentDayOption[]>([]);
  const [selectedDayId, setSelectedDayId] = useState(initialDayId);
  const [snapshot, setSnapshot] = useState<AdminTournamentDayWorkspaceSnapshot | null>(null);
  const [panelFocus, setPanelFocus] = useState<TournamentDayWorkspacePanelFocus>(initialPanel);
  const [drawFilter, setDrawFilter] = useState(initialDrawId || "all");
  const [focusedCourtId, setFocusedCourtId] = useState(initialCourtId);
  const [focusedGameId, setFocusedGameId] = useState(initialGameId);
  const [scoreEditor, setScoreEditor] = useState<ScoreEditor | null>(null);
  const [correctionEditor, setCorrectionEditor] = useState<CorrectionEditor | null>(null);
  const [outcomeEditor, setOutcomeEditor] = useState<OutcomeEditor | null>(null);
  const [playoffAdvanceCounts, setPlayoffAdvanceCounts] = useState<Record<string, string>>({});
  const [pendingCommand, setPendingCommand] = useState<PendingDayCommand | null>(null);
  const [busyKey, setBusyKey] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastRefresh, setLastRefresh] = useState<string | null>(null);
  const scopeKey = workspaceScopeKey(accessToken, tournamentId, selectedDayId);
  const snapshotRequest = useLatestRequestGuard(scopeKey);
  const actionRequest = useLatestRequestGuard(scopeKey);

  const gamesById = useMemo(
    () => new Map((snapshot?.games || []).map((game) => [game.id, game])),
    [snapshot]
  );
  const activeRecovery = useMemo(
    () => (snapshot?.operations || []).find((operation) => RECOVERY_STATUSES.has(operation.status.toLowerCase())) || null,
    [snapshot]
  );
  const writesFrozen = Boolean(pendingCommand || activeRecovery);
  const runtimeWritesEnabled = Boolean(
    status.writes_enabled
      && status.write_flag?.enabled
      && snapshot?.runtime?.writes_enabled !== false
  );
  const dayRunState = String(snapshot?.day_run.state || "DRAFT");
  const dayStarted = dayRunHasStarted(dayRunState);
  const dayActive = dayRunAcceptsLiveCommands(dayRunState);
  const visibleQueue = useMemo(
    () => visibleServerQueue(snapshot?.eligible_queue || [], drawFilter),
    [drawFilter, snapshot]
  );

  const replaceWorkspaceUrl = useCallback((next: {
    dayId?: string;
    drawId?: string;
    courtId?: string;
    gameId?: string;
    panel?: TournamentDayWorkspacePanelFocus;
  }) => {
    const dayId = next.dayId ?? selectedDayId;
    const drawId = next.drawId ?? (drawFilter === "all" ? "" : drawFilter);
    const courtId = next.courtId ?? focusedCourtId;
    const gameId = next.gameId ?? focusedGameId;
    const panel = next.panel ?? panelFocus;
    router.replace(
      tournamentRouteHref(
        pathname,
        {
          tournamentId,
          tournamentName: snapshot?.tournament.name || tournamentName,
          drawId,
          dayId
        },
        { panel, court: courtId || null, game: gameId || null }
      ),
      { scroll: false }
    );
  }, [drawFilter, focusedCourtId, focusedGameId, panelFocus, pathname, router, selectedDayId, snapshot, tournamentId, tournamentName]);

  const loadWorkspace = useCallback(async (options: { silent?: boolean; signal?: AbortSignal } = {}) => {
    if (!apiBase || !accessToken || !selectedDayId) return;
    const generation = snapshotRequest.begin();
    if (!options.silent) setLoading(true);
    setError(null);
    try {
      const payload = await fetchAdminTournamentDayWorkspace({
        apiBase,
        clubId,
        tournamentId,
        dayId: selectedDayId,
        accessToken,
        signal: options.signal
      });
      if (!snapshotRequest.isCurrent(generation) || options.signal?.aborted) return;
      assertWorkspaceSnapshotScope(payload, clubId, tournamentId, selectedDayId);
      setSnapshot(payload);
      setDayOptions(payload.day_scope.available_days || []);
      setLastRefresh(payload.generated_at || new Date().toISOString());
      const retained = readPendingCommand(clubId, tournamentId, selectedDayId);
      setPendingCommand(retained);
      if (!options.silent) setMessage("Authoritative tournament-day state loaded.");
    } catch (loadError) {
      if (!snapshotRequest.isCurrent(generation) || options.signal?.aborted) return;
      setError(loadError instanceof Error ? loadError.message : "Unable to load the tournament-day workspace.");
    } finally {
      if (snapshotRequest.isCurrent(generation) && !options.silent) setLoading(false);
    }
  }, [accessToken, apiBase, clubId, selectedDayId, snapshotRequest, tournamentId]);

  useEffect(() => {
    if (!apiBase || !accessToken || !tournamentId) return;
    const controller = new AbortController();
    void fetchAdminTournamentDayOptions({
      apiBase,
      clubId,
      tournamentId,
      accessToken,
      signal: controller.signal
    }).then((days) => {
      if (controller.signal.aborted) return;
      setDayOptions(days);
      if (selectedDayId) return;
      const inferred = days.length === 1 ? days[0] : null;
      if (inferred) {
        setSelectedDayId(inferred.id);
        router.replace(
          tournamentRouteHref(
            pathname,
            {
              tournamentId,
              tournamentName,
              drawId: "",
              dayId: inferred.id
            },
            { panel: initialPanel }
          ),
          { scroll: false }
        );
      }
    }).catch((loadError) => {
      if (!controller.signal.aborted && !selectedDayId) {
        setError(loadError instanceof Error ? loadError.message : "Unable to load tournament days.");
      }
    });
    return () => controller.abort();
  }, [accessToken, apiBase, clubId, initialPanel, pathname, router, selectedDayId, tournamentId, tournamentName]);

  useEffect(() => {
    if (!selectedDayId || !accessToken) {
      setSnapshot(null);
      return;
    }
    const controller = new AbortController();
    void loadWorkspace({ signal: controller.signal });
    return () => controller.abort();
  }, [accessToken, loadWorkspace, selectedDayId]);

  useEffect(() => {
    if (!selectedDayId || !accessToken) return;
    const refresh = () => {
      if (document.visibilityState === "visible") void loadWorkspace({ silent: true });
    };
    const interval = window.setInterval(refresh, 15_000);
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", refresh);
    };
  }, [accessToken, loadWorkspace, selectedDayId]);

  useEffect(() => {
    if (!snapshot) return;
    const expectedSnapshotChanged = (expected: AdminTournamentDayCommandExpected) => (
      expected.day_run_version !== snapshot.day_run.version
      || expected.state_fingerprint !== snapshot.state_fingerprint
      || String(expected.queue_version || "") !== String(snapshot.queue_version || "")
    );
    const drawVersionChanged = (reviewed: ReviewedGameTruth, expected: AdminTournamentDayCommandExpected) => (
      Boolean(expected.draw_version)
      && snapshot.draws.find((draw) => draw.id === reviewed.drawId)?.version !== expected.draw_version
    );
    const courtVersionChanged = (reviewed: ReviewedGameTruth, expected: AdminTournamentDayCommandExpected) => (
      Boolean(expected.court_version)
      && snapshot.courts.find((court) => court.id === reviewed.courtId)?.version !== expected.court_version
    );
    const queueEntryVersion = (gameId: string, game: AdminTournamentDayGame | undefined) => (
      game?.queue_entry_version
      || snapshot.eligible_queue.find((entry) => entry.game_id === gameId)?.version
      || snapshot.held_games.find((entry) => entry.game_id === gameId)?.version
      || snapshot.blocked_games.find((entry) => entry.game_id === gameId)?.version
      || ""
    );

    if (scoreEditor) {
      const game = snapshot.games.find((row) => row.id === scoreEditor.gameId);
      const court = snapshot.courts.find((row) => row.id === scoreEditor.courtId);
      const stale = expectedSnapshotChanged(scoreEditor.expected)
        || !reviewedGameStillCurrent(game, scoreEditor.reviewedGame)
        || drawVersionChanged(scoreEditor.reviewedGame, scoreEditor.expected)
        || courtVersionChanged(scoreEditor.reviewedGame, scoreEditor.expected)
        || court?.current_assignment?.game_id !== scoreEditor.gameId
        || court?.current_assignment?.version !== scoreEditor.reviewedAssignmentVersion;
      if (stale) {
        setScoreEditor(null);
        setError("Score editor closed because authoritative tournament-day state changed. Reopen the current court assignment and review it again.");
      }
    }

    if (correctionEditor) {
      const game = snapshot.games.find((row) => row.id === correctionEditor.gameId);
      const stale = expectedSnapshotChanged(correctionEditor.expected)
        || !reviewedGameStillCurrent(game, correctionEditor.reviewedGame)
        || drawVersionChanged(correctionEditor.reviewedGame, correctionEditor.expected);
      if (stale) {
        setCorrectionEditor(null);
        setError("Correction editor closed because the reviewed result or authoritative tournament-day version changed. Reopen the result before correcting it.");
      }
    }

    if (outcomeEditor) {
      const game = snapshot.games.find((row) => row.id === outcomeEditor.gameId);
      const stale = expectedSnapshotChanged(outcomeEditor.expected)
        || !reviewedGameStillCurrent(game, outcomeEditor.reviewedGame)
        || drawVersionChanged(outcomeEditor.reviewedGame, outcomeEditor.expected)
        || courtVersionChanged(outcomeEditor.reviewedGame, outcomeEditor.expected)
        || (Boolean(outcomeEditor.expected.queue_entry_version)
          && queueEntryVersion(outcomeEditor.gameId, game) !== outcomeEditor.expected.queue_entry_version);
      if (stale) {
        setOutcomeEditor(null);
        setError("Non-played outcome editor closed because the reviewed matchup or authoritative tournament-day version changed. Reopen it before confirming an outcome.");
      }
    }
  }, [correctionEditor, outcomeEditor, scoreEditor, snapshot]);

  function persistPending(next: PendingDayCommand | null) {
    if (!selectedDayId) return;
    const key = retainedDayCommandStorageKey(clubId, tournamentId, selectedDayId);
    if (next) window.localStorage.setItem(key, JSON.stringify(next));
    else window.localStorage.removeItem(key);
    setPendingCommand(next);
  }

  function expectedVersions(fields: Partial<AdminTournamentDayCommandExpected> = {}): AdminTournamentDayCommandExpected {
    if (!snapshot?.day_run.version || !snapshot.state_fingerprint) {
      throw new Error("Reload authoritative day state before submitting an operation.");
    }
    return {
      day_run_version: snapshot.day_run.version,
      state_fingerprint: snapshot.state_fingerprint,
      queue_version: snapshot.queue_version,
      ...fields
    };
  }

  async function executePending(command: PendingDayCommand, retry: boolean): Promise<ActionCompletion> {
    if (!apiBase || !accessToken) {
      return actionUncertain(
        "Tournament-day operation needs verification",
        "Admin authentication or the API base URL is unavailable. The exact request remains retained.",
        command.request.client_idempotency_key,
        "Retry exact request",
        () => executePending(command, true)
      );
    }
    const generation = actionRequest.begin();
    setBusyKey(command.request.client_idempotency_key);
    setError(null);
    setMessage(null);
    try {
      const result = await executeAdminTournamentDayCommand({
        apiBase,
        clubId,
        tournamentId,
        dayId: command.dayId,
        accessToken,
        request: command.request
      });
      assertWorkspaceSnapshotScope(result.snapshot, clubId, tournamentId, command.dayId);
      const completionText = command.request.action === "score_and_release"
        ? "Score saved, court released, and the authoritative day queue refreshed."
        : command.request.action === "correct_completed_score"
          ? "Completed score corrected. The authoritative day result and all reviewed versions were refreshed."
        : command.request.action === "record_non_played_result"
          ? "Non-played outcome recorded, court and participant claims released, and the authoritative day queue refreshed."
        : command.request.action === "close_day"
          ? "Tournament day closed. Its court, queue, claim, and operation history remains preserved."
          : result.command.idempotent_replay
          ? "The exact retained request returned its durable result."
          : retry
            ? "The exact retained tournament-day request completed."
            : "The server committed the reviewed tournament-day operation.";
      const completion = actionSuccess("Tournament-day operation complete", completionText);
      if (!actionRequest.isCurrent(generation)) return completion;
      persistPending(null);
      setSnapshot(result.snapshot);
      setDayOptions(result.snapshot.day_scope.available_days || []);
      setLastRefresh(result.snapshot.generated_at || new Date().toISOString());
      if (command.request.action === "score_and_release") {
        setScoreEditor(null);
        setFocusedCourtId("");
        setFocusedGameId("");
        replaceWorkspaceUrl({ courtId: "", gameId: "", panel: "board" });
      } else if (command.request.action === "correct_completed_score") {
        setCorrectionEditor(null);
        setFocusedGameId("");
        replaceWorkspaceUrl({ gameId: "", panel: "corrections" });
      } else if (command.request.action === "record_non_played_result") {
        setOutcomeEditor(null);
        setFocusedCourtId("");
        setFocusedGameId("");
        replaceWorkspaceUrl({ courtId: "", gameId: "", panel: "board" });
      }
      setMessage(completionText);
      return completion;
    } catch (commandError) {
      const stale = commandError instanceof AdminTournamentDayOpsApiError && commandError.status === 409;
      const detail = commandError instanceof Error ? commandError.message : "The operation outcome is unknown.";
      if (stale) await loadWorkspace({ silent: true });
      if (actionRequest.isCurrent(generation)) {
        setError(`${detail} ${stale ? "Full-day truth was reloaded; writes remain frozen until this exact request is retried or reconciled." : "The exact request remains retained for recovery."}`);
      }
      return actionUncertain(
        "Tournament-day operation needs verification",
        `Request ${command.request.client_idempotency_key} remains retained. Do not create a replacement operation until its outcome is known.`,
        command.request.client_idempotency_key,
        "Retry exact request",
        () => executePending(command, true)
      );
    } finally {
      if (actionRequest.isCurrent(generation)) setBusyKey("");
    }
  }

  function submitCommand(
    action: AdminTournamentDayCommandAction,
    confirmationText: string,
    payload: AdminTournamentDayCommandPayload,
    versions: Partial<AdminTournamentDayCommandExpected> = {}
  ) {
    if (!selectedDayId || writesFrozen) {
      throw new Error("Resolve the retained or recovery-required operation before another day write.");
    }
    const pending: PendingDayCommand = {
      clubId,
      tournamentId,
      dayId: selectedDayId,
      createdAt: new Date().toISOString(),
      request: {
        action,
        client_idempotency_key: globalThis.crypto.randomUUID(),
        confirmation_text: confirmationText,
        expected: expectedVersions(versions),
        payload
      }
    };
    persistPending(pending);
    return executePending(pending, false);
  }

  async function reconcileOperation(operation: AdminTournamentDayOperation, confirmationText: string): Promise<ActionCompletion> {
    if (!apiBase || !accessToken || !selectedDayId) {
      throw new Error("Reload the selected tournament day before reconciliation.");
    }
    const generation = actionRequest.begin();
    setBusyKey(operation.operation_key);
    setError(null);
    try {
      const result = await reconcileAdminTournamentDayOperation({
        apiBase,
        clubId,
        tournamentId,
        dayId: selectedDayId,
        accessToken,
        operationKey: operation.operation_key,
        confirmationText
      });
      assertWorkspaceSnapshotScope(result.snapshot, clubId, tournamentId, selectedDayId);
      const completion = actionSuccess(
        "Tournament-day recovery reconciled",
        "The server returned refreshed full-day truth without fabricating a replacement command."
      );
      if (!actionRequest.isCurrent(generation)) return completion;
      if (pendingCommand?.request.client_idempotency_key === operation.client_idempotency_key) persistPending(null);
      setSnapshot(result.snapshot);
      setDayOptions(result.snapshot.day_scope.available_days || []);
      setLastRefresh(result.snapshot.generated_at || new Date().toISOString());
      setMessage("Recovery reconciled. Review the refreshed court board and queue before continuing.");
      return completion;
    } catch (reconcileError) {
      if (actionRequest.isCurrent(generation)) {
        setError(reconcileError instanceof Error ? reconcileError.message : "Recovery still needs verification.");
      }
      return actionUncertain(
        "Tournament-day recovery needs verification",
        `Operation ${operation.operation_key} remains recovery-required.`,
        operation.operation_key,
        "Reconcile again",
        () => reconcileOperation(operation, confirmationText)
      );
    } finally {
      if (actionRequest.isCurrent(generation)) setBusyKey("");
    }
  }

  function selectDay(dayId: string) {
    if (!dayId || dayId === selectedDayId) return;
    if (writesFrozen) {
      setError("Resolve the retained or recovery-required operation before changing tournament days.");
      return;
    }
    snapshotRequest.invalidate();
    actionRequest.invalidate();
    setSelectedDayId(dayId);
    setSnapshot(null);
    setDrawFilter("all");
    setFocusedCourtId("");
    setFocusedGameId("");
    setScoreEditor(null);
    setCorrectionEditor(null);
    setOutcomeEditor(null);
    setPlayoffAdvanceCounts({});
    setPendingCommand(null);
    setMessage(null);
    setError(null);
    replaceWorkspaceUrl({ dayId, drawId: "", courtId: "", gameId: "", panel: panelFocus });
  }

  function setPanel(panel: TournamentDayWorkspacePanelFocus) {
    setPanelFocus(panel);
    replaceWorkspaceUrl({ panel });
    document.getElementById(`day-workspace-${panel}`)?.focus();
  }

  function chooseQueueDraw(drawId: string) {
    setDrawFilter(drawId);
    replaceWorkspaceUrl({ drawId: drawId === "all" ? "" : drawId, panel: "queue" });
  }

  function openScore(game: AdminTournamentDayGame, courtId: string) {
    const draw = snapshot?.draws.find((row) => row.id === game.draw_id);
    const court = snapshot?.courts.find((row) => row.id === courtId);
    const assignment = court?.current_assignment;
    if (!draw || !court || assignment?.game_id !== game.id) {
      setError("Reload the court board before opening score entry; the reviewed draw, court, or assignment version is unavailable.");
      return;
    }
    setCorrectionEditor(null);
    setOutcomeEditor(null);
    setScoreEditor({
      gameId: game.id,
      courtId,
      scoreA: game.score_a == null ? "" : String(game.score_a),
      scoreB: game.score_b == null ? "" : String(game.score_b),
      reviewing: false,
      error: "",
      unusualScoreAcknowledged: false,
      expected: expectedVersions({ draw_version: draw.version, game_version: game.version, court_version: court.version }),
      reviewedGame: reviewedGameTruth(game, { courtId }),
      reviewedAssignmentVersion: assignment.version
    });
    setFocusedGameId(game.id);
    setFocusedCourtId(courtId);
    replaceWorkspaceUrl({ gameId: game.id, courtId, panel: "board" });
  }

  function openCorrection(game: AdminTournamentDayGame) {
    const draw = snapshot?.draws.find((row) => row.id === game.draw_id);
    if (!draw) {
      setError("Reload the completed results before opening a correction; the reviewed draw version is unavailable.");
      return;
    }
    setScoreEditor(null);
    setOutcomeEditor(null);
    setCorrectionEditor({
      gameId: game.id,
      scoreA: game.score_a == null ? "" : String(game.score_a),
      scoreB: game.score_b == null ? "" : String(game.score_b),
      reviewing: false,
      error: "",
      unusualScoreAcknowledged: false,
      expected: expectedVersions({ draw_version: draw.version, game_version: game.version }),
      reviewedGame: reviewedGameTruth(game)
    });
    setFocusedGameId(game.id);
    replaceWorkspaceUrl({ gameId: game.id, courtId: "", panel: "corrections" });
  }

  function openOutcome(game: AdminTournamentDayGame) {
    const draw = snapshot?.draws.find((row) => row.id === game.draw_id);
    const court = game.court_id ? snapshot?.courts.find((row) => row.id === game.court_id) : undefined;
    const queueEntryVersion = game.queue_entry_version
      || snapshot?.eligible_queue.find((entry) => entry.game_id === game.id)?.version
      || snapshot?.held_games.find((entry) => entry.game_id === game.id)?.version
      || snapshot?.blocked_games.find((entry) => entry.game_id === game.id)?.version;
    if (!draw) {
      setError("Reload the tournament day before opening a non-played outcome; the reviewed draw version is unavailable.");
      return;
    }
    setScoreEditor(null);
    setCorrectionEditor(null);
    setOutcomeEditor({
      gameId: game.id,
      resultType: "NO_SHOW",
      winnerTeamId: "",
      resultNote: "",
      reviewing: false,
      error: "",
      expected: expectedVersions({
        draw_version: draw.version,
        game_version: game.version,
        court_version: court?.version,
        queue_entry_version: queueEntryVersion
      }),
      reviewedGame: reviewedGameTruth(game, {
        courtId: court?.id || "",
        queueEntryVersion: String(queueEntryVersion || "")
      })
    });
    setFocusedGameId(game.id);
    setFocusedCourtId(game.court_id || "");
    replaceWorkspaceUrl({ gameId: game.id, courtId: game.court_id || "" });
    window.requestAnimationFrame(() => {
      const editor = document.getElementById("non-played-outcome-editor");
      editor?.scrollIntoView({ behavior: "smooth", block: "center" });
      editor?.focus({ preventScroll: true });
    });
  }

  function reviewScore() {
    if (!scoreEditor) return;
    const game = gamesById.get(scoreEditor.gameId);
    const validation = validateDayScoreDraft(
      scoreEditor.scoreA,
      scoreEditor.scoreB,
      game?.scoring,
      scoreEditor.unusualScoreAcknowledged
    );
    if (!validation.ok) {
      setScoreEditor({ ...scoreEditor, reviewing: false, error: validation.message });
      return;
    }
    setScoreEditor({ ...scoreEditor, reviewing: true, error: "" });
  }

  function reviewCorrection() {
    if (!correctionEditor) return;
    const game = gamesById.get(correctionEditor.gameId);
    const validation = validateDayCorrectionDraft(
      correctionEditor.scoreA,
      correctionEditor.scoreB,
      game?.score_a,
      game?.score_b,
      game?.scoring,
      correctionEditor.unusualScoreAcknowledged
    );
    if (!validation.ok) {
      setCorrectionEditor({ ...correctionEditor, reviewing: false, error: validation.message });
      return;
    }
    setCorrectionEditor({ ...correctionEditor, reviewing: true, error: "" });
  }

  function reviewOutcome() {
    if (!outcomeEditor) return;
    const validation = validateNonPlayedOutcomeDraft(
      outcomeEditor.resultType,
      outcomeEditor.winnerTeamId,
      outcomeEditor.resultNote
    );
    if (!validation.ok) {
      setOutcomeEditor({ ...outcomeEditor, reviewing: false, error: validation.message });
      return;
    }
    setOutcomeEditor({ ...outcomeEditor, reviewing: true, error: "" });
  }

  if (sessionLoading) return <p className={styles.notice}>Restoring the admin session…</p>;
  if (!accessToken) return <p role="alert" className={`${styles.notice} ${styles.error}`}>Sign in at /admin/login to operate the tournament day. {sessionMessage || ""}</p>;
  if (!apiBase) return <p role="alert" className={`${styles.notice} ${styles.error}`}>The Tournament Admin API base URL is not configured.</p>;

  const selectedDay = snapshot?.day_scope.selected_day || dayOptions.find((day) => day.id === selectedDayId) || null;
  const activateDayReadiness = readinessOrBlocked(snapshot?.readiness.activate_day, "Day activation readiness is unavailable.");
  const fillReadiness = readinessOrBlocked(snapshot?.readiness.auto_fill_courts, "Court fill readiness is unavailable.");
  const closeDayReadiness = readinessOrBlocked(snapshot?.readiness.close_day, "Day closure readiness is unavailable.");
  const correctionReadiness = readinessOrBlocked(snapshot?.readiness.correct_completed_score, "Completed-score correction readiness is unavailable.");
  const dayCorrectionOpen = ["ACTIVE", "PAUSED"].includes(dayRunState.toUpperCase());
  const completedGames = (snapshot?.games || []).filter((game) => String(game.state).toUpperCase() === "COMPLETED");
  const selectedScoreGame = scoreEditor ? gamesById.get(scoreEditor.gameId) : undefined;
  const selectedScoreCourt = scoreEditor ? snapshot?.courts.find((court) => court.id === scoreEditor.courtId) : undefined;
  const selectedScoreDraw = selectedScoreGame
    ? snapshot?.draws.find((draw) => draw.id === selectedScoreGame.draw_id)
    : undefined;
  const selectedScoreAssignmentCurrent = Boolean(
    selectedScoreGame
      && selectedScoreCourt?.current_assignment?.game_id === selectedScoreGame.id
  );
  const selectedScoreValidation = scoreEditor ? validateDayScoreDraft(
    scoreEditor.scoreA,
    scoreEditor.scoreB,
    selectedScoreGame?.scoring,
    scoreEditor.unusualScoreAcknowledged
  ) : null;
  const selectedCorrectionGame = correctionEditor ? gamesById.get(correctionEditor.gameId) : undefined;
  const selectedCorrectionDraw = selectedCorrectionGame
    ? snapshot?.draws.find((draw) => draw.id === selectedCorrectionGame.draw_id)
    : undefined;
  const selectedCorrectionReadiness = readinessOrBlocked(
    selectedCorrectionGame?.correction_readiness,
    "This completed game's correction readiness is unavailable."
  );
  const selectedCorrectionValidation = correctionEditor && selectedCorrectionGame
    ? validateDayCorrectionDraft(
        correctionEditor.scoreA,
        correctionEditor.scoreB,
        selectedCorrectionGame.score_a,
        selectedCorrectionGame.score_b,
        selectedCorrectionGame.scoring,
        correctionEditor.unusualScoreAcknowledged
      )
    : null;
  const selectedOutcomeGame = outcomeEditor ? gamesById.get(outcomeEditor.gameId) : undefined;
  const selectedOutcomeValidation = outcomeEditor
    ? validateNonPlayedOutcomeDraft(
        outcomeEditor.resultType,
        outcomeEditor.winnerTeamId,
        outcomeEditor.resultNote
      )
    : null;

  return (
    <div className={styles.root} aria-busy={loading || undefined}>
      {error ? <p role="alert" className={`${styles.notice} ${styles.error}`}>{error}</p> : null}
      {message ? <p role="status" aria-live="polite" className={styles.notice}>{message}</p> : null}

      <section className={styles.scopeBar} aria-label="Tournament day scope">
        <div>
          <p className={styles.eyebrow}>Authoritative day scope</p>
          <h2>{snapshot?.tournament.name || tournamentName || "Selected tournament"}</h2>
          <p className={styles.muted}>Court assignments, queue eligibility, and progression below are server-owned for one tournament day.</p>
        </div>
        <label className={styles.field}>
          Tournament day
          <select value={selectedDayId} onChange={(event) => selectDay(event.target.value)} disabled={loading || Boolean(busyKey) || writesFrozen}>
            <option value="">Choose a day…</option>
            {dayOptions.map((day) => <option value={day.id} key={day.id}>{day.event_date ? `${day.label} · ${day.event_date}` : day.label}</option>)}
          </select>
        </label>
        <div className={styles.scopeStatus}>
          <span className={dayActive ? styles.successBadge : styles.neutralBadge}>{statusLabel(snapshot?.day_run.state || "Not started")}</span>
          <button type="button" className={styles.secondaryButton} onClick={() => void loadWorkspace()} disabled={!selectedDayId || loading}>Reload full day</button>
          <small>Updated {timestamp(lastRefresh)}</small>
        </div>
      </section>

      <nav aria-label="Day workspace views" className={styles.viewNav}>
        {(["board", "queue", "draws", "corrections"] as const).map((panel) => (
          <button type="button" key={panel} aria-pressed={panelFocus === panel} onClick={() => setPanel(panel)}>
            {panel === "board" ? "Court board" : panel === "queue" ? "Eligible queue" : panel === "draws" ? "Draws & progression" : "Corrections & recovery"}
          </button>
        ))}
      </nav>

      {!selectedDayId ? (
        <section className={styles.emptyState}>
          <h2>Select the tournament day</h2>
          <p>Day selection is required before court or draw state can load. No first day is assumed when several are available.</p>
        </section>
      ) : null}
      {loading && !snapshot ? <p className={styles.notice}>Loading authoritative tournament-day state…</p> : null}

      {snapshot ? (
        <>
          <section className={styles.summaryGrid} aria-label="Tournament day summary">
            <article><span>Courts</span><strong>{snapshot.summary.courts}</strong></article>
            <article><span>Available</span><strong>{snapshot.summary.available_courts}</strong></article>
            <article><span>Active draws</span><strong>{snapshot.summary.active_draws}</strong></article>
            <article><span>Eligible queue</span><strong>{snapshot.summary.eligible_games}</strong></article>
            <article><span>Held</span><strong>{snapshot.summary.held_games}</strong></article>
            <article><span>Completed</span><strong>{snapshot.summary.completed_games}</strong></article>
          </section>

          {!dayStarted ? (
            <section className={styles.activationCard} aria-labelledby="activate-day-title">
              <div>
                <p className={styles.eyebrow}>Day activation</p>
                <h2 id="activate-day-title">Start {selectedDay?.label || "this tournament day"}</h2>
                <p>Starting the day initializes the server-owned court inventory only. Draws stay inactive until you activate them one at a time from Draws &amp; progression.</p>
              </div>
              <ReadinessBlockers readiness={activateDayReadiness} />
              <ConfirmAction
                triggerLabel="Activate tournament day"
                title={`Activate ${selectedDay?.label || "this tournament day"}?`}
                description={`Initialize ${snapshot.courts.length} authoritative court${snapshot.courts.length === 1 ? "" : "s"}. No draw is activated by this command.`}
                preview={<p>{snapshot.draws.length} scheduled draw(s) remain available for individual review and activation after the day starts.</p>}
                confirmLabel="Yes, activate day"
                confirmationText={activateDayReadiness.confirmation || dayActionConfirmation("activate_day")}
                disabled={!runtimeWritesEnabled || !activateDayReadiness.ready || writesFrozen}
                disabledReason={!runtimeWritesEnabled ? "Tournament-day writes are unavailable." : writesFrozen ? "Resolve recovery first." : activateDayReadiness.blockers.map(blockerText).join(" ")}
                busy={Boolean(busyKey)}
                onConfirm={(confirmationText) => submitCommand("activate_day", confirmationText, {})}
              />
            </section>
          ) : null}

          {writesFrozen ? (
            <section className={styles.recoveryCard} aria-labelledby="day-recovery-title">
              <p className={styles.eyebrow}>Recovery required</p>
              <h2 id="day-recovery-title">Day writes are frozen</h2>
              <p>Resolve or replay the exact operation, then review a refreshed full-day snapshot before continuing.</p>
              {pendingCommand ? (
                <div className={styles.recoveryRow}>
                  <div>
                    <strong>Retained {statusLabel(pendingCommand.request.action)}</strong>
                    <p>{timestamp(pendingCommand.createdAt)}</p>
                    <details><summary>Technical request evidence</summary><code>{pendingCommand.request.client_idempotency_key}</code></details>
                  </div>
                  <button type="button" className={styles.primaryButton} disabled={Boolean(busyKey)} onClick={() => void executePending(pendingCommand, true)}>Retry exact request</button>
                </div>
              ) : null}
              {activeRecovery ? (
                <div className={styles.recoveryRow}>
                  <div><strong>{activeRecovery.entity_label || statusLabel(activeRecovery.action)}</strong><p>{statusLabel(activeRecovery.status)} · {timestamp(activeRecovery.updated_at)}</p></div>
                  <ConfirmAction
                    triggerLabel="Reconcile operation"
                    title="Reconcile this tournament-day operation?"
                    description="The server will use durable operation evidence and return refreshed full-day truth without repeating a proven mutation."
                    confirmLabel="Yes, reconcile"
                    confirmationText="RECONCILE DAY OPERATIONS"
                    busy={Boolean(busyKey)}
                    onConfirm={(confirmationText) => reconcileOperation(activeRecovery, confirmationText)}
                  />
                </div>
              ) : null}
            </section>
          ) : null}

          <section id="day-workspace-board" tabIndex={-1} className={`${styles.workspaceSection} ${panelFocus === "board" ? styles.focusedSection : ""}`} aria-label="Court board">
            <div className={styles.sectionHeading}>
              <div><p className={styles.eyebrow}>Live allocation</p><h2>Court board</h2><p className={styles.muted}>Physical courts come only from the authoritative day allocation—not bracket slot numbers.</p></div>
              <div>
                <ReadinessBlockers readiness={fillReadiness} />
                <ConfirmAction
                  triggerLabel="Fill available courts"
                  title="Fill every currently available court?"
                  description="The server will assign from the current unified eligible queue in its authoritative order."
                  preview={<p>{snapshot.summary.available_courts} available court(s) · {snapshot.eligible_queue.length} eligible matchup(s)</p>}
                  confirmLabel="Yes, fill courts"
                  confirmationText={fillReadiness.confirmation || dayActionConfirmation("auto_fill_courts")}
                  disabled={!dayActive || !runtimeWritesEnabled || !fillReadiness.ready || writesFrozen}
                  disabledReason={!dayActive ? "Activate the day first." : writesFrozen ? "Resolve recovery first." : "The server has no safe court assignments ready."}
                  busy={Boolean(busyKey)}
                  onConfirm={(confirmationText) => submitCommand("auto_fill_courts", confirmationText, {})}
                />
              </div>
            </div>
            <div className={styles.courtBoard}>
              {snapshot.courts.map((court) => {
                const assignment = court.current_assignment;
                const game = assignment ? gamesById.get(assignment.game_id) : undefined;
                const occupied = Boolean(assignment);
                return (
                  <article key={court.id} className={`${styles.courtCard} ${occupied ? styles.occupiedCourt : styles.availableCourt}`} aria-labelledby={`court-${court.id}`}>
                    <div className={styles.cardHeading}><h3 id={`court-${court.id}`}>{court.label}</h3><span className={occupied ? styles.activeBadge : styles.successBadge}>{statusLabel(occupied ? assignment?.state : court.state)}</span></div>
                    {game ? (
                      <>
                        <p className={styles.stage}>{gameStageLabel(game)}</p>
                        <p className={styles.matchup}>{matchupLabel(game)}</p>
                        <p className={styles.muted}>Assigned {timestamp(assignment?.assigned_at)}</p>
                        <div className={styles.buttonRow}>
                          <button type="button" className={styles.primaryButton} onClick={() => openScore(game, court.id)} disabled={!dayActive || !runtimeWritesEnabled || writesFrozen || Boolean(busyKey)} aria-label={`Enter score for ${matchupLabel(game)} on ${court.label}`}>Enter score</button>
                          <button type="button" className={styles.secondaryButton} onClick={() => openOutcome(game)} disabled={!dayActive || !runtimeWritesEnabled || writesFrozen || Boolean(busyKey)}>Record no-play outcome</button>
                        </div>
                      </>
                    ) : assignment ? (
                      <p role="alert">Assignment details are unavailable. Reload before taking action.</p>
                    ) : (
                      <><p className={styles.availableText}>Available for the next server-eligible matchup.</p><p className={styles.muted}>Use Fill available courts; assignments are never derived in the browser.</p></>
                    )}
                  </article>
                );
              })}
            </div>
            {!snapshot.courts.length ? <p className={styles.emptyState}>No authoritative courts are available for this day. Return to Tournament Builder.</p> : null}

            {scoreEditor && selectedScoreGame && selectedScoreCourt && selectedScoreDraw && selectedScoreAssignmentCurrent ? (
              <article className={styles.scoreEditor} aria-labelledby="court-score-title">
                <div className={styles.sectionHeading}><div><p className={styles.eyebrow}>Inline score and release</p><h3 id="court-score-title">{selectedScoreCourt.label} · {gameStageLabel(selectedScoreGame)}</h3></div><button type="button" className={styles.secondaryButton} onClick={() => setScoreEditor(null)}>Close editor</button></div>
                <div className={styles.scoreGrid}>
                  <label>{sideLabel(selectedScoreGame.team_a)} score<input aria-invalid={Boolean(scoreEditor.error) || undefined} aria-describedby={scoreEditor.error ? "day-score-error" : undefined} value={scoreEditor.scoreA} onChange={(event) => setScoreEditor({ ...scoreEditor, scoreA: event.target.value, reviewing: false, error: "", unusualScoreAcknowledged: false })} type="number" min={0} step={1} inputMode="numeric" /></label>
                  <span aria-hidden="true">–</span>
                  <label>{sideLabel(selectedScoreGame.team_b)} score<input aria-invalid={Boolean(scoreEditor.error) || undefined} aria-describedby={scoreEditor.error ? "day-score-error" : undefined} value={scoreEditor.scoreB} onChange={(event) => setScoreEditor({ ...scoreEditor, scoreB: event.target.value, reviewing: false, error: "", unusualScoreAcknowledged: false })} type="number" min={0} step={1} inputMode="numeric" /></label>
                </div>
                {scoreEditor.error ? <p id="day-score-error" role="alert" className={styles.errorText}>{scoreEditor.error}</p> : null}
                {!scoreEditor.reviewing ? <button type="button" className={styles.primaryButton} onClick={reviewScore} disabled={writesFrozen || Boolean(busyKey)}>Review score</button> : selectedScoreValidation?.ok ? (
                  <div className={styles.scoreConfirmation} aria-label="Score and court release confirmation">
                    <div><strong>{sideLabel(selectedScoreGame.team_a)}</strong><span>{selectedScoreValidation.scoreA}</span></div>
                    <p>Winner: <strong>{selectedScoreValidation.scoreA > selectedScoreValidation.scoreB ? sideLabel(selectedScoreGame.team_a) : sideLabel(selectedScoreGame.team_b)}</strong></p>
                    <div><strong>{sideLabel(selectedScoreGame.team_b)}</strong><span>{selectedScoreValidation.scoreB}</span></div>
                    <p>This atomic command finalizes the score, releases {selectedScoreCourt.label}, and refills from the server-ordered eligible queue. The refreshed state may immediately show the next assignment.</p>
                    {selectedScoreValidation.unusual ? (
                      <label className={styles.forfeitBoundary}>
                        <input type="checkbox" checked={scoreEditor.unusualScoreAcknowledged} onChange={(event) => setScoreEditor({ ...scoreEditor, unusualScoreAcknowledged: event.target.checked })} />
                        <span><strong>Unusual score:</strong> {selectedScoreValidation.reasons.join(" ")} I reviewed the configured {selectedScoreValidation.scoringFormat} format and confirm this exact result.</span>
                      </label>
                    ) : null}
                    <div className={styles.buttonRow}><button type="button" className={styles.secondaryButton} onClick={() => setScoreEditor({ ...scoreEditor, reviewing: false })}>Edit score</button><ConfirmAction
                      triggerLabel="Confirm & release court"
                      title="Confirm this score and release the court?"
                      description={`${matchupLabel(selectedScoreGame)} · ${selectedScoreValidation.scoreA}–${selectedScoreValidation.scoreB} on ${selectedScoreCourt.label}.`}
                      confirmLabel="Confirm & release court"
                      confirmationText={dayActionConfirmation("score_and_release")}
                      disabled={!dayActive || !runtimeWritesEnabled || writesFrozen || selectedScoreValidation.acknowledgementRequired}
                      disabledReason={selectedScoreValidation.acknowledgementRequired ? "Acknowledge the unusual score after reviewing the configured format." : !dayActive ? "The day must be active to save a score." : writesFrozen ? "Resolve day recovery first." : "Tournament-day writes are unavailable."}
                      busy={Boolean(busyKey)}
                      onConfirm={(confirmationText) => submitCommand(
                        "score_and_release",
                        confirmationText,
                        { game_id: selectedScoreGame.id, score_a: selectedScoreValidation.scoreA, score_b: selectedScoreValidation.scoreB, unusual_score_acknowledgement: scoreEditor.unusualScoreAcknowledged },
                        scoreEditor.expected
                      )}
                    /></div>
                  </div>
                ) : null}
              </article>
            ) : scoreEditor ? (
              <article className={styles.scoreEditor} role="alert">
                <h3>Score entry needs a fresh court assignment</h3>
                <p>The selected matchup is no longer the authoritative assignment on this court. Review the refreshed board before scoring.</p>
                <button type="button" className={styles.secondaryButton} onClick={() => setScoreEditor(null)}>Close stale score entry</button>
              </article>
            ) : null}
          </section>

          <div className={styles.lowerGrid}>
            <section id="day-workspace-queue" tabIndex={-1} autoFocus={initialPanel === "queue"} className={`${styles.workspaceSection} ${panelFocus === "queue" ? styles.focusedSection : ""}`} aria-label="Eligible match queue">
              <div className={styles.sectionHeading}><div><p className={styles.eyebrow}>Server order</p><h2>Unified eligible queue</h2><p className={styles.muted}>One authoritative order across every active draw. Filtering never renumbers priority.</p></div><label className={styles.field}>Visible draw<select value={drawFilter} onChange={(event) => chooseQueueDraw(event.target.value)}><option value="all">All eligible draws</option>{snapshot.draws.map((draw) => <option key={draw.id} value={draw.id}>{draw.name}</option>)}</select></label></div>
              <ol className={styles.queueList}>
                {visibleQueue.map((entry) => {
                  const game = gamesById.get(entry.game_id);
                  return <li value={entry.position} key={entry.game_id}><div><strong>{game ? matchupLabel(game) : "Matchup unavailable"}</strong><p>{game ? gameStageLabel(game) : "Draw details unavailable"}</p><small>{entry.reason || `Eligible since ${timestamp(entry.eligible_since)}`}</small>{game ? <button type="button" className={styles.secondaryButton} onClick={() => openOutcome(game)} disabled={!dayActive || !runtimeWritesEnabled || writesFrozen || Boolean(busyKey)}>Record no-play outcome</button> : null}</div><span className={styles.positionBadge}>#{entry.position}</span></li>;
                })}
              </ol>
              {!visibleQueue.length ? <p className={styles.emptyState}>No server-eligible matchups match this view.</p> : null}
            </section>

            <section className={styles.workspaceSection} aria-labelledby="held-blocked-title">
              <p className={styles.eyebrow}>Exceptions</p><h2 id="held-blocked-title">Held and blocked matches</h2>
              <p className={styles.muted}>Holds and reassignment remain authoritative read-only in this slice. Reload or use recovery; do not invent a client transition.</p>
              {[...snapshot.held_games, ...snapshot.blocked_games].map((entry) => {
                const game = gamesById.get(entry.game_id);
                return <article className={styles.exceptionCard} key={`${entry.state}:${entry.game_id}`}><div className={styles.cardHeading}><strong>{game ? matchupLabel(game) : "Matchup unavailable"}</strong><span className={styles.warningBadge}>{statusLabel(entry.state)}</span></div><p>{entry.note || (entry.blockers.length ? blockerText(entry.blockers[0]) : entry.reason) || "Server review is required."}</p>{entry.blockers.length ? <ul className={styles.blockers}>{entry.blockers.map((blocker, index) => <li key={`${blockerText(blocker)}:${index}`}>{blockerText(blocker)}</li>)}</ul> : null}{game ? <button type="button" className={styles.secondaryButton} onClick={() => openOutcome(game)} disabled={!dayActive || !runtimeWritesEnabled || writesFrozen || Boolean(busyKey)}>Record forfeit, no-show, or retirement</button> : null}</article>;
              })}
              {!snapshot.held_games.length && !snapshot.blocked_games.length ? <p className={styles.emptyState}>No held or blocked matchups.</p> : null}
              <p className={styles.forfeitBoundary}><strong>Use the non-played outcome command.</strong> It records a visibly labeled forfeit, no-show, or retirement, advances progression atomically, and excludes the synthetic progression score from official rating publication.</p>
              <p><strong>Substitution:</strong> Day Live does not offer partial substitute assignment. Update the authoritative draw roster before day activation; after activation, keep the matchup blocked and use the documented recovery workflow.</p>
              <div className={styles.buttonRow}>
                <Link className={styles.textLink} href={tournamentRouteHref("/admin/tournaments/live-operations/corrections", { tournamentId, tournamentName: snapshot.tournament.name, dayId: selectedDayId, drawId: drawFilter === "all" ? "" : drawFilter })}>Open Corrections &amp; recovery</Link>
                <Link className={styles.textLink} href={tournamentRouteHref("/admin/tournaments/live-operations/check-in", { tournamentId, tournamentName: snapshot.tournament.name, dayId: selectedDayId, drawId: drawFilter === "all" ? "" : drawFilter })}>Open Preflight &amp; check-in</Link>
                <Link className={styles.textLink} href="/admin/match-log">Open Match Log</Link>
                <Link className={styles.textLink} href="/admin/replay-history">Open replay evidence</Link>
              </div>
              {outcomeEditor && selectedOutcomeGame ? (
                <article id="non-played-outcome-editor" tabIndex={-1} className={styles.scoreEditor} aria-labelledby="non-played-outcome-title">
                  <div className={styles.sectionHeading}><div><p className={styles.eyebrow}>Reviewed non-played result</p><h3 id="non-played-outcome-title">{matchupLabel(selectedOutcomeGame)}</h3></div><button type="button" className={styles.secondaryButton} onClick={() => setOutcomeEditor(null)}>Close outcome</button></div>
                  <div className={styles.scoreGrid}>
                    <label>Outcome<select value={outcomeEditor.resultType} onChange={(event) => setOutcomeEditor({ ...outcomeEditor, resultType: event.target.value as OutcomeEditor["resultType"], reviewing: false, error: "" })}><option value="FORFEIT">Forfeit</option><option value="NO_SHOW">No-show</option><option value="RETIREMENT">Retirement</option></select></label>
                    <label>Winning team<select value={outcomeEditor.winnerTeamId} onChange={(event) => setOutcomeEditor({ ...outcomeEditor, winnerTeamId: event.target.value, reviewing: false, error: "" })}><option value="">Choose winner…</option><option value={selectedOutcomeGame.team_a.team_id || ""}>{sideLabel(selectedOutcomeGame.team_a)}</option><option value={selectedOutcomeGame.team_b.team_id || ""}>{sideLabel(selectedOutcomeGame.team_b)}</option></select></label>
                  </div>
                  <label className={styles.field}>Operator note<textarea value={outcomeEditor.resultNote} maxLength={500} onChange={(event) => setOutcomeEditor({ ...outcomeEditor, resultNote: event.target.value, reviewing: false, error: "" })} placeholder="State who was absent, withdrew, or retired and what was verified." /></label>
                  {outcomeEditor.error ? <p role="alert" className={styles.errorText}>{outcomeEditor.error}</p> : null}
                  {!outcomeEditor.reviewing ? <button type="button" className={styles.primaryButton} onClick={reviewOutcome}>Review non-played result</button> : selectedOutcomeValidation?.ok ? (
                    <div className={styles.scoreConfirmation}>
                      <p><strong>{statusLabel(selectedOutcomeValidation.resultType)}</strong> · Winner: <strong>{selectedOutcomeValidation.winnerTeamId === selectedOutcomeGame.team_a.team_id ? sideLabel(selectedOutcomeGame.team_a) : sideLabel(selectedOutcomeGame.team_b)}</strong></p>
                      <p>{selectedOutcomeValidation.resultNote}</p>
                      <p>This is not a played score. The server will create only a synthetic progression result, release any court and participant claims, resolve bracket dependencies, refill courts, and keep this game out of rating publication.</p>
                      <div className={styles.buttonRow}><button type="button" className={styles.secondaryButton} onClick={() => setOutcomeEditor({ ...outcomeEditor, reviewing: false })}>Edit outcome</button><ConfirmAction
                        triggerLabel="Confirm non-played result"
                        title="Confirm this non-played tournament result?"
                        description={`${statusLabel(selectedOutcomeValidation.resultType)} · ${matchupLabel(selectedOutcomeGame)}`}
                        confirmLabel="Confirm outcome"
                        confirmationText={dayActionConfirmation("record_non_played_result")}
                        disabled={!dayActive || !runtimeWritesEnabled || writesFrozen}
                        disabledReason={!dayActive ? "The day must be active." : writesFrozen ? "Resolve day recovery first." : "Tournament-day writes are unavailable."}
                        busy={Boolean(busyKey)}
                        onConfirm={(confirmationText) => submitCommand(
                          "record_non_played_result",
                          confirmationText,
                          { game_id: selectedOutcomeGame.id, result_type: selectedOutcomeValidation.resultType, winner_team_id: selectedOutcomeValidation.winnerTeamId, result_note: selectedOutcomeValidation.resultNote },
                          outcomeEditor.expected
                        )}
                      /></div>
                    </div>
                  ) : null}
                </article>
              ) : null}
            </section>
          </div>

          <section id="day-workspace-corrections" tabIndex={-1} autoFocus={initialPanel === "corrections"} className={`${styles.workspaceSection} ${panelFocus === "corrections" ? styles.focusedSection : ""}`} aria-label="Completed score corrections and recovery">
            <div className={styles.sectionHeading}>
              <div>
                <p className={styles.eyebrow}>Guarded day recovery</p>
                <h2>Corrections &amp; recovery</h2>
                <p className={styles.muted}>Only the server can mark a released, completed round-robin result safe to correct. Playoff, podium, published, and unsettled-operation boundaries remain blocked.</p>
              </div>
              <ReadinessBlockers readiness={correctionReadiness} />
            </div>
            <div className={styles.correctionGrid}>
              {completedGames.map((game) => {
                const gameReadiness = readinessOrBlocked(game.correction_readiness, "Correction readiness is unavailable for this result.");
                const playoffResetRequired = readinessHasCode(gameReadiness, "PLAYOFF_RESET_REQUIRED");
                return (
                  <article key={game.id} className={styles.correctionCard}>
                    <div className={styles.cardHeading}>
                      <div><strong>{matchupLabel(game)}</strong><p>{gameStageLabel(game)}</p></div>
                      <span className={gameReadiness.ready ? styles.successBadge : styles.warningBadge}>{gameReadiness.ready ? "Correctable" : "Blocked"}</span>
                    </div>
                    {String(game.result_type || "PLAYED").toUpperCase() === "PLAYED" ? (
                      <p><strong>Played final:</strong> {game.score_a ?? "—"}–{game.score_b ?? "—"} · Winner: {game.winner_name || "Unavailable"}</p>
                    ) : (
                      <p className={styles.forfeitBoundary}><strong>{resultTypeLabel(game)} — not played.</strong> Winner: {game.winner_name || "Unavailable"}. {game.result_note || "Operator note unavailable."} The stored score is synthetic progression evidence and is not rating-eligible.</p>
                    )}
                    <ReadinessBlockers readiness={gameReadiness} />
                    {playoffResetRequired ? <p className={styles.forfeitBoundary}><strong>Playoff reset required.</strong> This round-robin result cannot change after its bracket exists.</p> : null}
                    <button
                      type="button"
                      className={styles.secondaryButton}
                      onClick={() => openCorrection(game)}
                      disabled={!dayCorrectionOpen || !runtimeWritesEnabled || !gameReadiness.ready || writesFrozen || Boolean(busyKey)}
                      aria-label={`Correct completed score for ${matchupLabel(game)}`}
                    >
                      Correct completed score
                    </button>
                  </article>
                );
              })}
            </div>
            {!completedGames.length ? <p className={styles.emptyState}>No day-owned completed results are available for correction review.</p> : null}

            <div className={styles.operationHistory}>
              <h3>Recent day operations and recovery evidence</h3>
              {snapshot.operations.map((operation) => (
                <article key={operation.operation_key} className={styles.recoveryRow}>
                  <div><strong>{operation.entity_label || statusLabel(operation.action)}</strong><p>{statusLabel(operation.status)} · {timestamp(operation.updated_at)}</p>{operation.error_text ? <p className={styles.errorText}>{operation.error_text}</p> : null}</div>
                  <details className={styles.technicalDetails}><summary>Technical operation evidence</summary><code>{operation.operation_key}</code></details>
                </article>
              ))}
              {!snapshot.operations.length ? <p className={styles.emptyState}>No day operation evidence has been recorded yet.</p> : null}
            </div>

            {correctionEditor && selectedCorrectionGame && selectedCorrectionDraw ? (
              <article className={styles.scoreEditor} aria-labelledby="day-correction-title">
                <div className={styles.sectionHeading}>
                  <div><p className={styles.eyebrow}>Before / after review</p><h3 id="day-correction-title">Correct {gameStageLabel(selectedCorrectionGame)}</h3></div>
                  <button type="button" className={styles.secondaryButton} onClick={() => setCorrectionEditor(null)}>Close correction</button>
                </div>
                <div className={styles.correctionComparison}>
                  <section>
                    <h4>Before correction</h4>
                    <p className={styles.matchup}>{matchupLabel(selectedCorrectionGame)}</p>
                    <p className={styles.largeScore}>{selectedCorrectionGame.score_a ?? "—"}–{selectedCorrectionGame.score_b ?? "—"}</p>
                    <p>Winner: <strong>{selectedCorrectionGame.winner_name || "Unavailable"}</strong></p>
                  </section>
                  <section>
                    <h4>After correction</h4>
                    <div className={styles.scoreGrid}>
                      <label htmlFor="day-correction-score-a">{sideLabel(selectedCorrectionGame.team_a)} score<input id="day-correction-score-a" aria-invalid={Boolean(correctionEditor.error) || undefined} aria-describedby={correctionEditor.error ? "day-correction-score-error" : undefined} value={correctionEditor.scoreA} onChange={(event) => setCorrectionEditor({ ...correctionEditor, scoreA: event.target.value, reviewing: false, error: "", unusualScoreAcknowledged: false })} type="number" min={0} step={1} inputMode="numeric" /></label>
                      <span aria-hidden="true">–</span>
                      <label htmlFor="day-correction-score-b">{sideLabel(selectedCorrectionGame.team_b)} score<input id="day-correction-score-b" aria-invalid={Boolean(correctionEditor.error) || undefined} aria-describedby={correctionEditor.error ? "day-correction-score-error" : undefined} value={correctionEditor.scoreB} onChange={(event) => setCorrectionEditor({ ...correctionEditor, scoreB: event.target.value, reviewing: false, error: "", unusualScoreAcknowledged: false })} type="number" min={0} step={1} inputMode="numeric" /></label>
                    </div>
                    {correctionEditor.error ? <p id="day-correction-score-error" role="alert" className={styles.errorText}>{correctionEditor.error}</p> : null}
                    {selectedCorrectionValidation?.ok ? <p>Proposed winner: <strong>{selectedCorrectionValidation.scoreA > selectedCorrectionValidation.scoreB ? sideLabel(selectedCorrectionGame.team_a) : sideLabel(selectedCorrectionGame.team_b)}</strong></p> : null}
                  </section>
                </div>
                <ReadinessBlockers readiness={selectedCorrectionReadiness} />
                {!correctionEditor.reviewing ? (
                  <button type="button" className={styles.primaryButton} onClick={reviewCorrection} disabled={!selectedCorrectionReadiness.ready || writesFrozen || Boolean(busyKey)}>Review correction</button>
                ) : selectedCorrectionValidation?.ok ? (
                  <div className={styles.scoreConfirmation} aria-label="Completed score correction confirmation">
                    <div><strong>{sideLabel(selectedCorrectionGame.team_a)}</strong><span>{selectedCorrectionValidation.scoreA}</span></div>
                    <p>After correction winner: <strong>{selectedCorrectionValidation.scoreA > selectedCorrectionValidation.scoreB ? sideLabel(selectedCorrectionGame.team_a) : sideLabel(selectedCorrectionGame.team_b)}</strong></p>
                    <div><strong>{sideLabel(selectedCorrectionGame.team_b)}</strong><span>{selectedCorrectionValidation.scoreB}</span></div>
                    <p><strong>Before:</strong> {selectedCorrectionGame.score_a ?? "—"}–{selectedCorrectionGame.score_b ?? "—"}, {selectedCorrectionGame.winner_name || "winner unavailable"}. <strong>After:</strong> {selectedCorrectionValidation.scoreA}–{selectedCorrectionValidation.scoreB}.</p>
                    {selectedCorrectionValidation.unusual ? (
                      <label className={styles.forfeitBoundary}>
                        <input type="checkbox" checked={correctionEditor.unusualScoreAcknowledged} onChange={(event) => setCorrectionEditor({ ...correctionEditor, unusualScoreAcknowledged: event.target.checked })} />
                        <span><strong>Unusual correction:</strong> {selectedCorrectionValidation.reasons.join(" ")} I reviewed the configured {selectedCorrectionValidation.scoringFormat} format and confirm this exact result.</span>
                      </label>
                    ) : null}
                    <div className={styles.buttonRow}>
                      <button type="button" className={styles.secondaryButton} onClick={() => setCorrectionEditor({ ...correctionEditor, reviewing: false })}>Edit score</button>
                      <ConfirmAction
                        triggerLabel="Confirm correction"
                        title="Confirm this exact completed-score correction?"
                        description={`${matchupLabel(selectedCorrectionGame)}. Change ${selectedCorrectionGame.score_a ?? "—"}–${selectedCorrectionGame.score_b ?? "—"} to ${selectedCorrectionValidation.scoreA}–${selectedCorrectionValidation.scoreB}.`}
                        confirmLabel="Confirm & save correction"
                        confirmationText={selectedCorrectionReadiness.confirmation || dayActionConfirmation("correct_completed_score")}
                        disabled={!dayCorrectionOpen || !runtimeWritesEnabled || !selectedCorrectionReadiness.ready || writesFrozen || selectedCorrectionValidation.acknowledgementRequired}
                        disabledReason={selectedCorrectionValidation.acknowledgementRequired ? "Acknowledge the unusual corrected score after reviewing the configured format." : !dayCorrectionOpen ? "The tournament day must be active or paused." : writesFrozen ? "Resolve day recovery first." : selectedCorrectionReadiness.blockers.map(blockerText).join(" ")}
                        busy={Boolean(busyKey)}
                        onConfirm={(confirmationText) => submitCommand(
                          "correct_completed_score",
                          confirmationText,
                          { game_id: selectedCorrectionGame.id, score_a: selectedCorrectionValidation.scoreA, score_b: selectedCorrectionValidation.scoreB, unusual_score_acknowledgement: correctionEditor.unusualScoreAcknowledged },
                          correctionEditor.expected
                        )}
                      />
                    </div>
                  </div>
                ) : null}
              </article>
            ) : correctionEditor ? (
              <article className={styles.scoreEditor} role="alert"><h3>Correction needs refreshed day truth</h3><p>The selected completed game or draw is no longer in this day snapshot. Reload before taking another action.</p><button type="button" className={styles.secondaryButton} onClick={() => setCorrectionEditor(null)}>Close stale correction</button></article>
            ) : null}
          </section>

          <section id="day-workspace-draws" tabIndex={-1} autoFocus={initialPanel === "draws"} className={`${styles.workspaceSection} ${panelFocus === "draws" ? styles.focusedSection : ""}`} aria-label="Draw activation and progression">
            <div className={styles.sectionHeading}><div><p className={styles.eyebrow}>Active draw control</p><h2>Draws & progression</h2><p className={styles.muted}>Activation and playoff generation remain fenced by this exact day version.</p></div></div>
            <div className={styles.drawGrid}>
              {snapshot.draws.map((draw) => {
                const action = drawAction(draw);
                const actionReadiness = drawActionReadiness(draw, action);
                const assignmentReadiness = readinessOrBlocked(draw.readiness.assignments, "Court assignment evidence is unavailable.");
                const playoffs = readinessOrBlocked(draw.readiness.generate_playoffs, "Playoff readiness is unavailable.");
                const podium = readinessOrBlocked(draw.readiness.podium, "Podium readiness is unavailable.");
                const selectedAdvanceCount = advanceCountSelection(
                  draw.readiness.generate_playoffs.allowed_advance_counts,
                  draw.readiness.generate_playoffs.default_advance_count,
                  playoffAdvanceCounts[draw.id]
                );
                const reviewedAdvanceCount = selectedAdvanceCount ? Number(selectedAdvanceCount) : null;
                const advanceCountHelpId = `advance-count-help-${draw.id}`;
                return (
                  <article key={draw.id} className={styles.drawCard}>
                    <div className={styles.cardHeading}><h3>{draw.name}</h3><span className={String(draw.activation_state).toUpperCase() === "ACTIVE" ? styles.activeBadge : styles.neutralBadge}>{statusLabel(draw.activation_state || draw.state)}</span></div>
                    <p>{draw.finalized_games} of {draw.total_games} finalized · {draw.queued_games} queued · {draw.active_games} on court · {draw.held_games} held</p>
                    <h4>Court assignment evidence</h4><ReadinessBlockers readiness={assignmentReadiness} />
                    <h4>Draw activation</h4><ReadinessBlockers readiness={actionReadiness} />
                    <ConfirmAction
                      triggerLabel={drawActionLabel(action)}
                      title={`${drawActionLabel(action)} ${draw.name}?`}
                      description={action === "pause_draw" ? "Pause stops new assignments while preserving queued and on-court state." : "The server will recheck day, draw, participant, and court versions before changing activation."}
                      confirmLabel={`Yes, ${drawActionLabel(action).toLowerCase()}`}
                      confirmationText={actionReadiness.confirmation || dayActionConfirmation(action)}
                      disabled={!dayActive || !runtimeWritesEnabled || !actionReadiness.ready || writesFrozen}
                      disabledReason={!dayActive ? "Activate the day first." : writesFrozen ? "Resolve recovery first." : actionReadiness.blockers.map(blockerText).join(" ")}
                      busy={Boolean(busyKey)}
                      onConfirm={(confirmationText) => submitCommand(action, confirmationText, { draw_id: draw.id }, { draw_version: draw.version })}
                    />
                    <hr />
                    <h4>Progression</h4><p>Current stage: <strong>{statusLabel(draw.stage || "Not started")}</strong></p><ReadinessBlockers readiness={playoffs} />
                    <label className={styles.field} htmlFor={`advance-count-${draw.id}`}>
                      Advancing teams
                      <select
                        id={`advance-count-${draw.id}`}
                        value={selectedAdvanceCount}
                        aria-describedby={advanceCountHelpId}
                        disabled={!dayActive || writesFrozen || Boolean(busyKey) || !draw.readiness.generate_playoffs.allowed_advance_counts.length}
                        onChange={(event) => setPlayoffAdvanceCounts((current) => ({ ...current, [draw.id]: event.target.value }))}
                      >
                        <option value="">Choose advancing teams…</option>
                        {draw.readiness.generate_playoffs.allowed_advance_counts.map((count) => <option key={count} value={count}>Top {count} teams</option>)}
                      </select>
                    </label>
                    <p id={advanceCountHelpId} className={styles.muted}>No playoff format is assumed. Choose one of the server-reviewed values before confirmation.</p>
                    <ConfirmAction
                      triggerLabel="Generate playoffs"
                      title={`Generate playoffs for ${draw.name}?`}
                      description={reviewedAdvanceCount == null ? "Choose an advancing-team count before reviewing the bracket command." : `Generate a top-${reviewedAdvanceCount} bracket only inside this day fence after the server verifies every source result.`}
                      confirmLabel="Yes, generate playoffs"
                      confirmationText={playoffs.confirmation || dayActionConfirmation("generate_playoffs")}
                      disabled={!dayActive || !runtimeWritesEnabled || !playoffs.ready || writesFrozen || reviewedAdvanceCount == null}
                      disabledReason={!dayActive ? "The day must be active to generate playoffs." : writesFrozen ? "Resolve recovery first." : reviewedAdvanceCount == null ? "Choose a server-reviewed advancing-team count." : playoffs.blockers.map(blockerText).join(" ")}
                      busy={Boolean(busyKey)}
                      onConfirm={(confirmationText) => submitCommand("generate_playoffs", confirmationText, { draw_id: draw.id, advance_count: reviewedAdvanceCount as number }, { draw_version: draw.version })}
                    />
                    <ReadinessBlockers readiness={podium} />
                    {podium.ready ? (
                      <Link className={styles.textLink} href={tournamentRouteHref("/admin/tournaments/live-operations/podium", { tournamentId, tournamentName: snapshot.tournament.name, dayId: selectedDayId, drawId: draw.id })}>Open Podium & awards</Link>
                    ) : (
                      <span className={styles.disabledLink}>Podium & awards blocked</span>
                    )}
                  </article>
                );
              })}
            </div>
          </section>

          {dayStarted ? (
            <section className={styles.activationCard} aria-labelledby="close-day-title">
              <div>
                <p className={styles.eyebrow}>End-of-day control</p>
                <h2 id="close-day-title">Close tournament day</h2>
                <p>Closure is final for live allocation. The server preserves this day&apos;s queue, court, participant-claim, and operation history.</p>
              </div>
              <ReadinessBlockers readiness={closeDayReadiness} />
              <ConfirmAction
                triggerLabel="Close tournament day"
                title={`Close ${selectedDay?.label || "this tournament day"}?`}
                description="The server will recheck that no matchup or participant claim remains active and that every queued game has a terminal disposition."
                preview={<p>{snapshot.summary.completed_games} completed matchup(s) · {snapshot.summary.held_games} held · {snapshot.summary.active_draws} activated draw(s)</p>}
                confirmLabel="Yes, close tournament day"
                confirmationText={closeDayReadiness.confirmation || dayActionConfirmation("close_day")}
                disabled={!runtimeWritesEnabled || !closeDayReadiness.ready || writesFrozen}
                disabledReason={!runtimeWritesEnabled ? "Tournament-day writes are unavailable." : writesFrozen ? "Resolve recovery first." : closeDayReadiness.blockers.map(blockerText).join(" ")}
                busy={Boolean(busyKey)}
                onConfirm={(confirmationText) => submitCommand("close_day", confirmationText, {})}
              />
            </section>
          ) : null}

          <details className={styles.technicalDetails}><summary>Technical day evidence</summary><p><code>{snapshot.state_fingerprint}</code></p><p>Day version: <code>{snapshot.day_run.version}</code> · queue version: <code>{snapshot.queue_version}</code></p></details>
        </>
      ) : null}
    </div>
  );
}
