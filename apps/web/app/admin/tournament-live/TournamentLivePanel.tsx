"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, actionUncertain } from "@/components/interaction";
import type { ActionCompletion } from "@/components/interaction";
import type {
  AdminTournamentDraw,
  AdminTournamentLifecycleDraw,
  AdminTournamentLifecycleReadiness,
  AdminTournamentLiveOperation,
  AdminTournamentLiveReadiness,
  AdminTournamentLiveSnapshotResponse,
  AdminTournamentLiveStatusResponse,
  AdminTournamentOpsTeam,
  AdminTournamentWriteResponse
} from "@/lib/adminTournamentApi";
import { tournamentRouteHref } from "@/lib/tournamentRouteContext";
import { drawOperationalStatus, isInactiveTournamentDraw } from "@/lib/tournamentDrawOperationalStatus.mjs";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";
import styles from "./TournamentLivePanel.module.css";

export type TournamentOperatorView =
  | "overview"
  | "draws"
  | "scoring"
  | "corrections"
  | "podium"
  | "results"
  | "publish-overview"
  | "publish"
  | "closeout"
  | "status";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentLiveStatusResponse;
  initialTournamentId?: string;
  initialTournamentName?: string | null;
  initialDrawId?: string | null;
  initialDayId?: string;
  view?: TournamentOperatorView;
};
type LiveCommand = "save_score" | "generate_round_robin" | "generate_playoffs" | "generate_podium" | "award_podium" | "publish_official_matches";
type ReviewedRowVersion = { id: string; updated_at: string };
type LiveCommandBody = {
  command: LiveCommand;
  expected_state_fingerprint: string;
  idempotency_key: string;
  confirmation_text: string;
  expected_draw_updated_at: string;
  expected_game_updated_at?: string;
  expected_team_versions?: ReviewedRowVersion[];
  expected_source_game_versions?: ReviewedRowVersion[];
  game_id?: string;
  score_a?: number;
  score_b?: number;
  advance_count?: number;
  playoff_winner_bonus_elo?: number;
};
type PendingCommand = {
  clubId: string;
  tournamentId: string;
  drawId: string;
  createdAt: string;
  body: LiveCommandBody;
};
type Notice = { tone: "success" | "error" | "info"; text: string };

const CONFIRMATIONS: Record<LiveCommand, string> = {
  save_score: "SAVE SCORE",
  generate_round_robin: "GENERATE GAMES",
  generate_playoffs: "GENERATE PLAYOFFS",
  generate_podium: "GENERATE PODIUM",
  award_podium: "AWARD PODIUM",
  publish_official_matches: "PUBLISH MATCHES"
};
const ACTIVE_OPERATION_STATUSES: ReadonlySet<string> = new Set(["intent", "mutated", "recovery_required"]);

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function pendingStorageKey(clubId: string): string {
  return `jupr_tournament_live_pending_v2:${clubId}`;
}

function shortValue(value: unknown): string {
  if (value == null || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value).slice(0, 160);
  return String(value);
}

function numericValue(value: unknown): number | null {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function isScored(game: Record<string, unknown>): boolean {
  const scoreA = numericValue(game.score_a);
  const scoreB = numericValue(game.score_b);
  return scoreA != null && scoreB != null && scoreA !== scoreB && Boolean(game.winner_team_id);
}

function gameSortKey(game: Record<string, unknown>): string {
  const stage = String(game.stage || "");
  const rrRound = String(game.rr_round_number || "").padStart(3, "0");
  const rrSlot = String(game.rr_slot_number || "").padStart(3, "0");
  const playoff = String(game.playoff_game_code || game.playoff_round || "");
  return [stage === "ROUND_ROBIN" ? "0" : "1", rrRound, rrSlot, playoff, String(game.id || "")].join("|");
}

function gameLabel(game: Record<string, unknown>): string {
  const stage = String(game.stage || "").toUpperCase();
  const rr = game.rr_round_number ? `Round ${game.rr_round_number}` : "";
  const slot = game.rr_slot_number ? `Match slot ${game.rr_slot_number}` : "";
  const playoff = [game.playoff_round, game.playoff_game_code].map(shortValue).filter((item) => item !== "—").join(" ");
  return [stage === "PLAYOFF" ? "Playoff" : "", rr, slot, playoff].filter(Boolean).join(" · ") || "Scheduled game";
}

function drawLabel(draw: AdminTournamentDraw, lifecycleDraw?: AdminTournamentLifecycleDraw): string {
  // Operational progress comes from authoritative match and publish evidence; the setup status may remain DRAFT.
  return `${draw.name || "Unnamed draw"} · ${drawOperationalStatus(draw, lifecycleDraw)}`;
}

function playerLabel(snapshot: AdminTournamentLiveSnapshotResponse | null, playerId?: number | null): string {
  if (playerId == null) return "Player unavailable";
  const match = (snapshot?.players || []).find((player) => Number(player.id) === Number(playerId));
  return match?.name || "Player unavailable";
}

function teamLabel(team: AdminTournamentOpsTeam | undefined, snapshot: AdminTournamentLiveSnapshotResponse | null): string {
  if (!team) return "Unassigned";
  const p1 = playerLabel(snapshot, team.player1_id ?? null);
  const p2 = team.player2_id == null ? null : playerLabel(snapshot, team.player2_id);
  return p2 ? `${p1} / ${p2}` : p1;
}

function matchupLabel(
  game: Record<string, unknown>,
  teamsById: Map<string, AdminTournamentOpsTeam>,
  snapshot: AdminTournamentLiveSnapshotResponse | null
): string {
  const teamA = teamLabel(teamsById.get(String(game.team_a_id || "")), snapshot);
  const teamB = teamLabel(teamsById.get(String(game.team_b_id || "")), snapshot);
  const status = isScored(game) ? "Final" : "Open";
  return `${gameLabel(game)} — ${teamA} vs ${teamB} — ${status}`;
}

function compactKey(value: string | null | undefined): string {
  const text = String(value || "");
  return text ? `${text.slice(0, 12)}…` : "—";
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString();
}

function readPendingCommand(clubId: string, tournamentId: string, drawId: string): PendingCommand | null {
  try {
    const raw = window.localStorage.getItem(pendingStorageKey(clubId));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as PendingCommand;
    if (parsed.clubId !== clubId || parsed.tournamentId !== tournamentId || parsed.drawId !== drawId || !parsed.body?.idempotency_key) return null;
    return parsed;
  } catch {
    return null;
  }
}

function StatusChip({ scored }: { scored: boolean }) {
  return <span className={scored ? styles.successChip : styles.openChip}>{scored ? "Final" : "Open"}</span>;
}

function OperationChip({ status }: { status: string }) {
  const recovery = ACTIVE_OPERATION_STATUSES.has(status);
  return <span className={recovery ? styles.recoveryChip : status === "completed" ? styles.successChip : styles.neutralChip}>{status.replace(/_/g, " ")}</span>;
}

function CommandBlockers({ readiness }: { readiness: AdminTournamentLiveReadiness }) {
  const blockers = [...new Set(readiness.blockers || [])];
  if (!blockers.length) return <p className={styles.readyText}>Python preflight: ready.</p>;
  return (
    <ul className={styles.blockers}>
      {blockers.map((blocker) => <li key={`${readiness.confirmation}:${blocker}`}>{blocker}</li>)}
    </ul>
  );
}

function lifecycleBlockerMessages(readiness?: AdminTournamentLifecycleReadiness | null): string[] {
  return (readiness?.blockers || []).map((blocker) => blocker.message);
}

function LifecycleBlockers({ readiness }: { readiness?: AdminTournamentLifecycleReadiness | null }) {
  const blockers = lifecycleBlockerMessages(readiness);
  if (readiness?.ready) return <p className={styles.readyText}>Tournament readiness: ready.</p>;
  if (!blockers.length) return <p className={styles.muted}>Tournament readiness is unavailable. Reload before acting.</p>;
  return <ul className={styles.blockers}>{blockers.map((message) => <li key={message}>{message}</li>)}</ul>;
}

export default function TournamentLivePanel({
  apiBase,
  clubId,
  status,
  initialTournamentId = "",
  initialTournamentName = null,
  initialDrawId = "",
  initialDayId = "",
  view = "scoring"
}: Props) {
  const { accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const router = useRouter();
  const pathname = usePathname();
  const lockedTournamentId = initialTournamentId;
  const selectedTournamentId = lockedTournamentId;
  const [draws, setDraws] = useState<AdminTournamentDraw[]>([]);
  const [drawLifecycle, setDrawLifecycle] = useState<AdminTournamentLifecycleDraw[]>([]);
  const [selectedDrawId, setSelectedDrawId] = useState(initialDrawId || "");
  const [snapshot, setSnapshot] = useState<AdminTournamentLiveSnapshotResponse | null>(null);
  const [scoreGameId, setScoreGameId] = useState("");
  const [scoreA, setScoreA] = useState("");
  const [scoreB, setScoreB] = useState("");
  const [roundFilter, setRoundFilter] = useState("all");
  const [gameStatusFilter, setGameStatusFilter] = useState("all");
  const [scoreConfirmation, setScoreConfirmation] = useState(false);
  const [playoffAdvanceCount, setPlayoffAdvanceCount] = useState("4");
  const [publishBonusElo, setPublishBonusElo] = useState("0");
  const [pendingCommand, setPendingCommand] = useState<PendingCommand | null>(null);
  const [lastResult, setLastResult] = useState<AdminTournamentWriteResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);
  const drawsRequest = useLatestRequestGuard(`${accessToken}\u0000${lockedTournamentId}\u0000${initialDayId}`, clearProtectedLiveState);
  const boardRequest = useLatestRequestGuard(accessToken);
  const actionScope = `${accessToken}\u0000${selectedTournamentId}\u0000${selectedDrawId}`;
  const actionRequest = useLatestRequestGuard(actionScope);

  const selectedTournament = snapshot?.tournament || null;
  const drawLifecycleById = useMemo(
    () => new Map(drawLifecycle.map((draw) => [draw.draw_id, draw])),
    [drawLifecycle]
  );
  const { teamsById, sortedGames } = useMemo(() => {
    const games = [...(snapshot?.games || [])].sort((left, right) => gameSortKey(left).localeCompare(gameSortKey(right)));
    return {
      teamsById: new Map((snapshot?.teams || []).map((team) => [String(team.id || ""), team])),
      sortedGames: games
    };
  }, [snapshot]);
  const selectedGame = sortedGames.find((game) => String(game.id || "") === scoreGameId) || sortedGames.find((game) => !isScored(game)) || sortedGames[0] || null;
  const roundOptions = [...new Set(sortedGames.map((game) => String(game.rr_round_number || game.playoff_round || "Other")))];
  const filteredGames = sortedGames.filter((game) => {
    const round = String(game.rr_round_number || game.playoff_round || "Other");
    return (roundFilter === "all" || round === roundFilter)
      && (gameStatusFilter === "all" || (gameStatusFilter === "final") === isScored(game));
  });
  const lifecycle = snapshot?.lifecycle;
  const officialReadiness = lifecycle?.domain_readiness?.official_publish;
  const archiveReadiness = lifecycle?.domain_readiness?.archive;
  const activeOperations = (snapshot?.operations || []).filter((operation) => ACTIVE_OPERATION_STATUSES.has(operation.status));
  const pendingOperation = pendingCommand
    ? (snapshot?.operations || []).find((operation) => operation.client_idempotency_key === pendingCommand.body.idempotency_key) || null
    : null;

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Tournament Live.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function clearProtectedLiveState() {
    boardRequest.invalidate(); actionRequest.invalidate();
    setBusy(false); setNotice(null);
    setDraws([]); setDrawLifecycle([]); setSelectedDrawId(initialDrawId || ""); setSnapshot(null);
    setScoreGameId(""); setScoreA(""); setScoreB(""); setScoreConfirmation(false); setPendingCommand(null); setLastResult(null);
  }

  function seedScoreEditor(nextSnapshot: AdminTournamentLiveSnapshotResponse | null) {
    const games = [...(nextSnapshot?.games || [])].sort((left, right) => gameSortKey(left).localeCompare(gameSortKey(right)));
    const target = games.find((game) => !isScored(game)) || games[0] || null;
    setScoreGameId(target ? String(target.id || "") : "");
    setScoreA(target?.score_a == null ? "" : String(target.score_a));
    setScoreB(target?.score_b == null ? "" : String(target.score_b));
    setScoreConfirmation(false);
  }

  async function fetchBoard(tournamentId: string, drawId: string): Promise<AdminTournamentLiveSnapshotResponse> {
    if (tournamentId !== lockedTournamentId) {
      throw new Error("This retained request belongs to a different tournament workspace.");
    }
    const payload = await requestJson<AdminTournamentLiveSnapshotResponse>(
      `/admin/clubs/${encodeURIComponent(clubId)}/tournament-live/tournaments/${encodeURIComponent(tournamentId)}/snapshot?draw_id=${encodeURIComponent(drawId)}`
    );
    assertSnapshotIdentity(payload, tournamentId, drawId);
    return payload;
  }

  function assertSnapshotIdentity(
    payload: AdminTournamentLiveSnapshotResponse,
    tournamentId: string,
    drawId?: string
  ) {
    if (String(payload.tournament?.id || "") !== tournamentId) {
      throw new Error("The response belongs to a different tournament workspace. Refresh from Tournament Manager.");
    }
    if (drawId && (payload.scope !== "draw" || String(payload.draw_id || "") !== drawId)) {
      throw new Error("The returned draw-scoped snapshot does not match the working draw. Choose the draw again.");
    }
  }

  function hydrateBoard(payload: AdminTournamentLiveSnapshotResponse, tournamentId: string, drawId: string) {
    if (tournamentId !== lockedTournamentId) throw new Error("This retained request belongs to a different tournament workspace.");
    assertSnapshotIdentity(payload, tournamentId, drawId);
    setSelectedDrawId(drawId);
    setSnapshot(payload);
    setDrawLifecycle(payload.lifecycle?.draws || []);
    seedScoreEditor(payload);
    setPendingCommand(readPendingCommand(clubId, tournamentId, drawId));
  }

  function replaceDrawInUrl(drawId: string) {
    const nextContext = {
      tournamentId: lockedTournamentId,
      tournamentName: selectedTournament?.name || initialTournamentName || "",
      drawId,
      dayId: initialDayId
    };
    router.replace(tournamentRouteHref(pathname, nextContext), { scroll: false });
  }

  async function loadDraws(preferredDrawId = selectedDrawId, reloadBoard = true) {
    const generation = drawsRequest.begin();
    boardRequest.invalidate();
    actionRequest.invalidate();
    if (!lockedTournamentId) {
      setNotice({ tone: "error", text: "Return to Tournament Manager and select a tournament first." });
      return;
    }
    let nextSelectedDrawId = "";
    setBusy(true);
    setNotice(null);
    try {
      const payload = await requestJson<AdminTournamentLiveSnapshotResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournament-live/tournaments/${encodeURIComponent(lockedTournamentId)}/snapshot`
      );
      if (!drawsRequest.isCurrent(generation)) return;
      assertSnapshotIdentity(payload, lockedTournamentId);
      const nextDraws = (payload.draws || []).filter((draw) => (
        !initialDayId || String(draw.registration_day_id || "") === initialDayId
      ));
      setDraws(nextDraws);
      setDrawLifecycle(payload.lifecycle?.draws || []);
      const operableDrawIds = new Set((payload.lifecycle?.draws || []).map((draw) => draw.draw_id));
      const operableDraws = nextDraws.filter((draw) => (
        operableDrawIds.has(draw.id)
        && !isInactiveTournamentDraw(draw)
      ));
      const preferredDrawStillAvailable = Boolean(
        preferredDrawId && operableDraws.some((row) => row.id === preferredDrawId)
      );
      nextSelectedDrawId = preferredDrawStillAvailable
        ? preferredDrawId
        : !preferredDrawId && operableDraws.length === 1 ? operableDraws[0].id : "";
      setSelectedDrawId(nextSelectedDrawId);
      setNotice({ tone: operableDraws.length ? "success" : "info", text: operableDraws.length ? `Loaded ${operableDraws.length} prepared draw(s).` : "This tournament has no operable draws. Build or reactivate one in Tournament Ops." });
      if (nextSelectedDrawId !== selectedDrawId) replaceDrawInUrl(nextSelectedDrawId);
      if (!nextSelectedDrawId) {
        setSnapshot(null);
        setPendingCommand(null);
      }
    } catch (error) {
      if (drawsRequest.isCurrent(generation)) setNotice({ tone: "error", text: error instanceof Error ? error.message : "Unable to load tournament draws." });
    } finally {
      if (drawsRequest.isCurrent(generation)) setBusy(false);
    }
    if (reloadBoard && nextSelectedDrawId && drawsRequest.isCurrent(generation)) await loadLiveBoard(nextSelectedDrawId);
  }

  async function loadLiveBoard(drawId = selectedDrawId) {
    const generation = boardRequest.begin();
    if (!lockedTournamentId || !drawId) {
      setNotice({ tone: "error", text: "Choose a working draw first." });
      return;
    }
    setBusy(true);
    setNotice(null);
    try {
      const payload = await requestJson<AdminTournamentLiveSnapshotResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournament-live/tournaments/${encodeURIComponent(lockedTournamentId)}/snapshot?draw_id=${encodeURIComponent(drawId)}`
      );
      if (!boardRequest.isCurrent(generation)) return;
      assertSnapshotIdentity(payload, lockedTournamentId, drawId);
      setSelectedDrawId(drawId);
      setSnapshot(payload);
      setDrawLifecycle(payload.lifecycle?.draws || []);
      seedScoreEditor(payload);
      setPendingCommand(readPendingCommand(clubId, lockedTournamentId, drawId));
      setNotice({ tone: "success", text: "Authoritative draw state loaded from FastAPI." });
    } catch (error) {
      if (boardRequest.isCurrent(generation)) setNotice({ tone: "error", text: error instanceof Error ? error.message : "Unable to load Tournament Live board." });
    } finally {
      if (boardRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function selectDraw(drawId: string) {
    drawsRequest.invalidate();
    boardRequest.invalidate();
    actionRequest.invalidate();
    setSelectedDrawId(drawId);
    setSnapshot(null);
    setPendingCommand(null);
    replaceDrawInUrl(drawId);
    if (lockedTournamentId && drawId) void loadLiveBoard(drawId);
  }

  function selectScoreGame(game: Record<string, unknown>) {
    setScoreGameId(String(game.id || ""));
    setScoreA(game.score_a == null ? "" : String(game.score_a));
    setScoreB(game.score_b == null ? "" : String(game.score_b));
    setScoreConfirmation(false);
  }

  function commandReadiness(command: LiveCommand): AdminTournamentLiveReadiness {
    return snapshot?.readiness?.[command] || {
      ready: false,
      confirmation: CONFIRMATIONS[command],
      blockers: ["Reload the authoritative draw state before submitting a command."]
    };
  }

  function commandEndpoint(tournamentId: string, drawId: string): string {
    return `/admin/clubs/${encodeURIComponent(clubId)}/tournament-live/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/commands`;
  }

  function persistPending(command: PendingCommand | null) {
    if (command) window.localStorage.setItem(pendingStorageKey(clubId), JSON.stringify(command));
    else window.localStorage.removeItem(pendingStorageKey(clubId));
    setPendingCommand(command);
  }

  async function executePending(command: PendingCommand, replay: boolean): Promise<ActionCompletion> {
    const generation = actionRequest.begin();
    setBusy(true);
    setNotice(null);
    try {
      const result = await requestJson<AdminTournamentWriteResponse>(commandEndpoint(command.tournamentId, command.drawId), {
        method: "POST",
        body: JSON.stringify(command.body)
      });
      const completedGame = snapshot?.games.find((row) => String(row.id || "") === String(command.body.game_id || ""));
      const scoreCompletion = command.body.command === "save_score" && completedGame
        ? `Score saved for ${matchupLabel(completedGame, teamsById, snapshot)}: ${command.body.score_a}–${command.body.score_b}.`
        : "";
      const completionText = scoreCompletion || (result.reconciled
        ? `Operation ${compactKey(result.operation_key)} reconciled without repeating the domain write.`
        : result.idempotent_replay
          ? `Operation ${compactKey(result.operation_key)} returned its durable stored result.`
          : replay
            ? `Exact request ${compactKey(result.operation_key)} completed.`
            : `Tournament Live command completed as operation ${compactKey(result.operation_key)}.`);
      const completion = actionSuccess("Tournament Live command complete", completionText);
      if (!actionRequest.isCurrent(generation)) return completion;
      persistPending(null);
      setLastResult(result);
      try {
        const board = await fetchBoard(command.tournamentId, command.drawId);
        if (actionRequest.isCurrent(generation)) hydrateBoard(board, command.tournamentId, command.drawId);
      } catch {
        // The authoritative command response proves completion even when the follow-up board refresh fails.
      }
      if (actionRequest.isCurrent(generation)) setNotice({ tone: "success", text: completionText });
      return completion;
    } catch (error) {
      if (!actionRequest.isCurrent(generation)) {
        return actionUncertain("Tournament Live command needs verification", `Operation ${command.body.idempotency_key} needs reconciliation before retrying.`, command.body.idempotency_key, "Retry exact request", () => executePending(command, true));
      }
      try {
        const board = await fetchBoard(command.tournamentId, command.drawId);
        if (!actionRequest.isCurrent(generation)) return actionUncertain("Tournament Live command needs verification", `Operation ${command.body.idempotency_key} needs reconciliation before retrying.`, command.body.idempotency_key, "Retry exact request", () => executePending(command, true));
        hydrateBoard(board, command.tournamentId, command.drawId);
      } catch {
        // Preserve the exact local request even when the recovery read is unavailable.
      }
      const uncertain = actionUncertain(
        "Tournament Live command needs verification",
        `The exact request is retained as operation ${command.body.idempotency_key}. Retry that exact request to recover its durable result.`,
        command.body.idempotency_key,
        "Retry exact request",
        () => executePending(command, true)
      );
      if (!actionRequest.isCurrent(generation)) return uncertain;
      setNotice({
        tone: "error",
        text: `${error instanceof Error ? error.message : "Tournament Live command failed."} The exact request is retained below; do not create a replacement command until its operation state is known.`
      });
      return uncertain;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function submitCommand(
    command: LiveCommand,
    confirmationText: string,
    fields: Omit<
      LiveCommandBody,
      | "command"
      | "expected_state_fingerprint"
      | "idempotency_key"
      | "confirmation_text"
      | "expected_draw_updated_at"
      | "expected_game_updated_at"
      | "expected_team_versions"
      | "expected_source_game_versions"
    > = {}
  ) {
    if (!snapshot?.state_fingerprint || !selectedTournamentId || !selectedDrawId) {
      setNotice({ tone: "error", text: "Reload the selected draw before submitting a command." });
      throw new Error("Reload the selected draw before submitting a command.");
    }
    const readiness = commandReadiness(command);
    if (!readiness.ready) {
      setNotice({ tone: "error", text: `Python preflight blocks this command: ${readiness.blockers.join(" ")}` });
      throw new Error(`Python preflight blocks this command: ${readiness.blockers.join(" ")}`);
    }
    const reviewedDraw = snapshot.draws.find((draw) => String(draw.id || "") === selectedDrawId);
    if (!reviewedDraw?.updated_at) {
      setNotice({ tone: "error", text: "The draw has no reviewed version. Reload before submitting a command." });
      throw new Error("The draw has no reviewed version. Reload before submitting a command.");
    }
    const versionRows = (rows: Array<{ id?: unknown; updated_at?: unknown }>): ReviewedRowVersion[] | null => {
      const versions = rows.map((row) => ({ id: String(row.id || ""), updated_at: String(row.updated_at || "") }));
      return versions.every((row) => row.id && row.updated_at)
        ? versions.sort((left, right) => left.id.localeCompare(right.id))
        : null;
    };
    const body: LiveCommandBody = {
      command,
      expected_state_fingerprint: snapshot.state_fingerprint,
      idempotency_key: globalThis.crypto.randomUUID(),
      confirmation_text: confirmationText,
      expected_draw_updated_at: reviewedDraw.updated_at,
      ...fields
    };
    if (command === "save_score") {
      const game = snapshot.games.find((row) => String(row.id || "") === String(fields.game_id || ""));
      if (!game?.updated_at) {
        setNotice({ tone: "error", text: "The selected game has no reviewed version. Reload before saving." });
        throw new Error("The selected game has no reviewed version. Reload before saving.");
      }
      body.expected_game_updated_at = String(game.updated_at);
    } else {
      const teamVersions = versionRows(snapshot.teams);
      if (!teamVersions?.length) {
        setNotice({ tone: "error", text: "The reviewed team version set is incomplete. Reload before submitting." });
        throw new Error("The reviewed team version set is incomplete. Reload before submitting.");
      }
      body.expected_team_versions = teamVersions;
      if (command !== "generate_round_robin") {
        const gameVersions = versionRows(snapshot.games);
        if (!gameVersions?.length) {
          setNotice({ tone: "error", text: "The reviewed game version set is incomplete. Reload before submitting." });
          throw new Error("The reviewed game version set is incomplete. Reload before submitting.");
        }
        body.expected_source_game_versions = gameVersions;
      }
    }
    const pending: PendingCommand = {
      clubId,
      tournamentId: selectedTournamentId,
      drawId: selectedDrawId,
      createdAt: new Date().toISOString(),
      body
    };
    persistPending(pending);
    return executePending(pending, false);
  }

  function saveScore(confirmationText: string) {
    if (!selectedGame) {
      setNotice({ tone: "error", text: "Select a game before saving a score." });
      throw new Error("Select a game before saving a score.");
    }
    const a = Number(scoreA);
    const b = Number(scoreB);
    if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < 0 || a === b) {
      setNotice({ tone: "error", text: "Enter two non-tied, non-negative whole-number scores." });
      throw new Error("Enter two non-tied, non-negative whole-number scores.");
    }
    return submitCommand("save_score", confirmationText, { game_id: String(selectedGame.id || ""), score_a: a, score_b: b });
  }

  function validateScoreDraft() {
    if (!selectedGame) {
      setNotice({ tone: "error", text: "Select a matchup before reviewing a score." });
      return;
    }
    const a = Number(scoreA);
    const b = Number(scoreB);
    if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < 0 || a === b) {
      setScoreConfirmation(false);
      setNotice({ tone: "error", text: "Enter two non-tied, non-negative whole-number scores. A tied or invalid score cannot be reviewed or saved." });
      return;
    }
    setNotice(null);
    setScoreConfirmation(true);
  }

  async function reviewPodium(confirmationText: string): Promise<ActionCompletion> {
    const reviewedSnapshot = snapshot;
    const opsStateFingerprint = String(reviewedSnapshot?.ops_state_fingerprint || "");
    if (!reviewedSnapshot || !opsStateFingerprint || !selectedTournamentId || !selectedDrawId) {
      throw new Error("Podium review needs the current Tournament Ops fingerprint. Reload after the server exposes that guarded state version.");
    }
    const reviewedDraw = reviewedSnapshot.draws.find((draw) => draw.id === selectedDrawId);
    const versionRows = (rows: Array<{ id?: unknown; updated_at?: unknown }>) =>
      rows
        .map((row) => ({ id: String(row.id || ""), updated_at: String(row.updated_at || "") }))
        .sort((left, right) => left.id.localeCompare(right.id));
    const expectedTeamVersions = versionRows(reviewedSnapshot.teams);
    const expectedGameVersions = versionRows(reviewedSnapshot.games);
    if (!reviewedDraw?.updated_at || expectedTeamVersions.some((row) => !row.id || !row.updated_at) || expectedGameVersions.some((row) => !row.id || !row.updated_at)) {
      throw new Error("The reviewed draw, team, or game versions are incomplete. Reload before reviewing the podium.");
    }
    const generation = actionRequest.begin();
    setBusy(true);
    setNotice(null);
    try {
      const result = await requestJson<{ review_fingerprint?: string; reviewed?: boolean }>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/draws/${encodeURIComponent(selectedDrawId)}/podium/review`,
        {
          method: "POST",
          body: JSON.stringify({
            expected_state_fingerprint: opsStateFingerprint,
            expected_draw_updated_at: reviewedDraw.updated_at,
            expected_team_versions: expectedTeamVersions,
            expected_source_game_versions: expectedGameVersions,
            confirmation_text: confirmationText,
            source: "next_tournament_admin_review_podium"
          })
        }
      );
      const completion = actionSuccess("Podium reviewed", "The current teams, games, and podium are now tied to immutable review evidence.");
      if (!actionRequest.isCurrent(generation)) return completion;
      const board = await fetchBoard(selectedTournamentId, selectedDrawId);
      if (actionRequest.isCurrent(generation)) hydrateBoard(board, selectedTournamentId, selectedDrawId);
      if (actionRequest.isCurrent(generation)) setNotice({ tone: "success", text: result.reviewed ? "Podium review recorded. Any later team, game, or podium change will require a new review." : "Podium review response reconciled." });
      return completion;
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setNotice({ tone: "error", text: error instanceof Error ? error.message : "Unable to review the podium." });
      throw error;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function reconcileOperation(operation: AdminTournamentLiveOperation, confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) throw new Error("Select a tournament draw before reconciling an operation.");
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setNotice(null);
    try {
      const result = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournament-live/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/operations/${encodeURIComponent(operation.operation_key)}/reconcile`,
        { method: "POST", body: JSON.stringify({ confirmation_text: confirmationText }) }
      );
      const completionText = result.recovery_disposition === "not_applied"
        ? "Authoritative evidence proved the operation never changed draw state; its lock is closed. Reload and review before a new command."
        : "Authoritative evidence proved the operation completed; recovery was audited without repeating the mutation.";
      const completion = actionSuccess("Tournament Live operation reconciled", completionText);
      if (!actionRequest.isCurrent(generation)) return completion;
      setLastResult(result);
      if (pendingCommand?.body.idempotency_key === operation.client_idempotency_key) persistPending(null);
      const board = await fetchBoard(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return completion;
      hydrateBoard(board, tournamentId, drawId);
      setNotice({
        tone: "success",
        text: completionText
      });
      return completion;
    } catch (error) {
      const uncertain = actionUncertain(
        "Tournament Live recovery needs verification",
        `Operation ${operation.operation_key} still needs authoritative reconciliation.`,
        operation.operation_key,
        "Reconcile again",
        () => reconcileOperation(operation, confirmationText)
      );
      if (!actionRequest.isCurrent(generation)) return uncertain;
      try {
        const board = await fetchBoard(tournamentId, drawId);
        if (!actionRequest.isCurrent(generation)) return uncertain;
        hydrateBoard(board, tournamentId, drawId);
      } catch {
        // Keep the original recovery error as the operator-facing result.
      }
      if (actionRequest.isCurrent(generation)) setNotice({ tone: "error", text: error instanceof Error ? error.message : "Unable to reconcile this Tournament Live operation." });
      return uncertain;
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function clearLocalPending() {
    if (pendingOperation && ACTIVE_OPERATION_STATUSES.has(pendingOperation.status)) {
      setNotice({ tone: "error", text: "This request has an active server operation. Reconcile it before clearing the local copy." });
      return;
    }
    persistPending(null);
    setNotice({ tone: "info", text: "Cleared the local request copy. No active server operation was attached to it." });
  }

  useAuthenticatedAutoLoad(
    status.enabled ? accessToken : "",
    () => loadDraws(initialDrawId || "", true),
    `${lockedTournamentId}\u0000${initialDayId}\u0000${initialDrawId || ""}`
  );

  if (!status.enabled) {
    return (
      <article className={styles.card}>
        <h2>Tournament Live is disabled</h2>
        <p>{status.warnings?.[0] || "Enable Tournament Admin reads on FastAPI."}</p>
        <a href={status.streamlit_fallback_url} target="_blank" rel="noreferrer">Open Streamlit Tournament Live fallback</a>
      </article>
    );
  }

  const scoreReadiness = commandReadiness("save_score");
  const rrReadiness = commandReadiness("generate_round_robin");
  const playoffReadiness = commandReadiness("generate_playoffs");
  const podiumReadiness = commandReadiness("generate_podium");
  const awardReadiness = commandReadiness("award_podium");
  const publishReadiness = commandReadiness("publish_official_matches");
  const routeContext = {
    tournamentId: selectedTournamentId || initialTournamentId,
    tournamentName: selectedTournament?.name || initialTournamentName || "",
    drawId: selectedDrawId || initialDrawId || "",
    dayId: initialDayId
  };
  const counts = lifecycle?.counts;
  const totalGames = counts?.games ?? sortedGames.length;
  const finalizedGames = counts?.finalized_games ?? sortedGames.filter(isScored).length;
  const openGames = counts?.open_games ?? Math.max(0, totalGames - finalizedGames);
  const duplicatePublications = counts?.duplicate_publications ?? counts?.duplicate_official_links ?? 0;
  const uncertainOperations = counts?.uncertain_operations ?? counts?.recovery_required_operations ?? 0;
  const podiumEntryCount = counts?.podium_entries ?? (lifecycle?.draws || []).reduce((total, draw) => total + draw.podium.length, 0);
  const selectedLifecycleDraw = lifecycle?.draws.find((draw) => draw.draw_id === selectedDrawId);
  const selectedDrawCounts = selectedLifecycleDraw?.counts || drawLifecycleById.get(selectedDrawId)?.counts;
  const selectedTotalGames = selectedDrawCounts?.games ?? sortedGames.length;
  const selectedFinalizedGames = selectedDrawCounts?.finalized_games ?? sortedGames.filter(isScored).length;
  const selectedOpenGames = selectedDrawCounts?.open_games ?? Math.max(0, selectedTotalGames - selectedFinalizedGames);
  const currentPodiumReview = Boolean(
    selectedLifecycleDraw?.review_evidence
      && (selectedLifecycleDraw.review_evidence.current ?? selectedLifecycleDraw.review_evidence.reviewed)
  );
  const podiumOpsFingerprint = String(snapshot?.ops_state_fingerprint || "");
  const legacyDayCorrectionsBlocked = view === "corrections" && Boolean(initialDayId);
  const publishActuallyReady = Boolean(officialReadiness?.ready && publishReadiness.ready);
  const runtimeCanPublish = Boolean(
    status.writes_enabled
      && status.official_publish_writes_enabled
      && status.official_publish_write_flag?.enabled
      && status.service_role_ready
      && status.operation_store_ready
      && status.audit_store_ready
  );
  const archiveAtomicAvailable = Boolean(
    lifecycle?.runtime_capability?.archive_available
      && lifecycle?.runtime_capability?.archive_writes_enabled
      && lifecycle?.runtime_capability?.archive_atomic_commit_enabled
  );

  const gameCards = (games: Array<Record<string, unknown>>, editable: boolean) => (
    <div className={`${styles.gameCards} ${styles.mobileGames}`}>
      {games.map((game) => (
        <article key={String(game.id || matchupLabel(game, teamsById, snapshot))} className={styles.gameCard}>
          <div className={styles.headingRow}>
            <strong>{gameLabel(game)}</strong>
            <StatusChip scored={isScored(game)} />
          </div>
          <p className={styles.matchup}>{teamLabel(teamsById.get(String(game.team_a_id || "")), snapshot)} <span>vs</span> {teamLabel(teamsById.get(String(game.team_b_id || "")), snapshot)}</p>
          <p className={styles.resultLine}>{isScored(game) ? `${shortValue(game.score_a)}–${shortValue(game.score_b)}` : "Score pending"}</p>
          {editable ? <button type="button" className={styles.secondaryButton} onClick={() => selectScoreGame(game)}>{isScored(game) ? "Correct score" : "Enter score"}</button> : null}
        </article>
      ))}
    </div>
  );

  const operationEvidence = (
    <article className={styles.card}>
      <h2>Recent operations and reconciliation</h2>
      <p className={styles.muted}>Durable operation and audit evidence is retained for every guarded draw write.</p>
      {!snapshot?.operations.length ? <p className={styles.muted}>No draw operations are recorded.</p> : (
        <div className={styles.operationList}>
          {snapshot.operations.map((operation) => (
            <div key={operation.operation_key} className={styles.operationRow}>
              <div>
                <OperationChip status={operation.status} /> <strong>{operation.command?.replace(/_/g, " ") || operation.action}</strong>
                <p className={styles.muted}>{formatTimestamp(operation.updated_at)} · attempt {operation.attempt_count}</p>
                {operation.error_text ? <p className={styles.errorText}>{operation.error_text}</p> : null}
                <details>
                  <summary>Technical operation evidence</summary>
                  <p><code>{operation.operation_key}</code></p>
                  <p><code>{operation.client_idempotency_key}</code></p>
                  <p>Intent: {operation.audit_evidence.intent_present ? "yes" : "no"} · completion/reconcile: {operation.audit_evidence.completion_present ? "yes" : "no"} · failure: {operation.audit_evidence.failure_present ? "yes" : "no"}</p>
                </details>
              </div>
              {ACTIVE_OPERATION_STATUSES.has(operation.status) ? (
                <ConfirmAction triggerLabel="Reconcile operation" title="Reconcile this interrupted tournament operation?" description="The server verifies authoritative evidence and closes or completes recovery without repeating the domain mutation." confirmLabel="Yes, reconcile operation" confirmationText="RECONCILE TOURNAMENT LIVE" busy={busy} onConfirm={(confirmationText) => reconcileOperation(operation, confirmationText)} />
              ) : null}
            </div>
          ))}
        </div>
      )}
      {lastResult ? <details><summary>Last technical response</summary><code>{lastResult.operation_key}</code>{lastResult.idempotent_replay ? " · idempotent replay" : ""}{lastResult.reconciled ? " · reconciled" : ""}</details> : null}
    </article>
  );

  const scoreWorkspace = (
    <article className={styles.card}>
      <div className={styles.headingRow}>
        <div><p className={styles.eyebrow}>{view === "corrections" ? "Correction" : "Live score"}</p><h2>{selectedGame ? gameLabel(selectedGame) : "Select a matchup"}</h2></div>
        {selectedGame ? <StatusChip scored={isScored(selectedGame)} /> : null}
      </div>
      <CommandBlockers readiness={scoreReadiness} />
      {selectedGame ? (
        <>
          <div className={styles.scorecard}>
            <div><span>Team A</span><strong>{teamLabel(teamsById.get(String(selectedGame.team_a_id || "")), snapshot)}</strong></div>
            <label htmlFor="score-a">Team A score<input id="score-a" value={scoreA} onChange={(event) => { setScoreA(event.target.value); setScoreConfirmation(false); }} type="number" min={0} step={1} inputMode="numeric" className={styles.input} /></label>
            <div className={styles.versus}>vs</div>
            <label htmlFor="score-b">Team B score<input id="score-b" value={scoreB} onChange={(event) => { setScoreB(event.target.value); setScoreConfirmation(false); }} type="number" min={0} step={1} inputMode="numeric" className={styles.input} /></label>
            <div><span>Team B</span><strong>{teamLabel(teamsById.get(String(selectedGame.team_b_id || "")), snapshot)}</strong></div>
          </div>
          {isScored(selectedGame) ? <p><strong>Before correction:</strong> {shortValue(selectedGame.score_a)}–{shortValue(selectedGame.score_b)} · <strong>After correction:</strong> {scoreA || "—"}–{scoreB || "—"}</p> : null}
          {!scoreConfirmation ? <button type="button" className={styles.primaryButton} onClick={validateScoreDraft} disabled={!scoreReadiness.ready || busy}>Review score</button> : (
            <section className={styles.confirmationCard} aria-label="Score confirmation">
              <div className={styles.headingRow}><div><p className={styles.eyebrow}>Score review</p><h3>{gameLabel(selectedGame)}</h3></div><button type="button" className={styles.secondaryButton} onClick={() => setScoreConfirmation(false)}>Edit score</button></div>
              <p className={styles.matchup}>{teamLabel(teamsById.get(String(selectedGame.team_a_id || "")), snapshot)} <strong>{scoreA}</strong> – <strong>{scoreB}</strong> {teamLabel(teamsById.get(String(selectedGame.team_b_id || "")), snapshot)}</p>
              <p><strong>Proposed winner:</strong> {Number(scoreA) > Number(scoreB) ? teamLabel(teamsById.get(String(selectedGame.team_a_id || "")), snapshot) : teamLabel(teamsById.get(String(selectedGame.team_b_id || "")), snapshot)}</p>
              <ConfirmAction triggerLabel="Confirm & save" title="Confirm this exact tournament score?" description={`${matchupLabel(selectedGame, teamsById, snapshot)}. Save ${scoreA}–${scoreB}.`} confirmLabel="Confirm & save" confirmationText={scoreReadiness.confirmation || CONFIRMATIONS.save_score} disabled={!scoreReadiness.ready} busy={busy} onConfirm={async (confirmationText) => { const completion = await saveScore(confirmationText); setScoreConfirmation(false); return completion; }} />
            </section>
          )}
        </>
      ) : <p className={styles.muted}>No game is available in this draw.</p>}
    </article>
  );

  return (
    <section className={styles.root}>
      <article className={styles.card}>
        <div className={styles.headingRow}>
          <div><p className={styles.eyebrow}>Authoritative tournament state</p><h2>{selectedTournament?.name || initialTournamentName || "Selected tournament"}</h2></div>
        </div>
        <p className={styles.muted}>This draw-scoped tournament runner is not JUPR Live. Tournament games become official only through the guarded Publish workflow.</p>
        <section className={styles.scopePanel} aria-label="Tournament operating scope">
          <div className={styles.scopeHeading}>
            <div>
              <p className={styles.eyebrow}>Working scope</p>
              <h3>Choose the draw you are operating</h3>
            </div>
            <p>Changing the working draw keeps you inside this tournament.</p>
          </div>
          <div className={styles.scopeGrid}>
            <div className={styles.lockedTournament}>
              <span>Tournament</span>
              <strong>{selectedTournament?.name || initialTournamentName || "Selected tournament"}</strong>
              <small>Locked to this tournament workspace</small>
            </div>
            <label htmlFor="working-draw">
              Working draw
              <select
                id="working-draw"
                value={selectedDrawId}
                onChange={(event) => selectDraw(event.target.value)}
                className={styles.input}
                disabled={busy || !draws.length}
                aria-describedby="working-draw-help"
              >
                <option value="">Choose a draw…</option>
                {selectedDrawId && !draws.some((draw) => draw.id === selectedDrawId) ? <option value={selectedDrawId}>Loading selected draw…</option> : null}
                {draws.map((draw) => {
                  const lifecycleDraw = drawLifecycleById.get(draw.id);
                  const inactive = isInactiveTournamentDraw(draw);
                  return <option key={draw.id} value={draw.id} disabled={inactive || !lifecycleDraw}>{drawLabel(draw, lifecycleDraw)}</option>;
                })}
              </select>
            </label>
            <div className={styles.scopeActions}>
              <button type="button" className={styles.secondaryButton} onClick={() => void loadDraws(selectedDrawId, false)} disabled={busy || !accessToken}>{busy ? "Refreshing…" : "Refresh available draws"}</button>
              <button type="button" className={styles.secondaryButton} onClick={() => void loadLiveBoard()} disabled={busy || !selectedDrawId}>Reload selected draw</button>
            </div>
          </div>
          <p id="working-draw-help" className={styles.scopeHelp}>Choose the division or draw you want to review across schedule, scoring, corrections, podium, and publish.</p>
        </section>
        <div className={styles.statsGrid}>
          <div><span>Draw</span><strong>{draws.find((draw) => draw.id === selectedDrawId)?.name || selectedLifecycleDraw?.name || "Select a draw"}</strong></div>
          <div><span>Scores</span><strong>{selectedFinalizedGames} of {selectedTotalGames}</strong></div>
          <div><span>Open games</span><strong>{selectedOpenGames}</strong></div>
          <div><span>Official matches</span><strong>{selectedDrawCounts?.published_games ?? snapshot?.progression?.published_games ?? 0}</strong></div>
          <div><span>Awards</span><strong>{selectedDrawCounts?.verified_awards ?? snapshot?.progression?.verified_awards ?? 0} / {selectedDrawCounts?.expected_awards ?? snapshot?.progression?.expected_awards ?? 0}</strong></div>
        </div>
        {snapshot?.state_fingerprint ? <details><summary>Technical state version</summary><code>{snapshot.state_fingerprint}</code></details> : null}
        {!accessToken && !sessionLoading ? <p className={styles.sessionWarning}>Admin sign-in required. <Link href="/admin/login">Open admin login</Link></p> : null}
        {sessionMessage ? <p className={styles.errorText}>{sessionMessage}</p> : null}
      </article>

      {pendingCommand ? <article className={styles.recoveryCard}><h2>Interrupted request retained on this device</h2><p>The exact {pendingCommand.body.command.replace(/_/g, " ")} request from {formatTimestamp(pendingCommand.createdAt)} is retained until its outcome is proven.</p><div className={styles.buttonRow}><button type="button" className={styles.primaryButton} onClick={() => void executePending(pendingCommand, true)} disabled={busy}>Retry exact retained request</button><button type="button" className={styles.secondaryButton} onClick={clearLocalPending} disabled={busy || Boolean(pendingOperation && ACTIVE_OPERATION_STATUSES.has(pendingOperation.status))}>Clear local copy</button></div><details><summary>Technical operation evidence</summary><code>{pendingCommand.body.idempotency_key}</code></details></article> : null}

      {snapshot && view === "overview" ? (
        <div className={styles.moduleGrid}>
          {[
            ["Preflight & check-in", "/admin/tournaments/live-operations/check-in", "Attendance, payment, waiver, partner, substitute, court, time, and staffing readiness."],
            ["Draws & schedule", "/admin/tournaments/live-operations/draws", "Review prepared teams, rounds, courts, open games, and playoff progression."],
            ["Live scoring", "/admin/tournament-live", `${selectedFinalizedGames} of ${selectedTotalGames} games scored; ${selectedOpenGames} open.`],
            ["Corrections & recovery", "/admin/tournaments/live-operations/corrections", "Correct draw scores and reconcile failed or uncertain operations."],
            ["Podium draft", "/admin/tournaments/live-operations/podium", currentPodiumReview ? "The current podium has explicit review evidence." : "Podium review is incomplete or stale."]
          ].map(([title, path, detail]) => <Link key={path} href={tournamentRouteHref(path, routeContext)} className={styles.moduleCard}><h2>{title}</h2><p>{detail}</p></Link>)}
        </div>
      ) : null}

      {snapshot && view === "draws" ? (
        <>
          <article className={styles.card}><h2>Draws & schedule</h2><p className={styles.muted}>Round and court cards keep the tournament schedule readable without a page-wide technical table.</p>{gameCards(sortedGames, false)}</article>
          <article className={styles.card}><h2>Progress this draw</h2><div className={styles.commandGrid}><section className={styles.commandCard}><h3>Round robin</h3><CommandBlockers readiness={rrReadiness} /><ConfirmAction triggerLabel="Generate games" title="Generate round-robin games?" description="Create the reviewed Python schedule from the current teams." confirmLabel="Yes, generate games" confirmationText={rrReadiness.confirmation || CONFIRMATIONS.generate_round_robin} disabled={!rrReadiness.ready} busy={busy} onConfirm={(confirmationText) => submitCommand("generate_round_robin", confirmationText)} /></section><section className={styles.commandCard}><h3>Playoffs</h3><CommandBlockers readiness={playoffReadiness} /><label htmlFor="advance-count">Advance count<input id="advance-count" value={playoffAdvanceCount} onChange={(event) => setPlayoffAdvanceCount(event.target.value)} type="number" min={4} max={6} className={styles.input} /></label><ConfirmAction triggerLabel="Generate playoffs" title="Generate the playoff bracket?" description={`Advance the top ${playoffAdvanceCount} teams after every round-robin score is final.`} confirmLabel="Yes, generate playoffs" confirmationText={playoffReadiness.confirmation || CONFIRMATIONS.generate_playoffs} disabled={!playoffReadiness.ready} busy={busy} onConfirm={(confirmationText) => submitCommand("generate_playoffs", confirmationText, { advance_count: Number(playoffAdvanceCount) })} /></section></div></article>
        </>
      ) : null}

      {snapshot && legacyDayCorrectionsBlocked ? (
        <article className={styles.recoveryCard} role="alert">
          <h2>Use guarded tournament-day correction</h2>
          <p>Day-owned completed scores must use the guarded tournament-day correction workspace. The legacy draw correction command is closed because it cannot preserve day-run, queue, draw, and game versions together.</p>
          <Link className={styles.secondaryLink} href={tournamentRouteHref("/admin/tournaments/live-operations/corrections", routeContext)}>Open Corrections &amp; recovery</Link>
        </article>
      ) : null}

      {snapshot && (view === "scoring" || (view === "corrections" && !legacyDayCorrectionsBlocked)) ? (
        <>
          <article className={styles.card}><div className={styles.headingRow}><div><h2>{view === "corrections" ? "Scored games with current results" : "Round and court matchups"}</h2><p className={styles.muted}>{view === "corrections" ? "Choose a finalized game to review its before/after correction." : "Choose a matchup for inline score entry."}</p></div><div className={styles.filterRow}><label>Round<select className={styles.input} value={roundFilter} onChange={(event) => setRoundFilter(event.target.value)}><option value="all">All rounds</option>{roundOptions.map((round) => <option key={round} value={round}>{round}</option>)}</select></label><label>Status<select className={styles.input} value={gameStatusFilter} onChange={(event) => setGameStatusFilter(event.target.value)}><option value="all">All games</option><option value="open">Open</option><option value="final">Final</option></select></label></div></div>{gameCards(view === "corrections" ? filteredGames.filter(isScored) : filteredGames, true)}</article>
          {scoreWorkspace}
          {view === "corrections" ? <article className={styles.card}><h2>Correction boundaries</h2><p>Match Log corrections are for official published matches. Tournament draw corrections occur here before or after publication, and the server preserves replay and rating safety.</p><p className={styles.linkRow}><Link href={tournamentRouteHref("/admin/tournaments/live-operations/check-in", routeContext)}>Review substitution attendance</Link><Link href="/admin/match-log">Open Match Log</Link><Link href="/admin/replay-history">Open replay evidence</Link></p><p><strong>Forfeit or substitution:</strong> record attendance/substitution evidence first, then use the supported draw command or Match Log recovery path. Do not disguise either as an ordinary score correction.</p></article> : null}
          {view === "corrections" ? operationEvidence : null}
        </>
      ) : null}

      {snapshot && view === "podium" ? (
        <article className={styles.card}><h2>Podium draft</h2><p className={styles.muted}>Generate placements, review the current teams/games/podium explicitly, then award the reviewed podium.</p><div className={styles.podiumGrid}>{(snapshot.podium || []).map((row, index) => { const teamId = String(row.team_id || row.tournament_team_id || ""); return <div key={String(row.id || `${teamId}:${index}`)} className={styles.podiumCard}><span>Place {shortValue(row.place || row.placement || index + 1)}</span><strong>{teamLabel(teamsById.get(teamId), snapshot)}</strong></div>; })}</div>{!snapshot.podium.length ? <p>No podium entries yet.</p> : null}<div className={styles.commandGrid}><section className={styles.commandCard}><h3>Generate podium</h3><CommandBlockers readiness={podiumReadiness} /><ConfirmAction triggerLabel="Generate podium" title="Generate podium placements?" description="Calculate placements from the reviewed final results." confirmLabel="Yes, generate podium" confirmationText={podiumReadiness.confirmation || CONFIRMATIONS.generate_podium} disabled={!podiumReadiness.ready} busy={busy} onConfirm={(text) => submitCommand("generate_podium", text)} /></section><section className={styles.commandCard}><h3>Explicit review</h3><p>{currentPodiumReview ? "Current review evidence is present." : "A current explicit review is required; any team, game, or podium change makes it stale."}</p>{!podiumOpsFingerprint ? <p className={styles.errorText}>Podium review needs the current Tournament Ops fingerprint. Reload after the guarded snapshot exposes it.</p> : null}<ConfirmAction triggerLabel="Review podium" title="Review this exact podium?" description="This records immutable evidence for the current teams, games, and placements." confirmLabel="Yes, review podium" confirmationText="REVIEW PODIUM" disabled={!snapshot.podium.length || currentPodiumReview || !podiumOpsFingerprint} disabledReason={!podiumOpsFingerprint ? "The guarded Tournament Ops fingerprint is unavailable." : undefined} busy={busy} onConfirm={reviewPodium} /></section><section className={styles.commandCard}><h3>Awards</h3><CommandBlockers readiness={awardReadiness} /><ConfirmAction triggerLabel="Award podium" title="Award this reviewed podium?" description="Mint only the exact expected linked-player tournament awards." confirmLabel="Yes, award podium" confirmationText={awardReadiness.confirmation || CONFIRMATIONS.award_podium} disabled={!awardReadiness.ready || !currentPodiumReview} busy={busy} onConfirm={(text) => submitCommand("award_podium", text)} /></section></div></article>
      ) : null}

      {snapshot && view === "results" ? (
        <><article className={styles.card}><div className={styles.headingRow}><div><h2>Review results</h2><p className={styles.muted}>Human-readable division and draw state, standings, podium readiness, corrections, and exceptions.</p></div><Link href={tournamentRouteHref("/admin/tournaments/publish/import-results", routeContext)} className={styles.secondaryLink}>Import results</Link></div><div className={styles.drawCards}>{(lifecycle?.draws || []).map((draw) => <section key={draw.draw_id} className={styles.commandCard}><div className={styles.headingRow}><h3>{draw.name}</h3><span className={draw.readiness.official_publish.ready ? styles.successChip : styles.recoveryChip}>{draw.readiness.official_publish.ready ? "Ready" : "Blocked"}</span></div><p>{draw.counts.finalized_games || 0} of {draw.counts.games || 0} scores complete · {(draw.counts.open_games || 0)} missing</p><h4>Standings</h4>{draw.standings.length ? <ol>{draw.standings.slice(0, 8).map((row, index) => <li key={String(row.team_id || index)}>{teamLabel(teamsById.get(String(row.team_id || "")), snapshot)} · {shortValue(row.wins)} wins</li>)}</ol> : <p className={styles.muted}>Standings are not available yet.</p>}<h4>Podium readiness</h4><LifecycleBlockers readiness={draw.readiness.official_publish} /></section>)}</div>{!(lifecycle?.draws || []).length ? <p className={styles.muted}>Reload authoritative lifecycle state to review divisions.</p> : null}</article><article className={styles.card}><h2>Selected draw results</h2>{gameCards(sortedGames, false)}</article></>
      ) : null}

      {snapshot && view === "publish-overview" ? <div className={styles.moduleGrid}>{[["Review results", "/admin/tournaments/ops/results", `${finalizedGames} of ${totalGames} scores complete.`], ["Import results", "/admin/tournaments/publish/import-results", "Separate DUPR CSV preview and guarded import workspace."], ["Publish divisions", "/admin/tournaments/ops/publish", officialReadiness?.ready ? "Tournament prerequisites are complete." : "Publishing is blocked by tournament prerequisites."], ["Tournament closeout", "/admin/tournaments/publish/closeout", archiveAtomicAvailable ? "Review the final server-enforced closeout prerequisites." : "Archive is unavailable until atomic closeout commit is installed."]].map(([title, path, detail]) => <Link key={path} href={tournamentRouteHref(path, routeContext)} className={styles.moduleCard}><h2>{title}</h2><p>{detail}</p></Link>)}</div> : null}

      {snapshot && view === "publish" ? (
        <article className={styles.card}><h2>Publish divisions</h2><div className={styles.readinessColumns}><section className={styles.commandCard}><h3>Tournament readiness</h3><LifecycleBlockers readiness={officialReadiness} /></section><section className={styles.commandCard}><h3>Runtime capability</h3><p><span className={runtimeCanPublish ? styles.successChip : styles.recoveryChip}>{runtimeCanPublish ? "Available" : "Unavailable"}</span></p><p className={styles.muted}>The dedicated official-publish permission, service role, operation store, and audit store must all be available. Environment permission never means this tournament is ready.</p></section></div><h3>{selectedLifecycleDraw?.name || "Selected draw"}</h3><p>{selectedFinalizedGames} of {selectedTotalGames} games finalized · {selectedOpenGames} open · {selectedDrawCounts?.published_games || 0} official matches</p><div className={styles.dangerCard}><h3>Official rated matches</h3><LifecycleBlockers readiness={officialReadiness} /><CommandBlockers readiness={publishReadiness} /><label htmlFor="winner-bonus">Playoff winner bonus Elo<input id="winner-bonus" value={publishBonusElo} onChange={(event) => setPublishBonusElo(event.target.value)} type="number" min={0} max={40} step={1} className={styles.input} disabled={!publishActuallyReady || !runtimeCanPublish} /></label><ConfirmAction triggerLabel="Publish official matches" title="Publish all verified games as official rated matches?" description={`This terminal write publishes the exact reviewed tournament games and applies a ${publishBonusElo || "0"}-Elo playoff-winner bonus.`} confirmLabel="Yes, publish official matches" confirmationText={publishReadiness.confirmation || CONFIRMATIONS.publish_official_matches} tone="danger" disabled={!publishActuallyReady || !runtimeCanPublish} busy={busy} onConfirm={(text) => submitCommand("publish_official_matches", text, { playoff_winner_bonus_elo: Number(publishBonusElo) })} /></div></article>
      ) : null}

      {snapshot && view === "closeout" ? (
        <article className={styles.card}><h2>Tournament closeout</h2><p className={styles.muted}>Live status cards are authoritative where records exist; operational follow-ups remain clearly marked Needs review.</p><div className={styles.closeoutGrid}>{[
          ["Divisions", counts?.draws ? "Complete" : "Blocked", `${counts?.draws || 0} draw${counts?.draws === 1 ? "" : "s"}`],
          ["Scores", openGames === 0 && totalGames > 0 ? "Complete" : "Blocked", `${finalizedGames} finalized; ${openGames} open`],
          ["Podiums", lifecycle?.draws.length && lifecycle.draws.every((draw) => Boolean(draw.review_evidence && (draw.review_evidence.current ?? draw.review_evidence.reviewed))) ? "Complete" : "Blocked", `${podiumEntryCount} entries; explicit review required`],
          ["Awards", counts?.expected_awards === counts?.verified_awards && !counts?.unexpected_awards ? "Complete" : "Blocked", `${counts?.verified_awards || 0} of ${counts?.expected_awards || 0} verified`],
          ["Official matches", counts?.published_games === totalGames && !duplicatePublications ? "Complete" : "Blocked", `${counts?.published_games || 0} of ${totalGames} linked; ${duplicatePublications} duplicate`],
          ["Replay / audit evidence", !counts?.active_operations && !uncertainOperations ? "Complete" : "Blocked", `${counts?.active_operations || 0} active; ${uncertainOperations} uncertain`],
          ["Communications", "Needs review", "Confirm participant result communication before archive."],
          ["Payments, extras, and fulfillment", "Needs review", "Offline payment and fulfillment exceptions require operator review."],
          ["Archive readiness", archiveReadiness?.ready && archiveAtomicAvailable ? "Complete" : "Blocked", !archiveAtomicAvailable ? "Atomic archive commit is unavailable; no archive write is permitted." : archiveReadiness?.ready ? "All server-enforced archive prerequisites passed." : `${lifecycleBlockerMessages(archiveReadiness).length} blocker(s)`]
        ].map(([title, stateValue, detail]) => <section key={title} className={styles.closeoutCard}><div className={styles.headingRow}><h3>{title}</h3><span className={stateValue === "Complete" ? styles.successChip : stateValue === "Needs review" ? styles.neutralChip : styles.recoveryChip}>{stateValue}</span></div><p>{detail}</p></section>)}</div><section className={styles.dangerCard}><h3>Archive unavailable</h3><LifecycleBlockers readiness={archiveReadiness} /><p><strong>No archive write is available.</strong> The server requires an atomic database commit that rechecks all closeout evidence under lock; that commit surface is not installed.</p></section></article>
      ) : null}

      {snapshot && view === "status" ? <><article className={styles.card}><h2>Status and recovery</h2><p>Actual score and publication operations appear below with reconciliation status and optional technical evidence. Archive lifecycle controls live only in Tournament Closeout.</p><div className={styles.statsGrid}><div><span>Active operations</span><strong>{counts?.active_operations || activeOperations.length}</strong></div><div><span>Uncertain operations</span><strong>{uncertainOperations}</strong></div><div><span>Duplicate publications</span><strong>{duplicatePublications}</strong></div></div></article>{operationEvidence}</> : null}

      {notice ? <p role="status" aria-live="polite" className={notice.tone === "error" ? styles.noticeError : notice.tone === "success" ? styles.noticeSuccess : styles.noticeInfo}>{notice.text}</p> : null}
    </section>
  );
}
