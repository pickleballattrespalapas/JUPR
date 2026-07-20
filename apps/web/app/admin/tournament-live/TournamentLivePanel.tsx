"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import type {
  AdminTournament,
  AdminTournamentDraw,
  AdminTournamentListResponse,
  AdminTournamentLiveOperation,
  AdminTournamentLiveReadiness,
  AdminTournamentLiveSnapshotResponse,
  AdminTournamentLiveStatusResponse,
  AdminTournamentOpsTeam,
  AdminTournamentWriteResponse
} from "@/lib/adminTournamentApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import styles from "./TournamentLivePanel.module.css";

type Props = { apiBase: string | null; clubId: string; status: AdminTournamentLiveStatusResponse };
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
  const stage = String(game.stage || "Game").replace(/_/g, " ");
  const rr = game.rr_round_number ? `Round ${game.rr_round_number}` : "";
  const slot = game.rr_slot_number ? `Slot ${game.rr_slot_number}` : "";
  const playoff = [game.playoff_round, game.playoff_game_code].map(shortValue).filter((item) => item !== "—").join(" ");
  return [stage, rr, slot, playoff].filter(Boolean).join(" · ") || String(game.id || "Game");
}

function drawLabel(draw: AdminTournamentDraw): string {
  return `${draw.name || "Draw"} · ${draw.status || "draft"}`;
}

function playerLabel(snapshot: AdminTournamentLiveSnapshotResponse | null, playerId?: number | null): string {
  if (playerId == null) return "—";
  const match = (snapshot?.players || []).find((player) => Number(player.id) === Number(playerId));
  return match ? `${match.name} (#${match.id})` : `#${playerId}`;
}

function teamLabel(team: AdminTournamentOpsTeam | undefined, snapshot: AdminTournamentLiveSnapshotResponse | null): string {
  if (!team) return "Unassigned";
  const p1 = playerLabel(snapshot, team.player1_id ?? null);
  const p2 = team.player2_id == null ? null : playerLabel(snapshot, team.player2_id);
  return `Team ${team.team_number ?? "?"}: ${p2 ? `${p1} / ${p2}` : p1}`;
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
  return <span className={scored ? styles.successChip : styles.openChip}>{scored ? "scored" : "open"}</span>;
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

export default function TournamentLivePanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [includeArchived, setIncludeArchived] = useState(false);
  const [tournaments, setTournaments] = useState<AdminTournament[]>([]);
  const [draws, setDraws] = useState<AdminTournamentDraw[]>([]);
  const [selectedTournamentId, setSelectedTournamentId] = useState("");
  const [selectedDrawId, setSelectedDrawId] = useState("");
  const [snapshot, setSnapshot] = useState<AdminTournamentLiveSnapshotResponse | null>(null);
  const [scoreGameId, setScoreGameId] = useState("");
  const [scoreA, setScoreA] = useState("");
  const [scoreB, setScoreB] = useState("");
  const [scoreConfirm, setScoreConfirm] = useState("");
  const [roundRobinConfirm, setRoundRobinConfirm] = useState("");
  const [playoffAdvanceCount, setPlayoffAdvanceCount] = useState("4");
  const [playoffConfirm, setPlayoffConfirm] = useState("");
  const [podiumConfirm, setPodiumConfirm] = useState("");
  const [awardConfirm, setAwardConfirm] = useState("");
  const [publishConfirm, setPublishConfirm] = useState("");
  const [publishBonusElo, setPublishBonusElo] = useState("0");
  const [reconcileConfirm, setReconcileConfirm] = useState("");
  const [pendingCommand, setPendingCommand] = useState<PendingCommand | null>(null);
  const [lastResult, setLastResult] = useState<AdminTournamentWriteResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  const selectedTournament = tournaments.find((tournament) => tournament.id === selectedTournamentId) || snapshot?.tournament || null;
  const { teamsById, sortedGames, rrGames, playoffGames } = useMemo(() => {
    const games = [...(snapshot?.games || [])].sort((left, right) => gameSortKey(left).localeCompare(gameSortKey(right)));
    return {
      teamsById: new Map((snapshot?.teams || []).map((team) => [String(team.id || ""), team])),
      sortedGames: games,
      rrGames: games.filter((game) => String(game.stage || "").toUpperCase() === "ROUND_ROBIN"),
      playoffGames: games.filter((game) => String(game.stage || "").toUpperCase() === "PLAYOFF")
    };
  }, [snapshot]);
  const selectedGame = sortedGames.find((game) => String(game.id || "") === scoreGameId) || sortedGames.find((game) => !isScored(game)) || sortedGames[0] || null;
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

  function seedScoreEditor(nextSnapshot: AdminTournamentLiveSnapshotResponse | null) {
    const games = [...(nextSnapshot?.games || [])].sort((left, right) => gameSortKey(left).localeCompare(gameSortKey(right)));
    const target = games.find((game) => !isScored(game)) || games[0] || null;
    setScoreGameId(target ? String(target.id || "") : "");
    setScoreA(target?.score_a == null ? "" : String(target.score_a));
    setScoreB(target?.score_b == null ? "" : String(target.score_b));
    setScoreConfirm("");
  }

  function resetCommandConfirmations() {
    setRoundRobinConfirm("");
    setPlayoffConfirm("");
    setPodiumConfirm("");
    setAwardConfirm("");
    setPublishConfirm("");
  }

  async function fetchBoard(tournamentId: string, drawId: string): Promise<AdminTournamentLiveSnapshotResponse> {
    const payload = await requestJson<AdminTournamentLiveSnapshotResponse>(
      `/admin/clubs/${encodeURIComponent(clubId)}/tournament-live/tournaments/${encodeURIComponent(tournamentId)}/snapshot?draw_id=${encodeURIComponent(drawId)}`
    );
    setSelectedTournamentId(tournamentId);
    setSelectedDrawId(drawId);
    setSnapshot(payload);
    seedScoreEditor(payload);
    setPendingCommand(readPendingCommand(clubId, tournamentId, drawId));
    return payload;
  }

  async function loadTournaments() {
    setBusy(true);
    setNotice(null);
    setSnapshot(null);
    setDraws([]);
    setSelectedDrawId("");
    try {
      const suffix = includeArchived ? "?include_archived=true" : "";
      const payload = await requestJson<AdminTournamentListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/ops/tournaments${suffix}`);
      setTournaments(payload.tournaments || []);
      setNotice({ tone: "success", text: `Loaded ${payload.count ?? payload.tournaments?.length ?? 0} tournament(s).` });
    } catch (error) {
      setNotice({ tone: "error", text: error instanceof Error ? error.message : "Unable to load tournaments." });
    } finally {
      setBusy(false);
    }
  }

  async function loadDraws() {
    if (!selectedTournamentId) {
      setNotice({ tone: "error", text: "Select a tournament first." });
      return;
    }
    setBusy(true);
    setNotice(null);
    setSnapshot(null);
    try {
      const payload = await requestJson<AdminTournamentLiveSnapshotResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournament-live/tournaments/${encodeURIComponent(selectedTournamentId)}/snapshot`
      );
      const nextDraws = payload.draws || [];
      setDraws(nextDraws);
      setSelectedDrawId(nextDraws.length === 1 ? nextDraws[0].id : "");
      setNotice({ tone: nextDraws.length ? "success" : "info", text: nextDraws.length ? `Loaded ${nextDraws.length} prepared draw(s).` : "This tournament has no prepared draws. Build one in Tournament Ops." });
    } catch (error) {
      setNotice({ tone: "error", text: error instanceof Error ? error.message : "Unable to load tournament draws." });
    } finally {
      setBusy(false);
    }
  }

  async function loadLiveBoard() {
    if (!selectedTournamentId || !selectedDrawId) {
      setNotice({ tone: "error", text: "Select a tournament draw first." });
      return;
    }
    setBusy(true);
    setNotice(null);
    try {
      await fetchBoard(selectedTournamentId, selectedDrawId);
      resetCommandConfirmations();
      setReconcileConfirm("");
      setNotice({ tone: "success", text: "Authoritative draw state loaded from FastAPI." });
    } catch (error) {
      setNotice({ tone: "error", text: error instanceof Error ? error.message : "Unable to load Tournament Live board." });
    } finally {
      setBusy(false);
    }
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

  async function executePending(command: PendingCommand, replay: boolean) {
    setBusy(true);
    setNotice(null);
    try {
      const result = await requestJson<AdminTournamentWriteResponse>(commandEndpoint(command.tournamentId, command.drawId), {
        method: "POST",
        body: JSON.stringify(command.body)
      });
      persistPending(null);
      setLastResult(result);
      await fetchBoard(command.tournamentId, command.drawId);
      setNotice({
        tone: "success",
        text: result.reconciled
          ? `Operation ${compactKey(result.operation_key)} reconciled without repeating the domain write.`
          : result.idempotent_replay
            ? `Operation ${compactKey(result.operation_key)} returned its durable stored result.`
            : replay
              ? `Exact request ${compactKey(result.operation_key)} completed.`
              : `Tournament Live command completed as operation ${compactKey(result.operation_key)}.`
      });
    } catch (error) {
      try {
        await fetchBoard(command.tournamentId, command.drawId);
      } catch {
        // Preserve the exact local request even when the recovery read is unavailable.
      }
      setNotice({
        tone: "error",
        text: `${error instanceof Error ? error.message : "Tournament Live command failed."} The exact request is retained below; do not create a replacement command until its operation state is known.`
      });
    } finally {
      setBusy(false);
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
      return;
    }
    const readiness = commandReadiness(command);
    if (!readiness.ready) {
      setNotice({ tone: "error", text: `Python preflight blocks this command: ${readiness.blockers.join(" ")}` });
      return;
    }
    if (confirmationText.trim() !== CONFIRMATIONS[command]) {
      setNotice({ tone: "error", text: `Type ${CONFIRMATIONS[command]} exactly.` });
      return;
    }
    const reviewedDraw = snapshot.draws.find((draw) => String(draw.id || "") === selectedDrawId);
    if (!reviewedDraw?.updated_at) {
      setNotice({ tone: "error", text: "The draw has no reviewed version. Reload before submitting a command." });
      return;
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
        return;
      }
      body.expected_game_updated_at = String(game.updated_at);
    } else {
      const teamVersions = versionRows(snapshot.teams);
      if (!teamVersions?.length) {
        setNotice({ tone: "error", text: "The reviewed team version set is incomplete. Reload before submitting." });
        return;
      }
      body.expected_team_versions = teamVersions;
      if (command !== "generate_round_robin") {
        const gameVersions = versionRows(snapshot.games);
        if (!gameVersions?.length) {
          setNotice({ tone: "error", text: "The reviewed game version set is incomplete. Reload before submitting." });
          return;
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
    void executePending(pending, false);
  }

  function saveScore() {
    if (!selectedGame) {
      setNotice({ tone: "error", text: "Select a game before saving a score." });
      return;
    }
    const a = Number(scoreA);
    const b = Number(scoreB);
    if (!Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < 0 || a === b) {
      setNotice({ tone: "error", text: "Enter two non-tied, non-negative whole-number scores." });
      return;
    }
    submitCommand("save_score", scoreConfirm, { game_id: String(selectedGame.id || ""), score_a: a, score_b: b });
  }

  async function reconcileOperation(operation: AdminTournamentLiveOperation) {
    if (!selectedTournamentId || !selectedDrawId) return;
    if (reconcileConfirm.trim() !== "RECONCILE TOURNAMENT LIVE") {
      setNotice({ tone: "error", text: "Type RECONCILE TOURNAMENT LIVE exactly." });
      return;
    }
    setBusy(true);
    setNotice(null);
    try {
      const result = await requestJson<AdminTournamentWriteResponse>(
        `/admin/clubs/${encodeURIComponent(clubId)}/tournament-live/tournaments/${encodeURIComponent(selectedTournamentId)}/draws/${encodeURIComponent(selectedDrawId)}/operations/${encodeURIComponent(operation.operation_key)}/reconcile`,
        { method: "POST", body: JSON.stringify({ confirmation_text: reconcileConfirm }) }
      );
      setLastResult(result);
      if (pendingCommand?.body.idempotency_key === operation.client_idempotency_key) persistPending(null);
      await fetchBoard(selectedTournamentId, selectedDrawId);
      setReconcileConfirm("");
      setNotice({
        tone: "success",
        text: result.recovery_disposition === "not_applied"
          ? "Authoritative evidence proved the operation never changed draw state; its lock is closed. Reload and review before a new command."
          : "Authoritative evidence proved the operation completed; recovery was audited without repeating the mutation."
      });
    } catch (error) {
      try {
        await fetchBoard(selectedTournamentId, selectedDrawId);
      } catch {
        // Keep the original recovery error as the operator-facing result.
      }
      setNotice({ tone: "error", text: error instanceof Error ? error.message : "Unable to reconcile this Tournament Live operation." });
    } finally {
      setBusy(false);
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

  return (
    <section className={styles.root}>
      <article className={styles.card}>
        <div className={styles.headingRow}>
          <div>
            <h2>Tournament Live control room</h2>
            <p className={styles.muted}>Draw-scoped tournament scoring and progression, authorized by FastAPI/Python.</p>
          </div>
          <span className={status.writes_enabled ? styles.successChip : styles.recoveryChip}>
            {status.writes_enabled ? "staging writes open" : "read only"}
          </span>
        </div>
        <div className={styles.boundaryCallout}>
          <strong>Separate product boundary:</strong> this runner operates a prepared tournament draw. It is not JUPR Live, which remains the one-off Round Robin, League/Ladder, and Club Social product. <Link href="/admin/jupr-live">Open JUPR Live Admin</Link>.
        </div>
        <div className={accessToken ? styles.sessionReady : styles.sessionWarning}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p>{accessToken ? "Authenticated reads are available. Writes still require the dedicated staging gate shown above." : sessionLoading ? "Checking admin session…" : "Sign in before loading tournament data."}</p>
          {sessionMessage ? <p className={styles.errorText}>{sessionMessage}</p> : null}
          {!accessToken && !sessionLoading ? <Link href="/admin/login">Open admin login</Link> : null}
        </div>
        {status.warnings?.length ? (
          <ul className={styles.warnings}>{[...new Set(status.warnings)].map((warning) => <li key={warning}>{warning}</li>)}</ul>
        ) : null}
        <p className={styles.linkRow}>
          <a href={status.streamlit_fallback_url} target="_blank" rel="noreferrer">Open Streamlit Tournament Live fallback</a>
          <span aria-hidden="true"> · </span>
          <Link href="/admin/tournaments/ops">Tournament Ops setup/recovery</Link>
          <span aria-hidden="true"> · </span>
          <Link href="/admin/match-log">Match Log corrections</Link>
          <span aria-hidden="true"> · </span>
          <Link href="/admin/replay-history">Replay History</Link>
        </p>
      </article>

      <article className={styles.card}>
        <h2>1. Select a prepared draw</h2>
        <div className={styles.controlGrid}>
          <label className={styles.checkboxLabel}>
            <input type="checkbox" checked={includeArchived} onChange={(event) => setIncludeArchived(event.target.checked)} />
            Include archived tournaments
          </label>
          <button type="button" className={styles.primaryButton} onClick={loadTournaments} disabled={busy || !accessToken}>
            {busy ? "Working…" : "Load tournaments"}
          </button>
          <label>
            Tournament
            <select
              value={selectedTournamentId}
              onChange={(event) => {
                setSelectedTournamentId(event.target.value);
                setSelectedDrawId("");
                setDraws([]);
                setSnapshot(null);
                setPendingCommand(null);
              }}
              className={styles.input}
            >
              <option value="">Choose tournament…</option>
              {tournaments.map((tournament) => <option key={tournament.id} value={tournament.id}>{tournament.name} · {tournament.status}</option>)}
            </select>
          </label>
          <button type="button" className={styles.secondaryButton} onClick={loadDraws} disabled={busy || !selectedTournamentId || !accessToken}>Load prepared draws</button>
          <label>
            Draw
            <select value={selectedDrawId} onChange={(event) => { setSelectedDrawId(event.target.value); setSnapshot(null); setPendingCommand(null); }} className={styles.input} disabled={!draws.length}>
              <option value="">Choose draw…</option>
              {draws.map((draw) => <option key={draw.id} value={draw.id}>{drawLabel(draw)}</option>)}
            </select>
          </label>
          <button type="button" className={styles.primaryButton} onClick={loadLiveBoard} disabled={busy || !selectedTournamentId || !selectedDrawId || !accessToken}>Open authoritative board</button>
        </div>
      </article>

      {snapshot ? (
        <>
          <article className={styles.card}>
            <div className={styles.headingRow}>
              <div>
                <p className={styles.eyebrow}>Live draw state</p>
                <h2>{selectedTournament?.name || snapshot.tournament?.name || "Tournament"}</h2>
              </div>
              <button type="button" className={styles.secondaryButton} onClick={loadLiveBoard} disabled={busy}>Reload state</button>
            </div>
            <div className={styles.statsGrid}>
              <div><span>Status</span><strong>{snapshot.tournament?.status || "—"}</strong></div>
              <div><span>Phase</span><strong>{String(snapshot.progression?.phase || "—").replace(/_/g, " ")}</strong></div>
              <div><span>Teams</span><strong>{snapshot.summary.teams}</strong></div>
              <div><span>RR / playoff</span><strong>{rrGames.length} / {playoffGames.length}</strong></div>
              <div><span>Open games</span><strong>{snapshot.progression?.open_games ?? 0}</strong></div>
              <div><span>Official</span><strong>{snapshot.progression?.published_games ?? 0}</strong></div>
              <div><span>Awards</span><strong>{snapshot.progression?.verified_awards ?? 0} / {snapshot.progression?.expected_awards ?? 0}</strong></div>
            </div>
            <p className={styles.stateVersion}>Reviewed draw version: <code title={snapshot.state_fingerprint || ""}>{compactKey(snapshot.state_fingerprint)}</code></p>
            {snapshot.warnings?.length ? <ul className={styles.warnings}>{[...new Set(snapshot.warnings)].map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
          </article>

          {pendingCommand ? (
            <article className={styles.recoveryCard}>
              <h2>Interrupted request retained on this device</h2>
              <p>Command <strong>{pendingCommand.body.command.replace(/_/g, " ")}</strong>, idempotency UUID <code>{pendingCommand.body.idempotency_key}</code>, saved {formatTimestamp(pendingCommand.createdAt)}.</p>
              <p>{pendingOperation ? <>Server operation <code>{compactKey(pendingOperation.operation_key)}</code> is <strong>{pendingOperation.status}</strong>.</> : "No matching server operation is visible after the latest reload. The original request may not have reached FastAPI."}</p>
              <div className={styles.buttonRow}>
                <button type="button" className={styles.primaryButton} onClick={() => void executePending(pendingCommand, true)} disabled={busy}>Retry exact retained request</button>
                <button type="button" className={styles.secondaryButton} onClick={clearLocalPending} disabled={busy || Boolean(pendingOperation && ACTIVE_OPERATION_STATUSES.has(pendingOperation.status))}>Clear local copy</button>
              </div>
            </article>
          ) : null}

          {activeOperations.length ? (
            <article className={styles.recoveryCard}>
              <h2>Recovery required before another draw write</h2>
              <p>Reconciliation is read/verify/audit only: FastAPI will not repeat the domain mutation. Ambiguous partial publishing remains locked.</p>
              <label htmlFor="tournament-live-reconcile">Type RECONCILE TOURNAMENT LIVE</label>
              <input id="tournament-live-reconcile" className={styles.input} value={reconcileConfirm} onChange={(event) => setReconcileConfirm(event.target.value)} autoComplete="off" />
              <div className={styles.operationList}>
                {activeOperations.map((operation) => (
                  <div key={operation.operation_key} className={styles.operationRow}>
                    <div><OperationChip status={operation.status} /> <strong>{operation.command?.replace(/_/g, " ") || operation.action}</strong><br /><code>{compactKey(operation.operation_key)}</code>{operation.error_text ? <p className={styles.errorText}>{operation.error_text}</p> : null}</div>
                    <button type="button" className={styles.primaryButton} onClick={() => void reconcileOperation(operation)} disabled={busy || reconcileConfirm.trim() !== "RECONCILE TOURNAMENT LIVE"}>Reconcile operation</button>
                  </div>
                ))}
              </div>
            </article>
          ) : null}

          <article className={styles.card}>
            <h2>2. Enter live scores</h2>
            <CommandBlockers readiness={scoreReadiness} />
            {!sortedGames.length ? <p className={styles.muted}>No games exist. Generate round-robin games after Tournament Ops has finalized teams.</p> : (
              <div className={styles.scoreGrid}>
                <label className={styles.wideControl}>
                  Game
                  <select
                    value={scoreGameId || String(selectedGame?.id || "")}
                    onChange={(event) => {
                      const game = sortedGames.find((row) => String(row.id || "") === event.target.value) || null;
                      setScoreGameId(event.target.value);
                      setScoreA(game?.score_a == null ? "" : String(game.score_a));
                      setScoreB(game?.score_b == null ? "" : String(game.score_b));
                      setScoreConfirm("");
                    }}
                    className={styles.input}
                  >
                    {sortedGames.map((game) => <option key={String(game.id || gameLabel(game))} value={String(game.id || "")}>{gameLabel(game)} · {isScored(game) ? "scored" : "open"}</option>)}
                  </select>
                </label>
                <div><span className={styles.labelText}>Team A</span><strong>{teamLabel(teamsById.get(String(selectedGame?.team_a_id || "")), snapshot)}</strong></div>
                <div><span className={styles.labelText}>Team B</span><strong>{teamLabel(teamsById.get(String(selectedGame?.team_b_id || "")), snapshot)}</strong></div>
                <label htmlFor="score-a">Score A<input id="score-a" value={scoreA} onChange={(event) => setScoreA(event.target.value)} type="number" min={0} step={1} inputMode="numeric" className={styles.input} /></label>
                <label htmlFor="score-b">Score B<input id="score-b" value={scoreB} onChange={(event) => setScoreB(event.target.value)} type="number" min={0} step={1} inputMode="numeric" className={styles.input} /></label>
                <label htmlFor="score-confirm">Type SAVE SCORE<input id="score-confirm" value={scoreConfirm} onChange={(event) => setScoreConfirm(event.target.value)} className={styles.input} autoComplete="off" /></label>
                <button type="button" className={styles.primaryButton} onClick={saveScore} disabled={busy || !scoreReadiness.ready || scoreConfirm.trim() !== "SAVE SCORE"}>Save score</button>
              </div>
            )}
          </article>

          <article className={styles.card}>
            <h2>3. Progress this draw</h2>
            <div className={styles.commandGrid}>
              <section className={styles.commandCard} aria-labelledby="live-round-robin-heading">
                <h3 id="live-round-robin-heading">Round robin</h3>
                <p>Generate the Python schedule once, after teams are final.</p>
                <CommandBlockers readiness={rrReadiness} />
                <label htmlFor="rr-confirm">Type GENERATE GAMES<input id="rr-confirm" value={roundRobinConfirm} onChange={(event) => setRoundRobinConfirm(event.target.value)} className={styles.input} autoComplete="off" /></label>
                <button type="button" className={styles.secondaryButton} disabled={busy || !rrReadiness.ready || roundRobinConfirm.trim() !== "GENERATE GAMES"} onClick={() => submitCommand("generate_round_robin", roundRobinConfirm)}>Generate games</button>
              </section>

              <section className={styles.commandCard} aria-labelledby="live-playoffs-heading">
                <h3 id="live-playoffs-heading">Playoffs</h3>
                <p>Seed a Python bracket only after every round-robin result is final.</p>
                <CommandBlockers readiness={playoffReadiness} />
                <label htmlFor="advance-count">Advance count<input id="advance-count" value={playoffAdvanceCount} onChange={(event) => setPlayoffAdvanceCount(event.target.value)} type="number" min={4} max={6} step={1} className={styles.input} /></label>
                <label htmlFor="playoff-confirm">Type GENERATE PLAYOFFS<input id="playoff-confirm" value={playoffConfirm} onChange={(event) => setPlayoffConfirm(event.target.value)} className={styles.input} autoComplete="off" /></label>
                <button type="button" className={styles.secondaryButton} disabled={busy || !playoffReadiness.ready || playoffConfirm.trim() !== "GENERATE PLAYOFFS"} onClick={() => submitCommand("generate_playoffs", playoffConfirm, { advance_count: Number(playoffAdvanceCount) })}>Generate playoffs</button>
              </section>

              <section className={styles.commandCard} aria-labelledby="live-podium-heading">
                <h3 id="live-podium-heading">Podium</h3>
                <p>Create immutable draw placements, then mint the expected linked-player awards.</p>
                <CommandBlockers readiness={podiumReadiness} />
                <label htmlFor="podium-confirm">Type GENERATE PODIUM<input id="podium-confirm" value={podiumConfirm} onChange={(event) => setPodiumConfirm(event.target.value)} className={styles.input} autoComplete="off" /></label>
                <button type="button" className={styles.secondaryButton} disabled={busy || !podiumReadiness.ready || podiumConfirm.trim() !== "GENERATE PODIUM"} onClick={() => submitCommand("generate_podium", podiumConfirm)}>Generate podium</button>
                <hr />
                <CommandBlockers readiness={awardReadiness} />
                <label htmlFor="award-confirm">Type AWARD PODIUM<input id="award-confirm" value={awardConfirm} onChange={(event) => setAwardConfirm(event.target.value)} className={styles.input} autoComplete="off" /></label>
                <button type="button" className={styles.secondaryButton} disabled={busy || !awardReadiness.ready || awardConfirm.trim() !== "AWARD PODIUM"} onClick={() => submitCommand("award_podium", awardConfirm)}>Award podium</button>
              </section>

              <section className={styles.dangerCard} aria-labelledby="live-publish-heading">
                <h3 id="live-publish-heading">Official match publish</h3>
                <p>Terminal rated write: every game, podium award, and existing official link must verify first.</p>
                <CommandBlockers readiness={publishReadiness} />
                <label htmlFor="winner-bonus">Playoff winner bonus Elo<input id="winner-bonus" value={publishBonusElo} onChange={(event) => setPublishBonusElo(event.target.value)} type="number" min={0} max={40} step={1} className={styles.input} /></label>
                <label htmlFor="publish-confirm">Type PUBLISH MATCHES<input id="publish-confirm" value={publishConfirm} onChange={(event) => setPublishConfirm(event.target.value)} className={styles.input} autoComplete="off" /></label>
                <button type="button" className={styles.dangerButton} disabled={busy || !publishReadiness.ready || publishConfirm.trim() !== "PUBLISH MATCHES"} onClick={() => submitCommand("publish_official_matches", publishConfirm, { playoff_winner_bonus_elo: Number(publishBonusElo) })}>Publish official matches</button>
              </section>
            </div>
          </article>

          <article className={styles.card}>
            <h2>4. Games</h2>
            <div className={styles.tableWrap}>
              <table className={styles.gameTable}>
                <caption>Authoritative games for the selected tournament draw</caption>
                <thead><tr><th scope="col">Game</th><th scope="col">Team A</th><th scope="col">Team B</th><th scope="col">Score</th><th scope="col">Status</th></tr></thead>
                <tbody>
                  {sortedGames.map((game) => (
                    <tr key={String(game.id || gameLabel(game))}>
                      <th scope="row">{gameLabel(game)}</th>
                      <td>{teamLabel(teamsById.get(String(game.team_a_id || "")), snapshot)}</td>
                      <td>{teamLabel(teamsById.get(String(game.team_b_id || "")), snapshot)}</td>
                      <td>{shortValue(game.score_a)}–{shortValue(game.score_b)}</td>
                      <td><StatusChip scored={isScored(game)} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className={styles.mobileGames}>
              {sortedGames.map((game) => (
                <article key={`mobile:${String(game.id || gameLabel(game))}`} className={styles.gameCard}>
                  <div className={styles.headingRow}><strong>{gameLabel(game)}</strong><StatusChip scored={isScored(game)} /></div>
                  <dl><dt>Team A</dt><dd>{teamLabel(teamsById.get(String(game.team_a_id || "")), snapshot)}</dd><dt>Team B</dt><dd>{teamLabel(teamsById.get(String(game.team_b_id || "")), snapshot)}</dd><dt>Score</dt><dd>{shortValue(game.score_a)}–{shortValue(game.score_b)}</dd></dl>
                </article>
              ))}
            </div>
          </article>

          <article className={styles.card}>
            <h2>5. Durable operation and audit evidence</h2>
            {!snapshot.operations.length ? <p className={styles.muted}>No Tournament Live operations are recorded for this draw.</p> : (
              <div className={styles.operationList}>
                {snapshot.operations.map((operation) => (
                  <div key={operation.operation_key} className={styles.operationRow}>
                    <div>
                      <OperationChip status={operation.status} /> <strong>{operation.command?.replace(/_/g, " ") || operation.action}</strong>
                      <p><code title={operation.operation_key}>{compactKey(operation.operation_key)}</code> · attempt {operation.attempt_count} · {formatTimestamp(operation.updated_at)}</p>
                      <p className={styles.muted}>Intent audit: {operation.audit_evidence.intent_present ? "yes" : "no"} · completion/reconcile audit: {operation.audit_evidence.completion_present ? "yes" : "no"} · failure audit: {operation.audit_evidence.failure_present ? "yes" : "no"}</p>
                    </div>
                    <code title={operation.client_idempotency_key}>{compactKey(operation.client_idempotency_key)}</code>
                  </div>
                ))}
              </div>
            )}
            {lastResult ? <p className={styles.stateVersion}>Last response: operation <code>{compactKey(lastResult.operation_key)}</code>{lastResult.idempotent_replay ? " · idempotent replay" : ""}{lastResult.reconciled ? " · reconciled" : ""}</p> : null}
          </article>
        </>
      ) : null}

      {notice ? <p role="status" aria-live="polite" className={notice.tone === "error" ? styles.noticeError : notice.tone === "success" ? styles.noticeSuccess : styles.noticeInfo}>{notice.text}</p> : null}
    </section>
  );
}
