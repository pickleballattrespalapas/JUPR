"use client";

import Link from "next/link";
import { useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import type {
  AdminTournament,
  AdminTournamentListResponse,
  AdminTournamentOpsSnapshotResponse,
  AdminTournamentOpsTeam,
  AdminTournamentResultsImportPreviewResponse,
  AdminTournamentStatusResponse,
  AdminTournamentWriteResponse
} from "@/lib/adminTournamentApi";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

export type OpsWorkflow = "all" | "draws" | "import" | "results" | "publish";
type Props = { apiBase: string | null; clubId: string; status: AdminTournamentStatusResponse; workflow?: OpsWorkflow };
type TeamEditorRow = { editor_key: string; team_number: string; player1_id: string; player2_id: string; seed: string; notes: string };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function shortValue(value: unknown): string {
  if (value == null || value === "") return "—";
  if (typeof value === "object") return JSON.stringify(value).slice(0, 160);
  return String(value);
}

function eventOptionLabel(row: Record<string, unknown>): string {
  const family = String(row.event_family_label || "").trim();
  const division = String(row.division_name || row.label || "").trim();
  if (family && division && family !== division) return `${family} / ${division}`;
  return division || family || String(row.id || "Event");
}

function gameLabel(row: Record<string, unknown>): string {
  const stage = String(row.stage || "Game");
  const round = row.rr_round_number ? `R${row.rr_round_number}` : String(row.playoff_round || "");
  const slot = row.rr_slot_number ? `S${row.rr_slot_number}` : String(row.playoff_game_code || "");
  const teams = `${shortValue(row.team_a_id)} vs ${shortValue(row.team_b_id)}`;
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

function GenericRowsTable({ rows, preferredColumns }: { rows: Array<Record<string, unknown>>; preferredColumns: string[] }) {
  if (!rows.length) return <p style={{ color: "#64748b" }}>No rows loaded.</p>;
  const discovered = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  const columns = [...preferredColumns.filter((key) => discovered.includes(key)), ...discovered.filter((key) => !preferredColumns.includes(key)).slice(0, 6)];
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "860px" }}>
        <thead>
          <tr>{columns.map((column) => <th key={column} style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>{column}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={String(row.id || index)}>
              {columns.map((column) => <td key={column} style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0", maxWidth: 260, overflowWrap: "anywhere" }}>{shortValue(row[column])}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function TournamentOpsPanel({ apiBase, clubId, status, workflow = "all" }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [includeArchived, setIncludeArchived] = useState(false);
  const [tournaments, setTournaments] = useState<AdminTournament[]>([]);
  const [selectedTournamentId, setSelectedTournamentId] = useState("");
  const [selectedDrawId, setSelectedDrawId] = useState("");
  const [snapshot, setSnapshot] = useState<AdminTournamentOpsSnapshotResponse | null>(null);
  const [drawEventOptionId, setDrawEventOptionId] = useState("");
  const [drawName, setDrawName] = useState("");
  const [teamRows, setTeamRows] = useState<TeamEditorRow[]>(() => teamRowsFromTeams([], ""));
  const [registrationImportMode, setRegistrationImportMode] = useState("REPLACE");
  const [bulkTeamMode, setBulkTeamMode] = useState("REPLACE");
  const [bulkTeamText, setBulkTeamText] = useState("Player 1,Player 2,Seed,Notes\n");
  const [scoreGameId, setScoreGameId] = useState("");
  const [scoreA, setScoreA] = useState("");
  const [scoreB, setScoreB] = useState("");
  const [playoffAdvanceCount, setPlayoffAdvanceCount] = useState("4");
  const [publishBonusElo, setPublishBonusElo] = useState("0");
  const [resultsImportMode, setResultsImportMode] = useState("REPLACE");
  const [resultsRawText, setResultsRawText] = useState("playerA1,playerB1,teamAGame1,teamBGame1\n");
  const [resultsPreview, setResultsPreview] = useState<AdminTournamentResultsImportPreviewResponse | null>(null);
  const [resultsMappings, setResultsMappings] = useState<Record<string, { action?: string; player_id?: string | number | null }>>({});
  const [resultsMatchReviews, setResultsMatchReviews] = useState<Record<string, { include?: boolean; stage?: string }>>({});
  const [resultsPodiumRefs, setResultsPodiumRefs] = useState<Record<string, string | null>>({});
  const [allowDuplicateMapping, setAllowDuplicateMapping] = useState(false);
  const [resultsReviewDirty, setResultsReviewDirty] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const listRequest = useLatestRequestGuard(accessToken, clearProtectedOpsState);
  const snapshotRequest = useLatestRequestGuard(accessToken);
  const actionRequest = useLatestRequestGuard(accessToken);

  const operationsWriteReady = Boolean(
    status.mutation_runtime?.service_role_ready
    && status.mutation_runtime?.surface_flags?.operations?.enabled
    && status.operations_runtime?.operations_mutations_enabled
  );
  const reviewedState = snapshot?.state_fingerprint || "";
  const selectedDraw = snapshot?.draws?.find((row) => String(row.id || "") === selectedDrawId) || null;
  const reviewedDrawUpdatedAt = String(selectedDraw?.updated_at || "").trim();
  const reviewedTeamVersions = (snapshot?.teams || [])
    .filter((row) => String(row.draw_id || "") === selectedDrawId)
    .map((row) => ({ id: String(row.id || ""), updated_at: String(row.updated_at || "") }));
  const reviewedSourceGameVersions = (snapshot?.games || [])
    .filter((row) => String(row.draw_id || "") === selectedDrawId)
    .map((row) => ({ id: String(row.id || ""), updated_at: String(row.updated_at || "") }));
  const officialPublishReady = Boolean(snapshot?.operation_runtime?.official_publish_enabled);
  const guardedWriteDisabled = busy || !accessToken || !operationsWriteReady || !reviewedState;
  const drawCasWriteDisabled = guardedWriteDisabled || !reviewedDrawUpdatedAt;
  const teamSnapshotCasDisabled = drawCasWriteDisabled || !reviewedTeamVersions.length || reviewedTeamVersions.some((row) => !row.id || !row.updated_at);
  const gameSnapshotCasDisabled = teamSnapshotCasDisabled || !reviewedSourceGameVersions.length || reviewedSourceGameVersions.some((row) => !row.id || !row.updated_at);
  const shows = (name: Exclude<OpsWorkflow, "all">) => workflow === "all" || workflow === name;

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Tournament Ops.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = String(payload?.detail || `API error (${response.status})`);
      if (response.status === 409) {
        const recovery = /recovery|required|may already|identical request|response.?lost/i.test(detail);
        throw new Error(recovery
          ? `${detail} Keep this exact reviewed request intact and retry it only to reconcile, or use the Streamlit fallback.`
          : `${detail} Reload the authoritative Ops snapshot before submitting a new request.`);
      }
      throw new Error(detail);
    }
    return payload as T;
  }

  function clearProtectedOpsState() {
    snapshotRequest.invalidate();
    setBusy(false); setMessage(null);
    setTournaments([]); setSelectedTournamentId(""); setSelectedDrawId(""); setSnapshot(null);
    setDrawEventOptionId(""); setTeamRows(teamRowsFromTeams([], "")); setScoreGameId(""); setScoreA(""); setScoreB("");
    setResultsPreview(null); setResultsMappings({}); setResultsMatchReviews({}); setResultsPodiumRefs({}); setResultsReviewDirty(true);
  }

  function operationSuffix(payload: AdminTournamentWriteResponse): string {
    if (payload.reconciled) return ` Reconciled without repeating the domain write (${payload.operation_key?.slice(0, 12) || "operation"}).`;
    if (payload.idempotent_replay) return ` Idempotent replay; no second domain write (${payload.operation_key?.slice(0, 12) || "operation"}).`;
    return payload.operation_key ? ` Operation ${payload.operation_key.slice(0, 12)} recorded.` : "";
  }

  function resetTeamEditor(nextSnapshot: AdminTournamentOpsSnapshotResponse | null, drawId: string) {
    setTeamRows(teamRowsFromTeams(nextSnapshot?.teams || [], drawId));
  }

  function resetScoreEditor(nextSnapshot: AdminTournamentOpsSnapshotResponse | null) {
    const firstGame = (nextSnapshot?.games || [])[0] || null;
    setScoreGameId(firstGame ? String(firstGame.id || "") : "");
    setScoreA(firstGame?.score_a == null ? "" : String(firstGame.score_a));
    setScoreB(firstGame?.score_b == null ? "" : String(firstGame.score_b));
  }

  async function loadTournaments() {
    const selectedTournamentBeforeRefresh = selectedTournamentId;
    const selectedDrawBeforeRefresh = selectedDrawId;
    const generation = listRequest.begin();
    snapshotRequest.invalidate();
    setBusy(true);
    setMessage(null);
    setSnapshot(null);
    setResultsPreview(null);
    try {
      const suffix = includeArchived ? "?include_archived=true" : "";
      const payload = await requestJson<AdminTournamentListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/ops/tournaments${suffix}`);
      if (!listRequest.isCurrent(generation)) return;
      const nextTournaments = payload.tournaments || [];
      const selectionStillAvailable = Boolean(selectedTournamentBeforeRefresh && nextTournaments.some((row) => row.id === selectedTournamentBeforeRefresh));
      setTournaments(nextTournaments);
      setMessage(nextTournaments.length ? `Loaded ${payload.count ?? nextTournaments.length} tournament(s).` : "No tournaments match this view.");
      if (selectionStillAvailable) {
        const refreshedSnapshot = await loadOps(selectedTournamentBeforeRefresh, selectedDrawBeforeRefresh);
        if (
          selectedDrawBeforeRefresh
          && refreshedSnapshot
          && !refreshedSnapshot.draws.some((row) => row.id === selectedDrawBeforeRefresh)
          && listRequest.isCurrent(generation)
        ) {
          setSelectedDrawId("");
          await loadOps(selectedTournamentBeforeRefresh, "");
        }
      } else {
        setSelectedTournamentId("");
        setSelectedDrawId("");
        setDrawEventOptionId("");
      }
    } catch (error) {
      if (listRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load tournaments.");
    } finally {
      if (listRequest.isCurrent(generation)) setBusy(false);
    }
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
      setDrawEventOptionId((current) => payload.event_options?.some((row) => String(row.id || "") === current)
        ? current
        : String(payload.event_options?.[0]?.id || ""));
      resetTeamEditor(payload, drawId);
      resetScoreEditor(payload);
      setResultsPreview(null);
      setResultsReviewDirty(true);
      setMessage("Tournament operations snapshot loaded.");
      return payload;
    } catch (error) {
      if (snapshotRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to load tournament operations.");
      return null;
    } finally {
      if (snapshotRequest.isCurrent(generation)) setBusy(false);
    }
  }

  function selectTournament(tournamentId: string) {
    setSelectedTournamentId(tournamentId);
    setSelectedDrawId("");
    setSnapshot(null);
    setDrawEventOptionId("");
    setResultsPreview(null);
    setResultsReviewDirty(true);
    if (tournamentId) void loadOps(tournamentId, "");
    else snapshotRequest.invalidate();
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
      return;
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
      if (!actionRequest.isCurrent(generation)) return;
      const nextDrawId = payload.draw?.id || "";
      setSelectedDrawId(nextDrawId);
      await loadOps(tournamentId, nextDrawId);
      if (!actionRequest.isCurrent(generation)) return;
      setMessage(`Draw created${payload.draw?.name ? `: ${payload.draw.name}` : ""}.${operationSuffix(payload)}`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to create draw.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function importRegistrations(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before importing registrations.");
      return;
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/teams/import-registrations`, {
        method: "POST",
        body: JSON.stringify({ import_mode: registrationImportMode, expected_state_fingerprint: reviewedState, expected_draw_updated_at: reviewedDrawUpdatedAt, confirmation_text: confirmationText, source: "next_tournament_ops_import_registrations" })
      });
      if (!actionRequest.isCurrent(generation)) return;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return;
      setMessage(`Imported ${payload.updated_count ?? payload.teams?.length ?? 0} registration team(s) with ${payload.import_mode || registrationImportMode} mode.${operationSuffix(payload)}`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to import registrations.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function importBulkTeams(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before importing teams.");
      return;
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
      if (!actionRequest.isCurrent(generation)) return;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return;
      setMessage(`Imported ${payload.updated_count ?? payload.teams?.length ?? 0} bulk team(s) with ${payload.import_mode || bulkTeamMode} mode.${operationSuffix(payload)}`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to import bulk teams.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveTeams(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before saving teams.");
      return;
    }
    const teams = teamRows
      .filter((row) => row.player1_id.trim())
      .map((row, index) => ({ team_number: Number(row.team_number || index + 1), player1_id: Number(row.player1_id), player2_id: row.player2_id.trim() ? Number(row.player2_id) : null, seed: row.seed.trim() ? Number(row.seed) : null, source: "MANUAL", notes: row.notes }));
    if (!teams.length) {
      setMessage("Add at least one team with Player 1 before saving.");
      return;
    }
    if (teams.some((team) => !Number.isFinite(team.team_number) || !Number.isFinite(team.player1_id) || (team.player2_id !== null && !Number.isFinite(team.player2_id)))) {
      setMessage("Team number and player IDs must be numeric. Use player selectors when available.");
      return;
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
      if (!actionRequest.isCurrent(generation)) return;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return;
      setMessage(`Saved ${payload.updated_count ?? payload.teams?.length ?? teams.length} team(s).${operationSuffix(payload)}`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save teams.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function generateGames(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before generating games.");
      return;
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
      if (!actionRequest.isCurrent(generation)) return;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return;
      setMessage(`Generated ${payload.game_count ?? payload.games?.length ?? 0} round-robin game(s).${operationSuffix(payload)}`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to generate games.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function generatePlayoffs(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before generating playoffs.");
      return;
    }
    const advanceCount = Number(playoffAdvanceCount);
    if (!Number.isFinite(advanceCount)) {
      setMessage("Advance count must be numeric.");
      return;
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
      if (!actionRequest.isCurrent(generation)) return;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return;
      setMessage(`Generated ${payload.game_count ?? payload.games?.length ?? 0} playoff game(s).${operationSuffix(payload)}`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to generate playoffs.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function saveScore(confirmationText: string) {
    if (!selectedTournamentId || !scoreGameId) {
      setMessage("Select a game before saving a score.");
      return;
    }
    const nextA = Number(scoreA);
    const nextB = Number(scoreB);
    if (!Number.isFinite(nextA) || !Number.isFinite(nextB)) {
      setMessage("Both scores must be numeric.");
      return;
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    const gameId = scoreGameId;
    setBusy(true);
    setMessage(null);
    try {
      const selectedGame = (snapshot?.games || []).find((row) => String(row.id || "") === scoreGameId) || null;
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/games/${encodeURIComponent(gameId)}/score`, {
        method: "PATCH",
        body: JSON.stringify({ score_a: nextA, score_b: nextB, expected_state_fingerprint: reviewedState, expected_game_updated_at: String(selectedGame?.updated_at || "") || null, expected_draw_updated_at: reviewedDrawUpdatedAt, confirmation_text: confirmationText, source: "next_tournament_ops_score_game" })
      });
      if (!actionRequest.isCurrent(generation)) return;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return;
      setMessage(`Saved score for game ${String(payload.game?.id || gameId)}.${operationSuffix(payload)}`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to save score.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function generatePodium(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before generating a podium.");
      return;
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
      if (!actionRequest.isCurrent(generation)) return;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return;
      setMessage(`Generated ${payload.podium?.length ?? 0} ${payload.podium_source || "draw"} podium placement(s).${operationSuffix(payload)}`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to generate podium.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function awardPodium(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before awarding podium trophies.");
      return;
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/podium/awards`, {
        method: "POST",
        body: JSON.stringify({ expected_state_fingerprint: reviewedState, confirmation_text: confirmationText, source: "next_tournament_ops_award_podium" })
      });
      if (!actionRequest.isCurrent(generation)) return;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return;
      setMessage(`Awarded ${payload.awarded_count ?? 0} new badge(s) from ${payload.candidate_count ?? 0} podium candidate(s).${operationSuffix(payload)}`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to award podium trophies.");
    } finally {
      if (actionRequest.isCurrent(generation)) setBusy(false);
    }
  }

  async function publishOfficialMatches(confirmationText: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament and draw before publishing official matches.");
      return;
    }
    const bonusElo = Number(publishBonusElo || "0");
    if (!Number.isFinite(bonusElo) || bonusElo < 0) {
      setMessage("Playoff winner bonus must be a non-negative number.");
      return;
    }
    const generation = actionRequest.begin();
    const tournamentId = selectedTournamentId;
    const drawId = selectedDrawId;
    setBusy(true);
    setMessage(null);
    try {
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/draws/${encodeURIComponent(drawId)}/matches/publish`, {
        method: "POST",
        body: JSON.stringify({ confirmation_text: confirmationText, playoff_winner_bonus_elo: bonusElo, expected_state_fingerprint: reviewedState, source: "next_tournament_ops_publish_matches" })
      });
      if (!actionRequest.isCurrent(generation)) return;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return;
      setMessage(`Published ${payload.match_count ?? 0} official rating match(es). Bonus applied to ${payload.bonus_match_count ?? 0} medal-playoff match(es) at ${payload.playoff_winner_bonus_elo ?? bonusElo} Elo per winning player.${operationSuffix(payload)}`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to publish official tournament matches.");
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
      return;
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
          expected_review_fingerprint: resultsPreview.review_fingerprint,
          expected_state_fingerprint: reviewedState,
          expected_draw_updated_at: reviewedDrawUpdatedAt,
          confirmation_text: confirmationText,
          source: "next_tournament_ops_results_import"
        })
      });
      if (!actionRequest.isCurrent(generation)) return;
      await loadOps(tournamentId, drawId);
      if (!actionRequest.isCurrent(generation)) return;
      setResultsReviewDirty(true);
      setMessage(`Imported ${payload.game_count ?? 0} reviewed result(s), ${payload.team_count ?? 0} team(s), and ${payload.podium_count ?? 0} podium row(s).${operationSuffix(payload)}`);
    } catch (error) {
      if (actionRequest.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Unable to commit tournament results.");
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
    setScoreGameId(gameId);
    const game = (snapshot?.games || []).find((row) => String(row.id || "") === gameId) || null;
    setScoreA(game?.score_a == null ? "" : String(game.score_a));
    setScoreB(game?.score_b == null ? "" : String(game.score_b));
  }

  function updateTeamRow(index: number, patch: Partial<TeamEditorRow>) {
    setTeamRows((current) => current.map((row, rowIndex) => rowIndex === index ? { ...row, ...patch } : row));
  }

  function playerSelectValue(id: string) {
    return id || "";
  }

  useAuthenticatedAutoLoad(status.enabled ? accessToken : "", loadTournaments, includeArchived ? "archived" : "active");

  if (!status.enabled) {
    return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Next Tournament Admin is disabled</h2><p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the Tournament Admin pilot flag on FastAPI."}</p></article>;
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Tournament Ops</h2>
        <p style={{ color: "#475569" }}>Operations visibility plus guarded writes for creating draws, importing or maintaining teams, generating/scoring games, podiums, trophies, and official rating match publication.</p>
        <nav aria-label="Tournament operations workflows" style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          <Link href="/admin/tournaments/ops/draws">Draws and scoring</Link>
          <Link href="/admin/tournaments/ops/import">Team imports</Link>
          <Link href="/admin/tournaments/ops/results">Results CSV</Link>
          <Link href="/admin/tournaments/ops/publish">Official publish</Link>
          {workflow !== "all" ? <Link href="/admin/tournaments/ops">All operations</Link> : null}
        </nav>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>{accessToken ? "Ready to load guarded tournament operations data." : sessionLoading ? "Checking admin session…" : "Sign in before loading ops data."}</p>
          {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
          {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
        </div>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.75rem" }}><input type="checkbox" checked={includeArchived} onChange={(event) => setIncludeArchived(event.target.checked)} disabled={busy} />Include archived tournaments</label>
        <button type="button" onClick={loadTournaments} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Refreshing…" : "Refresh tournaments"}</button>
      </article>

      {!operationsWriteReady ? (
        <article data-testid="tournament-ops-read-only-banner" style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
          <h2 style={{ marginTop: 0 }}>Tournament Ops is read-only</h2>
          <p style={{ color: "#7c2d12", marginBottom: 0 }}>
            Tournament and draw snapshots remain available. POST-backed previews and mutation controls are hidden until the service role, dedicated Tournament Ops mutation flag, and operations runtime are all enabled in staging.
          </p>
        </article>
      ) : null}

      {tournaments.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Select tournament</h2>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) auto", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>Tournament</strong><br /><select value={selectedTournamentId} onChange={(event) => selectTournament(event.target.value)} disabled={busy} style={inputStyle}><option value="">Choose a tournament…</option>{tournaments.map((tournament) => <option key={tournament.id} value={tournament.id}>{tournament.name} · {tournament.status}</option>)}</select></label>
            <button type="button" onClick={() => loadOps()} disabled={busy || !selectedTournamentId} style={ghostButtonStyle}>Retry snapshot</button>
          </div>
        </article>
      ) : <article style={cardStyle}><p style={{ color: "#64748b" }}>{busy ? "Loading tournaments…" : "No tournaments match this view."}</p></article>}

      {operationsWriteReady && snapshot && shows("draws") ? (
        <article style={{ ...cardStyle, background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Create empty division draw</h2>
          <p style={{ color: "#475569" }}>This creates a DRAFT draw shell scoped to the selected registration division.</p>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(240px, 1fr) minmax(180px, 1fr)", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>Registration division</strong><br /><select value={drawEventOptionId} onChange={(event) => setDrawEventOptionId(event.target.value)} style={inputStyle}><option value="">Legacy / tournament-wide draw</option>{(snapshot.event_options || []).map((row) => <option key={String(row.id)} value={String(row.id)}>{eventOptionLabel(row)}</option>)}</select></label>
            <label><strong>Draw name</strong><br /><input value={drawName} onChange={(event) => setDrawName(event.target.value)} placeholder="optional" style={inputStyle} /></label>
          </div>
          <p><ConfirmAction triggerLabel="Create draw" title="Create this tournament draw?" description={`This creates a new draft draw${drawName.trim() ? ` named ${drawName.trim()}` : ""}${drawEventOptionId ? " for the selected registration division" : " for the tournament-wide legacy scope"}.`} confirmLabel="Yes, create draw" confirmationText="CREATE DRAW" disabled={!accessToken || !operationsWriteReady || !reviewedState} busy={busy} onConfirm={createDraw} /></p>
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
              <label><strong>Draw</strong><br /><select value={selectedDrawId} onChange={(event) => selectDraw(event.target.value)} disabled={busy} style={inputStyle}><option value="">Choose a draw…</option>{snapshot.draws.map((draw) => <option key={draw.id} value={draw.id}>{draw.name || draw.id}</option>)}</select></label>
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
            <p style={{ color: "#475569" }}>Paste CSV or TSV with headers like <code>Player 1, Player 2, Seed, Notes</code>. Player names or IDs must match the club roster. Import is blocked after games exist.</p>
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
            {selectedDrawId ? (
              <>
                <ConfirmAction triggerLabel="Save teams" title="Replace the draw's saved teams?" description="This saves the currently reviewed team rows as the authoritative team list for the selected draw." confirmLabel="Yes, save teams" confirmationText="SAVE TEAMS" tone="danger" disabled={drawCasWriteDisabled} busy={busy} onConfirm={saveTeams} />
                <div style={{ overflowX: "auto", marginTop: "1rem" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "920px" }}>
                    <thead><tr>{["Team #", "Player 1", "Player 2", "Seed", "Notes", "Action"].map((header) => <th key={header} style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>{header}</th>)}</tr></thead>
                    <tbody>{teamRows.map((row, index) => <tr key={row.editor_key}>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><input value={row.team_number} onChange={(event) => updateTeamRow(index, { team_number: event.target.value })} style={inputStyle} /></td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{snapshot.players?.length ? <select value={playerSelectValue(row.player1_id)} onChange={(event) => updateTeamRow(index, { player1_id: event.target.value })} style={inputStyle}><option value="">Choose player…</option>{snapshot.players.map((player) => <option key={player.id} value={String(player.id)}>{player.name}</option>)}</select> : <input value={row.player1_id} onChange={(event) => updateTeamRow(index, { player1_id: event.target.value })} placeholder="player id" style={inputStyle} />}</td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{snapshot.players?.length ? <select value={playerSelectValue(row.player2_id)} onChange={(event) => updateTeamRow(index, { player2_id: event.target.value })} style={inputStyle}><option value="">Singles / no partner</option>{snapshot.players.map((player) => <option key={player.id} value={String(player.id)}>{player.name}</option>)}</select> : <input value={row.player2_id} onChange={(event) => updateTeamRow(index, { player2_id: event.target.value })} placeholder="optional player id" style={inputStyle} />}</td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><input value={row.seed} onChange={(event) => updateTeamRow(index, { seed: event.target.value })} style={inputStyle} /></td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><input value={row.notes} onChange={(event) => updateTeamRow(index, { notes: event.target.value })} style={inputStyle} /></td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><button type="button" onClick={() => setTeamRows((current) => current.filter((_, rowIndex) => rowIndex !== index))} style={ghostButtonStyle}>Remove</button></td>
                    </tr>)}</tbody>
                  </table>
                </div>
                <p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}><button type="button" onClick={() => setTeamRows((current) => [...current, { editor_key: `new-team-${Date.now()}`, team_number: String(current.length + 1), player1_id: "", player2_id: "", seed: String(current.length + 1), notes: "" }])} style={ghostButtonStyle}>Add team row</button><button type="button" onClick={() => resetTeamEditor(snapshot, selectedDrawId)} style={ghostButtonStyle}>Reset from snapshot</button></p>
              </>
            ) : <p style={{ color: "#64748b" }}>Create or select a draw before editing teams.</p>}
          </article>

          <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Generate round-robin games</h2><p style={{ color: "#475569" }}>Generate the schedule only after teams are saved and team numbers are contiguous.</p><ConfirmAction triggerLabel="Generate games" title="Generate round-robin games?" description="This creates the schedule from the currently reviewed teams and draw version." confirmLabel="Yes, generate games" confirmationText="GENERATE GAMES" disabled={teamSnapshotCasDisabled || !selectedDrawId} busy={busy} onConfirm={generateGames} /></article>
          <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Score game</h2><p style={{ color: "#475569" }}>Select a game and enter the score. Ties are blocked; published, awarded, or downstream-finalized draws are locked.</p>{snapshot.games.length ? <div style={{ display: "grid", gridTemplateColumns: "minmax(260px, 1fr) minmax(100px, 140px) minmax(100px, 140px) auto", gap: "0.75rem", alignItems: "end" }}><label><strong>Game</strong><br /><select value={scoreGameId} onChange={(event) => selectScoreGame(event.target.value)} style={inputStyle}><option value="">Choose a game…</option>{snapshot.games.map((game) => <option key={String(game.id)} value={String(game.id)}>{gameLabel(game)}</option>)}</select></label><label><strong>Score A</strong><br /><input type="number" value={scoreA} onChange={(event) => setScoreA(event.target.value)} style={inputStyle} /></label><label><strong>Score B</strong><br /><input type="number" value={scoreB} onChange={(event) => setScoreB(event.target.value)} style={inputStyle} /></label><ConfirmAction triggerLabel="Save score" title="Save this game score?" description={`This records ${scoreA || "0"}–${scoreB || "0"} for the selected game.`} confirmLabel="Yes, save score" confirmationText="SAVE SCORE" disabled={drawCasWriteDisabled || !scoreGameId} busy={busy} onConfirm={saveScore} /></div> : <p style={{ color: "#64748b" }}>Generate games before scoring.</p>}</article>
          <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Generate playoffs</h2><p style={{ color: "#475569" }}>After all round-robin games are scored, choose how many teams advance.</p><div style={{ display: "grid", gridTemplateColumns: "minmax(140px, 180px) auto", gap: "0.75rem", alignItems: "end" }}><label><strong>Advance count</strong><br /><select value={playoffAdvanceCount} onChange={(event) => setPlayoffAdvanceCount(event.target.value)} style={inputStyle}><option value="4">4 teams</option><option value="5">5 teams</option><option value="6">6 teams</option></select></label><ConfirmAction triggerLabel="Generate playoffs" title="Generate the playoff bracket?" description={`This advances ${playoffAdvanceCount} teams from the reviewed round-robin results into the playoff bracket.`} confirmLabel="Yes, generate playoffs" confirmationText="GENERATE PLAYOFFS" disabled={gameSnapshotCasDisabled || !selectedDrawId} busy={busy} onConfirm={generatePlayoffs} /></div></article>
          <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Generate podium</h2><p style={{ color: "#475569" }}>Creates draw-scoped podium rows from finalized playoffs, or from completed round-robin standings when no playoffs exist.</p><ConfirmAction triggerLabel="Generate podium" title="Generate podium placements?" description="This calculates and stores podium rows from the currently reviewed final results." confirmLabel="Yes, generate podium" confirmationText="GENERATE PODIUM" disabled={gameSnapshotCasDisabled || !selectedDrawId} busy={busy} onConfirm={generatePodium} /></article>
          <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Award podium trophies</h2><p style={{ color: "#475569" }}>Awards draw-scoped tournament badges from generated podium rows. Re-running is idempotent for existing badge context.</p><ConfirmAction triggerLabel="Award podium" title="Award the generated podium trophies?" description="This mints draw-scoped tournament badges for the verified podium placements." confirmLabel="Yes, award podium" confirmationText="AWARD PODIUM" disabled={guardedWriteDisabled || !selectedDrawId} busy={busy} onConfirm={awardPodium} /></article>
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
              </p>

              {resultsPreview ? (
                <div style={{ display: "grid", gap: "1rem" }}>
                  <div style={{ padding: "0.75rem", background: resultsPreview.ok && !resultsReviewDirty ? "#f0fdf4" : "#fff7ed", borderRadius: "10px" }}>
                    <strong>{resultsPreview.ok ? "Preview parsed" : "Preview needs review"}</strong> · {resultsPreview.summary.imported_players} players · {resultsPreview.summary.teams} teams · {resultsPreview.summary.matches} matches · {resultsPreview.summary.create_players} proposed new players
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
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}>{String(player.display_name || player.name || importKey)}</td>
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
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}>{shortValue(match.team_a_label || match.team_a_ref)}</td>
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}>{shortValue(match.team_b_label || match.team_b_ref)}</td>
                          <td style={{ padding: "0.5rem", borderBottom: "1px solid #dbeafe" }}>{shortValue(match.score_a)}–{shortValue(match.score_b)}</td>
                        </tr>;
                      })}</tbody>
                    </table>
                  </div>

                  <div>
                    <h3>Podium review</h3>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>{["1", "2", "3"].map((placement) => <label key={placement}><strong>Place {placement}</strong><br /><select value={resultsPodiumRefs[placement] || ""} onChange={(event) => { setResultsPodiumRefs((current) => ({ ...current, [placement]: event.target.value || null })); setResultsReviewDirty(true); }} style={inputStyle}><option value="">No placement</option>{resultsPreview.podium_candidates.map((teamRef) => <option key={teamRef} value={teamRef}>{teamRef}</option>)}</select></label>)}</div>
                  </div>

                  {operationsWriteReady ? <div style={{ padding: "0.9rem", border: "1px solid #93c5fd", borderRadius: "10px", background: "white" }}>
                    <ConfirmAction triggerLabel="Commit reviewed results" title={`${resultsImportMode === "REPLACE" ? "Replace" : "Import"} the draw results?`} description={<>{resultsImportMode === "REPLACE" ? "This replaces the draw's teams, games, and podium with the exact reviewed CSV fingerprint." : "This appends the exact reviewed CSV results to the selected draw."}{resultsPreview.summary.create_players ? ` It also creates ${resultsPreview.summary.create_players} permanent player record${resultsPreview.summary.create_players === 1 ? "" : "s"} from the reviewed mappings.` : " It creates no new player records."}</>} confirmLabel={resultsImportMode === "REPLACE" ? "Yes, replace results" : "Yes, import results"} confirmationText={resultsImportMode === "REPLACE" ? "REPLACE RESULTS" : "IMPORT RESULTS"} tone={resultsImportMode === "REPLACE" ? "danger" : "default"} disabled={drawCasWriteDisabled || !selectedDrawId || !resultsPreview.ok || resultsReviewDirty} busy={busy} onConfirm={commitResultsImport} />
                  </div> : null}
                </div>
              ) : null}
            </article>
          ) : null}

          {operationsWriteReady && shows("publish") ? (
            <article style={{ ...cardStyle, background: "#fff7ed", borderColor: "#fed7aa" }}>
              <h2 style={{ marginTop: 0 }}>Publish official rating matches</h2>
              <p style={{ color: "#7c2d12" }}>Creates official Match Log rows from finalized tournament games and applies the regular rating path for both doubles and singles. Optional medal-playoff bonus adds Elo only to semifinal, bronze, and gold winners. Publishing needs both tournament and match-management permissions, a separate staging gate, and a safe email mode when automatic player updates are enabled.</p>
              {!officialPublishReady ? <p style={{ color: "#b91c1c", fontWeight: 700 }}>Official publish is gated off in this environment.</p> : null}
              <div style={{ display: "grid", gridTemplateColumns: "minmax(160px, 220px) auto", gap: "0.75rem", alignItems: "end" }}>
                <label><strong>Winner bonus Elo</strong><br /><input type="number" min="0" step="0.5" value={publishBonusElo} onChange={(event) => setPublishBonusElo(event.target.value)} style={inputStyle} /><small style={{ color: "#7c2d12" }}>4 Elo = +0.01 JUPR.</small></label>
                <ConfirmAction triggerLabel="Publish official matches" title="Publish these tournament games as official rated matches?" description={`This terminal write creates Match Log rows from every finalized game and applies a ${publishBonusElo || "0"}-Elo medal-playoff bonus to eligible winners.`} confirmLabel="Yes, publish official matches" confirmationText="PUBLISH MATCHES" tone="danger" disabled={guardedWriteDisabled || !officialPublishReady || !selectedDrawId} busy={busy} onConfirm={publishOfficialMatches} />
              </div>
            </article>
          ) : null}

          <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Draws</h2><GenericRowsTable rows={snapshot.draws} preferredColumns={["id", "name", "status", "registration_day_id", "event_option_id", "team_count"]} /></article>
          <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Teams</h2><GenericRowsTable rows={snapshot.teams} preferredColumns={["team_number", "player1_id", "player2_id", "source", "draw_id", "event_option_id"]} /></article>
          <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Games</h2><GenericRowsTable rows={snapshot.games} preferredColumns={["stage", "rr_round_number", "rr_slot_number", "playoff_game_code", "playoff_round", "team_a_id", "team_b_id", "score_a", "score_b", "winner_team_id", "status"]} /></article>
          <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Podium</h2><GenericRowsTable rows={snapshot.podium} preferredColumns={["placement", "team_id", "player1_id", "player2_id", "award_label", "draw_id"]} /></article>
        </>
      ) : null}

      {message ? <p role="status" aria-live="polite" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("error") || message.toLowerCase().includes("sign in") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
