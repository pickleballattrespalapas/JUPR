"use client";

import Link from "next/link";
import { useState } from "react";
import type {
  AdminTournament,
  AdminTournamentDetailResponse,
  AdminTournamentListResponse,
  AdminTournamentOpsSnapshotResponse,
  AdminTournamentOpsTeam,
  AdminTournamentStatusResponse,
  AdminTournamentWriteResponse
} from "@/lib/adminTournamentApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = { apiBase: string | null; clubId: string; status: AdminTournamentStatusResponse };
type TeamEditorRow = { team_number: string; player1_id: string; player2_id: string; seed: string; notes: string };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
function shortValue(value: unknown): string { if (value == null || value === "") return "—"; if (typeof value === "object") return JSON.stringify(value).slice(0, 160); return String(value); }
function eventOptionLabel(row: Record<string, unknown>): string { const family = String(row.event_family_label || "").trim(); const division = String(row.division_name || row.label || "").trim(); if (family && division && family !== division) return `${family} / ${division}`; return division || family || String(row.id || "Event"); }
function gameLabel(row: Record<string, unknown>): string { const stage = String(row.stage || "Game"); const round = row.rr_round_number ? `R${row.rr_round_number}` : String(row.playoff_round || ""); const slot = row.rr_slot_number ? `S${row.rr_slot_number}` : String(row.playoff_game_code || ""); const teams = `${shortValue(row.team_a_id)} vs ${shortValue(row.team_b_id)}`; return [stage, round, slot, teams].filter(Boolean).join(" · "); }

function teamRowsFromTeams(teams: AdminTournamentOpsTeam[], drawId: string): TeamEditorRow[] {
  const scoped = teams.filter((row) => !drawId || String(row.draw_id || "") === drawId).sort((left, right) => Number(left.team_number || 0) - Number(right.team_number || 0));
  if (!scoped.length) return [1, 2, 3, 4].map((teamNumber) => ({ team_number: String(teamNumber), player1_id: "", player2_id: "", seed: String(teamNumber), notes: "" }));
  return scoped.map((row, index) => ({ team_number: String(row.team_number || index + 1), player1_id: row.player1_id == null ? "" : String(row.player1_id), player2_id: row.player2_id == null ? "" : String(row.player2_id), seed: row.seed == null ? "" : String(row.seed), notes: row.notes || "" }));
}

function GenericRowsTable({ rows, preferredColumns }: { rows: Array<Record<string, unknown>>; preferredColumns: string[] }) {
  if (!rows.length) return <p style={{ color: "#64748b" }}>No rows loaded.</p>;
  const discovered = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
  const columns = [...preferredColumns.filter((key) => discovered.includes(key)), ...discovered.filter((key) => !preferredColumns.includes(key)).slice(0, 6)];
  return <div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "860px" }}><thead><tr>{columns.map((column) => <th key={column} style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>{column}</th>)}</tr></thead><tbody>{rows.map((row, index) => <tr key={String(row.id || index)}>{columns.map((column) => <td key={column} style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0", maxWidth: 260, overflowWrap: "anywhere" }}>{shortValue(row[column])}</td>)}</tr>)}</tbody></table></div>;
}

export default function TournamentOpsPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [includeArchived, setIncludeArchived] = useState(false);
  const [tournaments, setTournaments] = useState<AdminTournament[]>([]);
  const [selectedTournamentId, setSelectedTournamentId] = useState("");
  const [selectedDrawId, setSelectedDrawId] = useState("");
  const [detail, setDetail] = useState<AdminTournamentDetailResponse | null>(null);
  const [snapshot, setSnapshot] = useState<AdminTournamentOpsSnapshotResponse | null>(null);
  const [drawEventOptionId, setDrawEventOptionId] = useState("");
  const [drawName, setDrawName] = useState("");
  const [drawConfirm, setDrawConfirm] = useState("");
  const [teamRows, setTeamRows] = useState<TeamEditorRow[]>(() => teamRowsFromTeams([], ""));
  const [teamConfirm, setTeamConfirm] = useState("");
  const [gameConfirm, setGameConfirm] = useState("");
  const [scoreGameId, setScoreGameId] = useState("");
  const [scoreA, setScoreA] = useState("");
  const [scoreB, setScoreB] = useState("");
  const [scoreConfirm, setScoreConfirm] = useState("");
  const [playoffAdvanceCount, setPlayoffAdvanceCount] = useState("4");
  const [playoffConfirm, setPlayoffConfirm] = useState("");
  const [podiumConfirm, setPodiumConfirm] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Tournament Ops.");
    const headers = new Headers(options?.headers); headers.set("Authorization", `Bearer ${accessToken}`); if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function resetTeamEditor(nextSnapshot: AdminTournamentOpsSnapshotResponse | null, drawId: string) { setTeamRows(teamRowsFromTeams(nextSnapshot?.teams || [], drawId)); setTeamConfirm(""); }
  function resetScoreEditor(nextSnapshot: AdminTournamentOpsSnapshotResponse | null) { const firstGame = (nextSnapshot?.games || [])[0] || null; setScoreGameId(firstGame ? String(firstGame.id || "") : ""); setScoreA(firstGame?.score_a == null ? "" : String(firstGame.score_a)); setScoreB(firstGame?.score_b == null ? "" : String(firstGame.score_b)); setScoreConfirm(""); }

  async function loadTournaments() {
    setBusy(true); setMessage(null); setSnapshot(null); setDetail(null);
    try { const suffix = includeArchived ? "?include_archived=true" : ""; const payload = await requestJson<AdminTournamentListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments${suffix}`); setTournaments(payload.tournaments || []); setMessage(`Loaded ${payload.count ?? payload.tournaments?.length ?? 0} tournament(s).`); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load tournaments."); }
    finally { setBusy(false); }
  }

  async function loadTournamentDetail(tournamentId: string): Promise<AdminTournamentDetailResponse> { const payload = await requestJson<AdminTournamentDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`); setDetail(payload); if (!drawEventOptionId && payload.event_options?.length) setDrawEventOptionId(String(payload.event_options[0].id || "")); return payload; }

  async function loadOps(tournamentId = selectedTournamentId, drawId = selectedDrawId) {
    if (!tournamentId) { setMessage("Select a tournament first."); return; }
    setBusy(true); setMessage(null);
    try { await loadTournamentDetail(tournamentId); const params = new URLSearchParams(); if (drawId) params.set("draw_id", drawId); const suffix = params.toString() ? `?${params.toString()}` : ""; const payload = await requestJson<AdminTournamentOpsSnapshotResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/ops${suffix}`); setSnapshot(payload); resetTeamEditor(payload, drawId); resetScoreEditor(payload); setGameConfirm(""); setPlayoffConfirm(""); setPodiumConfirm(""); setMessage("Tournament operations snapshot loaded."); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load tournament operations."); }
    finally { setBusy(false); }
  }

  async function createDraw() {
    if (!selectedTournamentId) { setMessage("Select a tournament before creating a draw."); return; }
    setBusy(true); setMessage(null);
    try { const selectedEvent = detail?.event_options?.find((row) => String(row.id || "") === drawEventOptionId) || null; const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/draws`, { method: "POST", body: JSON.stringify({ event_option_id: drawEventOptionId || null, registration_day_id: String(selectedEvent?.registration_day_id || "") || null, name: drawName, confirmation_text: drawConfirm, source: "next_tournament_ops_create_draw" }) }); const nextDrawId = payload.draw?.id || ""; setSelectedDrawId(nextDrawId); setDrawConfirm(""); await loadOps(selectedTournamentId, nextDrawId); setMessage(`Draw created${payload.draw?.name ? `: ${payload.draw.name}` : ""}.`); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to create draw."); }
    finally { setBusy(false); }
  }

  async function saveTeams() {
    if (!selectedTournamentId || !selectedDrawId) { setMessage("Select a tournament and draw before saving teams."); return; }
    const teams = teamRows.filter((row) => row.player1_id.trim()).map((row, index) => ({ team_number: Number(row.team_number || index + 1), player1_id: Number(row.player1_id), player2_id: row.player2_id.trim() ? Number(row.player2_id) : null, seed: row.seed.trim() ? Number(row.seed) : null, source: "MANUAL", notes: row.notes }));
    if (!teams.length) { setMessage("Add at least one team with Player 1 before saving."); return; }
    if (teams.some((team) => !Number.isFinite(team.team_number) || !Number.isFinite(team.player1_id) || (team.player2_id !== null && !Number.isFinite(team.player2_id)))) { setMessage("Team number and player IDs must be numeric. Use player selectors when available."); return; }
    setBusy(true); setMessage(null);
    try { const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/draws/${encodeURIComponent(selectedDrawId)}/teams`, { method: "PUT", body: JSON.stringify({ teams, confirmation_text: teamConfirm, source: "next_tournament_ops_team_editor" }) }); await loadOps(selectedTournamentId, selectedDrawId); setTeamConfirm(""); setMessage(`Saved ${payload.updated_count ?? payload.teams?.length ?? teams.length} team(s).`); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to save teams."); }
    finally { setBusy(false); }
  }

  async function generateGames() {
    if (!selectedTournamentId || !selectedDrawId) { setMessage("Select a tournament and draw before generating games."); return; }
    setBusy(true); setMessage(null);
    try { const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/draws/${encodeURIComponent(selectedDrawId)}/games/round-robin`, { method: "POST", body: JSON.stringify({ confirmation_text: gameConfirm, source: "next_tournament_ops_generate_round_robin" }) }); await loadOps(selectedTournamentId, selectedDrawId); setGameConfirm(""); setMessage(`Generated ${payload.game_count ?? payload.games?.length ?? 0} round-robin game(s).`); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to generate games."); }
    finally { setBusy(false); }
  }

  async function generatePlayoffs() {
    if (!selectedTournamentId || !selectedDrawId) { setMessage("Select a tournament and draw before generating playoffs."); return; }
    const advanceCount = Number(playoffAdvanceCount);
    if (!Number.isFinite(advanceCount)) { setMessage("Advance count must be numeric."); return; }
    setBusy(true); setMessage(null);
    try { const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/draws/${encodeURIComponent(selectedDrawId)}/games/playoffs`, { method: "POST", body: JSON.stringify({ advance_count: advanceCount, confirmation_text: playoffConfirm, source: "next_tournament_ops_generate_playoffs" }) }); await loadOps(selectedTournamentId, selectedDrawId); setPlayoffConfirm(""); setMessage(`Generated ${payload.game_count ?? payload.games?.length ?? 0} playoff game(s).`); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to generate playoffs."); }
    finally { setBusy(false); }
  }

  async function saveScore() {
    if (!selectedTournamentId || !scoreGameId) { setMessage("Select a game before saving a score."); return; }
    const nextA = Number(scoreA); const nextB = Number(scoreB);
    if (!Number.isFinite(nextA) || !Number.isFinite(nextB)) { setMessage("Both scores must be numeric."); return; }
    setBusy(true); setMessage(null);
    try { const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/games/${encodeURIComponent(scoreGameId)}/score`, { method: "PATCH", body: JSON.stringify({ score_a: nextA, score_b: nextB, confirmation_text: scoreConfirm, source: "next_tournament_ops_score_game" }) }); await loadOps(selectedTournamentId, selectedDrawId); setScoreConfirm(""); setMessage(`Saved score for game ${String(payload.game?.id || scoreGameId)}.`); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to save score."); }
    finally { setBusy(false); }
  }

  async function generatePodium() {
    if (!selectedTournamentId || !selectedDrawId) { setMessage("Select a tournament and draw before generating a podium."); return; }
    setBusy(true); setMessage(null);
    try { const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/draws/${encodeURIComponent(selectedDrawId)}/podium`, { method: "POST", body: JSON.stringify({ confirmation_text: podiumConfirm, source: "next_tournament_ops_generate_podium" }) }); await loadOps(selectedTournamentId, selectedDrawId); setPodiumConfirm(""); setMessage(`Generated ${payload.podium?.length ?? 0} ${payload.podium_source || "draw"} podium placement(s).`); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to generate podium."); }
    finally { setBusy(false); }
  }

  function selectScoreGame(gameId: string) { setScoreGameId(gameId); const game = (snapshot?.games || []).find((row) => String(row.id || "") === gameId) || null; setScoreA(game?.score_a == null ? "" : String(game.score_a)); setScoreB(game?.score_b == null ? "" : String(game.score_b)); setScoreConfirm(""); }
  function updateTeamRow(index: number, patch: Partial<TeamEditorRow>) { setTeamRows((current) => current.map((row, rowIndex) => rowIndex === index ? { ...row, ...patch } : row)); }
  function playerSelectValue(id: string) { return id || ""; }

  if (!status.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Next Tournament Admin is disabled</h2><p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the Tournament Admin pilot flag on FastAPI."}</p></article>;

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Tournament Ops</h2>
        <p style={{ color: "#475569" }}>Operations visibility plus guarded writes for creating draws, maintaining teams, generating/scoring round-robin games, generating/scoring playoff games, and generating draw-scoped podiums. Awards remain Streamlit-only until their write contract is ported.</p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}><strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong><p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>{accessToken ? "Ready to load guarded tournament operations data." : sessionLoading ? "Checking admin session…" : "Sign in before loading ops data."}</p>{sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}{!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}</div>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.75rem" }}><input type="checkbox" checked={includeArchived} onChange={(event) => setIncludeArchived(event.target.checked)} />Include archived tournaments</label>
        <button type="button" onClick={loadTournaments} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Working…" : "Load tournaments"}</button>
      </article>

      {tournaments.length ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Select tournament</h2><div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) minmax(160px, 240px) auto", gap: "0.75rem", alignItems: "end" }}><label><strong>Tournament</strong><br /><select value={selectedTournamentId} onChange={(event) => { setSelectedTournamentId(event.target.value); setSelectedDrawId(""); setSnapshot(null); setDetail(null); setDrawEventOptionId(""); }} style={inputStyle}><option value="">Choose a tournament…</option>{tournaments.map((tournament) => <option key={tournament.id} value={tournament.id}>{tournament.name} · {tournament.status}</option>)}</select></label><label><strong>Draw ID filter</strong><br /><input value={selectedDrawId} onChange={(event) => setSelectedDrawId(event.target.value)} placeholder="optional" style={inputStyle} /></label><button type="button" onClick={() => loadOps()} disabled={busy || !selectedTournamentId} style={ghostButtonStyle}>Load ops snapshot</button></div></article> : null}

      {detail ? <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Create empty division draw</h2><p style={{ color: "#475569" }}>This creates a DRAFT draw shell scoped to the selected registration division. It does not import teams or generate games.</p><div style={{ display: "grid", gridTemplateColumns: "minmax(240px, 1fr) minmax(180px, 1fr) minmax(160px, 220px)", gap: "0.75rem", alignItems: "end" }}><label><strong>Registration division</strong><br /><select value={drawEventOptionId} onChange={(event) => setDrawEventOptionId(event.target.value)} style={inputStyle}><option value="">Legacy / tournament-wide draw</option>{detail.event_options.map((row) => <option key={String(row.id)} value={String(row.id)}>{eventOptionLabel(row)}</option>)}</select></label><label><strong>Draw name</strong><br /><input value={drawName} onChange={(event) => setDrawName(event.target.value)} placeholder="optional" style={inputStyle} /></label><label><strong>Type CREATE DRAW</strong><br /><input value={drawConfirm} onChange={(event) => setDrawConfirm(event.target.value)} style={inputStyle} /></label></div><p><button type="button" onClick={createDraw} disabled={busy || !accessToken || drawConfirm.trim().toUpperCase() !== "CREATE DRAW"} style={buttonStyle}>{busy ? "Creating…" : "Create draw"}</button></p></article> : null}

      {snapshot ? <>
        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>{snapshot.tournament.name}</h2><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem" }}><div><strong>Draws</strong><br />{snapshot.summary.draws}</div><div><strong>Teams</strong><br />{snapshot.summary.teams}</div><div><strong>Games</strong><br />{snapshot.summary.games}</div><div><strong>Completed games</strong><br />{snapshot.summary.completed_games ?? 0}</div><div><strong>Podium rows</strong><br />{snapshot.summary.podium}</div></div>{snapshot.warnings?.length ? <ul style={{ color: "#92400e" }}>{snapshot.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}</article>

        <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Team editor</h2><p style={{ color: "#475569" }}>Select a DRAFT draw, assign players, then type <code>SAVE TEAMS</code>. Saving replaces the current teams for that draw and is blocked once games exist.</p><div style={{ display: "grid", gridTemplateColumns: "minmax(240px, 1fr) minmax(160px, 220px) auto", gap: "0.75rem", alignItems: "end" }}><label><strong>Draw</strong><br /><select value={selectedDrawId} onChange={(event) => { setSelectedDrawId(event.target.value); resetTeamEditor(snapshot, event.target.value); }} style={inputStyle}><option value="">Choose a draw…</option>{snapshot.draws.map((draw) => <option key={draw.id} value={draw.id}>{draw.name || draw.id}</option>)}</select></label><label><strong>Type SAVE TEAMS</strong><br /><input value={teamConfirm} onChange={(event) => setTeamConfirm(event.target.value)} style={inputStyle} /></label><button type="button" onClick={() => selectedTournamentId && selectedDrawId ? loadOps(selectedTournamentId, selectedDrawId) : undefined} disabled={!selectedTournamentId || !selectedDrawId || busy} style={ghostButtonStyle}>Reload selected draw</button></div>{selectedDrawId ? <><div style={{ overflowX: "auto", marginTop: "1rem" }}><table style={{ width: "100%", borderCollapse: "collapse", minWidth: "920px" }}><thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Team #</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player 1</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Player 2</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Seed</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Notes</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Action</th></tr></thead><tbody>{teamRows.map((row, index) => <tr key={index}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><input value={row.team_number} onChange={(event) => updateTeamRow(index, { team_number: event.target.value })} style={inputStyle} /></td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{snapshot.players?.length ? <select value={playerSelectValue(row.player1_id)} onChange={(event) => updateTeamRow(index, { player1_id: event.target.value })} style={inputStyle}><option value="">Choose player…</option>{snapshot.players.map((player) => <option key={player.id} value={String(player.id)}>{player.name}</option>)}</select> : <input value={row.player1_id} onChange={(event) => updateTeamRow(index, { player1_id: event.target.value })} placeholder="player id" style={inputStyle} />}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{snapshot.players?.length ? <select value={playerSelectValue(row.player2_id)} onChange={(event) => updateTeamRow(index, { player2_id: event.target.value })} style={inputStyle}><option value="">Singles / no partner</option>{snapshot.players.map((player) => <option key={player.id} value={String(player.id)}>{player.name}</option>)}</select> : <input value={row.player2_id} onChange={(event) => updateTeamRow(index, { player2_id: event.target.value })} placeholder="optional player id" style={inputStyle} />}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><input value={row.seed} onChange={(event) => updateTeamRow(index, { seed: event.target.value })} style={inputStyle} /></td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><input value={row.notes} onChange={(event) => updateTeamRow(index, { notes: event.target.value })} style={inputStyle} /></td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}><button type="button" onClick={() => setTeamRows((current) => current.filter((_, rowIndex) => rowIndex !== index))} style={ghostButtonStyle}>Remove</button></td></tr>)}</tbody></table></div><p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}><button type="button" onClick={() => setTeamRows((current) => [...current, { team_number: String(current.length + 1), player1_id: "", player2_id: "", seed: String(current.length + 1), notes: "" }])} style={ghostButtonStyle}>Add team row</button><button type="button" onClick={() => resetTeamEditor(snapshot, selectedDrawId)} style={ghostButtonStyle}>Reset from snapshot</button><button type="button" onClick={saveTeams} disabled={busy || !accessToken || teamConfirm.trim().toUpperCase() !== "SAVE TEAMS"} style={buttonStyle}>{busy ? "Saving…" : "Save teams"}</button></p></> : <p style={{ color: "#64748b" }}>Create or select a draw before editing teams.</p>}</article>

        <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Generate round-robin games</h2><p style={{ color: "#475569" }}>After teams are saved and team numbers are contiguous, type <code>GENERATE GAMES</code>. Generation is blocked if games already exist for the selected draw.</p><div style={{ display: "grid", gridTemplateColumns: "minmax(180px, 260px) auto", gap: "0.75rem", alignItems: "end" }}><label><strong>Type GENERATE GAMES</strong><br /><input value={gameConfirm} onChange={(event) => setGameConfirm(event.target.value)} style={inputStyle} /></label><button type="button" onClick={generateGames} disabled={busy || !accessToken || !selectedDrawId || gameConfirm.trim().toUpperCase() !== "GENERATE GAMES"} style={buttonStyle}>{busy ? "Generating…" : "Generate games"}</button></div></article>
        <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Score game</h2><p style={{ color: "#475569" }}>Select a generated game, enter the score, then type <code>SAVE SCORE</code>. Ties are blocked. Playoff results feed forward through dependent bracket games.</p>{snapshot.games.length ? <div style={{ display: "grid", gridTemplateColumns: "minmax(260px, 1fr) minmax(100px, 140px) minmax(100px, 140px) minmax(160px, 220px) auto", gap: "0.75rem", alignItems: "end" }}><label><strong>Game</strong><br /><select value={scoreGameId} onChange={(event) => selectScoreGame(event.target.value)} style={inputStyle}><option value="">Choose a game…</option>{snapshot.games.map((game) => <option key={String(game.id)} value={String(game.id)}>{gameLabel(game)}</option>)}</select></label><label><strong>Score A</strong><br /><input type="number" value={scoreA} onChange={(event) => setScoreA(event.target.value)} style={inputStyle} /></label><label><strong>Score B</strong><br /><input type="number" value={scoreB} onChange={(event) => setScoreB(event.target.value)} style={inputStyle} /></label><label><strong>Type SAVE SCORE</strong><br /><input value={scoreConfirm} onChange={(event) => setScoreConfirm(event.target.value)} style={inputStyle} /></label><button type="button" onClick={saveScore} disabled={busy || !accessToken || !scoreGameId || scoreConfirm.trim().toUpperCase() !== "SAVE SCORE"} style={buttonStyle}>{busy ? "Saving…" : "Save score"}</button></div> : <p style={{ color: "#64748b" }}>Generate games before scoring.</p>}</article>
        <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Generate playoffs</h2><p style={{ color: "#475569" }}>After all round-robin games are scored, choose how many teams advance and type <code>GENERATE PLAYOFFS</code>. Generation is blocked if playoff games already exist.</p><div style={{ display: "grid", gridTemplateColumns: "minmax(140px, 180px) minmax(180px, 260px) auto", gap: "0.75rem", alignItems: "end" }}><label><strong>Advance count</strong><br /><select value={playoffAdvanceCount} onChange={(event) => setPlayoffAdvanceCount(event.target.value)} style={inputStyle}><option value="4">4 teams</option><option value="5">5 teams</option><option value="6">6 teams</option></select></label><label><strong>Type GENERATE PLAYOFFS</strong><br /><input value={playoffConfirm} onChange={(event) => setPlayoffConfirm(event.target.value)} style={inputStyle} /></label><button type="button" onClick={generatePlayoffs} disabled={busy || !accessToken || !selectedDrawId || playoffConfirm.trim().toUpperCase() !== "GENERATE PLAYOFFS"} style={buttonStyle}>{busy ? "Generating…" : "Generate playoffs"}</button></div></article>
        <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Generate draw podium</h2><p style={{ color: "#475569" }}>For playoff draws, Final and Bronze games must be scored. For round-robin-only draws, all round-robin games must be scored. Podium rows are scoped to the selected draw/division.</p><div style={{ display: "grid", gridTemplateColumns: "minmax(180px, 260px) auto", gap: "0.75rem", alignItems: "end" }}><label><strong>Type GENERATE PODIUM</strong><br /><input value={podiumConfirm} onChange={(event) => setPodiumConfirm(event.target.value)} style={inputStyle} /></label><button type="button" onClick={generatePodium} disabled={busy || !accessToken || !selectedDrawId || podiumConfirm.trim().toUpperCase() !== "GENERATE PODIUM"} style={buttonStyle}>{busy ? "Generating…" : "Generate podium"}</button></div></article>

        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Draws</h2><GenericRowsTable rows={snapshot.draws} preferredColumns={["id", "name", "status", "registration_day_id", "event_option_id", "team_count"]} /></article>
        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Teams</h2><GenericRowsTable rows={snapshot.teams} preferredColumns={["team_number", "player1_id", "player2_id", "source", "draw_id", "event_option_id"]} /></article>
        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Games</h2><GenericRowsTable rows={snapshot.games} preferredColumns={["stage", "rr_round_number", "rr_slot_number", "playoff_game_code", "playoff_round", "team_a_id", "team_b_id", "score_a", "score_b", "winner_team_id", "status"]} /></article>
        <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Podium</h2><GenericRowsTable rows={snapshot.podium} preferredColumns={["placement", "team_id", "source", "draw_id", "id"]} /></article>
      </> : null}

      {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("error") || message.toLowerCase().includes("sign in") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
