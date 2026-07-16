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

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const dangerButtonStyle = { ...buttonStyle, background: "#991b1b", borderColor: "#991b1b" };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
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
  return numericValue(game.score_a) != null && numericValue(game.score_b) != null && Boolean(game.winner_team_id);
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

function eventOptionLabel(row: Record<string, unknown>): string {
  const family = String(row.event_family_label || row.label || "").trim();
  const division = String(row.division_name || row.label || "").trim();
  if (family && division && family !== division) return `${family} / ${division}`;
  return division || family || String(row.id || "Division");
}

function drawLabel(draw: Record<string, unknown>, detail: AdminTournamentDetailResponse | null): string {
  const event = detail?.event_options?.find((row) => String(row.id || "") === String(draw.event_option_id || ""));
  const eventText = event ? eventOptionLabel(event) : "Unscoped draw";
  return `${draw.name || "Draw"} · ${eventText} · ${draw.status || "draft"}`;
}

function playerLabel(players: AdminTournamentOpsSnapshotResponse["players"], playerId?: number | null): string {
  if (playerId == null) return "—";
  const match = (players || []).find((player) => Number(player.id) === Number(playerId));
  return match ? `${match.name} (#${match.id})` : `#${playerId}`;
}

function teamLabel(team: AdminTournamentOpsTeam | undefined, snapshot: AdminTournamentOpsSnapshotResponse | null): string {
  if (!team) return "Unassigned";
  const p1 = playerLabel(snapshot?.players, team.player1_id ?? null);
  const p2 = team.player2_id == null ? null : playerLabel(snapshot?.players, team.player2_id);
  const members = p2 ? `${p1} / ${p2}` : p1;
  return `Team ${team.team_number ?? "?"}: ${members}`;
}

function statusChip(scored: boolean) {
  return scored
    ? <span style={{ border: "1px solid #bbf7d0", borderRadius: "999px", padding: "0.15rem 0.45rem", background: "#dcfce7" }}>scored</span>
    : <span style={{ border: "1px solid #fde68a", borderRadius: "999px", padding: "0.15rem 0.45rem", background: "#fef3c7" }}>open</span>;
}

export default function TournamentLivePanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [includeArchived, setIncludeArchived] = useState(false);
  const [tournaments, setTournaments] = useState<AdminTournament[]>([]);
  const [selectedTournamentId, setSelectedTournamentId] = useState("");
  const [selectedDrawId, setSelectedDrawId] = useState("");
  const [detail, setDetail] = useState<AdminTournamentDetailResponse | null>(null);
  const [snapshot, setSnapshot] = useState<AdminTournamentOpsSnapshotResponse | null>(null);
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
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const selectedTournament = tournaments.find((tournament) => tournament.id === selectedTournamentId) || detail?.tournament || null;
  const teamsById = new Map((snapshot?.teams || []).map((team) => [String(team.id || ""), team]));
  const sortedGames = [...(snapshot?.games || [])].sort((left, right) => gameSortKey(left).localeCompare(gameSortKey(right)));
  const rrGames = sortedGames.filter((game) => String(game.stage || "") === "ROUND_ROBIN");
  const playoffGames = sortedGames.filter((game) => String(game.stage || "") === "PLAYOFF");
  const completedCount = sortedGames.filter(isScored).length;
  const openCount = Math.max(0, sortedGames.length - completedCount);
  const nextOpenGame = sortedGames.find((game) => !isScored(game)) || sortedGames[0] || null;
  const selectedGame = sortedGames.find((game) => String(game.id || "") === scoreGameId) || nextOpenGame;

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

  function seedScoreEditor(nextSnapshot: AdminTournamentOpsSnapshotResponse | null) {
    const games = [...(nextSnapshot?.games || [])].sort((left, right) => gameSortKey(left).localeCompare(gameSortKey(right)));
    const target = games.find((game) => !isScored(game)) || games[0] || null;
    setScoreGameId(target ? String(target.id || "") : "");
    setScoreA(target?.score_a == null ? "" : String(target.score_a));
    setScoreB(target?.score_b == null ? "" : String(target.score_b));
    setScoreConfirm("");
  }

  async function loadTournaments() {
    setBusy(true);
    setMessage(null);
    setSnapshot(null);
    setDetail(null);
    try {
      const suffix = includeArchived ? "?include_archived=true" : "";
      const payload = await requestJson<AdminTournamentListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments${suffix}`);
      setTournaments(payload.tournaments || []);
      setMessage(`Loaded ${payload.count ?? payload.tournaments?.length ?? 0} tournament(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load tournaments.");
    } finally {
      setBusy(false);
    }
  }

  async function loadTournamentDetail(tournamentId: string): Promise<AdminTournamentDetailResponse> {
    const payload = await requestJson<AdminTournamentDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}`);
    setDetail(payload);
    return payload;
  }

  async function loadLiveBoard(tournamentId = selectedTournamentId, requestedDrawId = selectedDrawId) {
    if (!tournamentId) {
      setMessage("Select a tournament first.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const nextDetail = await loadTournamentDetail(tournamentId);
      const params = new URLSearchParams();
      if (requestedDrawId) params.set("draw_id", requestedDrawId);
      const initial = await requestJson<AdminTournamentOpsSnapshotResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/ops${params.toString() ? `?${params.toString()}` : ""}`);
      let effectiveSnapshot = initial;
      let effectiveDrawId = requestedDrawId;
      if (!effectiveDrawId && initial.draws?.length) {
        effectiveDrawId = String(initial.draws[0].id || "");
        const scoped = await requestJson<AdminTournamentOpsSnapshotResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/ops?draw_id=${encodeURIComponent(effectiveDrawId)}`);
        effectiveSnapshot = scoped;
      }
      setSelectedTournamentId(tournamentId);
      setSelectedDrawId(effectiveDrawId || "");
      setDetail(nextDetail);
      setSnapshot(effectiveSnapshot);
      seedScoreEditor(effectiveSnapshot);
      setRoundRobinConfirm("");
      setPlayoffConfirm("");
      setPodiumConfirm("");
      setAwardConfirm("");
      setPublishConfirm("");
      setMessage("Tournament Live board loaded.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load Tournament Live board.");
    } finally {
      setBusy(false);
    }
  }

  async function liveAction(path: string, body: Record<string, unknown>, success: string) {
    if (!selectedTournamentId || !selectedDrawId) {
      setMessage("Select a tournament draw before running Tournament Live actions.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<AdminTournamentWriteResponse>(path, { method: "POST", body: JSON.stringify(body) });
      await loadLiveBoard(selectedTournamentId, selectedDrawId);
      setMessage(success);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Tournament Live action failed.");
    } finally {
      setBusy(false);
    }
  }

  async function saveScore() {
    if (!selectedTournamentId || !selectedGame) {
      setMessage("Select a game before saving a score.");
      return;
    }
    const a = Number(scoreA);
    const b = Number(scoreB);
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isInteger(a) || !Number.isInteger(b) || a < 0 || b < 0 || a === b) {
      setMessage("Enter two non-tied whole-number scores before saving.");
      return;
    }
    if (scoreConfirm.trim().toUpperCase() !== "SAVE SCORE") {
      setMessage("Type SAVE SCORE to confirm this game result.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/games/${encodeURIComponent(String(selectedGame.id || ""))}/score`, {
        method: "PATCH",
        body: JSON.stringify({ score_a: a, score_b: b, confirmation_text: scoreConfirm, source: "next_tournament_live_score" })
      });
      await loadLiveBoard(selectedTournamentId, selectedDrawId);
      setMessage("Score saved and bracket dependencies updated if needed.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save score.");
    } finally {
      setBusy(false);
    }
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Tournament Live is disabled</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable Tournament Admin/Ops on FastAPI to use this live runner."}</p>
      </article>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Tournament Live session</h2>
        <p style={{ color: "#475569" }}>
          This is the tournament-specific live runner. Use Tournament Ops for setup/import/build tasks; use this page during play to run a selected draw, enter scores, progress brackets, create podiums, and publish official matches.
        </p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
            {accessToken ? "Ready to run Tournament Live actions through FastAPI." : sessionLoading ? "Checking admin session…" : "Sign in before using Tournament Live."}
          </p>
          {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
          {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
        </div>
        <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center", marginRight: "0.75rem" }}>
          <input type="checkbox" checked={includeArchived} onChange={(event) => setIncludeArchived(event.target.checked)} /> Include archived
        </label>
        <button type="button" onClick={loadTournaments} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Working…" : "Load tournaments"}</button>
      </article>

      {tournaments.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Select tournament draw</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>Tournament</strong><br />
              <select value={selectedTournamentId} onChange={(event) => { setSelectedTournamentId(event.target.value); setSelectedDrawId(""); setSnapshot(null); setDetail(null); }} style={inputStyle}>
                <option value="">Choose tournament…</option>
                {tournaments.map((tournament) => <option key={tournament.id} value={tournament.id}>{tournament.name} · {tournament.status}</option>)}
              </select>
            </label>
            {snapshot?.draws?.length ? (
              <label><strong>Draw</strong><br />
                <select value={selectedDrawId} onChange={(event) => { setSelectedDrawId(event.target.value); void loadLiveBoard(selectedTournamentId, event.target.value); }} style={inputStyle}>
                  {snapshot.draws.map((draw) => <option key={draw.id} value={draw.id}>{drawLabel(draw as Record<string, unknown>, detail)}</option>)}
                </select>
              </label>
            ) : null}
            <button type="button" onClick={() => loadLiveBoard()} disabled={busy || !selectedTournamentId || !accessToken} style={ghostButtonStyle}>Open live board</button>
          </div>
        </article>
      ) : null}

      {snapshot ? (
        <>
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>{selectedTournament?.name || snapshot.tournament?.name || "Tournament"}</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.75rem" }}>
              <div><strong>Status</strong><br />{snapshot.tournament?.status || "—"}</div>
              <div><strong>Teams</strong><br />{snapshot.summary.teams}</div>
              <div><strong>Games</strong><br />{snapshot.summary.games}</div>
              <div><strong>Scored</strong><br />{completedCount}</div>
              <div><strong>Open</strong><br />{openCount}</div>
              <div><strong>RR / Playoff</strong><br />{rrGames.length} / {playoffGames.length}</div>
              <div><strong>Podium rows</strong><br />{snapshot.summary.podium}</div>
            </div>
            {snapshot.warnings?.length ? <ul style={{ color: "#92400e" }}>{snapshot.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Live score entry</h2>
            {!sortedGames.length ? <p style={{ color: "#64748b" }}>No games exist for this draw yet. Generate round-robin games after teams are ready.</p> : null}
            {sortedGames.length ? (
              <div style={{ display: "grid", gap: "0.75rem" }}>
                <label><strong>Game</strong><br />
                  <select value={scoreGameId || String(selectedGame?.id || "")} onChange={(event) => {
                    const game = sortedGames.find((row) => String(row.id || "") === event.target.value) || null;
                    setScoreGameId(event.target.value);
                    setScoreA(game?.score_a == null ? "" : String(game.score_a));
                    setScoreB(game?.score_b == null ? "" : String(game.score_b));
                    setScoreConfirm("");
                  }} style={inputStyle}>
                    {sortedGames.map((game) => <option key={String(game.id || gameLabel(game))} value={String(game.id || "")}>{gameLabel(game)} · {isScored(game) ? "scored" : "open"}</option>)}
                  </select>
                </label>
                {selectedGame ? (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
                    <div><strong>Team A</strong><br />{teamLabel(teamsById.get(String(selectedGame.team_a_id || "")), snapshot)}</div>
                    <div><strong>Team B</strong><br />{teamLabel(teamsById.get(String(selectedGame.team_b_id || "")), snapshot)}</div>
                    <label><strong>Score A</strong><br /><input value={scoreA} onChange={(event) => setScoreA(event.target.value)} type="number" min={0} step={1} style={inputStyle} /></label>
                    <label><strong>Score B</strong><br /><input value={scoreB} onChange={(event) => setScoreB(event.target.value)} type="number" min={0} step={1} style={inputStyle} /></label>
                    <label><strong>Type SAVE SCORE</strong><br /><input value={scoreConfirm} onChange={(event) => setScoreConfirm(event.target.value)} style={inputStyle} /></label>
                    <button type="button" onClick={saveScore} disabled={busy || scoreConfirm.trim().toUpperCase() !== "SAVE SCORE"} style={buttonStyle}>Save score</button>
                  </div>
                ) : null}
              </div>
            ) : null}
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Draw progression</h2>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1rem" }}>
              <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
                <h3 style={{ marginTop: 0 }}>Round robin</h3>
                <p style={{ color: "#64748b" }}>Generate the initial games after teams are imported or entered in Tournament Ops.</p>
                <input value={roundRobinConfirm} onChange={(event) => setRoundRobinConfirm(event.target.value)} placeholder="GENERATE GAMES" style={inputStyle} />
                <p><button type="button" disabled={busy || roundRobinConfirm.trim().toUpperCase() !== "GENERATE GAMES"} onClick={() => liveAction(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/draws/${encodeURIComponent(selectedDrawId)}/games/round-robin`, { confirmation_text: roundRobinConfirm, source: "next_tournament_live_round_robin" }, "Round-robin games generated.")} style={ghostButtonStyle}>Generate games</button></p>
              </div>
              <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
                <h3 style={{ marginTop: 0 }}>Playoffs</h3>
                <p style={{ color: "#64748b" }}>Generate playoff bracket after all round-robin games are scored.</p>
                <label><strong>Advance count</strong><br /><input value={playoffAdvanceCount} onChange={(event) => setPlayoffAdvanceCount(event.target.value)} type="number" min={2} max={16} step={1} style={inputStyle} /></label>
                <input value={playoffConfirm} onChange={(event) => setPlayoffConfirm(event.target.value)} placeholder="GENERATE PLAYOFFS" style={{ ...inputStyle, marginTop: "0.5rem" }} />
                <p><button type="button" disabled={busy || playoffConfirm.trim().toUpperCase() !== "GENERATE PLAYOFFS"} onClick={() => liveAction(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/draws/${encodeURIComponent(selectedDrawId)}/games/playoffs`, { advance_count: Number(playoffAdvanceCount), confirmation_text: playoffConfirm, source: "next_tournament_live_playoffs" }, "Playoff bracket generated.")} style={ghostButtonStyle}>Generate playoffs</button></p>
              </div>
              <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
                <h3 style={{ marginTop: 0 }}>Podium</h3>
                <p style={{ color: "#64748b" }}>Generate draw-scoped podium rows, then award trophies/badges when reviewed.</p>
                <input value={podiumConfirm} onChange={(event) => setPodiumConfirm(event.target.value)} placeholder="GENERATE PODIUM" style={inputStyle} />
                <p><button type="button" disabled={busy || podiumConfirm.trim().toUpperCase() !== "GENERATE PODIUM"} onClick={() => liveAction(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/draws/${encodeURIComponent(selectedDrawId)}/podium`, { confirmation_text: podiumConfirm, source: "next_tournament_live_podium" }, "Podium generated.")} style={ghostButtonStyle}>Generate podium</button></p>
                <input value={awardConfirm} onChange={(event) => setAwardConfirm(event.target.value)} placeholder="AWARD PODIUM" style={inputStyle} />
                <p><button type="button" disabled={busy || awardConfirm.trim().toUpperCase() !== "AWARD PODIUM"} onClick={() => liveAction(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/draws/${encodeURIComponent(selectedDrawId)}/podium/awards`, { confirmation_text: awardConfirm, source: "next_tournament_live_awards" }, "Podium awards applied.")} style={ghostButtonStyle}>Award podium</button></p>
              </div>
              <div style={{ border: "1px solid #fecaca", borderRadius: "12px", padding: "0.75rem", background: "#fef2f2" }}>
                <h3 style={{ marginTop: 0 }}>Official publish</h3>
                <p style={{ color: "#7f1d1d" }}>Publish finalized tournament games as official rating matches. This can trigger automatic player update emails when enabled on FastAPI.</p>
                <label><strong>Winner bonus Elo</strong><br /><input value={publishBonusElo} onChange={(event) => setPublishBonusElo(event.target.value)} type="number" min={0} max={40} step={1} style={inputStyle} /></label>
                <input value={publishConfirm} onChange={(event) => setPublishConfirm(event.target.value)} placeholder="PUBLISH MATCHES" style={{ ...inputStyle, marginTop: "0.5rem" }} />
                <p><button type="button" disabled={busy || publishConfirm.trim().toUpperCase() !== "PUBLISH MATCHES"} onClick={() => liveAction(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/draws/${encodeURIComponent(selectedDrawId)}/matches/publish`, { playoff_winner_bonus_elo: Number(publishBonusElo), confirmation_text: publishConfirm, source: "next_tournament_live_publish_matches" }, "Official tournament matches published.")} style={dangerButtonStyle}>Publish official matches</button></p>
              </div>
            </div>
          </article>

          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>Games</h2>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "920px" }}>
                <thead>
                  <tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Game</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Team A</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Team B</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Score</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Status</th></tr>
                </thead>
                <tbody>
                  {sortedGames.map((game, index) => (
                    <tr key={String(game.id || index)}>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{gameLabel(game)}</td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{teamLabel(teamsById.get(String(game.team_a_id || "")), snapshot)}</td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{teamLabel(teamsById.get(String(game.team_b_id || "")), snapshot)}</td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{shortValue(game.score_a)}–{shortValue(game.score_b)}</td>
                      <td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{statusChip(isScored(game))}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </>
      ) : null}

      {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("failed") || message.toLowerCase().includes("sign in") || message.toLowerCase().includes("type") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
