"use client";

import Link from "next/link";
import { useState } from "react";
import type {
  AdminTournament,
  AdminTournamentDetailResponse,
  AdminTournamentListResponse,
  AdminTournamentOpsSnapshotResponse,
  AdminTournamentStatusResponse,
  AdminTournamentWriteResponse
} from "@/lib/adminTournamentApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminTournamentStatusResponse;
};

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
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Tournament Ops.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
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
    if (!drawEventOptionId && payload.event_options?.length) setDrawEventOptionId(String(payload.event_options[0].id || ""));
    return payload;
  }

  async function loadOps(tournamentId = selectedTournamentId, drawId = selectedDrawId) {
    if (!tournamentId) {
      setMessage("Select a tournament first.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      await loadTournamentDetail(tournamentId);
      const params = new URLSearchParams();
      if (drawId) params.set("draw_id", drawId);
      const suffix = params.toString() ? `?${params.toString()}` : "";
      const payload = await requestJson<AdminTournamentOpsSnapshotResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/ops${suffix}`);
      setSnapshot(payload);
      setMessage("Tournament operations snapshot loaded.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load tournament operations.");
    } finally {
      setBusy(false);
    }
  }

  async function createDraw() {
    if (!selectedTournamentId) {
      setMessage("Select a tournament before creating a draw.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const selectedEvent = detail?.event_options?.find((row) => String(row.id || "") === drawEventOptionId) || null;
      const payload = await requestJson<AdminTournamentWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(selectedTournamentId)}/draws`, {
        method: "POST",
        body: JSON.stringify({
          event_option_id: drawEventOptionId || null,
          registration_day_id: String(selectedEvent?.registration_day_id || "") || null,
          name: drawName,
          confirmation_text: drawConfirm,
          source: "next_tournament_ops_create_draw"
        })
      });
      const nextDrawId = payload.draw?.id || "";
      setSelectedDrawId(nextDrawId);
      setDrawConfirm("");
      await loadOps(selectedTournamentId, nextDrawId);
      setMessage(`Draw created${payload.draw?.name ? `: ${payload.draw.name}` : ""}.`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to create draw.");
    } finally {
      setBusy(false);
    }
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Next Tournament Admin is disabled</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the Tournament Admin pilot flag on FastAPI."}</p>
      </article>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Tournament Ops</h2>
        <p style={{ color: "#475569" }}>Operations visibility plus the first guarded write: creating an empty division draw. Team import, scheduling, scoring, podiums, and awards remain Streamlit-only until their write contracts are ported.</p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>{accessToken ? "Ready to load guarded tournament operations data." : sessionLoading ? "Checking admin session…" : "Sign in before loading ops data."}</p>
          {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
          {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
        </div>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginBottom: "0.75rem" }}>
          <input type="checkbox" checked={includeArchived} onChange={(event) => setIncludeArchived(event.target.checked)} />
          Include archived tournaments
        </label>
        <button type="button" onClick={loadTournaments} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Working…" : "Load tournaments"}</button>
      </article>

      {tournaments.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Select tournament</h2>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) minmax(160px, 240px) auto", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>Tournament</strong><br />
              <select value={selectedTournamentId} onChange={(event) => { setSelectedTournamentId(event.target.value); setSelectedDrawId(""); setSnapshot(null); setDetail(null); setDrawEventOptionId(""); }} style={inputStyle}>
                <option value="">Choose a tournament…</option>
                {tournaments.map((tournament) => <option key={tournament.id} value={tournament.id}>{tournament.name} · {tournament.status}</option>)}
              </select>
            </label>
            <label><strong>Draw ID filter</strong><br /><input value={selectedDrawId} onChange={(event) => setSelectedDrawId(event.target.value)} placeholder="optional" style={inputStyle} /></label>
            <button type="button" onClick={() => loadOps()} disabled={busy || !selectedTournamentId} style={ghostButtonStyle}>Load ops snapshot</button>
          </div>
        </article>
      ) : null}

      {detail ? (
        <article style={{ ...cardStyle, background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Create empty division draw</h2>
          <p style={{ color: "#475569" }}>This creates a DRAFT draw shell scoped to the selected registration division. It does not import teams or generate games.</p>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(240px, 1fr) minmax(180px, 1fr) minmax(160px, 220px)", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>Registration division</strong><br />
              <select value={drawEventOptionId} onChange={(event) => setDrawEventOptionId(event.target.value)} style={inputStyle}>
                <option value="">Legacy / tournament-wide draw</option>
                {detail.event_options.map((row) => <option key={String(row.id)} value={String(row.id)}>{eventOptionLabel(row)}</option>)}
              </select>
            </label>
            <label><strong>Draw name</strong><br /><input value={drawName} onChange={(event) => setDrawName(event.target.value)} placeholder="optional" style={inputStyle} /></label>
            <label><strong>Type CREATE DRAW</strong><br /><input value={drawConfirm} onChange={(event) => setDrawConfirm(event.target.value)} style={inputStyle} /></label>
          </div>
          <p><button type="button" onClick={createDraw} disabled={busy || !accessToken || drawConfirm.trim().toUpperCase() !== "CREATE DRAW"} style={buttonStyle}>{busy ? "Creating…" : "Create draw"}</button></p>
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
          </article>
          <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Draws</h2><GenericRowsTable rows={snapshot.draws} preferredColumns={["id", "name", "status", "registration_day_id", "event_option_id", "team_count"]} /></article>
          <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Teams</h2><GenericRowsTable rows={snapshot.teams} preferredColumns={["team_number", "player1_id", "player2_id", "source", "draw_id", "event_option_id"]} /></article>
          <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Games</h2><GenericRowsTable rows={snapshot.games} preferredColumns={["stage", "rr_round_number", "rr_slot_number", "team1_id", "team2_id", "score_team1", "score_team2", "winner_team_id", "status"]} /></article>
          <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Podium</h2><GenericRowsTable rows={snapshot.podium} preferredColumns={["placement", "team_id", "player1_id", "player2_id", "award_label", "draw_id"]} /></article>
        </>
      ) : null}

      {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("error") || message.toLowerCase().includes("sign in") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
