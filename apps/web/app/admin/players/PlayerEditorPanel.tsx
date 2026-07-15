"use client";

import Link from "next/link";
import { useState } from "react";
import type {
  AdminPlayerEditorDetailResponse,
  AdminPlayerEditorLeagueRating,
  AdminPlayerEditorListResponse,
  AdminPlayerEditorPlayer,
  AdminPlayerEditorStatusResponse,
  AdminPlayerEditorWriteResponse
} from "@/lib/adminPlayerEditorApi";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Props = {
  apiBase: string | null;
  clubId: string;
  status: AdminPlayerEditorStatusResponse;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const leagueRatingConfirmText = "SAVE LEAGUE RATING";

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

function juprLabel(value?: number | null): string {
  return value == null ? "—" : Number(value).toFixed(2);
}

export default function PlayerEditorPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [players, setPlayers] = useState<AdminPlayerEditorPlayer[]>([]);
  const [detail, setDetail] = useState<AdminPlayerEditorDetailResponse | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [newName, setNewName] = useState("");
  const [newStartingJupr, setNewStartingJupr] = useState("3.5");
  const [editName, setEditName] = useState("");
  const [editRating, setEditRating] = useState("3.5");
  const [editStartingRating, setEditStartingRating] = useState("3.5");
  const [editActive, setEditActive] = useState(true);
  const [selectedLeagueRatingId, setSelectedLeagueRatingId] = useState("");
  const [editLeagueRating, setEditLeagueRating] = useState("3.5");
  const [editLeagueStartingRating, setEditLeagueStartingRating] = useState("3.5");
  const [editLeagueActive, setEditLeagueActive] = useState(true);
  const [leagueRatingConfirm, setLeagueRatingConfirm] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  function requireReady(): boolean {
    if (!apiBase) {
      setMessage("API base URL is not configured.");
      return false;
    }
    if (!accessToken) {
      setMessage("Sign in at /admin/login before using the Player Editor.");
      return false;
    }
    if (!status.enabled) {
      setMessage("Next Player Editor is disabled on the API.");
      return false;
    }
    return true;
  }

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("API base URL is not configured.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using the Player Editor.");
    const headers = new Headers(options?.headers);
    headers.set("Content-Type", "application/json");
    headers.set("Authorization", `Bearer ${accessToken}`);
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(String(payload?.detail || `API error (${response.status})`));
    return payload as T;
  }

  function seedEditForm(player: AdminPlayerEditorPlayer) {
    setEditName(player.name || "");
    setEditRating(String(player.rating_jupr ?? 3.5));
    setEditStartingRating(String(player.starting_jupr ?? player.rating_jupr ?? 3.5));
    setEditActive(player.active !== false);
  }

  function seedLeagueRatingForm(row: AdminPlayerEditorLeagueRating | null) {
    setSelectedLeagueRatingId(row ? String(row.id) : "");
    setEditLeagueRating(String(row?.rating_jupr ?? 3.5));
    setEditLeagueStartingRating(String(row?.starting_jupr ?? row?.rating_jupr ?? 3.5));
    setEditLeagueActive(row?.is_active !== false);
    setLeagueRatingConfirm("");
  }

  async function loadPlayers() {
    setMessage(null);
    if (!requireReady()) return;
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorListResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/players`);
      setPlayers(payload.players || []);
      setMessage(`Loaded ${payload.count ?? payload.players?.length ?? 0} player(s).`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load players.");
    } finally {
      setSaving(false);
    }
  }

  async function loadDetail(playerId: string) {
    setSelectedId(playerId);
    setDetail(null);
    setMessage(null);
    if (!playerId || !requireReady()) return;
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorDetailResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/players/${encodeURIComponent(playerId)}`);
      setDetail(payload);
      seedEditForm(payload.player);
      seedLeagueRatingForm(payload.league_ratings?.[0] || null);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to load player detail.");
    } finally {
      setSaving(false);
    }
  }

  async function createPlayer() {
    setMessage(null);
    if (!requireReady()) return;
    const name = newName.trim();
    const starting = Number(newStartingJupr);
    if (!name || !Number.isFinite(starting) || starting < 1 || starting > 7) {
      setMessage("Enter a name and a Starting JUPR between 1.0 and 7.0.");
      return;
    }
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/players`, {
        method: "POST",
        body: JSON.stringify({ name, starting_jupr: starting, source: "next_player_editor_create" })
      });
      if (payload.player) {
        setPlayers((current) => [...current.filter((player) => player.id !== payload.player?.id), payload.player as AdminPlayerEditorPlayer].sort((left, right) => left.name.localeCompare(right.name)));
        setSelectedId(String(payload.player.id));
        await loadDetail(String(payload.player.id));
      }
      setNewName("");
      setNewStartingJupr("3.5");
      setMessage("Player created or confirmed.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to create player.");
    } finally {
      setSaving(false);
    }
  }

  async function savePlayer() {
    setMessage(null);
    if (!selectedId || !requireReady()) return;
    const rating = Number(editRating);
    const starting = Number(editStartingRating);
    if (!editName.trim() || !Number.isFinite(rating) || !Number.isFinite(starting) || rating < 1 || rating > 7 || starting < 1 || starting > 7) {
      setMessage("Name, Overall JUPR, and Starting JUPR are required. Ratings must be between 1.0 and 7.0.");
      return;
    }
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/players/${encodeURIComponent(selectedId)}`, {
        method: "PATCH",
        body: JSON.stringify({ name: editName.trim(), rating_jupr: rating, starting_jupr: starting, active: editActive, source: "next_player_editor_update" })
      });
      if (payload.player) {
        setPlayers((current) => current.map((player) => player.id === payload.player?.id ? payload.player as AdminPlayerEditorPlayer : player).sort((left, right) => left.name.localeCompare(right.name)));
        await loadDetail(String(payload.player.id));
      }
      setMessage("Player saved. Use Match Log and Replay History if downstream rating repair is needed.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save player.");
    } finally {
      setSaving(false);
    }
  }

  async function saveLeagueRating() {
    setMessage(null);
    if (!selectedId || !selectedLeagueRatingId || !requireReady()) return;
    const rating = Number(editLeagueRating);
    const starting = Number(editLeagueStartingRating);
    if (!Number.isFinite(rating) || !Number.isFinite(starting) || rating < 1 || rating > 7 || starting < 1 || starting > 7) {
      setMessage("League JUPR and league Starting JUPR must be between 1.0 and 7.0.");
      return;
    }
    if (leagueRatingConfirm.trim().toUpperCase() !== leagueRatingConfirmText) {
      setMessage(`Type ${leagueRatingConfirmText} to confirm league-rating edits.`);
      return;
    }
    setSaving(true);
    try {
      const payload = await requestJson<AdminPlayerEditorWriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/players/editor/players/${encodeURIComponent(selectedId)}/league-ratings/${encodeURIComponent(selectedLeagueRatingId)}`, {
        method: "PATCH",
        body: JSON.stringify({ rating_jupr: rating, starting_jupr: starting, is_active: editLeagueActive, confirmation_text: leagueRatingConfirm, source: "next_player_editor_league_rating" })
      });
      if (payload.league_ratings && detail) {
        setDetail({ ...detail, league_ratings: payload.league_ratings });
        const updated = payload.league_ratings.find((row) => String(row.id) === selectedLeagueRatingId) || payload.league_ratings[0] || null;
        seedLeagueRatingForm(updated);
      }
      setMessage("League rating saved and audit-flagged for review. Run Replay History if you need to rebuild derived history.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to save league rating.");
    } finally {
      setSaving(false);
    }
  }

  if (!status.enabled) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Next Player Editor is disabled</h2>
        <p style={{ color: "#475569" }}>{status.warnings?.[0] || "Enable the Player Editor pilot flag on FastAPI."}</p>
      </article>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Player Editor admin session</h2>
        <p style={{ color: "#475569" }}>This route supports roster/detail read, add player, basic player updates, and guarded league-rating edits through FastAPI. Merge and social identity linking stay on Streamlit for now.</p>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
          <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
          <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
            {accessToken ? "Ready to send authorized Player Editor requests." : sessionLoading ? "Checking admin session…" : "Sign in before using the Player Editor."}
          </p>
          {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
          {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
        </div>
        <button type="button" onClick={loadPlayers} disabled={saving || !accessToken} style={buttonStyle}>{saving ? "Working…" : "Load players"}</button>
        {status.warnings?.length ? <ul style={{ color: "#92400e" }}>{status.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Add new player</h2>
        <div style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) minmax(120px, 180px) auto", gap: "0.75rem", alignItems: "end" }}>
          <label><strong>Name</strong><br /><input value={newName} onChange={(event) => setNewName(event.target.value)} style={inputStyle} /></label>
          <label><strong>Starting JUPR</strong><br /><input value={newStartingJupr} onChange={(event) => setNewStartingJupr(event.target.value)} type="number" min={1} max={7} step={0.1} style={inputStyle} /></label>
          <button type="button" onClick={createPlayer} disabled={saving || !accessToken} style={ghostButtonStyle}>Add player</button>
        </div>
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Select player</h2>
        <select value={selectedId} onChange={(event) => loadDetail(event.target.value)} style={inputStyle} disabled={!accessToken}>
          <option value="">Choose a player</option>
          {players.map((player) => <option key={player.id} value={String(player.id)}>{player.name} #{player.id}{player.active === false ? " [inactive]" : ""}</option>)}
        </select>
      </article>

      {detail ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>Manage player</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
            <label><strong>Name</strong><br /><input value={editName} onChange={(event) => setEditName(event.target.value)} style={inputStyle} /></label>
            <label><strong>Overall JUPR</strong><br /><input value={editRating} onChange={(event) => setEditRating(event.target.value)} type="number" min={1} max={7} step={0.01} style={inputStyle} /></label>
            <label><strong>Starting JUPR</strong><br /><input value={editStartingRating} onChange={(event) => setEditStartingRating(event.target.value)} type="number" min={1} max={7} step={0.01} style={inputStyle} /></label>
            <label><strong>Active</strong><br /><select value={editActive ? "yes" : "no"} onChange={(event) => setEditActive(event.target.value === "yes")} style={inputStyle}><option value="yes">Active</option><option value="no">Inactive</option></select></label>
          </div>
          <p><button type="button" onClick={savePlayer} disabled={saving || !accessToken} style={buttonStyle}>{saving ? "Saving…" : "Save player"}</button></p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem", marginTop: "1rem" }}>
            <div><strong>Wins</strong><br />{detail.player.wins ?? 0}</div>
            <div><strong>Losses</strong><br />{detail.player.losses ?? 0}</div>
            <div><strong>Matches</strong><br />{detail.player.matches_played ?? 0}</div>
            <div><strong>Match refs</strong><br />{detail.match_reference_counts?.total ?? 0}</div>
          </div>
        </article>
      ) : null}

      {detail?.league_ratings?.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>League ratings</h2>
          <p style={{ color: "#475569" }}>League-rating edits are audit-flagged because they can diverge from replayed history. Use them for targeted corrections only, then run Replay History if needed.</p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
            <label><strong>League rating row</strong><br />
              <select value={selectedLeagueRatingId} onChange={(event) => seedLeagueRatingForm(detail.league_ratings.find((row) => String(row.id) === event.target.value) || null)} style={inputStyle}>
                <option value="">Choose league rating…</option>
                {detail.league_ratings.map((row) => <option key={row.id} value={String(row.id)}>{row.league_name} · {juprLabel(row.rating_jupr)}</option>)}
              </select>
            </label>
            <label><strong>League JUPR</strong><br /><input value={editLeagueRating} onChange={(event) => setEditLeagueRating(event.target.value)} type="number" min={1} max={7} step={0.01} style={inputStyle} /></label>
            <label><strong>League starting JUPR</strong><br /><input value={editLeagueStartingRating} onChange={(event) => setEditLeagueStartingRating(event.target.value)} type="number" min={1} max={7} step={0.01} style={inputStyle} /></label>
            <label><strong>League active</strong><br /><select value={editLeagueActive ? "yes" : "no"} onChange={(event) => setEditLeagueActive(event.target.value === "yes")} style={inputStyle}><option value="yes">Active</option><option value="no">Inactive</option></select></label>
          </div>
          <label style={{ display: "block", marginTop: "0.75rem" }}><strong>Type {leagueRatingConfirmText} to confirm</strong><br /><input value={leagueRatingConfirm} onChange={(event) => setLeagueRatingConfirm(event.target.value)} style={inputStyle} /></label>
          <p><button type="button" onClick={saveLeagueRating} disabled={saving || !accessToken || !selectedLeagueRatingId || leagueRatingConfirm.trim().toUpperCase() !== leagueRatingConfirmText} style={buttonStyle}>{saving ? "Saving…" : "Save league rating"}</button></p>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead><tr><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>League</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>JUPR</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Start</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>W-L</th><th style={{ textAlign: "left", padding: "0.5rem", borderBottom: "1px solid #cbd5e1" }}>Active</th></tr></thead>
              <tbody>
                {detail.league_ratings.map((row) => (
                  <tr key={row.id}><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.league_name}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{juprLabel(row.rating_jupr)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{juprLabel(row.starting_jupr)}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.wins ?? 0}-{row.losses ?? 0}</td><td style={{ padding: "0.5rem", borderBottom: "1px solid #e2e8f0" }}>{row.is_active ? "Yes" : "No"}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      ) : null}

      {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("required") || message.toLowerCase().includes("sign in") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
    </section>
  );
}
