"use client";

import { useMemo, useState } from "react";
import type { VerifiedUpdatePlayer, VerifiedUpdateRequestResponse } from "@/lib/verifiedUpdatesApi";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.7rem 1rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

async function apiError(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");
  if (!text) return `API error (${response.status}).`;
  try {
    const payload = JSON.parse(text) as { detail?: unknown };
    return String(payload.detail || text);
  } catch {
    return text.slice(0, 240);
  }
}

type Props = {
  apiBase: string | null;
  clubSlug: string;
  players: VerifiedUpdatePlayer[];
  initialPlayerId?: string | null;
};

export default function VerifiedUpdatesRequestForm({ apiBase, clubSlug, players, initialPlayerId }: Props) {
  const [playerId, setPlayerId] = useState(initialPlayerId || (players[0]?.id ? String(players[0].id) : ""));
  const [email, setEmail] = useState("");
  const [requestNote, setRequestNote] = useState("");
  const [website, setWebsite] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const selectedPlayer = useMemo(() => players.find((player) => String(player.id) === String(playerId)), [players, playerId]);

  async function submitRequest() {
    if (!apiBase) {
      setMessage("Missing JUPR API base URL.");
      return;
    }
    if (!playerId || !email.trim()) {
      setMessage("Choose a player and enter your email.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const response = await fetch(apiUrl(apiBase, `/clubs/${encodeURIComponent(clubSlug)}/verified-updates/request`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ player_id: Number(playerId), email, request_note: requestNote, website })
      });
      if (!response.ok) throw new Error(await apiError(response));
      const payload = (await response.json()) as VerifiedUpdateRequestResponse;
      setMessage(payload.player?.request_status === "active" ? "Verified updates are already active for this profile." : "Request submitted for admin review.");
      setEmail("");
      setRequestNote("");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to submit request.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Request verified player updates</h2>
      <p style={{ color: "#475569" }}>Choose the player profile you manage and submit the email address that should receive player update summaries after batch uploads.</p>
      <div style={{ display: "grid", gap: "0.75rem" }}>
        <label>Player<br />
          <select value={playerId} onChange={(event) => setPlayerId(event.target.value)} style={inputStyle}>
            {players.map((player) => <option key={player.id} value={player.id}>{player.name}{player.request_status ? ` · ${player.request_status.replace(/_/g, " ")}` : ""}</option>)}
          </select>
        </label>
        {selectedPlayer?.already_requested ? <p style={{ color: "#92400e" }}>This profile already has an open or active verified update request.</p> : null}
        <label>Email<br /><input type="email" value={email} onChange={(event) => setEmail(event.target.value)} style={inputStyle} /></label>
        <label>Note for admins, optional<br /><textarea value={requestNote} onChange={(event) => setRequestNote(event.target.value)} rows={4} style={inputStyle} /></label>
        <label style={{ position: "absolute", left: "-10000px" }}>Website<br /><input value={website} onChange={(event) => setWebsite(event.target.value)} tabIndex={-1} autoComplete="off" /></label>
        <button type="button" onClick={submitRequest} disabled={busy || !playerId || !email.trim()} style={buttonStyle}>{busy ? "Submitting…" : "Submit request"}</button>
        {message ? <p style={{ color: message.toLowerCase().includes("submitted") || message.toLowerCase().includes("active") ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      </div>
    </article>
  );
}
