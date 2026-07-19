"use client";

import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import type { VerifiedUpdatePlayer, VerifiedUpdateRequestResponse, VerifiedUpdateStatusResponse } from "@/lib/verifiedUpdatesApi";

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
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [selectedStatus, setSelectedStatus] = useState<string | null>(null);
  const [statusReady, setStatusReady] = useState(false);

  const selectedPlayer = players.find((player) => String(player.id) === String(playerId));
  const alreadyRequested = selectedStatus === "pending_admin_review" || selectedStatus === "active";

  useEffect(() => {
    const controller = new AbortController();
    setMessage(null);
    setSelectedStatus(selectedPlayer?.request_status || null);
    setStatusMessage(null);
    setStatusReady(false);
    if (!apiBase || !playerId) {
      setStatusMessage(apiBase ? "Choose a player to check request status." : "Missing JUPR API base URL.");
      return () => controller.abort();
    }

    fetch(apiUrl(apiBase, `/clubs/${encodeURIComponent(clubSlug)}/verified-updates/status?player_id=${encodeURIComponent(playerId)}`), {
      signal: controller.signal,
      cache: "no-store"
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(await apiError(response));
        return (await response.json()) as VerifiedUpdateStatusResponse;
      })
      .then((payload) => {
        setSelectedStatus(payload.player?.request_status || null);
        setStatusReady(true);
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setStatusMessage(error instanceof Error ? error.message : "Unable to confirm request status.");
      });

    return () => controller.abort();
  }, [apiBase, clubSlug, playerId, selectedPlayer?.request_status]);

  async function submitRequest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
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
      const nextStatus = payload.player?.request_status || payload.request_status || null;
      setSelectedStatus(nextStatus);
      setStatusReady(true);
      setMessage(
        nextStatus === "active"
          ? "Verified updates are already active for this profile."
          : payload.deduplicated
          ? "This request is already pending admin review; no duplicate was created."
          : "Request submitted for admin review."
      );
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
      <form onSubmit={submitRequest} style={{ display: "grid", gap: "0.75rem" }}>
        <label>Player<br />
          <select value={playerId} onChange={(event) => setPlayerId(event.target.value)} style={inputStyle}>
            {players.map((player) => <option key={player.id} value={player.id}>{player.name}{player.request_status ? ` · ${player.request_status.replace(/_/g, " ")}` : ""}</option>)}
          </select>
        </label>
        <div aria-live="polite">
          {!statusReady && !statusMessage ? <p style={{ color: "#475569" }}>Checking current request status…</p> : null}
          {statusMessage ? <p role="alert" style={{ color: "#b91c1c" }}>Request status unavailable. {statusMessage}</p> : null}
          {selectedStatus === "pending_admin_review" ? <p style={{ color: "#92400e" }}>A verified update request for this profile is pending admin review.</p> : null}
          {selectedStatus === "active" ? <p style={{ color: "#166534" }}>Verified player updates are active for this profile.</p> : null}
        </div>
        <label>Email<br /><input type="email" required maxLength={320} value={email} onChange={(event) => setEmail(event.target.value)} disabled={alreadyRequested} style={inputStyle} /></label>
        <label>Note for admins, optional<br /><textarea value={requestNote} maxLength={1000} onChange={(event) => setRequestNote(event.target.value)} disabled={alreadyRequested} rows={4} style={inputStyle} /></label>
        <label style={{ position: "absolute", left: "-10000px" }}>Website<br /><input value={website} onChange={(event) => setWebsite(event.target.value)} tabIndex={-1} autoComplete="off" /></label>
        <button type="submit" disabled={busy || !statusReady || alreadyRequested || !playerId || !email.trim()} style={buttonStyle}>{busy ? "Submitting…" : alreadyRequested ? "Request already recorded" : "Submit request"}</button>
        {message ? <p aria-live="polite" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("api error") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </form>
    </article>
  );
}
