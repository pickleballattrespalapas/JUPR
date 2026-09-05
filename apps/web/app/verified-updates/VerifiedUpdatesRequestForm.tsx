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

const REQUEST_STATUS_ERROR = "We couldn’t check this request right now. Please try again in a moment.";
const REQUEST_SUBMIT_ERROR = "We couldn’t send your request right now. Please try again later.";

function requestStatusLabel(value?: string | null): string | null {
  switch (String(value || "").toLowerCase()) {
    case "pending_admin_review":
      return "Awaiting club approval";
    case "active":
      return "Updates active";
    case "unsubscribed":
      return "Updates stopped";
    default:
      return null;
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
      setStatusMessage(apiBase ? "Choose a player to continue." : REQUEST_STATUS_ERROR);
      return () => controller.abort();
    }

    fetch(apiUrl(apiBase, `/clubs/${encodeURIComponent(clubSlug)}/verified-updates/status?player_id=${encodeURIComponent(playerId)}`), {
      signal: controller.signal,
      cache: "no-store"
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(REQUEST_STATUS_ERROR);
        return (await response.json()) as VerifiedUpdateStatusResponse;
      })
      .then((payload) => {
        setSelectedStatus(payload.player?.request_status || null);
        setStatusReady(true);
      })
      .catch(() => {
        if (controller.signal.aborted) return;
        setStatusMessage(REQUEST_STATUS_ERROR);
      });

    return () => controller.abort();
  }, [apiBase, clubSlug, playerId, selectedPlayer?.request_status]);

  async function submitRequest(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!apiBase) {
      setMessage(REQUEST_SUBMIT_ERROR);
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
      if (!response.ok) throw new Error(REQUEST_SUBMIT_ERROR);
      const payload = (await response.json()) as VerifiedUpdateRequestResponse;
      const nextStatus = payload.player?.request_status || payload.request_status || null;
      setSelectedStatus(nextStatus);
      setStatusReady(true);
      setMessage(
        nextStatus === "active"
          ? "Verified updates are already active for this profile."
          : payload.deduplicated
          ? "You’ve already sent a request for this profile. Club staff will review it soon."
          : "Request sent. Club staff will review it."
      );
      setEmail("");
      setRequestNote("");
    } catch {
      setMessage(REQUEST_SUBMIT_ERROR);
    } finally {
      setBusy(false);
    }
  }

  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Request verified player updates</h2>
      <p style={{ color: "#475569" }}>Choose a player and enter where you&apos;d like to receive updates when new results are posted.</p>
      <form onSubmit={submitRequest} style={{ display: "grid", gap: "0.75rem" }}>
        <label>Player<br />
          <select value={playerId} onChange={(event) => setPlayerId(event.target.value)} style={inputStyle}>
            {players.map((player) => {
              const status = requestStatusLabel(player.request_status);
              return <option key={player.id} value={player.id}>{player.name}{status ? ` · ${status}` : ""}</option>;
            })}
          </select>
        </label>
        <div aria-live="polite">
          {!statusReady && !statusMessage ? <p style={{ color: "#475569" }}>Checking for an existing request…</p> : null}
          {statusMessage ? <p role="alert" style={{ color: "#b91c1c" }}>{statusMessage}</p> : null}
          {selectedStatus === "pending_admin_review" ? <p style={{ color: "#92400e" }}>Club staff are reviewing your request for this profile.</p> : null}
          {selectedStatus === "active" ? <p style={{ color: "#166534" }}>Verified player updates are active for this profile.</p> : null}
        </div>
        <label>Email<br /><input type="email" required maxLength={320} value={email} onChange={(event) => setEmail(event.target.value)} disabled={alreadyRequested} style={inputStyle} /></label>
        <label>Note for club staff, optional<br /><textarea value={requestNote} maxLength={1000} onChange={(event) => setRequestNote(event.target.value)} disabled={alreadyRequested} rows={4} style={inputStyle} /></label>
        <label style={{ position: "absolute", left: "-10000px" }}>Website<br /><input value={website} onChange={(event) => setWebsite(event.target.value)} tabIndex={-1} autoComplete="off" /></label>
        <button type="submit" disabled={busy || !statusReady || alreadyRequested || !playerId || !email.trim()} style={buttonStyle}>{busy ? "Sending…" : alreadyRequested ? "Request already sent" : "Request updates"}</button>
        {message ? <p aria-live="polite" style={{ color: message.toLowerCase().includes("couldn’t") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </form>
    </article>
  );
}
