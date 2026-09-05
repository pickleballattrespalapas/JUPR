"use client";

import { FormEvent, useState } from "react";
import { submitPublicSupportRequest } from "@/lib/supportIntakeApi";

const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.65rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

type FormState = {
  requesterName: string;
  requesterEmail: string;
  playerName: string;
  playerId: string;
  matchId: string;
  tournamentId: string;
  subject: string;
  description: string;
  requestedAction: string;
  evidenceUrl: string;
  consent: boolean;
};

const initialState: FormState = {
  requesterName: "",
  requesterEmail: "",
  playerName: "",
  playerId: "",
  matchId: "",
  tournamentId: "",
  subject: "",
  description: "",
  requestedAction: "",
  evidenceUrl: "",
  consent: false
};

export default function DataCorrectionForm({ clubSlug = "tres-palapas" }: { clubSlug?: string }) {
  const [state, setState] = useState<FormState>(initialState);
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  function update<K extends keyof FormState>(key: K, value: FormState[K]) {
    setState((current) => ({ ...current, [key]: value }));
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage(null);
    setError(null);
    const formData = new FormData(event.currentTarget);
    const website = String(formData.get("website") ?? "").trim();
    if (!state.requesterName.trim() || !state.requesterEmail.includes("@") || !state.subject.trim() || !state.description.trim()) {
      setError("Name, email, subject, and request details are required.");
      return;
    }
    if (!state.consent) {
      setError("Please confirm staff may contact you about this correction request.");
      return;
    }
    setPending(true);
    const response = await submitPublicSupportRequest(clubSlug, {
      request_type: "data_correction",
      requester_name: state.requesterName,
      requester_email: state.requesterEmail,
      player_name: state.playerName || null,
      player_id: state.playerId || null,
      match_id: state.matchId || null,
      tournament_id: state.tournamentId || null,
      subject: state.subject,
      description: state.description,
      requested_action: state.requestedAction || null,
      evidence_url: state.evidenceUrl || null,
      consent_to_contact: state.consent,
      website,
      source: "next_data_corrections_form"
    });
    setPending(false);
    if (response.error) {
      setError(response.error);
      return;
    }
    setState(initialState);
    setMessage(response.data?.message || "Correction request received. Staff will review it before making any changes.");
  }

  return (
    <form onSubmit={onSubmit} style={{ display: "grid", gap: "0.85rem" }}>
      <input type="text" name="website" autoComplete="off" tabIndex={-1} aria-hidden="true" style={{ position: "absolute", left: "-10000px" }} />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
        <label>Your name<br /><input value={state.requesterName} onChange={(event) => update("requesterName", event.target.value)} style={inputStyle} /></label>
        <label>Your email<br /><input type="email" value={state.requesterEmail} onChange={(event) => update("requesterEmail", event.target.value)} style={inputStyle} /></label>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
        <label>Player name<br /><input value={state.playerName} onChange={(event) => update("playerName", event.target.value)} style={inputStyle} /></label>
        <label>Player ID, if known<br /><input value={state.playerId} onChange={(event) => update("playerId", event.target.value)} style={inputStyle} /></label>
        <label>Match ID, if known<br /><input value={state.matchId} onChange={(event) => update("matchId", event.target.value)} style={inputStyle} /></label>
        <label>Tournament ID, if known<br /><input value={state.tournamentId} onChange={(event) => update("tournamentId", event.target.value)} style={inputStyle} /></label>
      </div>
      <label>Short subject<br /><input value={state.subject} onChange={(event) => update("subject", event.target.value)} placeholder="Wrong score, duplicate match, profile name, tournament entry…" style={inputStyle} /></label>
      <label>What looks wrong?<br /><textarea value={state.description} onChange={(event) => update("description", event.target.value)} rows={5} style={inputStyle} /></label>
      <label>What should staff change after review?<br /><textarea value={state.requestedAction} onChange={(event) => update("requestedAction", event.target.value)} rows={4} style={inputStyle} /></label>
      <label>Evidence or screenshot link, optional<br /><input type="url" value={state.evidenceUrl} onChange={(event) => update("evidenceUrl", event.target.value)} placeholder="https://…" style={inputStyle} /></label>
      <label style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
        <input type="checkbox" checked={state.consent} onChange={(event) => update("consent", event.target.checked)} />
        <span>I understand that staff will review my request before making changes and may contact me with questions.</span>
      </label>
      <button type="submit" disabled={pending} style={buttonStyle}>{pending ? "Submitting…" : "Submit correction request"}</button>
      {message ? <p role="status" style={{ color: "#166534", margin: 0 }}>{message}</p> : null}
      {error ? <p role="alert" style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
    </form>
  );
}
