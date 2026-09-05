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
  requestKind: string;
  details: string;
  evidenceUrl: string;
  consent: boolean;
};

const initialState: FormState = {
  requesterName: "",
  requesterEmail: "",
  playerName: "",
  playerId: "",
  requestKind: "review_display_name",
  details: "",
  evidenceUrl: "",
  consent: false
};

const requestKindLabels: Record<string, string> = {
  review_display_name: "Review my public name",
  hide_profile: "Hide my public profile",
  anonymize_history: "Use an anonymous name in past results",
  contact_update: "Change my contact or profile details",
  other: "Something else"
};

export default function ProfilePrivacyRequestForm({ clubSlug = "tres-palapas" }: { clubSlug?: string }) {
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
    if (!state.requesterName.trim() || !state.requesterEmail.includes("@") || !state.playerName.trim() || !state.details.trim()) {
      setError("Name, email, player name, and request details are required.");
      return;
    }
    if (!state.consent) {
      setError("Please confirm staff may contact you about this privacy request.");
      return;
    }
    setPending(true);
    const response = await submitPublicSupportRequest(clubSlug, {
      request_type: "profile_privacy",
      requester_name: state.requesterName,
      requester_email: state.requesterEmail,
      player_name: state.playerName,
      player_id: state.playerId || null,
      subject: requestKindLabels[state.requestKind] || "Profile privacy request",
      description: state.details,
      requested_action: `${requestKindLabels[state.requestKind] || state.requestKind}. Staff review required before any public display changes.`,
      evidence_url: state.evidenceUrl || null,
      consent_to_contact: state.consent,
      website,
      source: "next_profile_privacy_form"
    });
    setPending(false);
    if (response.error) {
      setError(response.error);
      return;
    }
    setState(initialState);
    setMessage(response.data?.message || "Privacy request received. Staff will review it before changing your public profile.");
  }

  return (
    <form onSubmit={onSubmit} style={{ display: "grid", gap: "0.85rem" }}>
      <input type="text" name="website" autoComplete="off" tabIndex={-1} aria-hidden="true" style={{ position: "absolute", left: "-10000px" }} />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
        <label>Your name<br /><input value={state.requesterName} onChange={(event) => update("requesterName", event.target.value)} style={inputStyle} /></label>
        <label>Your email<br /><input type="email" value={state.requesterEmail} onChange={(event) => update("requesterEmail", event.target.value)} style={inputStyle} /></label>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
        <label>Player name<br /><input value={state.playerName} onChange={(event) => update("playerName", event.target.value)} style={inputStyle} /></label>
        <label>Player ID, if known<br /><input value={state.playerId} onChange={(event) => update("playerId", event.target.value)} style={inputStyle} /></label>
      </div>
      <label>Request type<br />
        <select value={state.requestKind} onChange={(event) => update("requestKind", event.target.value)} style={inputStyle}>
          {Object.entries(requestKindLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
        </select>
      </label>
      <label>Request details<br /><textarea value={state.details} onChange={(event) => update("details", event.target.value)} rows={5} style={inputStyle} /></label>
      <label>Evidence or relevant link, optional<br /><input type="url" value={state.evidenceUrl} onChange={(event) => update("evidenceUrl", event.target.value)} placeholder="https://…" style={inputStyle} /></label>
      <label style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
        <input type="checkbox" checked={state.consent} onChange={(event) => update("consent", event.target.checked)} />
        <span>I understand staff will review my request and may contact me to verify it.</span>
      </label>
      <button type="submit" disabled={pending} style={buttonStyle}>{pending ? "Submitting…" : "Submit privacy request"}</button>
      {message ? <p role="status" style={{ color: "#166534", margin: 0 }}>{message}</p> : null}
      {error ? <p role="alert" style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
    </form>
  );
}
