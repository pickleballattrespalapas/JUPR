"use client";

import { FormEvent, useState } from "react";
import { submitPublicSupportRequest } from "@/lib/supportIntakeApi";

const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.65rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };

type State = {
  requesterName: string;
  requesterEmail: string;
  playerName: string;
  subject: string;
  description: string;
  evidenceUrl: string;
  consent: boolean;
};

const initialState: State = {
  requesterName: "",
  requesterEmail: "",
  playerName: "",
  subject: "",
  description: "",
  evidenceUrl: "",
  consent: false
};

export default function SupportRequestForm({ clubSlug = "tres-palapas" }: { clubSlug?: string }) {
  const [state, setState] = useState<State>(initialState);
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  function update<K extends keyof State>(key: K, value: State[K]) {
    setState((current) => ({ ...current, [key]: value }));
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage(null);
    setError(null);
    const formData = new FormData(event.currentTarget);
    if (!state.requesterName.trim() || !state.requesterEmail.includes("@") || !state.subject.trim() || !state.description.trim()) {
      setError("Name, email, subject, and request details are required.");
      return;
    }
    if (!state.consent) {
      setError("Please confirm staff may contact you about this request.");
      return;
    }
    setPending(true);
    const response = await submitPublicSupportRequest(clubSlug, {
      request_type: "general_support",
      requester_name: state.requesterName,
      requester_email: state.requesterEmail,
      player_name: state.playerName || null,
      subject: state.subject,
      description: state.description,
      requested_action: "Staff response requested.",
      evidence_url: state.evidenceUrl || null,
      consent_to_contact: state.consent,
      website: String(formData.get("website") || ""),
      source: "next_general_support_form"
    });
    setPending(false);
    if (response.error) {
      setError(response.error);
      return;
    }
    setState(initialState);
    setMessage(response.data?.message || "Support request received. Staff will follow up by email.");
  }

  return (
    <form onSubmit={onSubmit} style={{ display: "grid", gap: "0.85rem" }}>
      <input type="text" name="website" autoComplete="off" tabIndex={-1} aria-hidden="true" style={{ position: "absolute", left: "-10000px" }} />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
        <label>Your name<br /><input value={state.requesterName} onChange={(event) => update("requesterName", event.target.value)} autoComplete="name" style={inputStyle} /></label>
        <label>Your email<br /><input type="email" value={state.requesterEmail} onChange={(event) => update("requesterEmail", event.target.value)} autoComplete="email" style={inputStyle} /></label>
      </div>
      <label>Player name, if relevant<br /><input value={state.playerName} onChange={(event) => update("playerName", event.target.value)} style={inputStyle} /></label>
      <label>Short subject<br /><input value={state.subject} onChange={(event) => update("subject", event.target.value)} style={inputStyle} /></label>
      <label>How can we help?<br /><textarea value={state.description} onChange={(event) => update("description", event.target.value)} rows={5} style={inputStyle} /></label>
      <label>Relevant http/https link, optional<br /><input type="url" value={state.evidenceUrl} onChange={(event) => update("evidenceUrl", event.target.value)} placeholder="https://…" style={inputStyle} /></label>
      <label style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
        <input type="checkbox" checked={state.consent} onChange={(event) => update("consent", event.target.checked)} />
        <span>Staff may contact me at the email above to resolve this request.</span>
      </label>
      <button type="submit" disabled={pending} style={buttonStyle}>{pending ? "Submitting…" : "Submit support request"}</button>
      {message ? <p role="status" style={{ color: "#166534", margin: 0 }}>{message}</p> : null}
      {error ? <p role="alert" style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
    </form>
  );
}

