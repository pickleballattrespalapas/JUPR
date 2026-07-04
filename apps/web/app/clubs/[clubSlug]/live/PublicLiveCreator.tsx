"use client";

import { useMemo, useState } from "react";

type PublicLiveCreatorProps = {
  apiBase: string | null;
  clubSlug: string;
};

const defaultNames = "Amy\nBrooke\nChris\nDana";

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

export default function PublicLiveCreator({ apiBase, clubSlug }: PublicLiveCreatorProps) {
  const [eventName, setEventName] = useState("Live Round Robin");
  const [participantText, setParticipantText] = useState(defaultNames);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const participantCount = useMemo(
    () => participantText.split(/\r?\n|,/).map((name) => name.trim()).filter(Boolean).length,
    [participantText]
  );

  async function createSession() {
    if (!apiBase) {
      setError("The public API base URL is not configured for this deployment.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const response = await fetch(apiUrl(apiBase, `/clubs/${clubSlug}/live-sessions`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          event_name: eventName,
          event_type: "round_robin",
          participant_names: participantText.split(/\r?\n|,/).map((name) => name.trim()).filter(Boolean)
        })
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(String(payload?.detail || `API error (${response.status})`));
      }
      const sessionKey = String(payload?.session?.session_key || "");
      const editToken = String(payload?.edit_token || "");
      if (!sessionKey || !editToken) {
        throw new Error("The API did not return a live session edit link.");
      }
      window.location.href = `/clubs/${clubSlug}/live/${sessionKey}?edit=${encodeURIComponent(editToken)}`;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to create live session.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <section style={{ border: "1px solid #bfdbfe", borderRadius: "14px", padding: "1rem", background: "#eff6ff", marginBottom: "1rem" }}>
      <h2 style={{ marginTop: 0, fontSize: "1.2rem" }}>Start your own live event</h2>
      <p style={{ color: "#334155" }}>
        Create a public round robin, enter scores from this browser, and share the scoreboard link with players.
      </p>
      <div style={{ display: "grid", gap: "0.75rem" }}>
        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Event name
          <input
            value={eventName}
            onChange={(event) => setEventName(event.target.value)}
            style={{ padding: "0.6rem", borderRadius: "8px", border: "1px solid #cbd5e1", font: "inherit" }}
          />
        </label>
        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Player names ({participantCount})
          <textarea
            value={participantText}
            onChange={(event) => setParticipantText(event.target.value)}
            rows={7}
            style={{ padding: "0.6rem", borderRadius: "8px", border: "1px solid #cbd5e1", font: "inherit" }}
          />
        </label>
        <div>
          <button
            type="button"
            onClick={createSession}
            disabled={submitting}
            style={{ border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#2563eb", color: "white", fontWeight: 800, cursor: submitting ? "default" : "pointer" }}
          >
            {submitting ? "Creating…" : "Create live event"}
          </button>
        </div>
        {error ? <p style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
      </div>
    </section>
  );
}
