"use client";

import { useEffect, useRef, useState } from "react";

type Props = { apiBase: string | null; clubSlug: string; teamId: string };

function key(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) return `partner:${crypto.randomUUID()}`;
  return `partner:${Date.now()}:${Math.random().toString(16).slice(2)}`;
}

export default function PartnerConfirmationPanel({ apiBase, clubSlug, teamId }: Props) {
  const token = useRef("");
  const operationKey = useRef(key());
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    const fragment = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    token.current = fragment.get("token") || "";
    window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}`);
    setReady(Boolean(token.current && teamId));
  }, [teamId]);

  async function respond(accept: boolean) {
    if (!apiBase || !token.current || !teamId) {
      setMessage("This invitation link is incomplete or expired.");
      return;
    }
    setBusy(true);
    setMessage(null);
    try {
      const response = await fetch(
        `${apiBase.replace(/\/$/, "")}/clubs/${encodeURIComponent(clubSlug)}/team-leagues/partner-confirmations`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            team_id: teamId,
            token: token.current,
            accept,
            idempotency_key: operationKey.current
          })
        }
      );
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = typeof payload?.detail === "object" ? payload.detail.message : payload?.detail;
        throw new Error(String(detail || `Unable to respond (${response.status}).`));
      }
      token.current = "";
      setReady(false);
      setMessage(String(payload.message || (accept ? "Team confirmed." : "Invitation declined.")));
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to respond.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <article style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white", maxWidth: "620px" }}>
      <h2 style={{ marginTop: 0 }}>Partner invitation</h2>
      <p style={{ color: "#475569" }}>
        Confirm only if you agreed to keep this partner for the team-league season.
      </p>
      {!ready && !message ? <p style={{ color: "#b91c1c" }}>This invitation link is incomplete or expired.</p> : null}
      {ready ? (
        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <button type="button" disabled={busy} onClick={() => respond(true)} style={{ padding: "0.7rem 1rem", border: 0, borderRadius: "999px", background: "#0f172a", color: "white", fontWeight: 800 }}>
            {busy ? "Saving…" : "Accept and join team"}
          </button>
          <button type="button" disabled={busy} onClick={() => respond(false)} style={{ padding: "0.7rem 1rem", border: "1px solid #cbd5e1", borderRadius: "999px", background: "white", color: "#991b1b", fontWeight: 800 }}>
            Decline
          </button>
        </div>
      ) : null}
      {message ? <p role="status">{message}</p> : null}
    </article>
  );
}
