"use client";

import { useState } from "react";
import { useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";

type Breadcrumb = { id: string; created_at: string; source: string; event: string; stage: string; message: string;
  error_code: string; selections: { event_option_id: string; event_label?: string; partner_name?: string; same_email_as_player?: boolean }[] };
type Attempt = { attempt_id: string; name: string; email: string; outcome: string; stage: string; last_seen_at: string; events: Breadcrumb[] };
type Props = { apiBase: string | null; clubId: string; tournamentId: string; accessToken: string | null };

export default function RegistrationAttemptHistory({ apiBase, clubId, tournamentId, accessToken }: Props) {
  const [search, setSearch] = useState("");
  const [attempts, setAttempts] = useState<Attempt[] | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const request = useLatestRequestGuard(`${apiBase}\u0000${clubId}\u0000${tournamentId}\u0000${accessToken}`, () => {
    setAttempts(null); setSearch(""); setMessage(""); setBusy(false);
  });

  async function load() {
    if (!apiBase || !accessToken) return;
    const generation = request.begin();
    setBusy(true); setMessage(""); setAttempts(null);
    try {
      const path = `/admin/clubs/${encodeURIComponent(clubId)}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/registration-attempts?search=${encodeURIComponent(search.trim())}`;
      const response = await fetch(`${apiBase.replace(/\/$/, "")}${path}`, {
        headers: { Authorization: `Bearer ${accessToken}` }, cache: "no-store"
      });
      const payload = await response.json();
      if (!response.ok) throw new Error("Registration support history is unavailable. Try again later.");
      if (!request.isCurrent(generation)) return;
      setAttempts(payload.attempts || []);
      if (payload.truncated) setMessage("Showing the latest matches. Search by full email or name to narrow the results.");
    } catch (error) {
      if (request.isCurrent(generation)) setMessage(error instanceof Error ? error.message : "Could not load support history.");
    } finally {
      if (request.isCurrent(generation)) setBusy(false);
    }
  }

  return <article style={{ border: "1px solid #e2e8f0", borderRadius: 14, padding: "1rem", background: "white" }}>
    <h2 style={{ marginTop: 0 }}>Registration attempt history</h2>
    <p>Search the last 30 days for people who had trouble registering. These support records do not put someone on the tournament roster.</p>
    <form onSubmit={event => { event.preventDefault(); void load(); }} style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", alignItems: "end" }}>
      <label style={{ flex: "1 1 220px" }}>Name or full email
        <input value={search} onChange={event => setSearch(event.target.value)} maxLength={160}
          style={{ display: "block", width: "100%", boxSizing: "border-box", padding: "0.6rem" }} />
      </label>
      <button type="submit" disabled={busy || !accessToken}>{busy ? "Searching…" : "Search attempts"}</button>
    </form>
    {message ? <p role="status">{message}</p> : null}
    {attempts?.length === 0 ? <p>No matching attempts were recorded. Browser or connection failures can prevent support records from reaching us.</p> : null}
    {attempts?.map(attempt => <details key={attempt.attempt_id} style={{ marginTop: "1rem", overflowWrap: "anywhere" }}>
      <summary style={{ cursor: "pointer" }}><strong>{attempt.name || "Name not entered"}</strong> · {attempt.email || "Email not entered"} · {attempt.outcome} · Last step: {attempt.stage}</summary>
      <p>Last activity: {new Date(attempt.last_seen_at).toLocaleString()}. Contact details in an unfinished form are unverified.</p>
      <ol>{attempt.events.map(event => <li key={event.id} style={{ marginTop: "0.6rem" }}>
        <time>{new Date(event.created_at).toLocaleString()}</time> · {event.stage} · {event.event.replaceAll("_", " ").toLowerCase()}
        {event.source === "BROWSER" ? " (browser reported)" : " (server recorded)"}
        {event.message ? <p style={{ margin: "0.25rem 0", color: "#b91c1c" }}>{event.message}</p> : null}
        {event.selections?.map(selection => <p key={selection.event_option_id} style={{ margin: "0.25rem 0" }}>
          Event: {selection.event_label || "Selected event"}{selection.partner_name ? ` · Partner: ${selection.partner_name}` : ""}
          {selection.same_email_as_player ? " · Same email entered for player and partner" : ""}
        </p>)}
      </li>)}</ol>
    </details>)}
  </article>;
}
