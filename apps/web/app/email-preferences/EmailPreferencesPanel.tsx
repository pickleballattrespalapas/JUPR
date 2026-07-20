"use client";

import { useState } from "react";
import type { PublicEmailPreferencesResponse, PublicEmailUnsubscribeResponse } from "@/lib/emailPreferencesApi";
import { unsubscribeEmailPreferences } from "@/lib/emailPreferencesApi";

type Props = {
  initial: PublicEmailPreferencesResponse | null;
  token?: string | null;
  ut?: string | null;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.65rem 0.9rem", borderRadius: "999px", border: "1px solid #991b1b", background: "#991b1b", color: "white", fontWeight: 800 };

function statusText(value?: string | null): string {
  return String(value || "unknown").replace(/_/g, " ");
}

export default function EmailPreferencesPanel({ initial, token, ut }: Props) {
  const [scope, setScope] = useState("player_updates");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<PublicEmailUnsubscribeResponse | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const subscription = result?.subscription || initial?.subscription || null;
  const currentStatus = subscription?.request_status || "";
  const alreadyUnsubscribed = String(currentStatus).toLowerCase() === "unsubscribed";
  const storedScope = String(subscription?.preferences_json?.unsubscribe_scope || "").toLowerCase();
  const globallyUnsubscribed = storedScope === "global" || subscription?.preferences_json?.optional_emails_enabled === false;
  const selectedScopeAlreadyApplied = alreadyUnsubscribed && (scope === "player_updates" || globallyUnsubscribed);

  async function unsubscribe() {
    setBusy(true);
    setMessage(null);
    try {
      const response = await unsubscribeEmailPreferences({ token, ut, scope });
      if (response.error) throw new Error(response.error);
      setResult(response.data);
      setMessage(response.data?.message || "Your preference has been updated.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to update email preferences.");
    } finally {
      setBusy(false);
    }
  }

  if (!initial?.found) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Preference link not found</h2>
        <p style={{ color: "#475569" }}>{initial?.message || "Use the unsubscribe or preference link from one of your player update emails."}</p>
        <p style={{ color: "#475569", marginBottom: 0 }}>Need help? Contact joe@juprleagues.com.</p>
      </article>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Verified player update subscription</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
          <div><strong>Email</strong><br />{subscription?.email_masked || "—"}</div>
          <div><strong>Player ID</strong><br />{subscription?.player_id ?? "—"}</div>
          <div><strong>Status</strong><br />{statusText(currentStatus)}</div>
          <div><strong>Verified</strong><br />{subscription?.verified_at ? String(subscription.verified_at).slice(0, 10) : "—"}</div>
        </div>
        {alreadyUnsubscribed ? <p style={{ color: "#166534" }}>This subscription is unsubscribed{globallyUnsubscribed ? " from all optional email categories" : " from player updates"}.</p> : null}
      </article>

      <article style={{ ...cardStyle, borderColor: "#fecaca", background: "#fef2f2" }}>
        <h2 style={{ marginTop: 0 }}>Unsubscribe</h2>
        <p style={{ color: "#7f1d1d" }}>
          You can unsubscribe from optional JUPR player update emails. Transactional emails, such as password reset and tournament registration confirmations, are not controlled by this preference link.
        </p>
        <label><strong>Scope</strong><br />
          <select value={scope} onChange={(event) => setScope(event.target.value)} style={inputStyle}>
            <option value="player_updates">Player update emails</option>
            <option value="global">All optional JUPR emails now and in the future</option>
          </select>
        </label>
        <p style={{ color: "#7f1d1d" }}>{scope === "global" ? "Global scope opts this address out of every optional category managed by JUPR, including categories added later. Transactional service messages remain enabled." : "Category scope disables verified player update digests only."}</p>
        <p><button type="button" disabled={busy || selectedScopeAlreadyApplied} onClick={unsubscribe} style={buttonStyle}>{busy ? "Updating…" : selectedScopeAlreadyApplied ? "Preference already applied" : alreadyUnsubscribed && scope === "global" ? "Expand to global opt-out" : "Unsubscribe"}</button></p>
        {message ? <p aria-live="polite" style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("api error") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>
    </section>
  );
}
