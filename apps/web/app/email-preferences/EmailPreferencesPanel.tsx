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
  switch (String(value || "").toLowerCase()) {
    case "active":
      return "Active";
    case "pending_admin_review":
      return "Awaiting club approval";
    case "rejected":
      return "Request not approved";
    case "unsubscribed":
      return "Unsubscribed";
    default:
      return "Unavailable";
  }
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
      if (response.error) throw new Error();
      setResult(response.data);
      setMessage("Your email preferences have been updated.");
    } catch {
      setMessage("We couldn’t update your email preferences. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  if (!initial?.found) {
    return (
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Preference link not found</h2>
        <p style={{ color: "#475569" }}>Use the unsubscribe or preferences link from one of your player update emails.</p>
        <p style={{ color: "#475569", marginBottom: 0 }}>Need help? Contact joe@juprleagues.com.</p>
      </article>
    );
  }

  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Your player updates</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
          <div><strong>Email</strong><br />{subscription?.email_masked || "—"}</div>
          <div><strong>Subscription status</strong><br />{statusText(currentStatus)}</div>
          <div><strong>Confirmed on</strong><br />{subscription?.verified_at ? String(subscription.verified_at).slice(0, 10) : "—"}</div>
        </div>
        {alreadyUnsubscribed ? <p style={{ color: "#166534" }}>You are no longer receiving {globallyUnsubscribed ? "optional JUPR emails" : "player update emails"}.</p> : null}
      </article>

      <article style={{ ...cardStyle, borderColor: "#fecaca", background: "#fef2f2" }}>
        <h2 style={{ marginTop: 0 }}>Unsubscribe</h2>
        <p style={{ color: "#7f1d1d" }}>
          You&apos;ll still receive essential emails, such as password resets and tournament confirmations.
        </p>
        <label><strong>Unsubscribe from</strong><br />
          <select value={scope} onChange={(event) => setScope(event.target.value)} style={inputStyle}>
            <option value="player_updates">Player update emails</option>
            <option value="global">All optional JUPR emails now and in the future</option>
          </select>
        </label>
        <p style={{ color: "#7f1d1d" }}>{scope === "global" ? "Stop all optional JUPR emails, including any added later. Essential service emails will still arrive." : "Stop player update emails only."}</p>
        <p><button type="button" disabled={busy || selectedScopeAlreadyApplied} onClick={unsubscribe} style={buttonStyle}>{busy ? "Updating…" : selectedScopeAlreadyApplied ? "Already unsubscribed" : alreadyUnsubscribed && scope === "global" ? "Unsubscribe from all optional emails" : "Unsubscribe"}</button></p>
        {message ? <p aria-live="polite" style={{ color: message.toLowerCase().includes("couldn’t") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>
    </section>
  );
}
