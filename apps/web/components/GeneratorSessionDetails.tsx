"use client";

import { useState } from "react";

type Details = { title: string; version: string | number; rating_mode?: "rated" | "unrated" };
type Props<T extends Details> = {
  apiBase: string | null; clubId: string; sessionKey: string; session: T;
  adminToken?: string; ownerToken?: string; locked: boolean; disabled?: boolean;
  scored?: boolean; onSaved: (session: T) => void;
};
const field = { padding: ".6rem .75rem", border: "1px solid #cbd5e1", borderRadius: 8, font: "inherit" };

export default function GeneratorSessionDetails<T extends Details>({ apiBase, clubId, sessionKey, session,
  adminToken, ownerToken, locked, disabled = false, scored = true, onSaved }: Props<T>) {
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(session.title);
  const [rating, setRating] = useState(session.rating_mode || "unrated");
  const [saving, setSaving] = useState(false);
  const [baseVersion, setBaseVersion] = useState(session.version);
  const [message, setMessage] = useState("");
  const cleanTitle = title.replace(/[<>]/g, "").trim().replace(/\s+/g, " ");
  const changed = cleanTitle !== session.title || rating !== (session.rating_mode || "unrated");
  const valid = cleanTitle.length > 0 && cleanTitle.length <= 160 && (scored || rating === "unrated");
  if (!adminToken && !ownerToken) return null;

  async function save() {
    if (!apiBase || saving || disabled || locked || !valid || !changed) return;
    setSaving(true); setMessage("");
    try {
      const prefix = adminToken ? "/admin/clubs" : "/clubs";
      const response = await fetch(`${apiBase.replace(/\/$/, "")}${prefix}/${encodeURIComponent(clubId)}/play-generators/sessions/${encodeURIComponent(sessionKey)}/details`, {
        method: "PATCH", headers: { "Content-Type": "application/json", ...(adminToken ? { Authorization: `Bearer ${adminToken}` } : {}) },
        body: JSON.stringify({ title: cleanTitle, rating_mode: rating, expected_version: baseVersion,
          ...(!adminToken ? { edit_token: ownerToken, idempotency_key: `details-${Date.now()}-${Math.random().toString(16).slice(2)}` } : {}) })
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) throw new Error(response.status < 500 && typeof payload?.detail === "string" ? payload.detail : "Could not save session details. Refresh and try again.");
      if (!payload?.session) throw new Error("Could not confirm the change. Refresh the session before trying again.");
      onSaved(payload.session as T); setEditing(false); setMessage("Session details saved.");
    } catch (error) { setMessage(error instanceof Error ? error.message : "Could not save session details."); }
    finally { setSaving(false); }
  }

  return <section aria-label="Session details" style={{ marginTop: ".8rem" }}>
    {locked ? <p style={{ color: "#64748b" }}>Session details are locked after submission. A club administrator can correct them while results are awaiting approval.</p>
      : editing ? <form aria-label="Edit session details" onSubmit={event => { event.preventDefault(); void save(); }} style={{ display: "grid", gap: ".65rem", maxWidth: 560 }}>
        <label style={{ display: "grid", gap: 4 }}>Session title<input aria-label="Session title" value={title} maxLength={160} required disabled={saving || disabled} onChange={event => setTitle(event.target.value)} style={field} /></label>
        <label style={{ display: "grid", gap: 4 }}>Session rating<select aria-label="Session rating" value={rating} disabled={saving || disabled} onChange={event => setRating(event.target.value as "rated" | "unrated")} style={field}>
          <option value="unrated">Unrated</option><option value="rated" disabled={!scored}>Rated</option>
        </select></label>
        <p style={{ margin: 0 }}>{rating === "rated" ? "Saved games will update club ratings after administrator approval." : "Saved games will not change player ratings."}</p>
        {!scored ? <p style={{ margin: 0 }}>Unscored sessions must stay unrated.</p> : null}
        <div style={{ display: "flex", gap: ".5rem" }}>
          <button type="submit" disabled={saving || disabled || !valid || !changed} style={field}>{saving ? "Saving…" : "Save session details"}</button>
          <button type="button" disabled={saving} onClick={() => { setEditing(false); setMessage(""); }} style={field}>Cancel</button>
        </div>
      </form> : <button type="button" disabled={disabled || !apiBase} style={field} onClick={() => { setBaseVersion(session.version); setTitle(session.title); setRating(session.rating_mode || "unrated"); setMessage(""); setEditing(true); }}>Edit session details</button>}
    {message ? <p role="status">{message}</p> : null}
  </section>;
}
