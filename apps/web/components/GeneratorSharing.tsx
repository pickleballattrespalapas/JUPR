"use client";

import { useState } from "react";

type Props = {
  apiBase: string | null;
  clubId: string;
  sessionKey: string;
  version: string | number;
  adminToken?: string;
  ownerToken?: string;
  shared?: boolean;
  locked?: boolean;
  disabled?: boolean;
  onChanged?: () => Promise<void>;
};

const buttonStyle = { padding: "0.5rem 0.8rem", border: "1px solid #cbd5e1", borderRadius: "999px", background: "white", color: "#0f172a", fontWeight: 800, cursor: "pointer" };

export default function GeneratorSharing({ apiBase, clubId, sessionKey, version, adminToken, ownerToken, shared = false, locked = false, disabled = false, onChanged }: Props) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [link, setLink] = useState("");
  const [linkKind, setLinkKind] = useState<"view" | "handoff">("view");
  const canManage = !shared && Boolean(adminToken || ownerToken);

  async function copy(url: string): Promise<boolean> {
    setLink(url);
    try {
      await navigator.clipboard.writeText(url);
      return true;
    } catch {
      return false;
    }
  }

  async function share(action: "view" | "handoff" | "revoke"): Promise<void> {
    if (busy || disabled) return;
    setBusy(true);
    setMessage("");
    setLink("");
    try {
      let slug = clubId;
      let token = "";
      if (!shared) {
        if (!apiBase || !canManage) throw new Error("Use the organizer link to manage sharing.");
        const prefix = adminToken ? `/admin/clubs/${encodeURIComponent(clubId)}` : `/clubs/${encodeURIComponent(clubId)}`;
        const response = await fetch(`${apiBase.replace(/\/$/, "")}${prefix}/play-generators/sessions/${encodeURIComponent(sessionKey)}/sharing`, {
          method: "POST", headers: { "Content-Type": "application/json", ...(adminToken ? { Authorization: `Bearer ${adminToken}` } : {}) },
          body: JSON.stringify({ action, expected_version: version, ...(ownerToken ? { edit_token: ownerToken } : {}) })
        });
        const payload = await response.json().catch(() => null);
        if (!response.ok) throw new Error(response.status === 409 ? "This RR changed. Refresh before sharing." : "We couldn’t update sharing. Refresh and try again.");
        slug = payload.sharing.club_slug;
        token = payload.sharing.handoff_token || "";
        await onChanged?.();
      }
      if (action === "revoke") {
        setMessage("Handoff link revoked. The standings link still works.");
        return;
      }
      if (action === "handoff" && !token) throw new Error("We couldn’t create the handoff link. Try again.");
      const base = `${window.location.origin}/clubs/${encodeURIComponent(slug)}/round-robin-generator/shared/${encodeURIComponent(sessionKey)}`;
      const url = action === "view" ? `${base}/standings` : `${base}#edit=${encodeURIComponent(token)}`;
      setLinkKind(action);
      const copied = await copy(url);
      setMessage(copied ? (action === "view" ? "Standings link copied." : "Private handoff link copied.") : "Select and copy the link below.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "We couldn’t create this link. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <article aria-label="Round-Robin sharing" style={{ border: "1px solid #e2e8f0", borderRadius: "14px", padding: "0.9rem", background: "#f8fafc" }}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
        <button type="button" disabled={busy || disabled} onClick={() => void share("view")} style={buttonStyle}>Share standings</button>
        {canManage ? <>
          <button type="button" disabled={busy || disabled || locked} onClick={() => void share("handoff")} style={buttonStyle}>Hand off RR</button>
          <button type="button" disabled={busy || disabled} onClick={() => void share("revoke")} style={buttonStyle}>Revoke handoff link</button>
        </> : null}
      </div>
      <p style={{ margin: "0.6rem 0 0", color: "#475569", fontSize: "0.88rem" }}>
        {canManage ? "Standings are view-only. Keep the handoff link private: it lets someone run this RR. You can revoke it here." : "Share the full standings with players. Results refresh automatically."}
      </p>
      {message ? <p role="status" style={{ marginBottom: 0 }}>{message}</p> : null}
      {link ? <input aria-label={linkKind === "view" ? "Standings link" : "Private handoff link"} readOnly value={link} onFocus={event => event.target.select()} style={{ width: "100%", boxSizing: "border-box", marginTop: "0.5rem", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px" }} /> : null}
    </article>
  );
}
