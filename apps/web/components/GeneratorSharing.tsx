"use client";

import { useEffect, useState } from "react";
import { QRCodeSVG } from "qrcode.react";

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
  const [standingsUrl, setStandingsUrl] = useState("");
  const canManage = !shared && Boolean(adminToken || ownerToken);

  useEffect(() => {
    setStandingsUrl(shared ? `${window.location.origin}/clubs/${encodeURIComponent(clubId)}/round-robin-generator/shared/${encodeURIComponent(sessionKey)}/standings` : "");
    setLink("");
  }, [shared, clubId, sessionKey]);

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
      let viewingEnabled = shared;
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
        viewingEnabled = Boolean(payload.sharing.enabled);
        await onChanged?.();
      }
      const base = `${window.location.origin}/clubs/${encodeURIComponent(slug)}/round-robin-generator/shared/${encodeURIComponent(sessionKey)}`;
      if (viewingEnabled) setStandingsUrl(`${base}/standings`);
      if (action === "revoke") {
        setMessage("Handoff link revoked. The standings link still works.");
        return;
      }
      if (action === "handoff" && !token) throw new Error("We couldn’t create the handoff link. Try again.");
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
      {standingsUrl ? (
        <section aria-label="Share standings by QR code" style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "1rem", marginTop: "1rem" }}>
          <div style={{ padding: "0.5rem", background: "white", border: "1px solid #e2e8f0", borderRadius: "10px", flexShrink: 0 }}>
            <QRCodeSVG value={standingsUrl} size={224} marginSize={4} level="M" bgColor="#ffffff" fgColor="#000000" role="img" aria-label="Standings QR code" title="Scan to open the full round-robin standings" />
          </div>
          <div style={{ flex: "1 1 14rem", minWidth: 0 }}>
            <h3 style={{ margin: "0 0 0.4rem" }}>Scan for live standings</h3>
            <p style={{ color: "#475569", margin: "0 0 0.65rem" }}>Show this code courtside. Players can scan it with their phone camera to follow the full standings.</p>
            <input aria-label="Standings link" readOnly value={standingsUrl} onFocus={event => event.target.select()} style={{ width: "100%", boxSizing: "border-box", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", background: "white" }} />
            <a href={standingsUrl} style={{ display: "inline-block", marginTop: "0.6rem", fontWeight: 700 }}>Open standings</a>
          </div>
        </section>
      ) : null}
      {link && linkKind === "handoff" ? <input aria-label="Private handoff link" readOnly value={link} onFocus={event => event.target.select()} style={{ width: "100%", boxSizing: "border-box", marginTop: "0.5rem", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px" }} /> : null}
    </article>
  );
}
