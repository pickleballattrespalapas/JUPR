"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import {
  ADMIN_PASSWORD_MIN_LENGTH,
  authorizeAdminSession,
  clearRecoveryArtifacts,
  consumeRecoverySession,
  finishPasswordRecovery,
  getAdminApiBaseUrl,
  getAdminAuthConfig,
  loadRecoverySession,
  saveRecoverySession,
  sendPasswordResetEmail,
  updateAdminPassword
} from "@/lib/adminAuthClient";
import type { AdminSession } from "@/lib/adminAuthClient";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.6rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#2563eb", color: "white", fontWeight: 800, cursor: "pointer" };
const ghostButtonStyle = { border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.65rem 1rem", background: "white", color: "#0f172a", fontWeight: 800, cursor: "pointer" };

function sessionLabel(session: AdminSession | null): string {
  return session?.user?.email || (session?.access_token ? "password reset session" : "not signed in");
}

export default function AdminResetPasswordForm() {
  const [email, setEmail] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [session, setSession] = useState<AdminSession | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [requestSent, setRequestSent] = useState(false);
  const config = getAdminAuthConfig();
  const apiBase = getAdminApiBaseUrl();

  useEffect(() => {
    let cancelled = false;
    async function restoreRecovery() {
      setBusy(true);
      let candidate: AdminSession | null = null;
      try {
        const callbackAttempted = typeof window !== "undefined" && (
          /(?:^|[?&])(code|token_hash|error|error_code)=/.test(window.location.search) ||
          /(?:^|[&#])(access_token|error|error_code)=/.test(window.location.hash)
        );
        const callbackSession = await consumeRecoverySession();
        candidate = callbackSession || loadRecoverySession();
        if (callbackAttempted && !candidate) {
          throw new Error("This password recovery link is invalid or expired. Request a new email.");
        }
        if (!candidate) return;
        // Keep a consumed one-time callback recoverable across a temporary API outage,
        // but never expose it as the general admin session.
        saveRecoverySession(candidate);
        const authorized = await authorizeAdminSession(candidate);
        if (cancelled) return;
        saveRecoverySession(authorized);
        setSession(authorized);
        setMessage("Password recovery link verified. Enter a new password below.");
      } catch (error) {
        const detail = error instanceof Error ? error.message : "";
        const terminal = detail.includes("invalid or expired") || detail.includes("not authorized");
        if (!candidate || terminal) clearRecoveryArtifacts();
        if (!cancelled) {
          setSession(null);
          setMessage(error instanceof Error ? error.message : "This password recovery link is invalid or expired. Request a new email.");
        }
      } finally {
        if (!cancelled) setBusy(false);
      }
    }
    void restoreRecovery();
    return () => { cancelled = true; };
  }, []);

  async function requestReset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setMessage(null);
    try {
      await sendPasswordResetEmail(email, typeof window === "undefined" ? undefined : `${window.location.origin}/admin/reset-password`);
      setRequestSent(true);
      setMessage("If this is an eligible admin account, a recovery email has been sent. Open the newest link in this browser.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to send password reset email.");
    } finally {
      setBusy(false);
    }
  }

  async function updatePassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setMessage(null);
    try {
      if (newPassword.length < ADMIN_PASSWORD_MIN_LENGTH) throw new Error(`New password must be at least ${ADMIN_PASSWORD_MIN_LENGTH} characters.`);
      if (newPassword !== confirmPassword) throw new Error("Password confirmation does not match.");
      await updateAdminPassword(newPassword, session || undefined);
      await finishPasswordRecovery(session);
      setNewPassword("");
      setConfirmPassword("");
      setSession(null);
      setMessage("Password updated. You can now sign in with the new password.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to update password.");
    } finally {
      setBusy(false);
    }
  }

  if (!config || !apiBase) {
    return (
      <article style={{ ...cardStyle, borderColor: "#fbbf24", background: "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Admin auth is not configured</h2>
        <p style={{ color: "#92400e" }}>
          Set the browser-safe Supabase URL/key and <code>NEXT_PUBLIC_JUPR_API_BASE_URL</code> in the Vercel environment before using password reset.
        </p>
        <p style={{ marginBottom: 0 }}><Link href="/admin/login">Back to admin login</Link></p>
      </article>
    );
  }

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Request password reset</h2>
        <p style={{ color: "#475569" }}>
          Send a Supabase Auth password recovery link to a staff admin email. The link should open this page in the same browser.
        </p>
        <form onSubmit={requestReset} style={{ display: "grid", gap: "0.75rem", maxWidth: "520px" }}>
          <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
            Email
            <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" required style={inputStyle} />
          </label>
          <button type="submit" disabled={busy || !email} style={{ ...buttonStyle, cursor: busy || !email ? "default" : "pointer", background: busy || !email ? "#94a3b8" : "#2563eb" }}>
            {busy ? "Working…" : requestSent ? "Resend recovery email" : "Send recovery email"}
          </button>
        </form>
        <p style={{ color: "#64748b", fontSize: "0.9rem", marginBottom: 0 }}>
          For privacy, this page gives the same response whether or not the address has an eligible account. Each resend replaces the prior PKCE recovery attempt; use the newest email.
        </p>
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Set new password</h2>
        <p style={{ color: "#475569" }}>
          Current browser recovery session: <strong>{sessionLabel(session)}</strong>
        </p>
        <p style={{ color: "#64748b", fontSize: "0.9rem" }}>
          Use at least {ADMIN_PASSWORD_MIN_LENGTH} characters. Supabase may require additional character types or reject leaked/reused passwords under the project policy.
        </p>
        <form onSubmit={updatePassword} style={{ display: "grid", gap: "0.75rem", maxWidth: "520px" }}>
          <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
            New password
            <input type="password" value={newPassword} onChange={(event) => setNewPassword(event.target.value)} autoComplete="new-password" required minLength={ADMIN_PASSWORD_MIN_LENGTH} style={inputStyle} />
          </label>
          <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
            Confirm new password
            <input type="password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} autoComplete="new-password" required minLength={ADMIN_PASSWORD_MIN_LENGTH} style={inputStyle} />
          </label>
          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <button type="submit" disabled={busy || !session?.access_token || !newPassword || !confirmPassword} style={{ ...buttonStyle, cursor: busy || !session?.access_token || !newPassword || !confirmPassword ? "default" : "pointer", background: busy || !session?.access_token || !newPassword || !confirmPassword ? "#94a3b8" : "#2563eb" }}>
              {busy ? "Working…" : "Update password"}
            </button>
            <Link href="/admin/login" style={ghostButtonStyle}>Back to login</Link>
          </div>
        </form>
        {message ? <p role="status" style={{ color: message.includes("sent") || message.includes("updated") || message.includes("verified") ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      </article>
    </div>
  );
}
