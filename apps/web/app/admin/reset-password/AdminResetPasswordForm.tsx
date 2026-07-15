"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import {
  consumeHashSession,
  getAdminAuthConfig,
  loadAdminSession,
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
  const config = getAdminAuthConfig();

  useEffect(() => {
    const fromHash = consumeHashSession();
    setSession(fromHash || loadAdminSession());
    if (fromHash?.access_token) setMessage("Password reset session loaded. Enter a new password below.");
  }, []);

  async function requestReset(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setMessage(null);
    try {
      await sendPasswordResetEmail(email, typeof window === "undefined" ? undefined : `${window.location.origin}/admin/reset-password`);
      setMessage("Password reset email sent. Open the link in this browser, then set your new password.");
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
      if (newPassword.length < 8) throw new Error("New password must be at least 8 characters.");
      if (newPassword !== confirmPassword) throw new Error("Password confirmation does not match.");
      await updateAdminPassword(newPassword, session || undefined);
      setNewPassword("");
      setConfirmPassword("");
      setMessage("Password updated. You can now sign in with the new password.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to update password.");
    } finally {
      setBusy(false);
    }
  }

  if (!config) {
    return (
      <article style={{ ...cardStyle, borderColor: "#fbbf24", background: "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Admin auth is not configured</h2>
        <p style={{ color: "#92400e" }}>
          Set <code>NEXT_PUBLIC_SUPABASE_URL</code> and <code>NEXT_PUBLIC_SUPABASE_ANON_KEY</code> in the Vercel environment before using password reset.
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
            {busy ? "Working…" : "Send reset email"}
          </button>
        </form>
      </article>

      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Set new password</h2>
        <p style={{ color: "#475569" }}>
          Current browser recovery session: <strong>{sessionLabel(session)}</strong>
        </p>
        <form onSubmit={updatePassword} style={{ display: "grid", gap: "0.75rem", maxWidth: "520px" }}>
          <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
            New password
            <input type="password" value={newPassword} onChange={(event) => setNewPassword(event.target.value)} autoComplete="new-password" required minLength={8} style={inputStyle} />
          </label>
          <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
            Confirm new password
            <input type="password" value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} autoComplete="new-password" required minLength={8} style={inputStyle} />
          </label>
          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <button type="submit" disabled={busy || !session?.access_token || !newPassword || !confirmPassword} style={{ ...buttonStyle, cursor: busy || !session?.access_token || !newPassword || !confirmPassword ? "default" : "pointer", background: busy || !session?.access_token || !newPassword || !confirmPassword ? "#94a3b8" : "#2563eb" }}>
              {busy ? "Working…" : "Update password"}
            </button>
            <Link href="/admin/login" style={ghostButtonStyle}>Back to login</Link>
          </div>
        </form>
        {message ? <p style={{ color: message.includes("sent") || message.includes("updated") || message.includes("loaded") ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      </article>
    </div>
  );
}
