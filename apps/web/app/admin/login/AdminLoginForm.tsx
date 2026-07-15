"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import {
  AdminSession,
  clearAdminSession,
  consumeHashSession,
  getAdminAuthConfig,
  loadAdminSession,
  refreshAdminSession,
  sendMagicLink,
  signInWithPassword,
  signOutAdminSession
} from "@/lib/adminAuthClient";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.6rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#2563eb", color: "white", fontWeight: 800, cursor: "pointer" };
const ghostButtonStyle = { border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.65rem 1rem", background: "white", color: "#0f172a", fontWeight: 800, cursor: "pointer" };

function sessionEmail(session: AdminSession | null): string {
  return session?.user?.email || "signed-in admin";
}

function expirationLabel(session: AdminSession | null): string {
  if (!session?.expires_at) return "unknown";
  const date = new Date(Number(session.expires_at));
  if (Number.isNaN(date.getTime())) return "unknown";
  return date.toLocaleString();
}

export default function AdminLoginForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [session, setSession] = useState<AdminSession | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const config = getAdminAuthConfig();

  useEffect(() => {
    const fromHash = consumeHashSession();
    const initial = fromHash || loadAdminSession();
    if (initial) setSession(initial);
    refreshAdminSession(initial || undefined)
      .then((refreshed) => {
        if (refreshed) setSession(refreshed);
      })
      .catch((error) => setMessage(error instanceof Error ? error.message : "Unable to refresh admin session."));
  }, []);

  async function onPasswordSignIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setMessage(null);
    try {
      const signedIn = await signInWithPassword(email, password);
      setSession(signedIn);
      setPassword("");
      setMessage("Signed in. Admin pages can now use this Supabase access token for FastAPI authorization.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to sign in.");
    } finally {
      setBusy(false);
    }
  }

  async function onMagicLink() {
    setBusy(true);
    setMessage(null);
    try {
      await sendMagicLink(email, typeof window === "undefined" ? undefined : `${window.location.origin}/admin/login`);
      setMessage("Magic link sent. Open it in this browser to store the admin session.");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Unable to send magic link.");
    } finally {
      setBusy(false);
    }
  }

  async function onSignOut() {
    setBusy(true);
    setMessage(null);
    try {
      await signOutAdminSession();
      setSession(null);
      setMessage("Signed out.");
    } finally {
      setBusy(false);
    }
  }

  if (!config) {
    return (
      <article style={{ ...cardStyle, borderColor: "#fbbf24", background: "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Admin auth is not configured</h2>
        <p style={{ color: "#92400e" }}>
          Set <code>NEXT_PUBLIC_SUPABASE_URL</code> and <code>NEXT_PUBLIC_SUPABASE_ANON_KEY</code> in the Vercel environment before using the Next admin login shell.
        </p>
        <p style={{ marginBottom: 0 }}><Link href="/admin">Back to operations cockpit</Link></p>
      </article>
    );
  }

  if (session?.access_token) {
    return (
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Signed in</h2>
        <p style={{ color: "#475569" }}>Current browser session: <strong>{sessionEmail(session)}</strong></p>
        <p style={{ color: "#64748b" }}>Token expiry: {expirationLabel(session)}</p>
        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <Link href="/admin/match-uploader">Open Match Uploader</Link>
          <Link href="/admin/match-log">Open Match Log</Link>
          <Link href="/admin/players">Open Player Editor</Link>
          <Link href="/admin">Operations cockpit</Link>
          <button type="button" onClick={onSignOut} disabled={busy} style={ghostButtonStyle}>Sign out</button>
          <button type="button" onClick={() => { clearAdminSession(); setSession(null); setMessage("Local session cleared."); }} style={ghostButtonStyle}>Clear local session</button>
        </div>
        {message ? <p style={{ color: message.includes("Signed") || message.includes("cleared") ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      </article>
    );
  }

  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Sign in with Supabase Auth</h2>
      <p style={{ color: "#475569" }}>
        This stores only the short-lived Supabase user access token in this browser and passes it to FastAPI admin routes as <code>Authorization: Bearer</code>. Service-role keys never go to the browser.
      </p>
      <form onSubmit={onPasswordSignIn} style={{ display: "grid", gap: "0.75rem", maxWidth: "520px" }}>
        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Email
          <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" required style={inputStyle} />
        </label>
        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Password
          <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" style={inputStyle} />
        </label>
        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <button type="submit" disabled={busy || !email || !password} style={{ ...buttonStyle, cursor: busy || !email || !password ? "default" : "pointer", background: busy || !email || !password ? "#94a3b8" : "#2563eb" }}>
            {busy ? "Working…" : "Sign in"}
          </button>
          <button type="button" onClick={onMagicLink} disabled={busy || !email} style={ghostButtonStyle}>Send magic link</button>
          <Link href="/admin/reset-password" style={ghostButtonStyle}>Reset password</Link>
        </div>
      </form>
      {message ? <p style={{ color: message.includes("sent") || message.includes("Signed") ? "#166534" : "#b91c1c" }}>{message}</p> : null}
    </article>
  );
}
