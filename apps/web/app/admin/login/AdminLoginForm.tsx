"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import {
  AdminSession,
  authorizeAndSaveAdminSession,
  consumeHashSession,
  getAdminApiBaseUrl,
  getAdminAuthConfig,
  restoreAuthorizedAdminSession,
  safeAdminNextPath,
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
  const apiBase = getAdminApiBaseUrl();
  const searchParams = useSearchParams();
  const nextPath = safeAdminNextPath(searchParams.get("next"));

  useEffect(() => {
    let cancelled = false;
    async function restore() {
      setBusy(true);
      setMessage(null);
      try {
        if (typeof window !== "undefined") {
          const hashParams = new URLSearchParams(window.location.hash.replace(/^#/, ""));
          if (hashParams.get("type") === "recovery") {
            window.location.replace(`/admin/reset-password${window.location.hash}`);
            return;
          }
        }
        const callbackSession = consumeHashSession();
        const authorized = callbackSession
          ? await authorizeAndSaveAdminSession(callbackSession)
          : await restoreAuthorizedAdminSession();
        if (cancelled) return;
        setSession(authorized);
        if (callbackSession && authorized) window.location.assign(nextPath);
      } catch (error) {
        if (!cancelled) {
          setSession(null);
          setMessage(error instanceof Error ? error.message : "Unable to verify admin session.");
        }
      } finally {
        if (!cancelled) setBusy(false);
      }
    }
    void restore();
    return () => { cancelled = true; };
  }, [nextPath]);

  async function onPasswordSignIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setMessage(null);
    try {
      const signedIn = await signInWithPassword(email, password);
      const authorized = await authorizeAndSaveAdminSession(signedIn);
      setSession(authorized);
      setPassword("");
      setMessage("Signed in and authorized. Opening the requested admin page…");
      window.location.assign(nextPath);
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
      const redirect = typeof window === "undefined"
        ? undefined
        : `${window.location.origin}/admin/login?next=${encodeURIComponent(nextPath)}`;
      await sendMagicLink(email, redirect);
      setMessage("If this is an eligible admin account, a sign-in link has been sent. Open the newest link in this browser.");
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

  if (!config || !apiBase) {
    return (
      <article style={{ ...cardStyle, borderColor: "#fbbf24", background: "#fffbeb" }}>
        <h2 style={{ marginTop: 0 }}>Admin auth is not configured</h2>
        <p style={{ color: "#92400e" }}>
          Set the browser-safe Supabase URL/key and <code>NEXT_PUBLIC_JUPR_API_BASE_URL</code> in the Vercel environment before using the Next admin login shell.
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
        <p style={{ color: "#475569" }}>
          Verified access: {session.capabilities?.assignments.map((item) => `${item.club_id} (${item.role})`).join(", ") || "assigned admin workspace"}
        </p>
        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <Link href="/admin/match-uploader">Open Match Uploader</Link>
          <Link href="/admin/match-log">Open Match Log</Link>
          <Link href="/admin/players">Open Player Editor</Link>
          <Link href="/admin">Operations cockpit</Link>
          <button type="button" onClick={onSignOut} disabled={busy} style={ghostButtonStyle}>Sign out</button>
        </div>
        {message ? <p style={{ color: message.includes("Signed") || message.includes("cleared") ? "#166534" : "#b91c1c" }}>{message}</p> : null}
      </article>
    );
  }

  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Sign in with Supabase Auth</h2>
      <p style={{ color: "#475569" }}>
        This browser stores the Supabase access and refresh session needed for staff continuity. FastAPI verifies the JWT and a club-scoped role assignment before the session becomes usable. Service-role keys never go to the browser.
      </p>
      <form onSubmit={onPasswordSignIn} style={{ display: "grid", gap: "0.75rem", maxWidth: "520px" }}>
        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Email
          <input type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" required style={inputStyle} />
        </label>
        <label style={{ display: "grid", gap: "0.25rem", fontWeight: 700 }}>
          Password
          <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" required style={inputStyle} />
        </label>
        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <button type="submit" disabled={busy || !email || !password} style={{ ...buttonStyle, cursor: busy || !email || !password ? "default" : "pointer", background: busy || !email || !password ? "#94a3b8" : "#2563eb" }}>
            {busy ? "Working…" : "Sign in"}
          </button>
          <button type="button" onClick={onMagicLink} disabled={busy || !email} style={ghostButtonStyle}>Send magic link</button>
          <Link href="/admin/reset-password" style={ghostButtonStyle}>Reset password</Link>
        </div>
      </form>
      {message ? <p style={{ color: message.includes("sent") || message.includes("Signed") || message.includes("eligible") ? "#166534" : "#b91c1c" }}>{message}</p> : null}
    </article>
  );
}
