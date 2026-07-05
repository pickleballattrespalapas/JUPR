"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { AdminSession, getAdminAuthConfig, loadAdminSession, refreshAdminSession, signOutAdminSession } from "@/lib/adminAuthClient";

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };

function emailLabel(session: AdminSession | null): string {
  return session?.user?.email || "signed-in admin";
}

export default function AdminSessionCard() {
  const [session, setSession] = useState<AdminSession | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const config = getAdminAuthConfig();

  useEffect(() => {
    const load = () => {
      const current = loadAdminSession();
      setSession(current);
      refreshAdminSession(current || undefined)
        .then((refreshed) => {
          if (refreshed) setSession(refreshed);
        })
        .catch((error) => setMessage(error instanceof Error ? error.message : "Unable to refresh admin session."));
    };
    load();
    window.addEventListener("jupr-admin-session-change", load);
    window.addEventListener("storage", load);
    return () => {
      window.removeEventListener("jupr-admin-session-change", load);
      window.removeEventListener("storage", load);
    };
  }, []);

  if (!config) {
    return (
      <article style={{ ...cardStyle, borderColor: "#fbbf24", background: "#fffbeb", marginBottom: "1rem" }}>
        <strong>Admin login configuration needed</strong>
        <p style={{ color: "#92400e" }}>Set <code>NEXT_PUBLIC_SUPABASE_URL</code> and <code>NEXT_PUBLIC_SUPABASE_ANON_KEY</code> in Vercel before staff can sign in through Next.</p>
        <Link href="/admin/login">Open login setup page</Link>
      </article>
    );
  }

  return (
    <article style={{ ...cardStyle, marginBottom: "1rem", background: session ? "#f0fdf4" : "#f8fafc", borderColor: session ? "#bbf7d0" : "#e2e8f0" }}>
      <strong>{session ? `Signed in as ${emailLabel(session)}` : "Admin session not signed in"}</strong>
      <p style={{ color: "#475569" }}>
        {session
          ? "Pilot admin pages can use this browser session token for FastAPI authorization. Feature flags still control every write workflow."
          : "Sign in before using guarded admin pilot workflows. Read-only status pages remain visible."}
      </p>
      <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
        <Link href="/admin/login">{session ? "Manage session" : "Sign in"}</Link>
        {session ? <button type="button" onClick={() => signOutAdminSession().then(() => { setSession(null); setMessage("Signed out."); })}>Sign out</button> : null}
      </div>
      {message ? <p style={{ color: message.includes("Signed") ? "#166534" : "#b91c1c" }}>{message}</p> : null}
    </article>
  );
}
