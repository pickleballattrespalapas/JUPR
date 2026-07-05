"use client";

import Link from "next/link";
import { useState } from "react";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type AdminPilotPreflightPanelProps = {
  apiBase: string | null;
  clubId: string;
};

type CheckResult = {
  name: string;
  ok: boolean;
  status?: number;
  detail: string;
};

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const buttonStyle = { border: 0, borderRadius: "999px", padding: "0.65rem 1rem", background: "#2563eb", color: "white", fontWeight: 800, cursor: "pointer" };

export default function AdminPilotPreflightPanel({ apiBase, clubId }: AdminPilotPreflightPanelProps) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<CheckResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function runAll() {
    setRunning(true);
    setResults([]);
    setError(null);
    try {
      const response = await fetch("/api/admin/pilot", {
        method: "POST",
        headers: { accept: "application/json", "content-type": "application/json" },
        body: JSON.stringify({ club_id: clubId, access_token: accessToken })
      });
      const payload = await response.json().catch(() => null) as { detail?: unknown; results?: CheckResult[] } | null;
      if (!response.ok) {
        throw new Error(String(payload?.detail || `Pilot check API error (${response.status}).`));
      }
      setResults(Array.isArray(payload?.results) ? payload.results : []);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to run pilot checks.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Pilot readiness checks</h2>
      <p style={{ color: "#475569" }}>
        These checks run through a same-origin Next route and then FastAPI, avoiding browser CORS issues while still using your signed-in Supabase admin session for permission checks.
      </p>
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
        <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
        <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
          {accessToken ? "Ready to run same-origin pilot checks." : sessionLoading ? "Checking admin session…" : "Sign in before running pilot checks."}
        </p>
        {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
      </div>
      {!apiBase ? <p style={{ color: "#b45309" }}>The server-side API base URL is not configured for this deployment.</p> : null}
      <button type="button" disabled={running || !accessToken || !apiBase} onClick={runAll} style={{ ...buttonStyle, background: accessToken && apiBase ? "#2563eb" : "#94a3b8", cursor: running || !accessToken || !apiBase ? "default" : "pointer" }}>
        {running ? "Running…" : "Run pilot checks"}
      </button>
      {error ? <p style={{ color: "#b91c1c" }}>{error}</p> : null}
      {results.length ? (
        <div style={{ display: "grid", gap: "0.5rem", marginTop: "1rem" }}>
          {results.map((result) => (
            <div key={result.name} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem", background: result.ok ? "#f0fdf4" : "#fef2f2" }}>
              <strong>{result.ok ? "PASS" : "FAIL"} · {result.name}</strong>
              {result.status ? <span style={{ color: "#64748b" }}> · HTTP {result.status}</span> : null}
              <p style={{ margin: "0.35rem 0 0", color: result.ok ? "#166534" : "#b91c1c" }}>{result.detail}</p>
            </div>
          ))}
        </div>
      ) : null}
    </article>
  );
}
