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

function apiUrl(apiBase: string, path: string): string {
  return `${apiBase.replace(/\/$/, "")}${path}`;
}

async function readText(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");
  if (!text) return "";
  try {
    const payload = JSON.parse(text) as { detail?: unknown; error?: unknown; message?: unknown };
    return String(payload.detail || payload.error || payload.message || text).slice(0, 240);
  } catch {
    return text.slice(0, 240);
  }
}

export default function AdminPilotPreflightPanel({ apiBase, clubId }: AdminPilotPreflightPanelProps) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [running, setRunning] = useState(false);
  const [results, setResults] = useState<CheckResult[]>([]);

  async function runCheck(name: string, request: () => Promise<CheckResult>): Promise<CheckResult> {
    try {
      return await request();
    } catch (error) {
      return { name, ok: false, detail: error instanceof Error ? error.message : "Check failed." };
    }
  }

  async function getJsonCheck(name: string, path: string, predicate: (payload: Record<string, unknown>) => string | null): Promise<CheckResult> {
    if (!apiBase) return { name, ok: false, detail: "API base URL is not configured." };
    const response = await fetch(apiUrl(apiBase, path), { headers: { accept: "application/json" } });
    const payload = await response.json().catch(() => null) as Record<string, unknown> | null;
    if (!response.ok || !payload) return { name, ok: false, status: response.status, detail: await readText(response) || `HTTP ${response.status}` };
    const problem = predicate(payload);
    return { name, ok: !problem, status: response.status, detail: problem || "Ready." };
  }

  async function authValidationCheck(name: string, method: "PATCH" | "POST", path: string, body: unknown, expectedText: string): Promise<CheckResult> {
    if (!apiBase) return { name, ok: false, detail: "API base URL is not configured." };
    if (!accessToken) return { name, ok: false, detail: "Sign in first." };
    const response = await fetch(apiUrl(apiBase, path), {
      method,
      headers: { accept: "application/json", "content-type": "application/json", Authorization: `Bearer ${accessToken}` },
      body: JSON.stringify(body)
    });
    const detail = await readText(response);
    const ok = response.status === 400 && detail.includes(expectedText);
    return { name, ok, status: response.status, detail: ok ? "Authorized validation reached the expected API guard." : detail || `HTTP ${response.status}` };
  }

  async function runAll() {
    setRunning(true);
    setResults([]);
    const checks: CheckResult[] = [];
    checks.push(await runCheck("Operations pilot mode", () => getJsonCheck("Operations pilot mode", "/admin/operations/status", (payload) => payload.write_pilot_enabled === true ? null : "Write pilot flag is not enabled.")));
    checks.push(await runCheck("Match Log flags", () => getJsonCheck("Match Log flags", `/admin/clubs/${encodeURIComponent(clubId)}/match-log?limit=25`, (payload) => payload.enabled === true && payload.apply_enabled === true ? null : "Match Log read/apply flags are not both enabled.")));
    checks.push(await runCheck("Replay flag", () => getJsonCheck("Replay flag", `/admin/clubs/${encodeURIComponent(clubId)}/replay-history`, (payload) => payload.enabled === true ? null : "Replay flag is not enabled.")));
    checks.push(await runCheck("Match Log auth", () => authValidationCheck("Match Log auth", "PATCH", `/admin/clubs/${encodeURIComponent(clubId)}/match-log/edits`, { patches: [], confirmation_text: "APPLY", correction_note: "pilot browser validation", source: "next_admin_pilot_browser_validation" }, "No patches provided")));
    checks.push(await runCheck("Replay auth", () => authValidationCheck("Replay auth", "POST", `/admin/clubs/${encodeURIComponent(clubId)}/replay-history`, { target_reset: "ALL (Full System Reset)", confirmation_text: "NOT_REPLAY", source: "next_admin_pilot_browser_validation" }, "Type REPLAY")));
    setResults(checks);
    setRunning(false);
  }

  return (
    <article style={cardStyle}>
      <h2 style={{ marginTop: 0 }}>Pilot readiness checks</h2>
      <div style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: accessToken ? "#f0fdf4" : "#fffbeb", marginBottom: "1rem" }}>
        <strong>{accessToken ? `Admin session: ${adminSessionLabel(session)}` : "Admin session required"}</strong>
        <p style={{ margin: "0.35rem 0 0", color: accessToken ? "#166534" : "#92400e" }}>
          {accessToken ? "Ready to run browser-based checks." : sessionLoading ? "Checking admin session…" : "Sign in before running pilot checks."}
        </p>
        {sessionMessage ? <p style={{ color: "#b91c1c", marginBottom: 0 }}>{sessionMessage}</p> : null}
        {!accessToken && !sessionLoading ? <p style={{ marginBottom: 0 }}><Link href="/admin/login">Open admin login</Link></p> : null}
      </div>
      <button type="button" disabled={running || !accessToken} onClick={runAll} style={{ ...buttonStyle, background: accessToken ? "#2563eb" : "#94a3b8", cursor: running || !accessToken ? "default" : "pointer" }}>
        {running ? "Running…" : "Run pilot checks"}
      </button>
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
