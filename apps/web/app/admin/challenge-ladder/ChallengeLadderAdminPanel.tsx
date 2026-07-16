"use client";

import { useState } from "react";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Challenge = { id: number; tier_id: string; status: string; bucket: string; challenger_name: string; defender_name: string; created_at?: string | null; accept_by?: string | null; play_by?: string | null };
type Tier = { tier_id: string; label: string; range: string; players: Array<{ player_id: number; player_name: string; rank?: number; status?: string; rating_jupr?: number | null }> };
type StatusResponse = { enabled: boolean; status: string; summary?: Record<string, number>; warnings?: string[] };
type DashboardResponse = { ok: boolean; summary: Record<string, number>; settings: Record<string, unknown>; settings_row?: Record<string, unknown>; tiers: Tier[]; challenges: Challenge[]; bucket_counts: Record<string, number> };

type Props = { apiBase: string | null; clubId: string; status: StatusResponse | null };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
async function apiError(response: Response): Promise<string> { const text = await response.text().catch(() => ""); try { return String((JSON.parse(text) as { detail?: unknown }).detail || text); } catch { return text || `API error (${response.status}).`; } }

export default function ChallengeLadderAdminPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [confirmations, setConfirmations] = useState<Record<number, string>>({});
  const [notes, setNotes] = useState<Record<number, string>>({});

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing JUPR API base URL.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Challenge Ladder Admin.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    if (!response.ok) throw new Error(await apiError(response));
    return (await response.json()) as T;
  }

  async function loadDashboard() {
    setBusy(true); setMessage(null);
    try { const payload = await requestJson<DashboardResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/dashboard`); setDashboard(payload); setMessage("Challenge Ladder dashboard loaded."); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to load dashboard."); }
    finally { setBusy(false); }
  }

  async function updateChallenge(challenge: Challenge, nextStatus: string) {
    setBusy(true); setMessage(null);
    try {
      await requestJson(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges/${challenge.id}`, { method: "PATCH", body: JSON.stringify({ status: nextStatus, admin_note: notes[challenge.id] || "", confirmation_text: confirmations[challenge.id] || "" }) });
      setMessage(`Challenge #${challenge.id} saved as ${nextStatus}.`);
      await loadDashboard();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to update challenge."); }
    finally { setBusy(false); }
  }

  if (!status?.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Challenge Ladder Admin is disabled</h2><p>{status?.warnings?.[0] || "Enable JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER on FastAPI."}</p></article>;

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2><p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>{sessionLoading ? <p>Checking session…</p> : null}{sessionMessage ? <p>{sessionMessage}</p> : null}
        <p style={{ color: "#475569" }}>Players: {status.summary?.active_player_count ?? "—"} · Active challenges: {status.summary?.active_challenge_count ?? "—"}</p>
      </article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Dashboard</h2>
        <button type="button" onClick={loadDashboard} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Loading…" : "Load Challenge Ladder"}</button>
        {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Summary</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem" }}>
          {Object.entries(dashboard.summary || {}).map(([key, value]) => <div key={key} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem" }}><strong>{key.replace(/_/g, " ")}</strong><br />{String(value)}</div>)}
        </div>
      </article> : null}
      {dashboard?.tiers?.length ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Roster by tier</h2>{dashboard.tiers.filter((tier) => tier.players.length).map((tier) => <section key={tier.tier_id} style={{ marginTop: "1rem" }}><h3>{tier.label} · {tier.range}</h3><div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><tbody>{tier.players.map((player) => <tr key={player.player_id}><td>{player.rank ?? "—"}</td><td>{player.player_name}</td><td>{player.rating_jupr ? player.rating_jupr.toFixed(3) : "—"}</td><td>{player.status || "Ready"}</td></tr>)}</tbody></table></div></section>)}</article> : null}
      {dashboard?.challenges?.length ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Challenges</h2>{dashboard.challenges.map((challenge) => <section key={challenge.id} style={{ borderTop: "1px solid #e2e8f0", paddingTop: "0.75rem", marginTop: "0.75rem" }}><h3 style={{ marginTop: 0 }}>#{challenge.id} · {challenge.bucket}</h3><p>{challenge.challenger_name} vs {challenge.defender_name} · {challenge.status} · {challenge.tier_id}</p><label>Admin note<br /><input value={notes[challenge.id] || ""} onChange={(e) => setNotes((current) => ({ ...current, [challenge.id]: e.target.value }))} style={inputStyle} /></label><label>Confirmation<br /><input value={confirmations[challenge.id] || ""} onChange={(e) => setConfirmations((current) => ({ ...current, [challenge.id]: e.target.value }))} placeholder="SAVE LADDER" style={inputStyle} /></label><p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}><button type="button" onClick={() => updateChallenge(challenge, "CANCELLED")} disabled={busy} style={ghostButtonStyle}>Cancel</button><button type="button" onClick={() => updateChallenge(challenge, "FORFEITED")} disabled={busy} style={ghostButtonStyle}>Forfeit</button></p></section>)}</article> : null}
    </div>
  );
}
