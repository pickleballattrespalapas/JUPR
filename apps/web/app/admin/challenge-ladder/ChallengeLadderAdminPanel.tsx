"use client";

import { useState } from "react";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Challenge = { id: number; tier_id: string; status: string; bucket: string; challenger_id?: number | null; defender_id?: number | null; challenger_name: string; defender_name: string; winner_name?: string | null; created_at?: string | null; accept_by?: string | null; play_by?: string | null; resolution_notes?: string | null };
type Player = { player_id: number; player_name: string; rank?: number; status?: string; rating_jupr?: number | null };
type Tier = { tier_id: string; label: string; range: string; players: Player[] };
type StatusResponse = { enabled: boolean; status: string; summary?: Record<string, number>; warnings?: string[] };
type DashboardResponse = { ok: boolean; summary: Record<string, number>; settings: Record<string, unknown>; settings_row?: Record<string, unknown>; tiers: Tier[]; challenges: Challenge[]; bucket_counts: Record<string, number> };
type ActionResponse = { ok: boolean; challenge?: Challenge; warnings?: string[]; rank_result?: Record<string, unknown>; official_matches?: Record<string, unknown>; preview?: Record<string, unknown> };

type Props = { apiBase: string | null; clubId: string; status: StatusResponse | null };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
async function apiError(response: Response): Promise<string> { const text = await response.text().catch(() => ""); try { return String((JSON.parse(text) as { detail?: unknown }).detail || text); } catch { return text || `API error (${response.status}).`; } }
function Pre({ value }: { value: unknown }) { return <pre style={{ whiteSpace: "pre-wrap", background: "#0f172a", color: "white", padding: "1rem", borderRadius: "12px", overflowX: "auto", fontSize: "0.82rem" }}>{JSON.stringify(value, null, 2)}</pre>; }
function parseGames(raw: string): number[][] { return raw.split(/[|,;]/).map((part) => part.trim()).filter(Boolean).map((part) => { const bits = part.split(/[-–—:\/]/).map((x) => Number(x.trim())); if (bits.length !== 2 || !Number.isFinite(bits[0]) || !Number.isFinite(bits[1])) throw new Error(`Invalid score: ${part}`); return [bits[0], bits[1]]; }); }
function activePlayers(tiers: Tier[] | undefined): Player[] { const rows: Player[] = []; for (const tier of tiers || []) for (const player of tier.players || []) rows.push(player); return rows; }

export default function ChallengeLadderAdminPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [confirmations, setConfirmations] = useState<Record<number, string>>({});
  const [notes, setNotes] = useState<Record<number, string>>({});
  const [createDraft, setCreateDraft] = useState({ challenger_id: "", defender_id: "", tier_id: "ADV", ledger_ref: "", override: false, start_clock: false, confirmation_text: "" });
  const [resultDraft, setResultDraft] = useState({ challenge_id: "", a_chal: "", a_def: "", b_chal: "", b_def: "", match_a_games: "11-0,11-0", match_b_games: "11-0,11-0", match_date: new Date().toISOString(), winner_override: "computed", publish_official_matches: true, confirmation_text: "" });
  const [forfeitDraft, setForfeitDraft] = useState({ challenge_id: "", forfeited_by_id: "", admin_note: "", confirmation_text: "" });
  const [lastResult, setLastResult] = useState<ActionResponse | null>(null);

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
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges/${challenge.id}`, { method: "PATCH", body: JSON.stringify({ status: nextStatus, admin_note: notes[challenge.id] || "", confirmation_text: confirmations[challenge.id] || "" }) });
      setLastResult(payload); setMessage(`Challenge #${challenge.id} saved as ${nextStatus}.`); await loadDashboard();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to update challenge."); }
    finally { setBusy(false); }
  }

  async function simpleAction(challenge: Challenge, action: "start-clock" | "accept", expected: string) {
    if ((confirmations[challenge.id] || "").trim().toUpperCase() !== expected) { setMessage(`Type ${expected} to continue.`); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges/${challenge.id}/${action}`, { method: "POST", body: JSON.stringify({ confirmation_text: confirmations[challenge.id] || "" }) });
      setLastResult(payload); setMessage(`Challenge #${challenge.id} updated.`); await loadDashboard();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to update challenge."); }
    finally { setBusy(false); }
  }

  async function createChallenge() {
    if (createDraft.confirmation_text.trim().toUpperCase() !== "CREATE LADDER CHALLENGE") { setMessage("Type CREATE LADDER CHALLENGE to create a challenge."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges`, { method: "POST", body: JSON.stringify({ ...createDraft, challenger_id: Number(createDraft.challenger_id), defender_id: Number(createDraft.defender_id) }) });
      setLastResult(payload); setMessage("Challenge created."); await loadDashboard();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to create challenge."); }
    finally { setBusy(false); }
  }

  async function recordForfeit() {
    if (forfeitDraft.confirmation_text.trim().toUpperCase() !== "RECORD LADDER FORFEIT") { setMessage("Type RECORD LADDER FORFEIT to record a forfeit."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges/${Number(forfeitDraft.challenge_id)}/forfeit`, { method: "POST", body: JSON.stringify({ forfeited_by_id: Number(forfeitDraft.forfeited_by_id), admin_note: forfeitDraft.admin_note, confirmation_text: forfeitDraft.confirmation_text }) });
      setLastResult(payload); setMessage("Forfeit recorded."); await loadDashboard();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to record forfeit."); }
    finally { setBusy(false); }
  }

  async function publishResult() {
    if (resultDraft.confirmation_text.trim().toUpperCase() !== "PUBLISH LADDER RESULT") { setMessage("Type PUBLISH LADDER RESULT to publish the ladder result."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<ActionResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/challenge-ladder/challenges/${Number(resultDraft.challenge_id)}/result`, { method: "POST", body: JSON.stringify({
        partner_a_challenger_id: Number(resultDraft.a_chal), partner_a_defender_id: Number(resultDraft.a_def), partner_b_challenger_id: Number(resultDraft.b_chal), partner_b_defender_id: Number(resultDraft.b_def),
        match_a_games: parseGames(resultDraft.match_a_games), match_b_games: parseGames(resultDraft.match_b_games), match_date: resultDraft.match_date, winner_override: resultDraft.winner_override, publish_official_matches: resultDraft.publish_official_matches, confirmation_text: resultDraft.confirmation_text
      }) });
      setLastResult(payload); setMessage("Ladder result published."); await loadDashboard();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to publish ladder result."); }
    finally { setBusy(false); }
  }

  if (!status?.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Challenge Ladder Admin is disabled</h2><p>{status?.warnings?.[0] || "Enable JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER on FastAPI."}</p></article>;

  const players = activePlayers(dashboard?.tiers);
  const openChallengeOptions = dashboard?.challenges || [];
  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2><p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>{sessionLoading ? <p>Checking session…</p> : null}{sessionMessage ? <p>{sessionMessage}</p> : null}
        <p style={{ color: "#475569" }}>Players: {status.summary?.active_player_count ?? "—"} · Active challenges: {status.summary?.active_challenge_count ?? "—"}</p>
      </article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Dashboard</h2>
        <button type="button" onClick={loadDashboard} disabled={busy || !accessToken} style={buttonStyle}>{busy ? "Loading…" : "Load Challenge Ladder"}</button>
        {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") || message.toLowerCase().includes("invalid") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Create challenge</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Tier<br /><select value={createDraft.tier_id} onChange={(e) => setCreateDraft((c) => ({ ...c, tier_id: e.target.value }))} style={inputStyle}>{dashboard.tiers.map((tier) => <option key={tier.tier_id} value={tier.tier_id}>{tier.label}</option>)}</select></label>
          <label>Challenger<br /><select value={createDraft.challenger_id} onChange={(e) => setCreateDraft((c) => ({ ...c, challenger_id: e.target.value }))} style={inputStyle}><option value="">Choose</option>{players.map((p) => <option key={p.player_id} value={p.player_id}>{p.player_name} #{p.player_id}</option>)}</select></label>
          <label>Defender<br /><select value={createDraft.defender_id} onChange={(e) => setCreateDraft((c) => ({ ...c, defender_id: e.target.value }))} style={inputStyle}><option value="">Choose</option>{players.map((p) => <option key={p.player_id} value={p.player_id}>{p.player_name} #{p.player_id}</option>)}</select></label>
          <label>Ledger/ref<br /><input value={createDraft.ledger_ref} onChange={(e) => setCreateDraft((c) => ({ ...c, ledger_ref: e.target.value }))} style={inputStyle} /></label>
          <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={createDraft.override} onChange={(e) => setCreateDraft((c) => ({ ...c, override: e.target.checked }))} /> Override eligibility</label>
          <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={createDraft.start_clock} onChange={(e) => setCreateDraft((c) => ({ ...c, start_clock: e.target.checked }))} /> Start clock now</label>
          <label>Confirmation<br /><input value={createDraft.confirmation_text} onChange={(e) => setCreateDraft((c) => ({ ...c, confirmation_text: e.target.value }))} placeholder="CREATE LADDER CHALLENGE" style={inputStyle} /></label>
          <button type="button" onClick={createChallenge} disabled={busy} style={buttonStyle}>Create challenge</button>
        </div>
      </article> : null}
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Publish played result</h2>
        <p style={{ color: "#475569" }}>Played ladder results insert two official rated matches and apply direct rank swap when the challenger wins. Forfeits do not create match rows.</p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Challenge<br /><select value={resultDraft.challenge_id} onChange={(e) => setResultDraft((c) => ({ ...c, challenge_id: e.target.value }))} style={inputStyle}><option value="">Choose</option>{openChallengeOptions.map((ch) => <option key={ch.id} value={ch.id}>#{ch.id} {ch.challenger_name} vs {ch.defender_name} ({ch.status})</option>)}</select></label>
          <label>A challenger partner<br /><select value={resultDraft.a_chal} onChange={(e) => setResultDraft((c) => ({ ...c, a_chal: e.target.value }))} style={inputStyle}><option value="">Choose</option>{players.map((p) => <option key={p.player_id} value={p.player_id}>{p.player_name}</option>)}</select></label>
          <label>A defender partner<br /><select value={resultDraft.a_def} onChange={(e) => setResultDraft((c) => ({ ...c, a_def: e.target.value }))} style={inputStyle}><option value="">Choose</option>{players.map((p) => <option key={p.player_id} value={p.player_id}>{p.player_name}</option>)}</select></label>
          <label>B challenger partner<br /><select value={resultDraft.b_chal} onChange={(e) => setResultDraft((c) => ({ ...c, b_chal: e.target.value }))} style={inputStyle}><option value="">Choose</option>{players.map((p) => <option key={p.player_id} value={p.player_id}>{p.player_name}</option>)}</select></label>
          <label>B defender partner<br /><select value={resultDraft.b_def} onChange={(e) => setResultDraft((c) => ({ ...c, b_def: e.target.value }))} style={inputStyle}><option value="">Choose</option>{players.map((p) => <option key={p.player_id} value={p.player_id}>{p.player_name}</option>)}</select></label>
          <label>Match A games<br /><input value={resultDraft.match_a_games} onChange={(e) => setResultDraft((c) => ({ ...c, match_a_games: e.target.value }))} placeholder="11-7,8-11,11-6" style={inputStyle} /></label>
          <label>Match B games<br /><input value={resultDraft.match_b_games} onChange={(e) => setResultDraft((c) => ({ ...c, match_b_games: e.target.value }))} placeholder="11-7,8-11,11-6" style={inputStyle} /></label>
          <label>Winner override<br /><select value={resultDraft.winner_override} onChange={(e) => setResultDraft((c) => ({ ...c, winner_override: e.target.value }))} style={inputStyle}><option value="computed">Computed</option><option value="challenger">Challenger</option><option value="defender">Defender</option></select></label>
          <label>Match date ISO<br /><input value={resultDraft.match_date} onChange={(e) => setResultDraft((c) => ({ ...c, match_date: e.target.value }))} style={inputStyle} /></label>
          <label style={{ display: "inline-flex", gap: "0.5rem", alignItems: "center" }}><input type="checkbox" checked={resultDraft.publish_official_matches} onChange={(e) => setResultDraft((c) => ({ ...c, publish_official_matches: e.target.checked }))} /> Publish official matches</label>
          <label>Confirmation<br /><input value={resultDraft.confirmation_text} onChange={(e) => setResultDraft((c) => ({ ...c, confirmation_text: e.target.value }))} placeholder="PUBLISH LADDER RESULT" style={inputStyle} /></label>
          <button type="button" onClick={publishResult} disabled={busy} style={buttonStyle}>Publish result</button>
        </div>
      </article> : null}
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Record forfeit</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", alignItems: "end" }}>
          <label>Challenge<br /><select value={forfeitDraft.challenge_id} onChange={(e) => setForfeitDraft((c) => ({ ...c, challenge_id: e.target.value }))} style={inputStyle}><option value="">Choose</option>{openChallengeOptions.map((ch) => <option key={ch.id} value={ch.id}>#{ch.id} {ch.challenger_name} vs {ch.defender_name}</option>)}</select></label>
          <label>Forfeited by player ID<br /><input value={forfeitDraft.forfeited_by_id} onChange={(e) => setForfeitDraft((c) => ({ ...c, forfeited_by_id: e.target.value }))} style={inputStyle} /></label>
          <label>Note<br /><input value={forfeitDraft.admin_note} onChange={(e) => setForfeitDraft((c) => ({ ...c, admin_note: e.target.value }))} style={inputStyle} /></label>
          <label>Confirmation<br /><input value={forfeitDraft.confirmation_text} onChange={(e) => setForfeitDraft((c) => ({ ...c, confirmation_text: e.target.value }))} placeholder="RECORD LADDER FORFEIT" style={inputStyle} /></label>
          <button type="button" onClick={recordForfeit} disabled={busy} style={ghostButtonStyle}>Record forfeit</button>
        </div>
      </article> : null}
      {dashboard ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>Summary</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "0.75rem" }}>
          {Object.entries(dashboard.summary || {}).map(([key, value]) => <div key={key} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem" }}><strong>{key.replace(/_/g, " ")}</strong><br />{String(value)}</div>)}
        </div>
      </article> : null}
      {dashboard?.tiers?.length ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Roster by tier</h2>{dashboard.tiers.filter((tier) => tier.players.length).map((tier) => <section key={tier.tier_id} style={{ marginTop: "1rem" }}><h3>{tier.label} · {tier.range}</h3><div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><tbody>{tier.players.map((player) => <tr key={player.player_id}><td>{player.rank ?? "—"}</td><td>{player.player_name}</td><td>{player.rating_jupr ? player.rating_jupr.toFixed(3) : "—"}</td><td>{player.status || "Ready"}</td></tr>)}</tbody></table></div></section>)}</article> : null}
      {dashboard?.challenges?.length ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Challenges</h2>{dashboard.challenges.map((challenge) => <section key={challenge.id} style={{ borderTop: "1px solid #e2e8f0", paddingTop: "0.75rem", marginTop: "0.75rem" }}><h3 style={{ marginTop: 0 }}>#{challenge.id} · {challenge.bucket}</h3><p>{challenge.challenger_name} vs {challenge.defender_name} · {challenge.status} · {challenge.tier_id}</p><label>Admin note<br /><input value={notes[challenge.id] || ""} onChange={(e) => setNotes((current) => ({ ...current, [challenge.id]: e.target.value }))} style={inputStyle} /></label><label>Confirmation<br /><input value={confirmations[challenge.id] || ""} onChange={(e) => setConfirmations((current) => ({ ...current, [challenge.id]: e.target.value }))} placeholder="SAVE LADDER / START LADDER CLOCK / ACCEPT LADDER CHALLENGE" style={inputStyle} /></label><p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}><button type="button" onClick={() => simpleAction(challenge, "start-clock", "START LADDER CLOCK")} disabled={busy} style={ghostButtonStyle}>Start clock</button><button type="button" onClick={() => simpleAction(challenge, "accept", "ACCEPT LADDER CHALLENGE")} disabled={busy} style={ghostButtonStyle}>Accept</button><button type="button" onClick={() => updateChallenge(challenge, "CANCELLED")} disabled={busy} style={ghostButtonStyle}>Cancel</button></p></section>)}</article> : null}
      {lastResult ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>Last action result</h2><Pre value={lastResult} /></article> : null}
    </div>
  );
}
