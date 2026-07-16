"use client";

import { useMemo, useState } from "react";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";

type Player = { id: number; name: string; rating?: number | null; is_active?: boolean | null };
type MatchRow = { row_id: string; match_index: number; round?: number | null; court?: number | null; team_1: Player[]; team_2: Player[]; expected_win_pct_t1: number; expected_score: string; t1_p1: number; t1_p2: number; t2_p1: number; t2_p2: number };
type StatusResponse = { enabled: boolean; status: string; players?: Player[]; league_options?: string[]; warnings?: string[] };
type PreviewResponse = { ok: boolean; matches: MatchRow[]; players: Player[]; league_options?: string[] };
type ScoreDraft = { score_t1: string; score_t2: string };

type Props = { apiBase: string | null; clubId: string; status: StatusResponse | null };

const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
function validScore(score: ScoreDraft | undefined): boolean {
  const a = Number(score?.score_t1 ?? "");
  const b = Number(score?.score_t2 ?? "");
  return Number.isInteger(a) && Number.isInteger(b) && a >= 0 && b >= 0 && a !== b && a + b > 0;
}
async function apiError(response: Response): Promise<string> {
  const text = await response.text().catch(() => "");
  if (!text) return `API error (${response.status}).`;
  try { return String((JSON.parse(text) as { detail?: unknown }).detail || text); } catch { return text.slice(0, 240); }
}

export default function MoneyballPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [ratingContext, setRatingContext] = useState("OVERALL");
  const [winRate, setWinRate] = useState("5");
  const [pointRate, setPointRate] = useState("2");
  const [leagueName, setLeagueName] = useState("Moneyball");
  const [weekTag, setWeekTag] = useState(`Moneyball ${new Date().toISOString().slice(0, 10)}`);
  const [matchType, setMatchType] = useState("Moneyball RR");
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [scores, setScores] = useState<Record<string, ScoreDraft>>({});
  const [confirmation, setConfirmation] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const players = status?.players || [];
  const selectedCount = selectedIds.length;
  const validScoreCount = (preview?.matches || []).filter((match) => validScore(scores[match.row_id])).length;
  const selectedPlayerNames = useMemo(() => selectedIds.map((id) => players.find((p) => String(p.id) === id)?.name || `#${id}`), [players, selectedIds]);

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
    if (!apiBase) throw new Error("Missing JUPR API base URL.");
    if (!accessToken) throw new Error("Sign in at /admin/login before using Moneyball.");
    const headers = new Headers(options?.headers);
    headers.set("Authorization", `Bearer ${accessToken}`);
    if (options?.body) headers.set("Content-Type", "application/json");
    const response = await fetch(apiUrl(apiBase, path), { ...options, headers });
    if (!response.ok) throw new Error(await apiError(response));
    return (await response.json()) as T;
  }

  async function generatePreview() {
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<PreviewResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/moneyball/preview`, {
        method: "POST",
        body: JSON.stringify({ player_ids: selectedIds.map(Number), rating_context: ratingContext, win_rate: Number(winRate), point_rate: Number(pointRate) })
      });
      const nextScores: Record<string, ScoreDraft> = {};
      for (const match of payload.matches || []) nextScores[match.row_id] = { score_t1: "", score_t2: "" };
      setPreview(payload); setScores(nextScores); setMessage(`Generated ${payload.matches?.length || 0} Moneyball matches.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to generate preview."); }
    finally { setBusy(false); }
  }

  async function submitMoneyball() {
    if (confirmation.trim().toUpperCase() !== "SAVE MONEYBALL") { setMessage("Type SAVE MONEYBALL to save rated Moneyball matches."); return; }
    const scoreRows = Object.entries(scores).filter(([, score]) => validScore(score)).map(([row_id, score]) => ({ row_id, score_t1: Number(score.score_t1), score_t2: Number(score.score_t2) }));
    if (!scoreRows.length) { setMessage("Enter at least one valid non-tied score."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<{ ok: boolean; submitted_count?: number; settlement?: { standings?: Array<{ player_id: number; net: number }> } }>(`/admin/clubs/${encodeURIComponent(clubId)}/moneyball/submit`, {
        method: "POST",
        body: JSON.stringify({ player_ids: selectedIds.map(Number), scores: scoreRows, rating_context: ratingContext, league_name: leagueName, week_tag: weekTag, match_type: matchType, win_rate: Number(winRate), point_rate: Number(pointRate), confirmation_text: confirmation })
      });
      setConfirmation("");
      setMessage(`Saved ${payload.submitted_count || scoreRows.length} Moneyball match(es).`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to save Moneyball."); }
    finally { setBusy(false); }
  }

  if (!status?.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Moneyball is disabled</h2><p>{status?.warnings?.[0] || "Enable JUPR_ENABLE_NEXT_ADMIN_MONEYBALL on FastAPI."}</p></article>;

  return (
    <div style={{ display: "grid", gap: "1rem" }}>
      <article style={{ ...cardStyle, background: "#f8fafc" }}>
        <h2 style={{ marginTop: 0 }}>Admin session</h2><p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>{sessionLoading ? <p>Checking session…</p> : null}{sessionMessage ? <p>{sessionMessage}</p> : null}
      </article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>1. Setup</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}>
          <label>Rating context<br /><select value={ratingContext} onChange={(e) => setRatingContext(e.target.value)} style={inputStyle}>{(status.league_options || ["OVERALL"]).map((option) => <option key={option}>{option}</option>)}</select></label>
          <label>Win rate<br /><input value={winRate} onChange={(e) => setWinRate(e.target.value)} style={inputStyle} /></label>
          <label>Point rate<br /><input value={pointRate} onChange={(e) => setPointRate(e.target.value)} style={inputStyle} /></label>
          <label>League to store<br /><input value={leagueName} onChange={(e) => setLeagueName(e.target.value)} style={inputStyle} /></label>
          <label>Week tag<br /><input value={weekTag} onChange={(e) => setWeekTag(e.target.value)} style={inputStyle} /></label>
          <label>Match type<br /><input value={matchType} onChange={(e) => setMatchType(e.target.value)} style={inputStyle} /></label>
        </div>
      </article>
      <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>2. Select exactly 8 players</h2>
        <select multiple value={selectedIds} onChange={(e) => setSelectedIds(Array.from(e.currentTarget.selectedOptions).map((option) => option.value).slice(0, 8))} style={{ ...inputStyle, minHeight: "220px" }}>
          {players.filter((p) => p.is_active !== false).map((player) => <option key={player.id} value={player.id}>{player.name} · {(Number(player.rating || 0) / 400).toFixed(3)}</option>)}
        </select>
        <p style={{ color: selectedCount === 8 ? "#166534" : "#92400e" }}>Selected {selectedCount}/8: {selectedPlayerNames.join(", ") || "—"}</p>
        <button type="button" onClick={generatePreview} disabled={busy || selectedCount !== 8} style={buttonStyle}>Generate Moneyball schedule</button>
        {message ? <p style={{ color: message.toLowerCase().includes("unable") || message.toLowerCase().includes("type") ? "#b91c1c" : "#166534" }}>{message}</p> : null}
      </article>
      {preview ? <article style={cardStyle}>
        <h2 style={{ marginTop: 0 }}>3. Score and save</h2>
        <div style={{ display: "grid", gap: "0.5rem" }}>{preview.matches.map((match) => <div key={match.row_id} style={{ display: "grid", gridTemplateColumns: "1fr 80px 80px", gap: "0.5rem", alignItems: "center", borderTop: "1px solid #f1f5f9", paddingTop: "0.5rem" }}><div><strong>R{match.round} · Ct {match.court}</strong><br />{match.team_1.map((p) => p.name).join(" / ")} vs {match.team_2.map((p) => p.name).join(" / ")}<br /><span style={{ color: "#64748b" }}>T1 win exp {match.expected_win_pct_t1}% · expected {match.expected_score}</span></div><input placeholder="T1" value={scores[match.row_id]?.score_t1 || ""} onChange={(e) => setScores((current) => ({ ...current, [match.row_id]: { ...(current[match.row_id] || { score_t1: "", score_t2: "" }), score_t1: e.target.value } }))} style={inputStyle} /><input placeholder="T2" value={scores[match.row_id]?.score_t2 || ""} onChange={(e) => setScores((current) => ({ ...current, [match.row_id]: { ...(current[match.row_id] || { score_t1: "", score_t2: "" }), score_t2: e.target.value } }))} style={inputStyle} /></div>)}</div>
        <p>Valid scored matches: {validScoreCount}/{preview.matches.length}</p>
        <label>Confirmation<br /><input value={confirmation} onChange={(e) => setConfirmation(e.target.value)} placeholder="SAVE MONEYBALL" style={inputStyle} /></label>
        <p><button type="button" onClick={submitMoneyball} disabled={busy || !validScoreCount} style={buttonStyle}>Save Moneyball to JUPR</button> <button type="button" onClick={generatePreview} disabled={busy} style={ghostButtonStyle}>Regenerate</button></p>
      </article> : null}
    </div>
  );
}
