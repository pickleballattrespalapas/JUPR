"use client";

import { useMemo, useRef, useState } from "react";
import { adminSessionLabel, useAdminSession } from "@/lib/useAdminSession";
import { deriveLiveLadderOperationKey, idempotencyKeyFor, rotateIdempotencyKey } from "@/lib/liveLadderOperations";

type Player = { id: number; name: string; rating?: number | null; is_active?: boolean | null };
type MatchRow = { row_id: string; match_index: number; round?: number | null; court?: number | null; team_1: Player[]; team_2: Player[]; expected_win_pct_t1: number; expected_score: string; t1_p1: number; t1_p2: number; t2_p1: number; t2_p2: number };
type StatusResponse = { enabled: boolean; writes_enabled?: boolean; status: string; players?: Player[]; league_options?: string[]; warnings?: string[] };
type PreviewResponse = { ok: boolean; authority?: string; preview_fingerprint: string; matches: MatchRow[]; players: Player[] };
type SettlementRow = { player_id: number; player_name: string; gp: number; wins: number; losses: number; net: number; settlement_direction: "receives" | "owes" | "even"; settlement_amount: number };
type SettlementResponse = { ok: boolean; settlement_fingerprint: string; would_publish_count: number; settlement: { standings: SettlementRow[]; tie_matches: number[]; net_total: number }; preview: PreviewResponse };
type WriteResponse = { ok: boolean; submitted_count?: number; operation_key?: string; idempotent_replay?: boolean; recovery?: { match_log_url?: string; replay_history_url?: string; instructions?: string }; correction?: { match_log_url?: string; replay_history_url?: string; instructions?: string } };
type ScoreDraft = { score_t1: string; score_t2: string };
type Props = { apiBase: string | null; clubId: string; status: StatusResponse | null };

const EMPTY_PLAYERS: Player[] = [];
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "white" };
const inputStyle = { width: "100%", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };

function apiUrl(apiBase: string, path: string): string { return `${apiBase.replace(/\/$/, "")}${path}`; }
function validScore(score: ScoreDraft | undefined): boolean { const a = Number(score?.score_t1 ?? ""); const b = Number(score?.score_t2 ?? ""); return Number.isInteger(a) && Number.isInteger(b) && a >= 0 && b >= 0 && a !== b && a + b > 0; }
async function apiError(response: Response): Promise<string> { const text = await response.text().catch(() => ""); if (!text) return `API error (${response.status}).`; try { return String((JSON.parse(text) as { detail?: unknown }).detail || text); } catch { return text.slice(0, 240); } }

export default function MoneyballPanel({ apiBase, clubId, status }: Props) {
  const { session, accessToken, loading: sessionLoading, message: sessionMessage } = useAdminSession();
  const operationKeys = useRef<Record<string, string>>({});
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [ratingContext, setRatingContext] = useState("OVERALL");
  const [winRate, setWinRate] = useState("5");
  const [pointRate, setPointRate] = useState("2");
  const [leagueName, setLeagueName] = useState("Moneyball");
  const [weekTag, setWeekTag] = useState(`Moneyball ${new Date().toISOString().slice(0, 10)}`);
  const [matchType, setMatchType] = useState("Moneyball RR");
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [settlement, setSettlement] = useState<SettlementResponse | null>(null);
  const [scores, setScores] = useState<Record<string, ScoreDraft>>({});
  const [confirmation, setConfirmation] = useState("");
  const [reconcileConfirm, setReconcileConfirm] = useState("");
  const [lastOperationKey, setLastOperationKey] = useState("");
  const [lastResult, setLastResult] = useState<WriteResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const players = status?.players || EMPTY_PLAYERS;
  const selectedCount = selectedIds.length;
  const selectedPlayerNames = useMemo(() => selectedIds.map((id) => players.find((player) => String(player.id) === id)?.name || `#${id}`), [players, selectedIds]);
  const scoreRows = useMemo(() => Object.entries(scores).filter(([, score]) => validScore(score)).map(([row_id, score]) => ({ row_id, score_t1: Number(score.score_t1), score_t2: Number(score.score_t2) })), [scores]);
  const writesEnabled = status?.writes_enabled === true;

  async function requestJson<T>(path: string, options?: RequestInit): Promise<T> { if (!apiBase) throw new Error("Missing JUPR API base URL."); if (!accessToken) throw new Error("Sign in at /admin/login before using Moneyball."); const headers = new Headers(options?.headers); headers.set("Authorization", `Bearer ${accessToken}`); if (options?.body) headers.set("Content-Type", "application/json"); const response = await fetch(apiUrl(apiBase, path), { ...options, headers }); if (!response.ok) throw new Error(await apiError(response)); return (await response.json()) as T; }

  function moveSelected(index: number, offset: number) { setSelectedIds((current) => { const destination = index + offset; if (destination < 0 || destination >= current.length) return current; const next = [...current]; [next[index], next[destination]] = [next[destination], next[index]]; return next; }); setPreview(null); setSettlement(null); }

  async function generatePreview() {
    setBusy(true); setMessage(null); setSettlement(null);
    try {
      const payload = await requestJson<PreviewResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/moneyball/preview`, { method: "POST", body: JSON.stringify({ player_ids: selectedIds.map(Number), rating_context: ratingContext, win_rate: Number(winRate), point_rate: Number(pointRate) }) });
      const nextScores: Record<string, ScoreDraft> = {}; for (const match of payload.matches || []) nextScores[match.row_id] = { score_t1: "", score_t2: "" };
      setPreview(payload); setScores(nextScores); setMessage(`Python generated ${payload.matches?.length || 0} Moneyball matches.`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to generate preview."); } finally { setBusy(false); }
  }

  async function reviewSettlement() {
    if (!scoreRows.length) { setMessage("Enter at least one valid non-tied score before reviewing settlement."); return; }
    setBusy(true); setMessage(null);
    try {
      const payload = await requestJson<SettlementResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/moneyball/settlement`, { method: "POST", body: JSON.stringify({ player_ids: selectedIds.map(Number), scores: scoreRows, rating_context: ratingContext, win_rate: Number(winRate), point_rate: Number(pointRate) }) });
      setSettlement(payload); setMessage(`Reviewed settlement for ${payload.would_publish_count} official match(es).`);
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to review Moneyball settlement."); } finally { setBusy(false); }
  }

  async function submitMoneyball() {
    if (!writesEnabled) { setMessage("Next Moneyball writes are guarded off. Use the Streamlit fallback."); return; }
    if (!settlement) { setMessage("Review the Python settlement before official publish."); return; }
    if (confirmation.trim().toUpperCase() !== "SAVE MONEYBALL") { setMessage("Type SAVE MONEYBALL to publish rated Moneyball matches."); return; }
    const scope = `publish:${weekTag}`;
    const idempotencyKey = idempotencyKeyFor(operationKeys.current, scope);
    let operationKey = "";
    setBusy(true); setMessage(null);
    try {
      operationKey = await deriveLiveLadderOperationKey({ clubId, surface: "moneyball", operationType: "official_publish", entityId: weekTag || "moneyball", idempotencyKey });
      setLastOperationKey(operationKey);
      const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/moneyball/submit`, { method: "POST", body: JSON.stringify({ player_ids: selectedIds.map(Number), scores: scoreRows, rating_context: ratingContext, league_name: leagueName, week_tag: weekTag, match_type: matchType, win_rate: Number(winRate), point_rate: Number(pointRate), settlement_fingerprint: settlement.settlement_fingerprint, expected_version: settlement.settlement_fingerprint, idempotency_key: idempotencyKey, confirmation_text: confirmation }) });
      rotateIdempotencyKey(operationKeys.current, scope); setLastResult(payload); setConfirmation("");
      setMessage(`${payload.idempotent_replay ? "Recovered" : "Published"} ${payload.submitted_count || scoreRows.length} Moneyball match(es).`);
    } catch (error) { setMessage(`${error instanceof Error ? error.message : "Moneyball publish outcome is uncertain."} Do not blindly resubmit; reconcile operation ${operationKey || "shown below"}.`); } finally { setBusy(false); }
  }

  async function reconcileOperation() {
    if (!lastOperationKey) return;
    if (reconcileConfirm.trim().toUpperCase() !== "RECONCILE MONEYBALL") { setMessage("Type RECONCILE MONEYBALL to inspect/recover the stored response."); return; }
    setBusy(true); setMessage(null);
    try { const payload = await requestJson<WriteResponse>(`/admin/clubs/${encodeURIComponent(clubId)}/moneyball/operations/${encodeURIComponent(lastOperationKey)}/reconcile`, { method: "POST", body: JSON.stringify({ confirmation_text: reconcileConfirm }) }); setLastResult(payload); setMessage(payload.ok ? "Recovered the durable Moneyball response; no match was republished." : "The outcome remains uncertain. Use Match Log and Replay History."); }
    catch (error) { setMessage(error instanceof Error ? error.message : "Unable to reconcile Moneyball operation."); } finally { setBusy(false); }
  }

  if (!status?.enabled) return <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Moneyball is disabled</h2><p>{status?.warnings?.[0] || "Enable JUPR_ENABLE_NEXT_ADMIN_MONEYBALL on FastAPI."}</p></article>;

  return <div style={{ display: "grid", gap: "1rem" }}>
    <article style={{ ...cardStyle, background: "#f8fafc" }}><h2 style={{ marginTop: 0 }}>Admin session</h2><p style={{ color: "#475569" }}>{adminSessionLabel(session)}</p>{sessionLoading ? <p>Checking session…</p> : null}{sessionMessage ? <p>{sessionMessage}</p> : null}{!writesEnabled ? <p role="alert" style={{ color: "#92400e" }}>{status.warnings?.[0]} Streamlit Moneyball remains the write fallback.</p> : null}</article>
    <article style={cardStyle}><h2 style={{ marginTop: 0 }}>1. Setup</h2><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem" }}><label>Rating context<br /><select value={ratingContext} onChange={(event) => { setRatingContext(event.target.value); setPreview(null); setSettlement(null); }} style={inputStyle}>{(status.league_options || ["OVERALL"]).map((option) => <option key={option}>{option}</option>)}</select></label><label>Win rate<br /><input value={winRate} onChange={(event) => { setWinRate(event.target.value); setSettlement(null); }} inputMode="decimal" style={inputStyle} /></label><label>Point rate<br /><input value={pointRate} onChange={(event) => { setPointRate(event.target.value); setSettlement(null); }} inputMode="decimal" style={inputStyle} /></label><label>League to store<br /><input value={leagueName} onChange={(event) => setLeagueName(event.target.value)} style={inputStyle} /></label><label>Week tag<br /><input value={weekTag} onChange={(event) => setWeekTag(event.target.value)} style={inputStyle} /></label><label>Match type<br /><input value={matchType} onChange={(event) => setMatchType(event.target.value)} style={inputStyle} /></label></div></article>
    <article style={cardStyle}><h2 style={{ marginTop: 0 }}>2. Select and order exactly 8 players</h2><select aria-label="Moneyball player selection" multiple value={selectedIds} onChange={(event) => { setSelectedIds(Array.from(event.currentTarget.selectedOptions).map((option) => option.value).slice(0, 8)); setPreview(null); setSettlement(null); }} style={{ ...inputStyle, minHeight: "220px" }}>{players.filter((player) => player.is_active !== false).map((player) => <option key={player.id} value={player.id}>{player.name} · {(Number(player.rating || 0) / 400).toFixed(3)}</option>)}</select><p style={{ color: selectedCount === 8 ? "#166534" : "#92400e" }}>Selected {selectedCount}/8.</p><ol>{selectedPlayerNames.map((name, index) => <li key={selectedIds[index]} style={{ marginBottom: "0.35rem" }}><strong>P{index + 1}: {name}</strong> <button type="button" aria-label={`Move ${name} earlier`} onClick={() => moveSelected(index, -1)} disabled={index === 0} style={ghostButtonStyle}>↑</button> <button type="button" aria-label={`Move ${name} later`} onClick={() => moveSelected(index, 1)} disabled={index === selectedIds.length - 1} style={ghostButtonStyle}>↓</button></li>)}</ol><button type="button" onClick={generatePreview} disabled={busy || selectedCount !== 8} style={buttonStyle}>Generate Python schedule</button>{message ? <p role="status" aria-live="polite" style={{ color: /unable|type|uncertain|guarded/i.test(message) ? "#b91c1c" : "#166534" }}>{message}</p> : null}</article>
    {preview ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>3. Score and review settlement</h2><div style={{ display: "grid", gap: "0.5rem" }}>{preview.matches.map((match) => <fieldset key={match.row_id} style={{ display: "grid", gridTemplateColumns: "minmax(220px, 1fr) 90px 90px", gap: "0.5rem", alignItems: "center", border: 0, borderTop: "1px solid #f1f5f9", padding: "0.5rem 0 0" }}><legend style={{ position: "absolute", width: 1, height: 1, overflow: "hidden" }}>{match.team_1.map((player) => player.name).join(" and ")} versus {match.team_2.map((player) => player.name).join(" and ")}</legend><div><strong>R{match.round} · Ct {match.court}</strong><br />{match.team_1.map((player) => player.name).join(" / ")} vs {match.team_2.map((player) => player.name).join(" / ")}<br /><span style={{ color: "#64748b" }}>T1 win exp {match.expected_win_pct_t1}% · expected {match.expected_score}</span></div><label>Team 1<input aria-label={`${match.row_id} team 1 score`} value={scores[match.row_id]?.score_t1 || ""} onChange={(event) => { setScores((current) => ({ ...current, [match.row_id]: { ...(current[match.row_id] || { score_t1: "", score_t2: "" }), score_t1: event.target.value } })); setSettlement(null); }} inputMode="numeric" style={inputStyle} /></label><label>Team 2<input aria-label={`${match.row_id} team 2 score`} value={scores[match.row_id]?.score_t2 || ""} onChange={(event) => { setScores((current) => ({ ...current, [match.row_id]: { ...(current[match.row_id] || { score_t1: "", score_t2: "" }), score_t2: event.target.value } })); setSettlement(null); }} inputMode="numeric" style={inputStyle} /></label></fieldset>)}</div><p>Valid scored matches: {scoreRows.length}/{preview.matches.length}</p><button type="button" onClick={reviewSettlement} disabled={busy || !scoreRows.length} style={buttonStyle}>Review Python settlement</button></article> : null}
    {settlement ? <article style={cardStyle}><h2 style={{ marginTop: 0 }}>4. Settlement and official publish</h2>{settlement.settlement.tie_matches.length ? <p role="alert" style={{ color: "#92400e" }}>Tied matches are excluded: {settlement.settlement.tie_matches.join(", ")}.</p> : null}<div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><caption style={{ textAlign: "left", fontWeight: 800, marginBottom: "0.5rem" }}>Owes / receives settlement (net total {settlement.settlement.net_total.toFixed(2)})</caption><thead><tr><th scope="col" align="left">Player</th><th scope="col" align="right">Games</th><th scope="col" align="right">Net</th><th scope="col" align="left">Direction</th></tr></thead><tbody>{settlement.settlement.standings.map((row) => <tr key={row.player_id}><th scope="row" align="left">{row.player_name}</th><td align="right">{row.gp}</td><td align="right">{row.net.toFixed(2)}</td><td>{row.settlement_direction} {row.settlement_amount.toFixed(2)}</td></tr>)}</tbody></table></div><label>Confirmation<br /><input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} placeholder="SAVE MONEYBALL" style={inputStyle} /></label><p><button type="button" onClick={submitMoneyball} disabled={busy || !writesEnabled || confirmation.trim().toUpperCase() !== "SAVE MONEYBALL"} style={buttonStyle}>Publish reviewed official matches</button> <button type="button" onClick={reviewSettlement} disabled={busy} style={ghostButtonStyle}>Refresh settlement</button></p></article> : null}
    {lastOperationKey ? <article style={{ ...cardStyle, background: "#fff7ed" }}><h2 style={{ marginTop: 0 }}>Recovery</h2><p>Operation <code>{lastOperationKey}</code>. A timeout is not proof of failure. Reconcile this exact operation before any new publish.</p><label>Reconcile confirmation<br /><input value={reconcileConfirm} onChange={(event) => setReconcileConfirm(event.target.value)} placeholder="RECONCILE MONEYBALL" style={inputStyle} /></label><p><button type="button" onClick={reconcileOperation} disabled={busy || reconcileConfirm.trim().toUpperCase() !== "RECONCILE MONEYBALL"} style={ghostButtonStyle}>Reconcile stored response</button></p><p><a href={lastResult?.correction?.match_log_url || lastResult?.recovery?.match_log_url || "/admin/match-log"}>Match Log correction</a> · <a href={lastResult?.correction?.replay_history_url || lastResult?.recovery?.replay_history_url || "/admin/replay-history"}>Replay History verification</a></p></article> : null}
  </div>;
}
