"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";
import { useAdminSession } from "@/lib/useAdminSession";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess } from "@/components/interaction";

type ReviewRequest = { expected_version: number; action: "approve" | "reject"; player_ids: Record<string, number>; match_date: string; reason: string };
type Submission = {
  id: string; session_key: string; title: string; version: number; status: string;
  organizer_name: string; rating_mode: "rated" | "unrated"; match_date: string; match_count: number; approved_mode?: string;
  review?: { request: ReviewRequest };
  participants: Array<{ id: string; name: string; player_id?: number | null }>;
  matches: Array<{ id: string; round_number: number; sideA?: string[]; sideB?: string[]; teamA?: string[]; teamB?: string[]; scoreA: number; scoreB: number }>;
};
type Queue = { submissions: Submission[]; players: Array<{ id: number; name: string }> };
const field = { padding: "0.6rem", border: "1px solid #cbd5e1", borderRadius: 8, font: "inherit", maxWidth: "100%" };

function PlayerProfileChoice({ name, value, players, disabled, onChange }: {
  name: string; value?: number; players: Queue["players"]; disabled: boolean; onChange: (value: number) => void;
}) {
  const [search, setSearch] = useState(name);
  const selected = players.find(player => player.id === value);
  const choices = players.filter(player => player.name.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase())).slice(0, 25);
  if (selected && !choices.some(player => player.id === value)) choices.unshift(selected);
  return <div style={{ display: "grid", gap: 4 }}>
    <strong>{name}</strong>
    {!disabled ? <input type="search" aria-label={`Search club profiles for ${name}`} value={search} onChange={event => setSearch(event.target.value)} style={field} /> : null}
    <select aria-label={`Club profile for ${name}`} value={value || ""} disabled={disabled} style={field} onChange={event => onChange(Number(event.target.value))}>
      <option value="">Choose a club player</option>{choices.map(player => <option key={player.id} value={player.id}>{player.name}</option>)}
    </select>
  </div>;
}

export default function GeneratorSubmissions({ clubId, apiBase }: { clubId: string; apiBase: string }) {
  const { accessToken } = useAdminSession();
  const [queue, setQueue] = useState<Queue>({ submissions: [], players: [] });
  const [status, setStatus] = useState("pending");
  const [selected, setSelected] = useState<Submission | null>(null);
  const [mapping, setMapping] = useState<Record<string, number>>({});
  const [matchDate, setMatchDate] = useState("");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const generation = useRef(0);
  const request = useCallback(async (path: string, body?: unknown, signal?: AbortSignal) => {
    if (!apiBase || !accessToken) throw new Error("Sign in to review generator results.");
    const response = await fetch(`${apiBase.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/play-generators/${path}`, {
      method: body ? "POST" : "GET", headers: { Authorization: `Bearer ${accessToken}`, "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined, cache: "no-store", signal
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Could not load generator results.");
    return payload;
  }, [accessToken, apiBase, clubId]);
  const refresh = useCallback(async () => {
    const current = ++generation.current;
    setBusy(true);
    try { const payload = await request(`submissions?status=${status}`); if (current === generation.current) { setQueue(payload); setSelected(null); } }
    catch (error) { if (current === generation.current) setMessage(error instanceof Error ? error.message : "Could not refresh results."); }
    finally { if (current === generation.current) setBusy(false); }
  }, [request, status]);
  useEffect(() => {
    const controller = new AbortController();
    const current = ++generation.current;
    setQueue({ submissions: [], players: [] }); setSelected(null); setMessage("");
    if (!accessToken) return;
    setBusy(true);
    request(`submissions?status=${status}`, undefined, controller.signal).then(payload => {
      if (!controller.signal.aborted && current === generation.current) setQueue(payload);
    }).catch(error => { if (!controller.signal.aborted) setMessage(error.message); }).finally(() => { if (current === generation.current) setBusy(false); });
    return () => { controller.abort(); };
  }, [accessToken, request, status]);
  function select(item: Submission) {
    setSelected(item); setReason(item.review?.request.reason || ""); setMatchDate(item.review?.request.match_date || item.match_date);
    const ids: Record<string, number> = {};
    for (const participant of item.participants) {
      const exact = queue.players.filter(p => p.name.trim().toLocaleLowerCase() === participant.name.trim().toLocaleLowerCase());
      const chosen = participant.player_id || (exact.length === 1 ? exact[0].id : 0);
      if (chosen) ids[participant.id] = chosen;
    }
    setMapping(item.review?.request.player_ids || ids);
  }
  async function decide(action: ReviewRequest["action"]) {
    if (!selected) throw new Error("Choose a submission first.");
    setBusy(true); setMessage("");
    try {
      const body = selected.status === "processing" && selected.review ? selected.review.request : { expected_version: selected.version, action, player_ids: mapping, match_date: matchDate, reason };
      const payload = await request(`sessions/${encodeURIComponent(selected.session_key)}/review`, { ...body, expected_version: selected.version });
      const text = payload.status === "rejected" ? "Submission rejected. Club records are unchanged." : `Approved as ${selected.rating_mode}. Games are now included in player stats and available for weekly recaps.`;
      await refresh(); setMessage(text);
      return actionSuccess("Review saved", text);
    } catch (error) {
      setMessage(`${error instanceof Error ? error.message : "Could not confirm the review."} Refresh to check the result. An interrupted approval can be resumed safely.`);
      throw error;
    } finally { setBusy(false); }
  }
  const editable = selected?.status === "pending";
  const allMapped = selected?.participants.every(p => mapping[p.id]);
  const names = new Map(selected?.participants.map(p => [p.id, p.name]) || []);
  return <div style={{ display: "grid", gap: "1rem" }}>
    <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
      <label>Show <select value={status} disabled={busy} onChange={e => setStatus(e.target.value)} style={field}><option value="pending">Awaiting approval</option><option value="approved">Approved</option><option value="rejected">Rejected</option></select></label>
      <button onClick={() => void refresh()} disabled={busy || !accessToken} style={field}>Refresh</button>
    </div>
    {message ? <p role="status">{message}</p> : null}
    {busy ? <p role="status">Loading…</p> : null}
    {!busy && !queue.submissions.length ? <p>No submissions in this view.</p> : null}
    <div style={{ display: "grid", gap: "0.75rem", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 260px), 1fr))" }}>
      {queue.submissions.map(item => <button key={item.id} onClick={() => select(item)} disabled={busy} style={{ ...field, textAlign: "left", background: selected?.id === item.id ? "#eff6ff" : "white" }}>
        <strong>{item.title}</strong><br />{item.match_date} · {item.match_count} games · {item.rating_mode === "rated" ? "Rated" : "Unrated"}<br />Submitted by {item.organizer_name}<br />{item.status === "processing" ? "Approval interrupted — resume review" : item.approved_mode || item.status}
      </button>)}
    </div>
    {selected ? <article style={{ padding: "1rem", border: "1px solid #cbd5e1", borderRadius: 14 }}>
      <h2>{selected.title} · {selected.rating_mode === "rated" ? "Rated" : "Unrated"}</h2>
      <p>{selected.match_count} scored games · Submitted by {selected.organizer_name}</p>
      <label>Date played <input type="date" value={matchDate} disabled={!editable || busy} onChange={e => setMatchDate(e.target.value)} style={field} /></label>
      <h3>Match players to club profiles</h3>
      <p>Check each player before approving. <Link href="/admin/players" target="_blank" rel="noopener noreferrer">Add a missing player</Link>, then refresh this list.</p>
      <div style={{ display: "grid", gap: "0.75rem", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 260px), 1fr))" }}>
        {selected.participants.map(p => <PlayerProfileChoice key={`${selected.id}:${p.id}`} name={p.name} value={mapping[p.id]} players={queue.players} disabled={!editable || busy}
          onChange={value => setMapping(old => { const next = { ...old }; if (value) next[p.id] = value; else delete next[p.id]; return next; })} />)}
      </div>
      <h3>Saved scores</h3>
      <div style={{ overflowX: "auto" }}><table style={{ width: "100%", textAlign: "left", borderSpacing: "0 0.5rem" }}><thead><tr><th>Round</th><th>Team 1</th><th>Score</th><th>Team 2</th></tr></thead><tbody>
        {selected.matches.map(m => <tr key={m.id}><td>{m.round_number}</td><td>{(m.sideA || m.teamA || []).map(id => names.get(id) || "Unknown player").join(" & ")}</td><td style={{ whiteSpace: "nowrap" }}>{m.scoreA} – {m.scoreB}</td><td>{(m.sideB || m.teamB || []).map(id => names.get(id) || "Unknown player").join(" & ")}</td></tr>)}
      </tbody></table></div>
      {editable ? <>
        <p>The organizer chose {selected.rating_mode} before starting. {selected.rating_mode === "rated" ? "Approval updates the appropriate singles or doubles ratings." : "Approval leaves ratings unchanged."} Approved games count toward stats and recaps.</p>
        {!allMapped ? <p>Choose a club profile for every player before approving.</p> : null}
        <label style={{ display: "grid", gap: 4 }}>Reason if rejecting (optional)<textarea value={reason} disabled={busy} maxLength={500} onChange={e => setReason(e.target.value)} style={field} /></label>
        <div style={{ display: "flex", gap: "0.7rem", flexWrap: "wrap", marginTop: "1rem" }}>
          {(["approve", "reject"] as const).map(action => <ConfirmAction key={action}
            triggerLabel={action === "reject" ? "Reject" : `Approve ${selected.rating_mode} results`} title={action === "reject" ? "Reject these results?" : `Approve ${selected.match_count} ${selected.rating_mode} games?`}
            description={action === "reject" ? "These games will stay out of club records." : selected.rating_mode === "rated" ? "These games will update player ratings and count toward stats and recaps." : "These games will count toward stats and recaps. Ratings stay unchanged."}
            confirmLabel={action === "reject" ? "Reject results" : `Approve ${selected.rating_mode} results`} confirmationText="" disabled={busy || (action !== "reject" && (!allMapped || !matchDate))} onConfirm={() => decide(action)} />)}
        </div>
      </> : null}
      {selected.status === "processing" && selected.review ? <button disabled={busy} style={field} onClick={() => void decide(selected.review!.request.action).catch(() => {})}>Resume {selected.review.request.action === "reject" ? "rejection" : `${selected.rating_mode} approval`}</button> : null}
    </article> : null}
  </div>;
}
