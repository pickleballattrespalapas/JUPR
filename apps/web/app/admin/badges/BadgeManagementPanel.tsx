"use client";

import { useRef, useState } from "react";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAuthenticatedAutoLoad, useLatestRequestGuard } from "@/lib/useAuthenticatedAutoLoad";

type Season = { id: string; name: string; start_date: string; end_date: string; timezone: string; revision: number };
type Badge = { id: string; name: string; requirement: string; available: boolean; criteria: Record<string, string> };
type Award = { id: string; player_id: number; badge_id: string; earned_at: string; revoked_at: string | null; value_json: { recognition_note?: string; contribution_date?: string } };
type Options = { players: { id: number; name: string }[]; badges: Badge[]; seasons: Season[]; recent_awards: Award[]; write_enabled: boolean };
type Pending = { kind: "awards" | "seasons"; payload: Record<string, unknown> };
const card = { border: "1px solid #cbd5e1", borderRadius: 14, padding: "1rem", background: "white", marginBottom: "1rem" };
const input = { width: "100%", padding: ".55rem", border: "1px solid #94a3b8", borderRadius: 7, font: "inherit" };
const button = { padding: ".6rem 1rem", border: 0, borderRadius: 8, background: "#0f172a", color: "white", fontWeight: 700 };
const grid = { display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(200px,1fr))", gap: "1rem" };

export default function BadgeManagementPanel({ apiBase, clubId }: { apiBase: string | null; clubId: string }) {
  const { session, accessToken } = useAdminSession();
  const [options, setOptions] = useState<Options | null>(null);
  const [player, setPlayer] = useState("");
  const [badgeId, setBadgeId] = useState("good_sport");
  const [criteria, setCriteria] = useState<string[]>([]);
  const [note, setNote] = useState("");
  const [day, setDay] = useState("");
  const emptySeason = (): Season => ({ id: "", name: "", start_date: "", end_date: "", timezone: clubId === "tres_palapas" ? "America/Mazatlan" : "UTC", revision: 0 });
  const [season, setSeason] = useState<Season>(emptySeason);
  const [pending, setPending] = useState<Pending | null>(null);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const scope = `${clubId}:${session?.user?.id || session?.user?.email || ""}`;
  const storageKey = `badge-save:${scope}`;
  const requestGuard = useLatestRequestGuard(`${accessToken}:${clubId}`, () => { setOptions(null); setPending(null); });
  const selectedBadge = options?.badges.find(badge => badge.id === badgeId);

  async function request(path: string, body?: Record<string, unknown>) {
    if (!apiBase || !accessToken) throw new Error("Sign in to manage badges.");
    const response = await fetch(`${apiBase.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/badge-management${path}`, {
      method: body ? "POST" : "GET", headers: { Authorization: `Bearer ${accessToken}`, ...(body ? { "Content-Type": "application/json" } : {}) },
      body: body ? JSON.stringify(body) : undefined, cache: "no-store"
    });
    const data = await response.json();
    if (!response.ok) {
      const error = new Error(typeof data.detail === "string" ? data.detail : "Unable to save. Check the selected player, action, and dates.") as Error & { status: number };
      error.status = response.status;
      throw error;
    }
    return data;
  }

  async function load() {
    const generation = requestGuard.begin();
    setError("");
    try {
      const data = await request("") as Options;
      if (!requestGuard.isCurrent(generation)) return;
      setOptions(data);
      const stored = sessionStorage.getItem(storageKey);
      if (stored) {
        const saved = JSON.parse(stored) as Pending;
        if (["awards", "seasons"].includes(saved.kind) && saved.payload?.operation_id) setPending(saved);
      }
    } catch (err) { if (requestGuard.isCurrent(generation)) setError(err instanceof Error ? err.message : "Unable to load badges."); }
  }
  useAuthenticatedAutoLoad(accessToken, load, clubId);

  async function save(kind: Pending["kind"], payload: Record<string, unknown>) {
    if (busyRef.current) return;
    const submission = pending || { kind, payload: { ...payload, operation_id: crypto.randomUUID() } };
    busyRef.current = true; setBusy(true); setError(""); setMessage("");
    const generation = requestGuard.begin();
    try {
      sessionStorage.setItem(storageKey, JSON.stringify(submission));
      setPending(submission);
      await request(`/${submission.kind}`, submission.payload);
      sessionStorage.removeItem(storageKey);
      if (!requestGuard.isCurrent(generation)) return;
      setPending(null);
      if (submission.kind === "awards") { setNote(""); setCriteria([]); setDay(""); setMessage("Badge awarded. The contribution has been recorded."); }
      else { setSeason(emptySeason()); setMessage("Season saved."); }
      const refreshed = await request("") as Options;
      if (requestGuard.isCurrent(generation)) setOptions(refreshed);
    } catch (err) {
      if (!requestGuard.isCurrent(generation)) return;
      const status = (err as { status?: number }).status;
      if (status && status >= 400 && status < 500) { sessionStorage.removeItem(storageKey); setPending(null); }
      setError(err instanceof Error ? err.message : "Unable to verify the save. Retry to check its result.");
    } finally { busyRef.current = false; setBusy(false); }
  }

  if (!accessToken) return <p>Sign in as a club administrator to award community badges and manage seasons.</p>;
  const disabled = busy || Boolean(pending) || !options?.write_enabled;
  return <div>
    {error ? <p role="alert" style={{ color: "#b91c1c" }}>{error} <button type="button" onClick={() => void load()}>Refresh</button></p> : null}
    {message ? <p role="status" style={{ color: "#166534" }}>{message}</p> : null}
    {pending ? <div style={card}><p>A save is awaiting confirmation. Retrying checks the same request and will not create a duplicate.</p><button style={button} disabled={busy} onClick={() => void save(pending.kind, pending.payload)}>{busy ? "Saving…" : "Retry save"}</button></div> : null}
    {options && !options.write_enabled ? <p>Badge changes are currently paused.</p> : null}
    <section style={card} aria-labelledby="community-badge-heading">
      <h2 id="community-badge-heading" style={{ marginTop: 0 }}>Award a community badge</h2>
      <p>Recognize a specific contribution. Players can receive these badges again for separate contributions.</p>
      <form onSubmit={event => { event.preventDefault(); void save("awards", { player_id: Number(player), badge_id: badgeId, criteria, note, contribution_date: day }); }}>
        <fieldset disabled={disabled} style={{ border: 0, padding: 0, margin: 0 }}>
          <div style={grid}>
            <label>Player<select required style={input} value={player} onChange={event => setPlayer(event.target.value)}><option value="">Choose a player</option>{options?.players.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}</select></label>
            <label>Badge<select style={input} value={badgeId} onChange={event => { setBadgeId(event.target.value); setCriteria([]); }}>{options?.badges.map(b => <option key={b.id} value={b.id} disabled={!b.available}>{b.name}{b.available ? "" : " (paused)"}</option>)}</select></label>
            <label>Contribution date<input style={input} type="date" required value={day} onChange={event => setDay(event.target.value)} /></label>
          </div>
          <fieldset style={{ margin: "1rem 0", border: "1px solid #cbd5e1", borderRadius: 8 }}><legend>Qualifying actions</legend>{Object.entries(selectedBadge?.criteria || {}).map(([key, label]) => <label key={key} style={{ display: "block", margin: ".5rem 0" }}><input type="checkbox" checked={criteria.includes(key)} onChange={event => setCriteria(current => event.target.checked ? [...current, key] : current.filter(item => item !== key))} /> {label}</label>)}</fieldset>
          <label>What did they do?<textarea style={input} required maxLength={1000} rows={3} value={note} onChange={event => setNote(event.target.value)} placeholder="Describe the contribution you are recognizing." /></label>
          <button style={{ ...button, marginTop: ".75rem" }} disabled={!selectedBadge?.available || !criteria.length} type="submit">Award badge</button>
        </fieldset>
      </form>
    </section>
    <section style={card} aria-labelledby="badge-season-heading">
      <h2 id="badge-season-heading" style={{ marginTop: 0 }}>Badge seasons</h2>
      <p>Set the dates used by Battle Tested, Consistency, Steady Hand, and Mr. Reliable. Dates include the full first and last day.</p>
      {!options?.seasons.length ? <p>No seasons have been set. Season-based badges will begin once you add one.</p> : <ul>{options.seasons.map(s => <li key={s.id} style={{ marginBottom: ".5rem" }}><strong>{s.name}</strong> · {s.start_date} to {s.end_date} · {s.timezone} <button disabled={disabled} onClick={() => setSeason(s)}>Edit</button></li>)}</ul>}
      <form onSubmit={event => { event.preventDefault(); void save("seasons", { id: season.id || crypto.randomUUID(), name: season.name, start_date: season.start_date, end_date: season.end_date, timezone: season.timezone, expected_revision: season.revision }); }}>
        <fieldset disabled={disabled} style={{ border: 0, padding: 0, margin: 0 }}>
          <legend style={{ fontWeight: 700, marginBottom: ".75rem" }}>{season.id ? "Edit season" : "Add a season"}</legend>
          <div style={grid}>
            <label>Season name<input style={input} required maxLength={100} value={season.name} onChange={event => setSeason({ ...season, name: event.target.value })} placeholder="2026–27 season" /></label>
            <label>Start date<input style={input} type="date" required value={season.start_date} onChange={event => setSeason({ ...season, start_date: event.target.value })} /></label>
            <label>End date<input style={input} type="date" required min={season.start_date} value={season.end_date} onChange={event => setSeason({ ...season, end_date: event.target.value })} /></label>
            <label>Timezone<input style={input} required list="badge-timezones" value={season.timezone} onChange={event => setSeason({ ...season, timezone: event.target.value })} /><datalist id="badge-timezones"><option value="America/Mazatlan" /><option value="America/Los_Angeles" /><option value="America/New_York" /><option value="UTC" /></datalist></label>
          </div>
          <p><small>Season dates can be edited until a badge has been awarded for that season.</small></p>
          <button style={button} type="submit">Save season</button>{season.id ? <button type="button" onClick={() => setSeason(emptySeason())} style={{ marginLeft: ".75rem" }}>Cancel edit</button> : null}
        </fieldset>
      </form>
    </section>
    {options?.recent_awards.length ? <section style={card}><h2 style={{ marginTop: 0 }}>Recent community awards</h2><ul>{options.recent_awards.map(award => <li key={award.id} style={{ marginBottom: ".8rem" }}><strong>{options.players.find(p => p.id === award.player_id)?.name || `Player ${award.player_id}`}</strong> · {options.badges.find(b => b.id === award.badge_id)?.name}{award.revoked_at ? " · Revoked" : ""}<br /><small>{award.value_json.contribution_date || award.earned_at.slice(0, 10)} · {award.value_json.recognition_note}</small></li>)}</ul></section> : null}
  </div>;
}
