"use client";
import Link from "next/link";
import { useRef, useState } from "react";
import { managedRequest, managedState, type ManagedContext, type ManagedDetails, type ManagedMeet, type ManagedPlan } from "@/lib/managedInterclub";
import { CompetitionStandings } from "@/components/PublicInterclubCompetition";
import ManagedScoreboard from "@/components/interclub/ManagedScoreboard";
import TeamsEditor from "./TeamsEditor";
import MeetPlanner from "./MeetPlanner";
import styles from "../setup.module.css";

const tabs = ["Teams & Players", "Schedule", "Enter Results", "Standings"] as const;
export default function ManagedSeasonWorkspace({ root, accessToken, initial, onClose }: { root: string; accessToken: string; initial: ManagedContext; onClose: () => void }) {
  const [data, setData] = useState(initial), [tab, setTab] = useState<typeof tabs[number]>("Teams & Players");
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [status, setStatus] = useState("");
  const [planner, setPlanner] = useState<{ meet?: ManagedMeet } | null>(null), [selected, setSelected] = useState(initial.meets[0]?.id || "");
  const [hostLinks, setHostLinks] = useState<Record<string, string>>({}), [reason, setReason] = useState("");
  const [scoreDirty, setScoreDirty] = useState(false);
  const pending = useRef(false);
  const details = data.season.details, current = data.meets.find(m => m.id === selected);
  const clubName = (id: string) => details.clubs.find(c => c.id === id)?.name || id;
  const when = (value: string) => new Date(value).toLocaleString(undefined, { timeZone: details.timezone, dateStyle: "medium", timeStyle: "short" });
  async function request(path: string, method: string, body?: unknown) {
    if (pending.current) throw new Error("A save is still finishing. Please try again in a moment.");
    pending.current = true; setBusy(true); setError(""); setStatus("");
    try {
      const saved = await managedRequest<ManagedContext>(root + path, accessToken, method, body);
      setData(saved); setStatus("Saved."); return saved;
    } catch (reason) { const message = reason instanceof Error ? reason.message : "Could not confirm the save."; setError(message); throw reason; }
    finally { pending.current = false; setBusy(false); }
  }
  const safe = (action: () => Promise<unknown>) => { void action().catch(() => {}); };
  async function saveDetails(updated: ManagedDetails) { await request("", "PUT", { expected_revision: data.season.revision, details: updated }); }
  async function saveMeet(id: string, revision: number, plan: ManagedPlan) {
    await request(`/meets/${id}`, "PUT", { expected_revision: revision, season_revision: data.season.revision, details: plan });
    setPlanner(null); setSelected(id);
  }
  async function action(meet: ManagedMeet, name: string) {
    const saved = await request(`/meets/${meet.id}/${name}`, "POST", { expected_revision: meet.revision, ...(name === "reopen" ? { reason } : {}) });
    if (saved.host_token) setHostLinks(old => ({ ...old, [meet.id]: `${window.location.origin}/interclub/host#token=${saved.host_token}` }));
    if (name === "revoke") setHostLinks(old => { const next = { ...old }; delete next[meet.id]; return next; });
    if (name === "generate") { setSelected(meet.id); setTab("Enter Results"); }
  }
  const league = { document: { name: details.name, start_date: details.start_date || "", end_date: details.end_date || "", timezone: details.timezone,
    divisions: details.divisions, clubs: details.clubs.map(c => ({ id: c.id, name: c.name })), meets: [], results: [], scoring_version: 1 as const },
    standings: details.divisions.map(division => ({ division, rows: data.standings.divisions[division] || [] })), qualification: data.standings.qualification,
    club_cup: { ...data.club_cup, standings: data.club_cup.standings.map(row => ({ ...row, regular_points: row.regular_points || 0, championship_points: row.championship_points || 0 })) } };
  return <section className={styles.page}>
    <header className={styles.header}><div><p className={styles.eyebrow}>Organizer workspace</p><h1>{details.name}</h1><p>One league, one account. Hosts use their private meet links.</p></div>
      <button disabled={busy || scoreDirty} onClick={onClose}>All interclub leagues</button></header>
    <nav aria-label="League workspace" className={styles.toolbar}>{tabs.map(label => <button key={label} aria-current={tab === label ? "page" : undefined}
      className={tab === label ? styles.primary : undefined} disabled={busy || scoreDirty} onClick={() => { setTab(label); setPlanner(null); }}>{label}</button>)}</nav>
    {scoreDirty && <p>Keep this meet open until its scores are saved before switching sections.</p>}
    {error && <div role="alert" className={styles.error}><p>{error}</p><button disabled={busy} onClick={() => safe(() => request("", "GET"))}>Reload saved league</button></div>}
    <p role="status">{busy ? "Saving…" : status}</p>
    {tab === "Teams & Players" && <>
      <SeasonSettings key={JSON.stringify({ ...details, clubs: [] })} details={details} disabled={busy} onSave={saveDetails} />
      <TeamsEditor details={details} disabled={busy} onSave={saveDetails} />
    </>}
    {tab === "Schedule" && <>
      <h2>Meet schedule</h2><p>Choose the host, date and lineups. Generate the court schedule before sharing the host link.</p>
      {!planner && <button disabled={busy || details.clubs.length < 2} onClick={() => setPlanner({})}>Add meet</button>}
      {details.clubs.length < 2 && <p>Add at least two participating clubs first.</p>}
      {planner && <MeetPlanner key={planner.meet?.id || "new"} details={details} meet={planner.meet} disabled={busy} onSave={saveMeet} onCancel={() => setPlanner(null)} />}
      <div className={styles.cards}>{data.meets.map(meet => <article className={styles.card} key={meet.id}><h3>{when(meet.details.starts_at)}</h3>
        <p>Host: <strong>{clubName(meet.details.host_club_id)}</strong> · {meet.details.courts} courts</p><p>{meet.details.club_ids.map(clubName).join(" · ")}</p><p>{managedState[meet.state]}</p>
        <div className={styles.toolbar}>{["planning", "draft"].includes(meet.state) && <button disabled={busy} onClick={() => setPlanner({ meet })}>Edit date & lineups</button>}
          {meet.state === "planning" ? <button disabled={busy || !meet.details.teams.length} onClick={() => safe(() => action(meet, "generate"))}>Generate court schedule</button> :
            <button disabled={busy} onClick={() => { setSelected(meet.id); setTab("Enter Results"); }}>Open meet & host link</button>}</div>
      </article>)}</div>
    </>}
    {tab === "Enter Results" && <>
      <h2>Meet schedules & results</h2><label className={styles.field}>Meet<select disabled={busy || scoreDirty} value={selected} onChange={e => { setSelected(e.target.value); setReason(""); }}>
        <option value="">Choose a meet</option>{data.meets.map(meet => <option key={meet.id} value={meet.id}>{when(meet.details.starts_at)} · {clubName(meet.details.host_club_id)} · {managedState[meet.state]}</option>)}</select></label>
      {!data.meets.length && <p>Add your first meet in Schedule.</p>}
      {current && <>
        {current.document && <section className={styles.panel}><h3>Host access</h3><p>Send this link directly to the person running this meet. It allows printing and score entry without signing in.</p>
          <div className={styles.toolbar}><button disabled={busy || scoreDirty} onClick={() => safe(() => action(current, "share"))}>{current.host_expires_at ? "Replace host link" : "Create host link"}</button>
            {current.host_expires_at && <button disabled={busy || scoreDirty} onClick={() => safe(() => action(current, "revoke"))}>Disable host link</button>}</div>
          {current.host_expires_at && <p>Host access active until {when(current.host_expires_at)}. Replacing the link disables the previous one.</p>}
          {hostLinks[current.id] && <label className={styles.field}>Private host link<input aria-label="Private host link" readOnly value={hostLinks[current.id]} onFocus={e => e.target.select()} /><button type="button" onClick={() => {
            void navigator.clipboard.writeText(hostLinks[current.id]).then(() => setStatus("Host link copied.")).catch(() => setStatus("Select and copy the host link above."));
          }}>Copy host link</button></label>}
        </section>}
        {current.state === "planning" ? <p>Finish the lineups and generate the court schedule in Schedule.</p> : <ManagedScoreboard key={current.id} meet={current} seasonName={details.name} timezone={details.timezone} clubName={clubName} onDirtyChange={setScoreDirty}
          onWrite={async (name, revision, document) => {
            const saved = await request(`/meets/${current.id}/${name}`, name === "scores" ? "PUT" : "POST", { expected_revision: revision, ...(document ? { document } : {}) });
            return saved.meets.find(m => m.id === current.id)!;
          }} />}
        {current.state === "submitted" && <section className={styles.panel}><h3>Review completed meet</h3><p>Check the scores above. Approval updates the official standings and the published league page.</p>
          <button className={styles.primary} disabled={busy} onClick={() => safe(() => action(current, "approve"))}>Approve official results</button></section>}
        {["submitted", "approved"].includes(current.state) && <details className={styles.panel}><summary>Correct or return results</summary><label className={styles.field}>Reason<input value={reason} onChange={e => setReason(e.target.value)} maxLength={500} /></label>
          <p>Existing official results remain visible until you approve the corrected meet.</p><button disabled={busy || reason.trim().length < 3} onClick={() => safe(() => action(current, "reopen"))}>Reopen score entry</button></details>}
      </>}
    </>}
    {tab === "Standings" && <>
      <section className={styles.panel}><h2>Public league page</h2><p>Share this viewing link on Facebook. Only organizer-approved results affect these standings.</p>
        <button disabled={busy} onClick={() => safe(() => request(`/visibility/${data.season.published ? "unpublish" : "publish"}`, "POST", { expected_revision: data.season.revision }))}>
          {data.season.published ? "Unpublish league page" : "Publish league page"}</button>
        {data.season.published && <p><Link href={`/interclub/managed/${data.season.id}`} target="_blank">Open public schedule, results & standings</Link><button onClick={() => {
          void navigator.clipboard.writeText(`${window.location.origin}/interclub/managed/${data.season.id}`).then(() => setStatus("Public league link copied.")).catch(() => setStatus("Open the public page and copy its address."));
        }}>Copy public link</button></p>}</section>
      <CompetitionStandings league={league} />
    </>}
  </section>;
}

function SeasonSettings({ details, disabled, onSave }: { details: ManagedDetails; disabled: boolean; onSave: (details: ManagedDetails) => Promise<void> }) {
  const [draft, setDraft] = useState(details), [divisions, setDivisions] = useState(details.divisions.join(", "));
  return <details className={styles.panel} open={!details.start_date}><summary>Season name, dates, divisions & rules</summary><form onSubmit={e => {
    e.preventDefault(); void onSave({ ...draft, clubs: details.clubs, divisions: divisions.split(",").map(s => s.trim()).filter(Boolean) }).catch(() => {});
  }}><fieldset className={styles.form} disabled={disabled}><div className={styles.grid}>
    <label className={styles.field}>Season name<input required value={draft.name} maxLength={120} onChange={e => setDraft({ ...draft, name: e.target.value })} /></label>
    <label className={styles.field}>Divisions<input required value={divisions} onChange={e => setDivisions(e.target.value)} placeholder="3.0, 3.5, 4.0, 4.5" /></label>
    <label className={styles.field}>Start date<input type="date" value={draft.start_date || ""} onChange={e => setDraft({ ...draft, start_date: e.target.value || null })} /></label>
    <label className={styles.field}>End date<input type="date" value={draft.end_date || ""} onChange={e => setDraft({ ...draft, end_date: e.target.value || null })} /></label>
    <label className={styles.field}>Timezone<select value={draft.timezone} onChange={e => setDraft({ ...draft, timezone: e.target.value })}>
      {[...new Set(["America/Mazatlan", "America/Tijuana", "America/Mexico_City", "America/Los_Angeles", "UTC", draft.timezone])].map(zone => <option key={zone}>{zone}</option>)}</select></label>
    <label className={`${styles.field} ${styles.wide}`}>League information & rules<textarea maxLength={10000} rows={5} value={draft.rules} onChange={e => setDraft({ ...draft, rules: e.target.value })} /><small>Shown on the public league page and host page.</small></label>
  </div><button className={styles.primary}>Save season details</button></fieldset></form></details>;
}
