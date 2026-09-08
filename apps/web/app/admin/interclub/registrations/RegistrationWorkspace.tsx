"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import { InterclubTeam, MeetRegistrationDetail, RegistrationDetail, RegistrationSeason, RosterVersion, apiError, composition, rosterStatus } from "@/lib/interclubRegistration";
import styles from "./registrations.module.css";

export default function RegistrationWorkspace({ initialSeasonId }: { initialSeasonId: string }) {
  const { session, accessToken, loading } = useAdminSession();
  const { clubId } = useAdminWorkspace();
  const canManage = session?.capabilities?.assignments.some(a => a.club_id === clubId && ["administrator", "club_owner", "super_admin"].includes(a.role));
  if (loading) return <p>Checking club access…</p>;
  if (!canManage) return <p>Sign in as a club administrator to manage participation and rosters. <Link href="/admin/login">Sign in</Link></p>;
  return <ClubRegistrations key={`${clubId}:${session?.user?.id || session?.user?.email}`} clubId={clubId} accessToken={accessToken} initialSeasonId={initialSeasonId} />;
}

function ClubRegistrations({ clubId, accessToken, initialSeasonId }: { clubId: string; accessToken: string; initialSeasonId: string }) {
  const api = getAdminApiBaseUrl();
  const [seasons, setSeasons] = useState<RegistrationSeason[]>([]);
  const [selected, setSelected] = useState(initialSeasonId);
  const [error, setError] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [reload, setReload] = useState(0);
  const token = useRef(accessToken); token.current = accessToken;
  useEffect(() => {
    const controller = new AbortController(); setLoaded(false); setError("");
    if (!api) { setError("Interclub registration is unavailable."); return; }
    fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub/registrations`, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal })
      .then(async response => {
        const data = await response.json(); if (!response.ok) throw new Error(apiError(data, "Unable to load club invitations."));
        if (!controller.signal.aborted) { setSeasons(data.seasons); setSelected(old => data.seasons.some((s: RegistrationSeason) => s.id === old) ? old : data.seasons[0]?.id || ""); setLoaded(true); }
      }).catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [api, clubId, reload]);
  return <section className={styles.page}>
    <h1>Interclub season workspace</h1>
    <p>Accept season invitations, then choose your available players for each upcoming meet.</p>
    <p><Link href="/admin/interclub">Back to interclub leagues and setup</Link></p>
    <div className={styles.toolbar}>
      <label>Season <select value={selected} onChange={e => setSelected(e.target.value)} disabled={!loaded}>
        {!seasons.length && <option value="">No open invitations</option>}
        {seasons.map(s => <option key={s.id} value={s.id}>{s.details.name} · {s.details.start_date}{s.organizer_club_id === clubId ? " · Organizer" : ""}</option>)}
      </select></label>
      <button onClick={() => setReload(n => n + 1)}>Refresh invitations</button>
    </div>
    {error && <p role="alert">{error}</p>}
    {loaded && !seasons.length && <p>Your club has no season invitations yet. Organizers open invitations from a saved season plan.</p>}
    {loaded && selected && api && <SeasonRegistration key={`${selected}:${reload}`} api={api} clubId={clubId} accessToken={accessToken} seasonId={selected} />}
  </section>;
}

function SeasonRegistration({ api, clubId, accessToken, seasonId }: { api: string; clubId: string; accessToken: string; seasonId: string }) {
  const root = `${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub/registrations/${seasonId}`;
  const token = useRef(accessToken); token.current = accessToken;
  const [data, setData] = useState<RegistrationDetail | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [blocked, setBlocked] = useState(false);
  const [reload, setReload] = useState(0);
  const [selectedMeet, setSelectedMeet] = useState("");
  const pending = useRef(false), mutation = useRef<AbortController | null>(null);
  useEffect(() => () => { mutation.current?.abort(); }, [root]);
  useEffect(() => {
    const controller = new AbortController(); setData(null); setBlocked(false);
    fetch(root, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal })
      .then(async response => {
        const next = await response.json(); if (!response.ok) throw new Error(apiError(next, "Unable to load this season."));
        if (!controller.signal.aborted) { setData(next); setSelectedMeet(old => next.meets.some((m: { id: string }) => m.id === old) ? old : next.meets.find((m: { roster_open: boolean }) => m.roster_open)?.id || next.meets[0]?.id || ""); }
      }).catch(e => { if (!controller.signal.aborted) setMessage(e.message); });
    return () => controller.abort();
  }, [root, reload]);

  async function change(path: string, method: string, body: object, success: string): Promise<boolean> {
    if (pending.current || blocked) return false;
    pending.current = true; setBusy(true); setMessage("");
    const controller = new AbortController(); mutation.current = controller;
    try {
      const response = await fetch(`${root}${path}`, { method, signal: controller.signal,
        headers: { Authorization: `Bearer ${token.current}`, "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const result = await response.json();
      if (controller.signal.aborted) return false;
      if (!response.ok) {
        if ([409, 503].includes(response.status)) setBlocked(true);
        throw new Error(apiError(result, "Unable to save this change."));
      }
      setMessage(success); setReload(n => n + 1); return true;
    } catch (e) { if (!controller.signal.aborted) { setMessage(e instanceof Error ? e.message : "Unable to save this change."); if (e instanceof TypeError) setBlocked(true); } return false; }
    finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }
  async function moreTeams() {
    if (!data || data.next_team_offset === null || pending.current) return;
    pending.current = true; setBusy(true);
    const controller = new AbortController(); mutation.current = controller;
    try {
      const response = await fetch(`${root}?team_offset=${data.next_team_offset}`, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal });
      const next = await response.json(); if (!response.ok) throw new Error(apiError(next, "Unable to load more teams."));
      if (!controller.signal.aborted) setData(old => old ? { ...old, teams: Array.from(new Map([...old.teams, ...next.teams].map((t: InterclubTeam) => [t.id, t])).values()), next_team_offset: next.next_team_offset } : old);
    } catch (e) { if (!controller.signal.aborted) setMessage(e instanceof Error ? e.message : "Unable to load more teams."); }
    finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }
  const disabled = busy || blocked;
  const clubName = (id: string) => data?.clubs.find(c => c.id === id)?.name || id;
  const when = (value: string) => new Date(value).toLocaleString(undefined, { timeZone: data?.season.details.timezone });
  return <>
    <div className={styles.toolbar}><button disabled={busy} onClick={() => { setMessage(""); setReload(n => n + 1); }}>Reload season</button>{blocked && <span>Reload before making another change. Your draft remains below for reference.</span>}</div>
    {message && <p role="status" className={styles.notice}>{message}</p>}
    {!data ? <p>Loading season…</p> : <>
      <h2>{data.season.details.name}</h2>
      <p>Organized by {clubName(data.season.organizer_club_id)} · {data.season.details.start_date} to {data.season.details.end_date}</p>
      <ol className={styles.nextSteps} aria-label="Season next steps">
        <li><strong>1. Club responses</strong><p>{data.is_organizer ? `${data.participations.filter(p => p.status === "accepted").length} of ${data.participations.length} clubs have accepted.` : data.own_participation?.status === "accepted" ? "Your club has accepted the season invitation." : data.own_participation?.status === "invited" ? "Respond to your club’s invitation below." : "Your club has not joined. Ask the organizer for a new invitation."}</p><a href="#club-responses">Review responses</a></li>
        <li><strong>2. Prepare a meet</strong><p>Choose an upcoming meet. The organizer can set its roster deadline before teams submit.</p><a href="#meet-rosters">Choose a meet</a></li>
        <li><strong>3. Submit meet rosters</strong><p>Each accepted club chooses four available players per team. Lineups can change at the next meet.</p></li>
      </ol>
      <details className={styles.card}><summary>Season details and eligibility rules</summary>
      <div className={styles.scroll}><table className={styles.table}><caption>Season eligibility rules</caption><thead><tr><th>Division</th><th>Minimum rating</th><th>Maximum rating</th><th>Team</th></tr></thead><tbody>
        {Object.entries(data.season.rules).map(([division, rule]) => <tr key={division}><td>{division}</td><td>{rule.min_rating ?? "No minimum"}</td><td>{rule.max_rating ?? "No maximum"}</td><td>{composition(rule)}</td></tr>)}
      </tbody></table></div>
      <p>Starting ratings come from the represented club when each player first enters this season. Those starting ratings stay fixed for this season’s eligibility checks.</p>
      </details>
      {data.own_participation && <section id={data.is_organizer ? undefined : "club-responses"} className={styles.card}><h3>{clubName(clubId)} participation</h3><p className={styles.tag}>{data.own_participation.status}</p>
        {data.own_participation.status === "invited" && <div className={styles.toolbar}>
          <button disabled={disabled} onClick={() => void change(`/participations/${encodeURIComponent(clubId)}`, "POST", { action: "accept", expected_revision: data.own_participation!.revision }, "Your club accepted the invitation. Choose an upcoming meet to submit its roster.")}>Accept season invitation</button>
          <button disabled={disabled} onClick={() => void change(`/participations/${encodeURIComponent(clubId)}`, "POST", { action: "decline", expected_revision: data.own_participation!.revision }, "Your club declined the invitation.")}>Decline invitation</button>
        </div>}
      </section>}
      {data.is_organizer && <section id="club-responses" className={styles.card}><h3>Club responses</h3><div className={styles.scroll}><table className={styles.table}><thead><tr><th>Club</th><th>Response</th><th>Invitation</th></tr></thead><tbody>
        {data.participations.map(p => <tr key={p.club_id}><td>{clubName(p.club_id)}</td><td>{p.status}</td><td>
          {p.status === "invited" && <button disabled={disabled} onClick={() => void change(`/participations/${encodeURIComponent(p.club_id)}`, "POST", { action: "cancel", expected_revision: p.revision }, "Club invitation cancelled.")}>Cancel invitation</button>}
          {["declined", "cancelled"].includes(p.status) && <button disabled={disabled} onClick={() => void change(`/participations/${encodeURIComponent(p.club_id)}`, "POST", { action: "reinvite", expected_revision: p.revision }, "Club invited again.")}>Invite again</button>}
        </td></tr>)}
      </tbody></table></div></section>}
      <h3 id="meet-rosters">Meet rosters</h3>
      <p>Choose four players for each meet. Your lineup can change from one meet to the next.</p>
      {!data.meets.length ? <p>No meets are scheduled for your club in this season.</p> : <>
        <label>Meet <select aria-label="Meet" value={selectedMeet} onChange={e => setSelectedMeet(e.target.value)}>
          {data.meets.map(meet => <option key={meet.id} value={meet.id}>{when(meet.starts_at)} · {clubName(meet.host_club_id)}{meet.roster_open ? "" : " · History"}</option>)}
        </select></label>
        {selectedMeet && <MeetRegistration key={selectedMeet} root={`${root}/meets/${selectedMeet}`} accessToken={accessToken} clubId={clubId} seasonData={data} />}
      </>}
      {data.teams.length > 0 && <details className={styles.card}><summary>Earlier season submissions (reference only)</summary>
        <p>These were submitted before rosters moved to individual meets. Choose players again for each upcoming meet.</p>
        {data.teams.map(team => <TeamCard key={`${team.id}:${team.revision}`} team={team} clubName={clubName(team.club_id)} root={root} accessToken={accessToken} own={false} organizer={false} disabled={true}
          edit={() => {}} withdraw={async () => false} decide={async () => false} />)}
        {data.next_team_offset !== null && <button disabled={disabled} onClick={() => void moreTeams()}>Load earlier submissions</button>}
      </details>}
    </>}
  </>;
}

function MeetRegistration({ root, accessToken, clubId, seasonData }: { root: string; accessToken: string; clubId: string; seasonData: RegistrationDetail }) {
  const token = useRef(accessToken); token.current = accessToken;
  const [data, setData] = useState<MeetRegistrationDetail | null>(null);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [blocked, setBlocked] = useState(false);
  const [reload, setReload] = useState(0);
  const [editor, setEditor] = useState<{ team: InterclubTeam | null } | null>(null);
  const pending = useRef(false), mutation = useRef<AbortController | null>(null);
  useEffect(() => () => { mutation.current?.abort(); }, [root]);
  useEffect(() => {
    const controller = new AbortController(); setData(null); setEditor(null); setBlocked(false);
    fetch(root, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal })
      .then(async response => {
        const next = await response.json(); if (!response.ok) throw new Error(apiError(next, "Unable to load this meet."));
        if (!controller.signal.aborted) setData(next);
      }).catch(e => { if (!controller.signal.aborted) setMessage(e.message); });
    return () => controller.abort();
  }, [root, reload]);

  async function change(path: string, method: string, body: object, success: string): Promise<boolean> {
    if (pending.current || blocked) return false;
    pending.current = true; setBusy(true); setMessage("");
    const controller = new AbortController(); mutation.current = controller;
    try {
      const response = await fetch(`${root}${path}`, { method, signal: controller.signal,
        headers: { Authorization: `Bearer ${token.current}`, "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const result = await response.json();
      if (controller.signal.aborted) return false;
      if (!response.ok) {
        if ([409, 503].includes(response.status)) setBlocked(true);
        throw new Error(apiError(result, "Unable to save this change."));
      }
      setMessage(success); setReload(n => n + 1); return true;
    } catch (e) { if (!controller.signal.aborted) { setMessage(e instanceof Error ? e.message : "Unable to save this change."); if (e instanceof TypeError) setBlocked(true); } return false; }
    finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }
  async function moreTeams() {
    if (!data || data.next_team_offset === null || pending.current) return;
    pending.current = true; setBusy(true);
    const controller = new AbortController(); mutation.current = controller;
    try {
      const response = await fetch(`${root}?team_offset=${data.next_team_offset}`, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal });
      const next = await response.json(); if (!response.ok) throw new Error(apiError(next, "Unable to load more teams."));
      if (!controller.signal.aborted) setData(old => old ? { ...old, teams: Array.from(new Map([...old.teams, ...next.teams].map((t: InterclubTeam) => [t.id, t])).values()), next_team_offset: next.next_team_offset } : old);
    } catch (e) { if (!controller.signal.aborted) setMessage(e instanceof Error ? e.message : "Unable to load more teams."); }
    finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }
  const [deadline, setDeadline] = useState("");
  const disabled = busy || blocked || !data?.meet.roster_open;
  const clubName = (id: string) => seasonData.clubs.find(c => c.id === id)?.name || id;
  const when = (value: string) => new Date(value).toLocaleString(undefined, { timeZone: seasonData.season.details.timezone });
  const ownMeet = Boolean(data?.meet.club_ids.includes(clubId) && seasonData.own_participation?.status === "accepted");
  const revision = { expected_meet_revision: data?.meet.revision };
  return <section>
    <div className={styles.toolbar}><button disabled={busy} onClick={() => { setMessage(""); setDeadline(""); setReload(n => n + 1); }}>Reload meet</button>{blocked && <span>Reload before making another change. Your draft remains below for reference.</span>}</div>
    {message && <p role="status" className={styles.notice}>{message}</p>}
    {!data ? <p>Loading meet…</p> : <>
      <h3>{when(data.meet.starts_at)} · {clubName(data.meet.host_club_id)}</h3>
      <p>{data.meet.club_ids.map(clubName).join(", ")} · {data.meet.courts} courts</p>
      <p>Roster deadline for this meet: {when(data.meet.roster_deadline)} ({seasonData.season.details.timezone}).</p>
      {data.meet.roster_open ? <p>Valid substitutions are allowed after the deadline, until this meet starts. A new team submitted after the deadline needs an organizer exception.</p>
        : <p>This meet has started. Its rosters and decisions are available as history.</p>}
      {seasonData.is_organizer && data.meet.deadline_editable && <form className={styles.card} onSubmit={e => { e.preventDefault(); void change("/deadline", "PUT", { expected_revision: data.meet.revision, roster_deadline: new Date(deadline).toISOString() }, "Meet roster deadline saved."); }}>
        <label>Roster deadline (your device’s local time) <input required disabled={disabled} type="datetime-local" value={deadline} onChange={e => setDeadline(e.target.value)} /></label>
        <p>The default is the meet’s start time. Set an earlier deadline before the first team submits.</p>
        <button disabled={disabled || !deadline} type="submit">Save meet deadline</button>
      </form>}
      {ownMeet && data.meet.roster_open && <button disabled={disabled} onClick={() => setEditor({ team: null })}>Add a team for this meet</button>}
      {editor && <RosterEditor key={editor.team ? `${editor.team.id}:${editor.team.revision}` : "new"} root={root} accessToken={accessToken} clubName={clubName(clubId)} divisions={seasonData.season.details.divisions} team={editor.team} disabled={disabled}
        cancel={() => setEditor(null)} save={(id, body) => change(`/teams/${id}`, "PUT", { ...body, ...revision }, "Roster submitted for this meet.")} />}
      <h4>{seasonData.is_organizer ? "Teams for this meet" : "Your club’s teams for this meet"}</h4>
      {!data.teams.length && <p>No teams submitted for this meet yet.</p>}
      {data.teams.map(team => <TeamCard key={`${team.id}:${team.revision}:${team.status}`} team={team} clubName={clubName(team.club_id)} root={root} accessToken={accessToken} own={ownMeet && team.club_id === clubId && data.meet.roster_open} organizer={seasonData.is_organizer && data.meet.roster_open} disabled={disabled}
        edit={() => setEditor({ team })}
        withdraw={() => change(`/teams/${team.id}/withdraw`, "POST", { expected_revision: team.revision, ...revision }, "Team withdrawn from this meet. Earlier rosters remain in its history.")}
        decide={(approve, reason) => change(`/teams/${team.id}/eligibility`, "POST", { expected_revision: team.revision, ...revision, approve, reason }, approve ? "Exception approved for this meet roster." : "Exception declined for this meet roster.")} />)}
      {data.next_team_offset !== null && <button disabled={busy} onClick={() => void moreTeams()}>Load more teams</button>}
    </>}
  </section>;
}

type PlayerChoice = { id: string; name: string; starting_rating: number | null };
function RosterEditor({ root, accessToken, clubName, divisions, team, disabled, cancel, save }: {
  root: string; accessToken: string; clubName: string; divisions: string[]; team: InterclubTeam | null; disabled: boolean; cancel: () => void;
  save: (id: string, body: object) => Promise<boolean>;
}) {
  const [id] = useState(() => team?.id || crypto.randomUUID());
  const [name, setName] = useState(team?.name || "");
  const [division, setDivision] = useState(team?.division || divisions[0] || "");
  const [selected, setSelected] = useState<PlayerChoice[]>(() => (team?.roster || []).map(p => ({ id: p.player_id!, name: p.name, starting_rating: p.starting_rating })));
  const [query, setQuery] = useState("");
  const [offset, setOffset] = useState(0);
  const [choices, setChoices] = useState<PlayerChoice[]>([]);
  const [next, setNext] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const token = useRef(accessToken); token.current = accessToken;
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setError("");
    fetch(`${root}/players?q=${encodeURIComponent(query)}&offset=${offset}`, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal })
      .then(async response => {
        const data = await response.json(); if (!response.ok) throw new Error(apiError(data, "Unable to load your club players."));
        if (!controller.signal.aborted) { setChoices(old => offset ? Array.from(new Map([...old, ...data.players].map(p => [p.id, p])).values()) : data.players); setNext(data.next_offset); }
      }).catch(e => { if (!controller.signal.aborted) setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [root, query, offset]);
  function toggle(player: PlayerChoice) {
    setSelected(old => old.some(p => p.id === player.id) ? old.filter(p => p.id !== player.id) : old.length < 4 ? [...old, player] : old);
  }
  return <form className={styles.card} onSubmit={e => { e.preventDefault(); if (selected.length === 4) void save(id, { expected_revision: team?.revision || 0, name, division, player_ids: selected.map(p => p.id) }); }}>
    <h3>{team ? "Update team roster" : "New four-player team"}</h3>
    <fieldset disabled={disabled} className={styles.form}><legend>{clubName}</legend>
      <label>Team name<input required maxLength={80} value={name} onChange={e => setName(e.target.value)} /></label>
      <label>Division<select disabled={Boolean(team) || disabled} value={division} onChange={e => setDivision(e.target.value)}>{divisions.map(d => <option key={d} value={d}>{d}</option>)}</select></label>
      <p>{selected.length} of 4 players selected. A player may be on only one of your club’s teams in each division for this meet.</p>
      {selected.length > 0 && <ul>{selected.map(player => <li key={player.id}>{player.name} · {player.starting_rating?.toFixed(3) ?? "No rating"} <button type="button" onClick={() => toggle(player)}>Remove {player.name}</button></li>)}</ul>}
      <label>Find a player in {clubName}<input type="search" maxLength={80} value={query} onChange={e => { setQuery(e.target.value); setOffset(0); }} /></label>
      {error && <p role="alert">{error}</p>}
      <div className={styles.choices}>{choices.map(player => <label key={player.id}>
        <input type="checkbox" checked={selected.some(p => p.id === player.id)} disabled={disabled || (selected.length === 4 && !selected.some(p => p.id === player.id))} onChange={() => toggle(player)} />
        {player.name} · {player.starting_rating?.toFixed(3) ?? "No rating"}
      </label>)}{loading && <p>Loading players…</p>}{!loading && !error && !choices.length && <p>No active players found.</p>}</div>
      {next !== null && <button type="button" disabled={loading || disabled} onClick={() => setOffset(next)}>More players</button>}
      <div className={styles.toolbar}><button type="submit" disabled={disabled || selected.length !== 4}>Submit four-player roster</button><button type="button" onClick={cancel}>Cancel roster edit</button></div>
    </fieldset>
  </form>;
}

function RosterTable({ version }: { version: RosterVersion }) {
  return <><div className={styles.scroll}><table className={styles.table}><thead><tr><th>Player</th><th>Starting rating</th><th>Gender</th></tr></thead><tbody>
    {version.roster.map((player, index) => <tr key={player.entry_id || index}><td>{player.name}</td><td>{player.starting_rating.toFixed(3)}</td><td>{player.gender === "female" ? "Female" : player.gender === "male" ? "Male" : "Not set"}</td></tr>)}
  </tbody></table></div>
    {version.issues.length > 0 && <ul className={styles.issues}>{version.issues.map((issue, index) => <li key={index}>{issue.message}</li>)}</ul>}
    {version.decision_reason && <p>Organizer decision: {version.decision_reason}</p>}
  </>;
}

function TeamCard({ team, clubName, root, accessToken, own, organizer, disabled, edit, withdraw, decide }: {
  team: InterclubTeam; clubName: string; root: string; accessToken: string; own: boolean; organizer: boolean; disabled: boolean;
  edit: () => void; withdraw: () => Promise<boolean>; decide: (approve: boolean, reason: string) => Promise<boolean>;
}) {
  const [reason, setReason] = useState("");
  const [history, setHistory] = useState(false);
  return <article className={styles.card}>
    <h4>{team.name} · {clubName} · {team.division}</h4>
    <p className={styles.tag}>{rosterStatus[team.status] || team.status} · Roster {team.revision}{team.late_change ? " · Updated after the deadline" : ""}</p>
    <RosterTable version={team} />
    <div className={styles.toolbar}>
      {own && <button disabled={disabled} onClick={edit}>{team.withdrawn ? "Restore with an updated roster" : "Edit roster"}</button>}
      {own && !team.withdrawn && <ConfirmAction disabled={disabled} triggerLabel="Withdraw team" title="Withdraw this team?" description="The players will be available for another team at this meet. Earlier rosters stay in the history." confirmationText="" confirmLabel="Withdraw team" tone="danger" onConfirm={async () => {
        if (!await withdraw()) throw new Error("The team could not be withdrawn. Check the message above and reload if needed.");
        return { status: "success", title: "Team withdrawn", description: "Roster history has been preserved." };
      }} />}
      <button onClick={() => setHistory(value => !value)}>{history ? "Hide history" : "Roster history"}</button>
    </div>
    {organizer && team.status === "needs_exception" && <div className={styles.form}>
      <label>Explain your eligibility decision<textarea value={reason} maxLength={500} disabled={disabled} onChange={e => setReason(e.target.value)} /></label>
      <div className={styles.toolbar}><button disabled={disabled || !reason.trim()} onClick={() => void decide(true, reason)}>Approve exception</button><button disabled={disabled || !reason.trim()} onClick={() => void decide(false, reason)}>Decline exception</button></div>
      <p>The decision applies only to this roster for this meet. A different lineup or meet needs its own review.</p>
    </div>}
    {history && <RosterHistory root={root} teamId={team.id} accessToken={accessToken} />}
  </article>;
}

function RosterHistory({ root, teamId, accessToken }: { root: string; teamId: string; accessToken: string }) {
  const [rows, setRows] = useState<RosterVersion[]>([]), [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${root}/teams/${teamId}/history`, { headers: { Authorization: `Bearer ${accessToken}` }, cache: "no-store", signal: controller.signal })
      .then(async response => { const data = await response.json(); if (!response.ok) throw new Error(apiError(data, "Unable to load roster history.")); if (!controller.signal.aborted) setRows(data.history); })
      .catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [root, teamId, accessToken]);
  return <section><h5>Roster history (latest 50 versions)</h5>{error && <p role="alert">{error}</p>}{rows.map(row => <details key={row.revision}><summary>Roster {row.revision} · {rosterStatus[row.status]} · {new Date(row.submitted_at).toLocaleString()}</summary><RosterTable version={row} /></details>)}</section>;
}
