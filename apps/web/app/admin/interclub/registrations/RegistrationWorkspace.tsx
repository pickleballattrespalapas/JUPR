"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import { InterclubTeam, MeetRegistrationDetail, RegistrationDetail, RegistrationSeason, RosterVersion, apiError, composition, rosterStatus } from "@/lib/interclubRegistration";
import { divisionEligibilityLabel, emptyRule, meetLocalTime, meetUtcTime } from "@/lib/interclubSetup";
import styles from "./registrations.module.css";
import { SeasonPlayerPool, MeetAvailability } from "./PlayerPoolPanels";
import SeasonEligibilityApprovals from "./SeasonEligibilityApprovals";
import InterclubWorkflow, { RegistrationStep, workflowHref } from "../InterclubWorkflow";
import SeasonRegistrationWindow from "./SeasonRegistrationWindow";
import { useRegistrationWindow } from "@/lib/useRegistrationWindow";
import SeasonMeetSchedule from "./SeasonMeetSchedule";

function loadErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && !(error instanceof TypeError) && !(error instanceof SyntaxError) ? error.message : fallback;
}

export default function RegistrationWorkspace({ initialSeasonId, initialMeetId = "", initialStep = "pool" }: { initialSeasonId: string; initialMeetId?: string; initialStep?: RegistrationStep }) {
  const { session, accessToken, loading } = useAdminSession();
  const { clubId } = useAdminWorkspace();
  const canManage = session?.capabilities?.assignments.some(a => a.club_id === clubId && ["administrator", "club_owner", "super_admin"].includes(a.role));
  if (loading) return <p>Checking club access…</p>;
  if (!canManage) return <p>Sign in as a club administrator to manage participation and rosters. <Link href="/admin/login">Sign in</Link></p>;
  return <ClubRegistrations key={`${clubId}:${session?.user?.id || session?.user?.email}`} clubId={clubId} accessToken={accessToken} initialSeasonId={initialSeasonId} initialMeetId={initialMeetId} initialStep={initialStep} />;
}

function ClubRegistrations({ clubId, accessToken, initialSeasonId, initialMeetId, initialStep }: { clubId: string; accessToken: string; initialSeasonId: string; initialMeetId: string; initialStep: RegistrationStep }) {
  const api = getAdminApiBaseUrl();
  const [seasons, setSeasons] = useState<RegistrationSeason[]>([]);
  const [selected, setSelected] = useState(initialSeasonId);
  const [error, setError] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [loading, setLoading] = useState(true);
  const [reload, setReload] = useState(0);
  const token = useRef(accessToken); token.current = accessToken;
  useEffect(() => {
    const controller = new AbortController(); setLoaded(false); setLoading(true); setError("");
    if (!api) { setError("Interclub registration is unavailable."); setLoading(false); return; }
    fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub/registrations`, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal })
      .then(async response => {
        const data = await response.json(); if (!response.ok) throw new Error(apiError(data, "Unable to load club invitations."));
        if (!controller.signal.aborted) { setSeasons(data.seasons); setSelected(old => data.seasons.some((s: RegistrationSeason) => s.id === old) ? old : data.seasons[0]?.id || ""); setLoaded(true); }
      }).catch(e => { if (!controller.signal.aborted) setError(loadErrorMessage(e, "Unable to load club invitations. Try again.")); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [api, clubId, reload]);
  return <section className={styles.page}>
    <p className={styles.back}><Link href="/admin/interclub">← Interclub leagues</Link></p>
    <h1>League workspace</h1>
    <p>Register players for the season first. Meet planning opens after the commissioner’s registration period closes.</p>
    <div className={styles.toolbar}>
      <label>Season <select value={selected} onChange={e => setSelected(e.target.value)} disabled={!loaded}>
        {!seasons.length && <option value="">{loading ? "Loading invitations…" : loaded ? "No open invitations" : "Choose a season"}</option>}
        {seasons.map(s => <option key={s.id} value={s.id}>{s.details.name} · {s.details.start_date}{s.organizer_club_id === clubId ? " · Organizer" : ""}</option>)}
      </select></label>
      <button disabled={loading} onClick={() => setReload(n => n + 1)}>{error ? "Retry loading invitations" : "Refresh invitations"}</button>
    </div>
    {error && <p role="alert">{error}</p>}
    {loading && <p role="status">Loading club invitations…</p>}
    {loaded && !seasons.length && <p>Your club has no season invitations yet. Organizers open invitations from a saved season plan.</p>}
    {loaded && selected && api && <SeasonRegistration key={`${selected}:${reload}`} api={api} clubId={clubId} accessToken={accessToken} seasonId={selected} initialMeetId={selected === initialSeasonId ? initialMeetId : ""} initialStep={initialStep} />}
  </section>;
}

function SeasonRegistration({ api, clubId, accessToken, seasonId, initialMeetId, initialStep }: { api: string; clubId: string; accessToken: string; seasonId: string; initialMeetId: string; initialStep: RegistrationStep }) {
  const root = `${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub/registrations/${seasonId}`;
  const token = useRef(accessToken); token.current = accessToken;
  const [data, setData] = useState<RegistrationDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [blocked, setBlocked] = useState(false);
  const [reload, setReload] = useState(0);
  const [selectedMeet, setSelectedMeet] = useState(initialMeetId);
  const [step, setStep] = useState<RegistrationStep>(initialStep);
  const [poolRefreshKey, setPoolRefreshKey] = useState(0), [approvalRefreshKey, setApprovalRefreshKey] = useState(0);
  const registrationCheck = useRef<AbortController | null>(null);
  const { meetPlanningOpen } = useRegistrationWindow(data?.season.registration, () => void recheckRegistration());
  const stepFocus = useRef(false);
  const poolSection = useRef<HTMLElement | null>(null);
  const invitationHeading = useRef<HTMLHeadingElement | null>(null);
  const followedInvitation = useRef(false);
  const [responding, setResponding] = useState<"accept" | "decline" | null>(null);
  const responseFocus = useRef(false), confirmation = useRef<HTMLHeadingElement | null>(null), rosters = useRef<HTMLElement | null>(null);
  const pending = useRef(false), mutation = useRef<AbortController | null>(null);
  useEffect(() => {
    if (data?.own_participation?.status !== "invited" || followedInvitation.current || typeof window === "undefined" ||
        window.location?.hash !== "#invitation-title" || !invitationHeading.current) return;
    const linkedSeason = new URLSearchParams(window.location.search).get("season");
    if (linkedSeason && linkedSeason !== seasonId) return;
    followedInvitation.current = true;
    invitationHeading.current.scrollIntoView({ block: "start" });
    invitationHeading.current.focus({ preventScroll: true });
  }, [data?.own_participation?.status, seasonId]);
  useEffect(() => { if (data && !meetPlanningOpen) setStep("pool"); }, [data, meetPlanningOpen]);
  useEffect(() => {
    if (responseFocus.current && data?.own_participation?.status !== "invited") {
      confirmation.current?.focus({ preventScroll: true });
      responseFocus.current = false;
    }
  }, [data?.own_participation?.status]);
  useEffect(() => () => { mutation.current?.abort(); }, [root]);
  useEffect(() => () => registrationCheck.current?.abort(), [root]);
  useEffect(() => {
    if (!stepFocus.current) return;
    const section = step === "pool" ? poolSection.current : rosters.current;
    section?.focus({ preventScroll: true }); section?.scrollIntoView({ block: "start" });
    stepFocus.current = false;
  }, [step]);
  useEffect(() => {
    registrationCheck.current?.abort();
    const controller = new AbortController(); setData(null); setLoading(true); setLoadError(""); setBlocked(false);
    fetch(root, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal })
      .then(async response => {
        const next = await response.json(); if (!response.ok) throw new Error(apiError(next, "Unable to load this season."));
        if (!controller.signal.aborted) { setData(next); setSelectedMeet(old => !next.meets.length || next.meets.some((m: { id: string }) => m.id === old) ? old : next.meets.find((m: { roster_open: boolean; club_ids: string[] }) => m.roster_open && m.club_ids.includes(clubId))?.id || next.meets.find((m: { roster_open: boolean }) => m.roster_open)?.id || next.meets[0]?.id || ""); }
      }).catch(e => { if (!controller.signal.aborted) setLoadError(loadErrorMessage(e, "Unable to load this season. Try again.")); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [root, reload, clubId]);

  async function recheckRegistration() {
    if (!data || pending.current) return;
    registrationCheck.current?.abort();
    const controller = new AbortController(); registrationCheck.current = controller;
    try {
      const response = await fetch(root, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal });
      const next: RegistrationDetail = await response.json();
      if (!response.ok || next.season?.id !== seasonId) throw new Error("Unable to refresh season registration. Select Retry loading season to check again.");
      if (!controller.signal.aborted) {
        setData(current => current && (current.season.registration?.revision || 0) <= (next.season.registration?.revision || 0) ? next : current);
        setSelectedMeet(old => !next.meets.length || next.meets.some(meet => meet.id === old) ? old : next.meets.find(meet => meet.roster_open && meet.club_ids.includes(clubId))?.id || next.meets[0]?.id || "");
        setLoadError("");
      }
    } catch { if (!controller.signal.aborted) setLoadError("Unable to refresh season registration. Select Retry loading season to check again."); }
  }

  async function change(path: string, method: string, body: object, success: string, participationAction?: "accept" | "decline"): Promise<boolean> {
    if (pending.current || blocked) return false;
    registrationCheck.current?.abort();
    pending.current = true; setBusy(true); setMessage(""); setResponding(participationAction || null);
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
      if (participationAction) {
        if (result.participation?.club_id !== clubId || result.participation?.season_id !== seasonId) {
          setBlocked(true); throw new Error("Could not confirm your club’s response. Reload the season to check its status.");
        }
        responseFocus.current = true;
        setData(old => old ? { ...old, own_participation: result.participation,
          participations: old.participations.map(p => p.club_id === clubId ? result.participation : p) } : old);
      } else { setMessage(success); setReload(n => n + 1); }
      return true;
    } catch (e) { if (!controller.signal.aborted) { setMessage(e instanceof Error ? e.message : "Unable to save this change."); if (e instanceof TypeError) setBlocked(true); } return false; }
    finally { pending.current = false; if (!controller.signal.aborted) { setBusy(false); setResponding(null); } }
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
  const when = (value: string) => new Date(value).toLocaleString(undefined, { timeZone: data?.season.details.timezone, dateStyle: "medium", timeStyle: "short" });
  const date = (value: string) => new Date(`${value}T12:00:00Z`).toLocaleDateString(undefined, { timeZone: "UTC", dateStyle: "medium" });
  const status = data?.own_participation?.status;
  const hasWorkspace = data?.is_organizer || status === "accepted";
  const showRosters = hasWorkspace && meetPlanningOpen;
  const activeStep = !meetPlanningOpen ? "pool" : status !== "accepted" && step !== "lineups" ? "lineups" : step;
  const nextMeet = data?.meets.find(meet => meet.roster_open && meet.club_ids.includes(clubId));
  const schedule = data?.meet_schedule || data?.meets || [];
  function selectStep(value: RegistrationStep) { if (value !== "pool" && !meetPlanningOpen) return; stepFocus.current = true; setStep(value); }
  return <>
    {loadError && <p role="alert" className={styles.notice}>{loadError}</p>}
    {message && <p role="status" className={styles.notice}>{message}</p>}
    {(!data || blocked || loadError) && <div className={styles.toolbar}><button disabled={busy || loading} onClick={() => { setMessage(""); setReload(n => n + 1); }}>{loadError ? "Retry loading season" : "Reload season"}</button>{blocked && <span>Reload to check the latest response before trying again.</span>}</div>}
    {loading && <p role="status">Loading season…</p>}
    {data && <>
      <header className={styles.seasonHeader}>
        <h2>{data.season.details.name}</h2>
        <p>{date(data.season.details.start_date)} – {date(data.season.details.end_date)} · Organized by {clubName(data.season.organizer_club_id)}</p>
        {data.is_organizer && <p><a href="#meet-schedule">Meet schedule</a></p>}
      </header>
      {status === "invited" && <section className={`${styles.card} ${styles.invitation}`} aria-labelledby="invitation-title">
        <p className={styles.eyebrow}>Invitation to your club</p>
        <h3 id="invitation-title" ref={invitationHeading} tabIndex={-1} style={{ scrollMarginTop: "1rem" }}>{clubName(clubId)} is invited</h3>
        <p>Accept to join {data.season.details.name}. You’ll choose available players separately for each meet.</p>
        <div className={styles.toolbar}>
          <button className={styles.primary} disabled={disabled} aria-busy={responding === "accept" || undefined} onClick={() => void change(`/participations/${encodeURIComponent(clubId)}`, "POST", { action: "accept", expected_revision: data.own_participation!.revision }, "", "accept")}>{responding === "accept" ? "Accepting…" : "Accept invitation"}</button>
          <button disabled={disabled} aria-busy={responding === "decline" || undefined} onClick={() => void change(`/participations/${encodeURIComponent(clubId)}`, "POST", { action: "decline", expected_revision: data.own_participation!.revision }, "", "decline")}>{responding === "decline" ? "Declining…" : "Decline invitation"}</button>
        </div>
        <p className={styles.muted}>No player roster is required to accept.</p>
      </section>}
      <SeasonRegistrationWindow root={root} accessToken={accessToken} season={data.season} commissioner={data.is_organizer} firstMeetAt={data.first_meet_at}
        onSaved={season => { registrationCheck.current?.abort(); setData(current => current ? { ...current, season } : current); void recheckRegistration(); }} onReload={() => setReload(value => value + 1)} />
      {hasWorkspace && <InterclubWorkflow seasonId={seasonId} meetId={selectedMeet} current={activeStep} meetPlanningOpen={meetPlanningOpen}
        unavailable={status === "accepted" ? [] : ["pool", "availability"]}
        onSelect={value => selectStep(value as RegistrationStep)} />}
      {hasWorkspace && !meetPlanningOpen && <p className={styles.notice} role="status">Meet planning opens after registration closes. Stay with the season player pool for now; availability, lineups, and meet settings are locked for every club.</p>}
      {status === "accepted" && <section hidden={activeStep !== "pool"} className={`${styles.card} ${styles.success}`} aria-labelledby="participation-confirmed">
        <p className={styles.eyebrow}>Invitation accepted</p>
        <h3 id="participation-confirmed" ref={confirmation} tabIndex={-1}>{clubName(clubId)} has joined</h3>
        <p>Your place in {data.season.details.name} is confirmed. {meetPlanningOpen ? "Season registration is closed. Check who is available for each meet, then choose that meet’s lineup." : "Use the season player pool to register your club’s players during the commissioner’s registration period."}</p>
        {meetPlanningOpen && <div className={styles.toolbar}>
          {data.meets.length > 0 && <button onClick={() => { if (nextMeet) setSelectedMeet(nextMeet.id); selectStep("availability"); }}>Next: check meet availability</button>}
        </div>}
        {meetPlanningOpen && nextMeet && <p><strong>Next meet:</strong> {when(nextMeet.starts_at)} · {clubName(nextMeet.host_club_id)}</p>}
        {meetPlanningOpen && !data.meets.length && <p>The organizer will share your meet schedule here.</p>}
      </section>}
      {status === "accepted" && <section id="season-player-pool" hidden={activeStep !== "pool"} ref={poolSection} tabIndex={-1} className={styles.rosters} aria-label="Season player pool workspace">
        <SeasonPlayerPool root={root} accessToken={accessToken} clubName={clubName(clubId)} season={data.season} refreshKey={poolRefreshKey} onLateRequested={() => setApprovalRefreshKey(value => value + 1)} />
      </section>}
      {status && !["invited", "accepted"].includes(status) && <section className={styles.card}>
        <h3 ref={confirmation} tabIndex={-1}>{status === "declined" ? "Invitation declined" : "Invitation cancelled"}</h3>
        <p>{clubName(clubId)} has not joined {data.season.details.name}. Ask {clubName(data.season.organizer_club_id)} for a new invitation if you want to take part.</p>
      </section>}
      {data.is_organizer && <SeasonMeetSchedule root={`${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub/competition/${encodeURIComponent(seasonId)}`} clubId={clubId} accessToken={accessToken}
        seasonData={data} meetPlanningOpen={meetPlanningOpen} disabled={disabled} onSaved={() => void recheckRegistration()} />}
      {!data.is_organizer && <details className={styles.card}>
        <summary>Planned meet dates ({schedule.length})</summary>
        <p>Meet times are shown in {data.season.details.timezone}.</p>
        {schedule.length ? <ul className={styles.schedule}>{schedule.map(meet => <li key={meet.id}><strong>{when(meet.starts_at)}</strong><span>Hosted by {clubName(meet.host_club_id)}</span></li>)}</ul> : <p>No meets are scheduled for your club yet.</p>}
      </details>}
      {showRosters && <section id="meet-rosters" hidden={activeStep === "pool"} ref={rosters} tabIndex={-1} className={styles.rosters} aria-label="Meet rosters">
        <h3>{activeStep === "availability" ? "Meet availability" : "Lineups"}</h3>
        <p>{activeStep === "availability" ? "Ask players from your season pool who can attend this meet. Joining the pool does not confirm availability." : "Choose a meet, then add a team with four available players. Your lineup can change from one meet to the next."}</p>
        {!data.meets.length ? <p>No meets are scheduled for your club in this season.</p> : <>
          <label>Meet <select aria-label="Meet" value={selectedMeet} onChange={e => setSelectedMeet(e.target.value)}>
            {data.meets.map(meet => <option key={meet.id} value={meet.id}>{when(meet.starts_at)} · {clubName(meet.host_club_id)}{meet.roster_open ? "" : " · History"}</option>)}
          </select></label>
          {selectedMeet && <MeetRegistration key={selectedMeet} root={`${root}/meets/${selectedMeet}`} accessToken={accessToken} clubId={clubId} seasonData={data} step={activeStep} onStep={selectStep} playerRefreshKey={poolRefreshKey} />}
        </>}
        {data.teams.length > 0 && <details className={styles.card}><summary>Earlier season submissions (reference only)</summary>
          <p>These were submitted before rosters moved to individual meets. Choose players again for each upcoming meet.</p>
          {data.teams.map(team => <TeamCard key={`${team.id}:${team.revision}`} team={team} clubName={clubName(team.club_id)} root={root} accessToken={accessToken} own={false} organizer={false} disabled={true}
            edit={() => {}} withdraw={async () => false} decide={async () => false} />)}
          {data.next_team_offset !== null && <button disabled={disabled} onClick={() => void moreTeams()}>Load earlier submissions</button>}
        </details>}
      </section>}
      <details className={styles.card}><summary>Divisions and eligibility rules</summary>
        <div className={styles.scroll}><table className={styles.table}><caption>Season eligibility rules</caption><thead><tr><th>Division</th><th>League rating</th><th>Team</th></tr></thead><tbody>
          {[...data.season.details.divisions].sort((a, b) => Number.parseFloat(a) - Number.parseFloat(b) || a.localeCompare(b, undefined, { numeric: true })).map(division => {
            const rule = emptyRule(division);
            return <tr key={division}><td>{division}</td><td>{divisionEligibilityLabel(division)}</td><td>{composition(rule)}</td></tr>;
          })}
        </tbody></table></div>
        <p>Interclub ratings start from the represented club’s rating, then change with approved league results. Each meet locks the player’s league rating at its roster deadline. Players may play up: a 2.9 player can enter 3.0 or a higher division. Each numeric division requires a rating below its listed limit; Open divisions accept any positive rating. Every team has two women and two men.</p>
      </details>
      {data.is_organizer && <SeasonEligibilityApprovals root={root} accessToken={accessToken} clubs={data.clubs} refreshKey={approvalRefreshKey} onDecision={() => setPoolRefreshKey(value => value + 1)} />}
      {data.is_organizer && <section id="club-responses" className={styles.card}><h3>Club responses</h3>
        <p>{data.participations.filter(p => p.status === "accepted").length} of {data.participations.length} clubs have accepted. Each club chooses its players separately for each meet.</p>
        <div className={styles.scroll}><table className={styles.table}><thead><tr><th>Club</th><th>Response</th><th>Invitation</th></tr></thead><tbody>
        {data.participations.map(p => <tr key={p.club_id}><td>{clubName(p.club_id)}</td><td>{p.status}</td><td>
          {p.status === "invited" && <button disabled={disabled} onClick={() => void change(`/participations/${encodeURIComponent(p.club_id)}`, "POST", { action: "cancel", expected_revision: p.revision }, "Club invitation cancelled.")}>Cancel invitation</button>}
          {["declined", "cancelled"].includes(p.status) && <button disabled={disabled} onClick={() => void change(`/participations/${encodeURIComponent(p.club_id)}`, "POST", { action: "reinvite", expected_revision: p.revision }, "Club invited again.")}>Invite again</button>}
        </td></tr>)}
      </tbody></table></div></section>}
      {!blocked && <div className={styles.toolbar}><button disabled={busy || loading} onClick={() => { setMessage(""); setReload(n => n + 1); }}>Reload season</button></div>}
    </>}
  </>;
}

function MeetRegistration({ root, accessToken, clubId, seasonData, step, onStep, playerRefreshKey }: {
  root: string; accessToken: string; clubId: string; seasonData: RegistrationDetail;
  step: RegistrationStep; onStep: (step: RegistrationStep) => void; playerRefreshKey: number;
}) {
  const token = useRef(accessToken); token.current = accessToken;
  const [data, setData] = useState<MeetRegistrationDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [blocked, setBlocked] = useState(false);
  const [reload, setReload] = useState(0);
  const [editor, setEditor] = useState<{ team: InterclubTeam | null } | null>(null);
  const [availabilityVisited, setAvailabilityVisited] = useState(step === "availability");
  const [responses, setResponses] = useState<{ member_id: string; player_id: string | null; name: string; status: string; member_status?: string }[] | null>(null);
  const pending = useRef(false), mutation = useRef<AbortController | null>(null);
  const currentMeetRevision = seasonData.meets.find(meet => meet.id === data?.meet.id)?.revision;
  const loadedMeetRevision = data?.meet.revision;
  const editingRoster = Boolean(editor);
  useEffect(() => {
    if (loadedMeetRevision == null || currentMeetRevision == null || currentMeetRevision <= loadedMeetRevision) return;
    if (editingRoster) {
      setBlocked(true); setMessage("This meet’s schedule changed. Your lineup draft is kept below. Reload the meet before saving it.");
    } else setReload(value => value + 1);
  }, [currentMeetRevision, loadedMeetRevision, editingRoster]);
  useEffect(() => { if (step === "availability") setAvailabilityVisited(true); }, [step]);
  useEffect(() => () => { mutation.current?.abort(); }, [root]);
  useEffect(() => {
    const controller = new AbortController(); setData(null); setLoading(true); setLoadError(""); setEditor(null); setBlocked(false);
    fetch(root, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal })
      .then(async response => {
        const next = await response.json(); if (!response.ok) throw new Error(apiError(next, "Unable to load this meet."));
        if (!controller.signal.aborted) setData(next);
      }).catch(e => { if (!controller.signal.aborted) setLoadError(loadErrorMessage(e, "Unable to load this meet. Try again.")); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
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
    <div className={styles.toolbar}><button disabled={busy || loading} onClick={() => { setMessage(""); setDeadline(""); setReload(n => n + 1); }}>{loadError ? "Retry loading meet" : "Reload meet"}</button>{blocked && <span>Reload before making another change. Your draft remains below for reference.</span>}</div>
    {loadError && <p role="alert" className={styles.notice}>{loadError}</p>}
    {message && <p role="status" className={styles.notice}>{message}</p>}
    {loading && <p role="status">Loading meet…</p>}
    {data && <>
      <h3>{when(data.meet.starts_at)} · {clubName(data.meet.host_club_id)}</h3>
      <p>{data.meet.club_ids.map(clubName).join(", ")} · {data.meet.courts} courts</p>
      <p>Roster deadline for this meet: {when(data.meet.roster_deadline)} ({seasonData.season.details.timezone}).</p>
      {data.meet.roster_open ? <p>Valid substitutions are allowed after the deadline, until this meet starts. A new team submitted after the deadline needs an organizer exception.</p>
        : <p>This meet has started. Its rosters and decisions are available as history.</p>}
      {seasonData.is_organizer && data.meet.deadline_editable && <details className={styles.card}><summary>Change this meet’s roster deadline</summary><form onSubmit={e => { e.preventDefault(); try { const cutoff = meetUtcTime(deadline, seasonData.season.details.timezone); if (!cutoff) throw new Error("Choose a valid roster deadline."); void change("/deadline", "PUT", { expected_revision: data.meet.revision, roster_deadline: cutoff }, "Meet roster deadline saved."); } catch (cause) { setMessage(cause instanceof Error ? cause.message : "Choose a valid roster deadline."); } }}>
        <label>Roster deadline ({seasonData.season.details.timezone}) <input required disabled={disabled} type="datetime-local" max={meetLocalTime(data.meet.starts_at, seasonData.season.details.timezone)} value={deadline} onChange={e => setDeadline(e.target.value)} /></label>
        <p>The default is the meet’s start time. Set an earlier deadline before the first team submits.</p>
        <button disabled={disabled || !deadline} type="submit">Save meet deadline</button>
      </form></details>}
      <div hidden={step !== "availability"}>
        {ownMeet ? <>
          <p className={styles.notice}>Players who confirmed verbally or by email can go straight into your lineup, even without an email address. Sending availability invitations is optional. <button onClick={() => onStep("lineups")}>Choose lineups now</button></p>
          {availabilityVisited && <MeetAvailability meetRoot={root} accessToken={accessToken} clubName={clubName(clubId)} season={seasonData.season} meet={data.meet} onResponses={setResponses} />}
          <div className={styles.toolbar}><button className={styles.primary} onClick={() => onStep("lineups")}>Next: choose lineups</button></div>
        </> : <p>Your club is not participating in this meet. Each participating club manages its own player availability. <button onClick={() => onStep("lineups")}>View meet lineups</button></p>}
      </div>
      <div hidden={step !== "lineups"}>
      {ownMeet && data.meet.roster_open && <button disabled={disabled} onClick={() => setEditor({ team: null })}>Add a team for this meet</button>}
      {editor && <RosterEditor key={editor.team ? `${editor.team.id}:${editor.team.revision}` : "new"} root={root} accessToken={accessToken} clubName={clubName(clubId)} divisions={[...seasonData.season.details.divisions].sort((a, b) => Number.parseFloat(a) - Number.parseFloat(b) || a.localeCompare(b, undefined, { numeric: true }))} team={editor.team} disabled={disabled}
        responses={responses} allowMissingPairing={(data.meet.competition_phase || "regular") === "regular"} refreshKey={playerRefreshKey}
        cancel={() => setEditor(null)} save={(id, body) => change(`/teams/${id}`, "PUT", { ...body, ...revision }, "Roster submitted for this meet.")} />}
      <h4>{seasonData.is_organizer ? "Teams for this meet" : "Your club’s teams for this meet"}</h4>
      {!data.teams.length && <p>No teams submitted for this meet yet. {ownMeet ? "Add a team above using the players who can attend." : "Each participating club needs to submit its lineup before the host can prepare pairings."}</p>}
      {data.teams.map(team => <TeamCard key={`${team.id}:${team.revision}:${team.status}`} team={team} clubName={clubName(team.club_id)} root={root} accessToken={accessToken} own={ownMeet && team.club_id === clubId && data.meet.roster_open} organizer={seasonData.is_organizer && data.meet.roster_open} disabled={disabled}
        edit={() => setEditor({ team })}
        withdraw={() => change(`/teams/${team.id}/withdraw`, "POST", { expected_revision: team.revision, ...revision }, "Team withdrawn from this meet. Earlier rosters remain in its history.")}
        decide={(approve, reason) => change(`/teams/${team.id}/eligibility`, "POST", { expected_revision: team.revision, ...revision, approve, reason }, approve ? "Exception approved for this meet roster." : "Exception declined for this meet roster.")} />)}
      {data.next_team_offset !== null && <button disabled={busy} onClick={() => void moreTeams()}>Load more teams</button>}
      <div className={styles.toolbar}>
        {ownMeet && <button onClick={() => onStep("availability")}>Review player availability</button>}
        <Link className={styles.button} href={workflowHref("run", seasonData.season.id, data.meet.id)}>Next: prepare meet pairings →</Link>
      </div>
      <p className={styles.muted}>The host or organizer can prepare pairings once the participating clubs have approved lineups.</p>
      </div>
    </>}
  </section>;
}

type PlayerChoice = { id: string; name: string; starting_rating: number | null; eligibility_rating?: number; rating_locked?: boolean; rating_deadline?: string };
function RosterEditor({ root, accessToken, clubName, divisions, team, disabled, cancel, save, responses, allowMissingPairing, refreshKey }: {
  root: string; accessToken: string; clubName: string; divisions: string[]; team: InterclubTeam | null; disabled: boolean; cancel: () => void;
  save: (id: string, body: object) => Promise<boolean>;
  responses: { player_id: string | null; status: string; member_status?: string }[] | null;
  allowMissingPairing: boolean; refreshKey: number;
}) {
  const [id] = useState(() => team?.id || crypto.randomUUID());
  const [name, setName] = useState(team?.name || "");
  const [division, setDivision] = useState(team?.division || divisions[0] || "");
  const [selected, setSelected] = useState<PlayerChoice[]>(() => (team?.roster || []).map(p => ({ ...p, id: p.player_id! })));
  const [missingPairing, setMissingPairing] = useState(allowMissingPairing && team?.roster.length === 2);
  const requiredPlayers = missingPairing ? 2 : 4;
  const [query, setQuery] = useState("");
  const [offset, setOffset] = useState(0);
  const [choices, setChoices] = useState<PlayerChoice[]>([]);
  const [next, setNext] = useState<number | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [availableOnly, setAvailableOnly] = useState(false);
  const responseByPlayer = new Map((responses || []).filter(r => r.player_id && r.member_status !== "withdrawn").map(r => [String(r.player_id), r.status]));
  const visibleChoices = availableOnly ? choices.filter(player => responseByPlayer.get(player.id) === "available" || selected.some(p => p.id === player.id)) : choices;
  const token = useRef(accessToken); token.current = accessToken;
  useEffect(() => { setOffset(0); }, [refreshKey]);
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setError("");
    fetch(`${root}/players?q=${encodeURIComponent(query)}&offset=${offset}`, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal })
      .then(async response => {
        const data = await response.json(); if (!response.ok) throw new Error(apiError(data, "Unable to load your club players."));
        if (!controller.signal.aborted) { setChoices(old => offset ? Array.from(new Map([...old, ...data.players].map(p => [p.id, p])).values()) : data.players); setNext(data.next_offset); }
      }).catch(e => { if (!controller.signal.aborted) setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [root, query, offset, refreshKey]);
  function toggle(player: PlayerChoice) {
    setSelected(old => old.some(p => p.id === player.id) ? old.filter(p => p.id !== player.id) : old.length < requiredPlayers ? [...old, player] : old);
  }
  return <form className={styles.card} onSubmit={e => { e.preventDefault(); if (selected.length === requiredPlayers) void save(id, { expected_revision: team?.revision || 0, name, division, player_ids: selected.map(p => p.id), ...(missingPairing ? { missing_pairing_forfeit: true } : {}) }); }}>
    <h3>{team ? "Update team roster" : "New meet roster"}</h3>
    <fieldset disabled={disabled} className={styles.form}><legend>{clubName}</legend>
      <label>Team name<input required maxLength={80} value={name} onChange={e => setName(e.target.value)} /></label>
      <label>Division<select disabled={Boolean(team) || disabled} value={division} onChange={e => setDivision(e.target.value)}>{divisions.map(d => <option key={d} value={d}>{d}</option>)}</select></label>
      <p>{selected.length} of {requiredPlayers} players selected. {missingPairing ? "Choose the two players who will compete in the remaining doubles pairing." : "Choose two women and two men from your approved season pool."} A player may represent one skill level at this meet.</p>
      <p>Eligibility uses the player’s league rating, locked at this meet’s roster deadline.</p>
      {selected.length > 0 && <ul>{selected.map(player => <li key={player.id}>{player.name} · {(player.eligibility_rating ?? player.starting_rating)?.toFixed(3) ?? "No rating"} <button type="button" onClick={() => toggle(player)}>Remove {player.name}</button></li>)}</ul>}
      <label>Find a player in {clubName}<input type="search" maxLength={80} value={query} onChange={e => { setQuery(e.target.value); setOffset(0); }} /></label>
      {responses !== null && <label><input type="checkbox" checked={availableOnly} onChange={e => setAvailableOnly(e.target.checked)} />Show only players who said they are available</label>}
      {responses === null && <p>Choose “Meet availability” above to see responses alongside player names. Your lineup draft will stay here.</p>}
      {error && <p role="alert">{error}</p>}
      <div className={styles.choices}>{visibleChoices.map(player => <label key={player.id}>
        <input type="checkbox" checked={selected.some(p => p.id === player.id)} disabled={disabled || (selected.length >= requiredPlayers && !selected.some(p => p.id === player.id))} onChange={() => toggle(player)} />
        {player.name} · {(player.eligibility_rating ?? player.starting_rating)?.toFixed(3) ?? "No rating"}{player.rating_locked ? " · Locked for this meet" : ""}
        {responseByPlayer.has(player.id) && <span> · {({available: "Available", maybe: "Unsure", unavailable: "Unavailable", invited: "Not replied", pending: "Not replied"} as Record<string, string>)[responseByPlayer.get(player.id)!] || "Not replied"}</span>}
      </label>)}{loading && <p>Loading players…</p>}{!loading && !error && !visibleChoices.length && <p>{availableOnly ? "No available players in these results. Load more players or clear the filter." : "No active players found."}</p>}</div>
      {allowMissingPairing && <label><input type="checkbox" aria-label="Missing pairing forfeit" checked={missingPairing} onChange={e => setMissingPairing(e.target.checked)} />We can field only one pairing; the missing pairing will forfeit all three games.</label>}
      {missingPairing && <p role="status">Enter only the two players who will compete. They must form a women’s or men’s pairing at a gender-doubles meet, or one woman and one man at a mixed-doubles meet. The selected meet format is checked when its schedule is prepared. Championship teams still need four players.</p>}
      {next !== null && <button type="button" disabled={loading || disabled} onClick={() => setOffset(next)}>More players</button>}
      <div className={styles.toolbar}><button type="submit" disabled={disabled || selected.length !== requiredPlayers}>{missingPairing ? "Submit two-player roster with forfeit" : "Submit four-player roster"}</button><button type="button" onClick={cancel}>Cancel roster edit</button></div>
    </fieldset>
  </form>;
}

function RosterTable({ version }: { version: RosterVersion }) {
  return <>{version.roster.length === 2 && <p className={styles.notice}>Two-player roster: the missing pairing forfeits its three games.</p>}<div className={styles.scroll}><table className={styles.table}><thead><tr><th>Player</th><th>Eligibility rating</th><th>Gender</th></tr></thead><tbody>
    {version.roster.map((player, index) => <tr key={player.entry_id || index}><td>{player.name}</td><td>{(player.eligibility_rating ?? player.starting_rating).toFixed(3)}</td><td>{player.gender === "female" ? "Female" : player.gender === "male" ? "Male" : "Not set"}</td></tr>)}
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
