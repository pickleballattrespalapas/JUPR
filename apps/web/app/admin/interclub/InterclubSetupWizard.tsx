"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import type { DivisionRule, RegistrationSeason } from "@/lib/interclubRegistration";
import { apiError, composition } from "@/lib/interclubRegistration";
import { ClubChoice, PlanningDraft, PlanningMeet, PlanningSeason, divisionChoices, emptyRule, firstIncompleteStep, meetLocalTime, meetUtcTime, normalizeDraft, setupSteps, stepIssues } from "@/lib/interclubSetup";
import styles from "./setup.module.css";
import ClubInvitationPanel, { ClubJoinInvitation, InviteClubInput } from "./ClubInvitationPanel";

class SetupRequestError extends Error {
  constructor(message: string, readonly needsReload: boolean) { super(message); }
}
export function SetupNextSteps() {
  return <ol className={styles.nextSteps}>
    <li><strong>1. Clubs respond</strong><span>Each club’s administrator accepts the season invitation in PCS.</span></li>
    <li><strong>2. Choose a meet</strong><span>Set its roster deadline and see which clubs are attending.</span></li>
    <li><strong>3. Clubs choose players</strong><span>Each club submits four available players per team for that meet.</span></li>
  </ol>;
}

export default function InterclubSetupWizard({ api, club, accessToken, initialSeason, choices, onSaved, onOpened, onClose }: {
  api: string; club: ClubChoice; accessToken: string; initialSeason: PlanningSeason; choices: ClubChoice[];
  onSaved: (season: PlanningSeason) => void; onOpened: (season: RegistrationSeason) => void; onClose: () => void;
}) {
  const [season, setSeason] = useState<PlanningSeason>(() => ({ ...initialSeason, draft: normalizeDraft(initialSeason.draft) }));
  const [step, setStep] = useState(() => Math.min(initialSeason.draft.setup_step || 0, firstIncompleteStep(normalizeDraft(initialSeason.draft))));
  const [furthestStep, setFurthestStep] = useState(initialSeason.draft.setup_step || 0);
  const [savedDraft, setSavedDraft] = useState(() => JSON.stringify(normalizeDraft(initialSeason.draft)));
  const [busy, setBusy] = useState(false), [checking, setChecking] = useState(initialSeason.revision > 0);
  const [blocked, setBlocked] = useState(false), [reviewed, setReviewed] = useState(false);
  const [errors, setErrors] = useState<string[]>([]), [message, setMessage] = useState("");
  const [search, setSearch] = useState("");
  const [clubOptions, setClubOptions] = useState(choices);
  const [clubInvitations, setClubInvitations] = useState<ClubJoinInvitation[]>([]);
  const [invitationsLoading, setInvitationsLoading] = useState(false), [invitationError, setInvitationError] = useState("");
  const [invitationReload, setInvitationReload] = useState(0);
  const [opened, setOpened] = useState<RegistrationSeason | null>(null);
  const token = useRef(accessToken); token.current = accessToken;
  const request = useRef<AbortController | null>(null), pending = useRef(false), heading = useRef<HTMLHeadingElement | null>(null);
  const root = `${api}/admin/clubs/${encodeURIComponent(club.id)}/interclub`;
  const dirty = season.revision === 0 || JSON.stringify(season.draft) !== savedDraft;
  const disabled = busy || blocked || checking;
  const draft = season.draft;
  const clubName = (id: string) => clubOptions.find(c => c.id === id)?.name || id;
  const selectedClubs = clubOptions.filter(c => draft.club_ids.includes(c.id));
  const invitationRoot = `${root}/setup/${season.id}/club-invitations`;
  const when = (value: string | null) => value ? new Date(value).toLocaleString(undefined, { timeZone: draft.timezone, dateStyle: "medium", timeStyle: "short" }) : "Date not set";

  useEffect(() => { setClubOptions(choices); }, [choices]);
  useEffect(() => {
    if ((step !== 1 && !opened) || season.revision === 0 || checking) return;
    const controller = new AbortController();
    setInvitationsLoading(true); setInvitationError("");
    fetch(invitationRoot, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal })
      .then(async response => {
        const data = await response.json();
        if (!response.ok) throw new Error("Could not load club invitations. Refresh the club list to try again.");
        if (!controller.signal.aborted) {
          setClubInvitations(data.invitations);
          setClubOptions(old => [...old.filter(c => !data.clubs.some((club: ClubChoice) => club.id === c.id)), ...data.clubs]);
          if (opened) setBlocked(false);
        }
      }).catch(error => { if (!controller.signal.aborted) setInvitationError(error.message); })
      .finally(() => { if (!controller.signal.aborted) setInvitationsLoading(false); });
    return () => controller.abort();
  }, [invitationRoot, step, checking, opened, invitationReload, season.revision]);
  useEffect(() => {
    const controller = new AbortController();
    if (initialSeason.revision > 0) {
      fetch(`${root}/registrations/${initialSeason.id}`, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal })
        .then(async response => {
          if (response.ok) { const data = await response.json(); if (!controller.signal.aborted) setOpened(data.season); }
          else if (response.status !== 404) throw new Error("Unable to check whether invitations are open. Reload before editing.");
        }).catch(error => { if (!controller.signal.aborted) { setErrors([error.message]); setBlocked(true); } })
        .finally(() => { if (!controller.signal.aborted) setChecking(false); });
    }
    return () => { controller.abort(); request.current?.abort(); };
  }, [root, initialSeason.id, initialSeason.revision]);
  useEffect(() => { heading.current?.focus(); }, [step, checking]);
  useEffect(() => {
    if (!dirty || opened) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty, opened]);

  function edit(patch: Partial<PlanningDraft>) {
    setSeason(old => ({ ...old, draft: { ...old.draft, ...patch } })); setReviewed(false); setErrors([]); setMessage("");
  }
  function editMeet(index: number, patch: Partial<PlanningMeet>) { edit({ meets: draft.meets.map((meet, i) => i === index ? { ...meet, ...patch } : meet) }); }
  function toggleClub(id: string) {
    const club_ids = draft.club_ids.includes(id) ? draft.club_ids.filter(c => c !== id) : [...draft.club_ids, id];
    edit({ club_ids, meets: draft.meets.map(meet => ({ ...meet, club_ids: meet.club_ids.filter(c => club_ids.includes(c)), host_club_id: club_ids.includes(meet.host_club_id) ? meet.host_club_id : "" })) });
  }
  function editRule(division: string, field: keyof DivisionRule, value: string) { edit({ registration_rules: { ...draft.registration_rules, [division]: { ...(draft.registration_rules[division] || emptyRule()), [field]: value === "" ? null : Number(value) } } }); }
  function goTo(target: number) { if (target > step) { const missing = firstIncompleteStep(draft); if (missing < target) { setStep(missing); setErrors(stepIssues(draft, missing)); return; } } setStep(target); setErrors([]); setMessage(""); setReviewed(false); }

  async function jsonRequest(url: string, controller: AbortController, body?: object, method?: string) {
    const response = await fetch(url, { method: method || (body ? (url.endsWith("/open") ? "POST" : "PUT") : "GET"), headers: { Authorization: `Bearer ${token.current}`, ...(body ? { "Content-Type": "application/json" } : {}) },
      ...(body ? { body: JSON.stringify(body) } : {}), cache: "no-store", signal: controller.signal });
    const data = await response.json();
    if (!response.ok) throw new SetupRequestError(apiError(data, "Unable to save setup."), response.status === 409 || response.status >= 500);
    return data;
  }
  async function run(action: (controller: AbortController) => Promise<void>, allowReload = false) {
    if (pending.current || (blocked && !allowReload)) return;
    const controller = new AbortController(); request.current = controller; pending.current = true;
    setBusy(true); setErrors([]); setMessage("");
    try { await action(controller); }
    catch (error) { if (!controller.signal.aborted) { setErrors([error instanceof Error ? error.message : "Could not confirm the save. Reload before retrying."]); setBlocked(error instanceof SetupRequestError ? error.needsReload : true); } }
    finally { pending.current = false; if (!controller.signal.aborted) setBusy(false); }
  }
  async function persist(controller: AbortController, nextStep: number) {
    const nextDraft = normalizeDraft({ ...draft, setup_step: nextStep });
    const result = await jsonRequest(`${root}/setup`, controller, { season_id: season.id, expected_revision: season.revision, draft: nextDraft });
    if (controller.signal.aborted) return null;
    const saved = { ...result.season, draft: normalizeDraft(result.season.draft) } as PlanningSeason;
    setSeason(saved); setSavedDraft(JSON.stringify(saved.draft)); onSaved(saved); return saved;
  }
  function continueSetup() {
    const issues = stepIssues(draft, step);
    if (issues.length) { setErrors(issues); return; }
    void run(async controller => { if (await persist(controller, step + 1)) { setStep(step + 1); setFurthestStep(old => Math.max(old, step + 1)); setReviewed(false); } });
  }
  function finishSetup() {
    if (!reviewed) return;
    const incomplete = firstIncompleteStep(draft);
    if (incomplete < 4) { setStep(incomplete); setErrors(stepIssues(draft, incomplete)); setReviewed(false); return; }
    void run(async controller => {
      const saved = await persist(controller, 4); if (!saved) return;
      const result = await jsonRequest(`${root}/registrations/${season.id}/open`, controller, { expected_revision: saved.revision, rules: saved.draft.registration_rules });
      if (!controller.signal.aborted) { setOpened(result.season); onOpened(result.season); }
    });
  }
  function reloadSaved() {
    void run(async controller => {
      const response = await fetch(`${root}/registrations/${season.id}`, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal });
      if (response.ok) { const data = await response.json(); if (!controller.signal.aborted) setOpened(data.season); return; }
      if (response.status !== 404) throw new Error("Unable to check club invitations. Try reloading again.");
      const data = await jsonRequest(`${root}/setup`, controller);
      if (controller.signal.aborted) return;
      const row = data.seasons.find((s: PlanningSeason) => s.id === season.id);
      if (row) { const next = { ...row, draft: normalizeDraft(row.draft) }; setSeason(next); setSavedDraft(JSON.stringify(next.draft)); setStep(Math.min(next.draft.setup_step, firstIncompleteStep(next.draft))); onSaved(next); }
      setReviewed(false); setBlocked(false); setInvitationReload(n => n + 1); setMessage(row ? "Saved setup reloaded." : "This draft has not been saved yet. You can try saving again.");
    }, true);
  }

  async function inviteClub(input: InviteClubInput): Promise<boolean> {
    let created = false;
    await run(async controller => {
      const data = await jsonRequest(invitationRoot, controller, { ...input, expected_revision: season.revision, draft: normalizeDraft({ ...draft, setup_step: 1 }) }, "POST");
      if (controller.signal.aborted) return;
      const saved = { ...data.season, draft: normalizeDraft(data.season.draft) } as PlanningSeason;
      setSeason(saved); setSavedDraft(JSON.stringify(saved.draft)); onSaved(saved);
      setClubOptions(old => [...old.filter(c => c.id !== data.club.id), data.club]);
      setClubInvitations(old => [...old.filter(i => i.id !== data.invitation.id), data.invitation]);
      setSearch(""); setReviewed(false); setMessage(`${data.club.name} is selected and saved. Share its invitation link below.`); created = true;
    });
    return created;
  }
  function updateClubInvitation(invitation: ClubJoinInvitation, action: "cancel" | "renew", email: string) {
    void run(async controller => {
      const data = await jsonRequest(`${invitationRoot}/${invitation.id}`, controller, { action, email, expected_revision: invitation.revision }, "POST");
      if (!controller.signal.aborted) {
        setClubInvitations(old => old.map(i => i.id === invitation.id ? data.invitation : i));
        setMessage(action === "cancel" ? (opened ? "Account invitation cancelled. Season participation is unchanged." : "Account invitation cancelled. Uncheck the club above if it will not participate.") : "Invitation updated. Share its link with the administrator.");
      }
    });
  }
  function refreshClubs() {
    void run(async controller => {
      try {
        const all: ClubChoice[] = []; let offset: number | null = 0;
        while (offset !== null) {
          const data = await jsonRequest(`${root}/club-choices?offset=${offset}`, controller);
          if (controller.signal.aborted) return;
          all.push(...data.clubs); offset = data.next_offset;
        }
        setClubOptions(all); setInvitationReload(n => n + 1); setMessage("Club list refreshed.");
      } catch { throw new SetupRequestError("Could not refresh the club list. Try again.", false); }
    });
  }

  if (opened) return <section className={styles.panel}>
    <p className={styles.eyebrow}>Season setup complete</p><h2>{opened.details.name}</h2>
    <div className={styles.success}><strong>Club invitations are open.</strong><p>Continue in the season workspace to see responses and prepare each meet.</p></div>
    <SetupNextSteps />
    <button disabled={busy || invitationsLoading} onClick={() => setInvitationReload(n => n + 1)}>Refresh club account invitations</button>
    {invitationError && <p role="alert">{invitationError}</p>}
    {errors.map(error => <p role="alert" key={error}>{error}</p>)}
    {message && <p role="status">{message}</p>}
    <ClubInvitationPanel allowCreate={false} disabled={disabled || invitationsLoading || Boolean(invitationError)} atClubLimit invitations={clubInvitations} onInvite={inviteClub} onUpdate={updateClubInvitation} />
    <div className={styles.toolbar}><Link className={`${styles.button} ${styles.primary}`} href={`/admin/interclub/registrations?season=${opened.id}`}>Manage this season</Link><button onClick={onClose}>Back to interclub leagues</button></div>
  </section>;

  return <section aria-label="Interclub setup wizard">
    <header className={styles.header}><div><p className={styles.eyebrow}>Organized by {club.name}</p><h1>{draft.name || "Set up an interclub season"}</h1><p className={styles.muted}>Five steps to invite clubs and prepare your meets.</p></div>
      <div><button disabled={disabled} onClick={() => void run(async controller => { if (await persist(controller, step)) onClose(); })}>{busy ? "Saving…" : "Save and exit"}</button><p className={styles.saved} role="status">{dirty ? "Unsaved changes" : "Progress saved"}</p></div>
    </header>
    <nav aria-label="Setup steps"><ol className={styles.steps}>{setupSteps.map((label, index) => <li key={label}><button type="button" disabled={disabled || index > Math.max(step, furthestStep)} aria-current={index === step ? "step" : undefined} onClick={() => goTo(index)}><span className={styles.stepNumber}>{index + 1}</span>{label}</button></li>)}</ol></nav>
    <div className={styles.panel}>
      {checking ? <p>Checking saved setup…</p> : <>
        <p className={styles.progress}>Step {step + 1} of 5</p><h2 ref={heading} tabIndex={-1}>{setupSteps[step]}</h2>
        {step === 0 && <>
          <p className={styles.muted}>Name the season and choose its dates. Your current club is the organizer.</p>
          <fieldset disabled={disabled} className={styles.form}><div className={styles.grid}>
            <label className={`${styles.field} ${styles.wide}`}>Season name<input aria-label="Season name" maxLength={120} placeholder="e.g. Southern BCS · Winter 2026–27" value={draft.name} onChange={e => edit({ name: e.target.value })} /></label>
            <label className={styles.field}>Start date<input aria-label="Season start date" type="date" value={draft.start_date || ""} onChange={e => edit({ start_date: e.target.value || null })} /></label>
            <label className={styles.field}>End date<input aria-label="Season end date" type="date" min={draft.start_date || undefined} value={draft.end_date || ""} onChange={e => edit({ end_date: e.target.value || null })} /></label>
            <label className={`${styles.field} ${styles.wide}`}>Meet timezone<select aria-label="Meet timezone" value={draft.timezone} onChange={e => edit({ timezone: e.target.value })}>
              {[...new Set(["America/Mazatlan", "America/Tijuana", "America/Mexico_City", "America/Los_Angeles", "America/Phoenix", "America/Denver", "America/Chicago", "America/New_York", "UTC", draft.timezone])].map(zone => <option key={zone} value={zone}>{zone === "America/Mazatlan" ? "Baja California Sur · America/Mazatlan" : zone.replaceAll("_", " ")}</option>)}
            </select><small>All meet dates and times in this setup use this timezone.</small></label>
          </div></fieldset>
        </>}
        {step === 1 && <>
          <p className={styles.muted}>Select existing clubs or invite a new club below. Include {club.name} if your club will also play. Choose at least two clubs to continue.</p>
          <fieldset disabled={disabled} className={styles.form}>
            <label className={styles.field}>Find an existing club<input aria-label="Find an existing club" type="search" value={search} onChange={e => setSearch(e.target.value)} /></label>
            <div className={styles.choices}>{clubOptions.filter(c => c.name.toLowerCase().includes(search.toLowerCase())).map(c => <label className={styles.choice} data-selected={draft.club_ids.includes(c.id)} key={c.id}><input aria-label={c.name} type="checkbox" disabled={disabled || (!draft.club_ids.includes(c.id) && draft.club_ids.length >= 32)} checked={draft.club_ids.includes(c.id)} onChange={() => toggleClub(c.id)} /><span>{c.name}{c.id === club.id && <small>Your club · organizer</small>}</span></label>)}</div>
            {!clubOptions.some(c => c.name.toLowerCase().includes(search.toLowerCase())) && <p>No clubs match. Invite a new club below.</p>}
          </fieldset>
          <div className={styles.toolbar}><p><strong>{draft.club_ids.length} {draft.club_ids.length === 1 ? "club" : "clubs"} selected</strong></p><button disabled={disabled} onClick={refreshClubs}>Refresh club list</button></div>
          {invitationsLoading && <p role="status">Loading club invitations…</p>}
          {invitationError && <p role="alert">{invitationError}</p>}
          <ClubInvitationPanel disabled={disabled || invitationsLoading || Boolean(invitationError)} atClubLimit={draft.club_ids.length >= 32} invitations={clubInvitations} onInvite={inviteClub} onUpdate={updateClubInvitation} />
          <p className={styles.muted}>New club invitations let administrators join PCS now. After you review the divisions and meet schedule in Step 5, every selected club receives the season invitation in its workspace.</p>
        </>}
        {step === 2 && <>
          <p className={styles.muted}>Choose your divisions and who can play in each. All teams have four players.</p>
          <fieldset disabled={disabled} className={styles.form}>
            <div className={styles.divisionChoices}>{[...divisionChoices, ...(draft.divisions.includes("4.5/Open") ? ["4.5/Open"] : [])].map(division => <label className={styles.choice} data-selected={draft.divisions.includes(division)} key={division}><input aria-label={`Include ${division} division`} type="checkbox" checked={draft.divisions.includes(division)} onChange={() => edit({ divisions: draft.divisions.includes(division) ? draft.divisions.filter(d => d !== division) : [...draft.divisions, division] })} />{division}</label>)}</div>
            <div className={styles.note}>Division names don’t set rating limits automatically. Leave a limit blank if you don’t want one. A player’s starting rating comes from the club they represent when they first enter the season.</div>
            {draft.divisions.map(division => { const rule = draft.registration_rules[division] || emptyRule(); return <fieldset key={division} className={styles.rule}><legend>{division} division</legend><div className={styles.ruleGrid}>
              <label className={styles.field}>Minimum rating<input aria-label={`${division} minimum rating`} type="number" min={1} max={7} step="0.001" placeholder="No minimum" value={rule.min_rating ?? ""} onChange={e => editRule(division, "min_rating", e.target.value)} /></label>
              <label className={styles.field}>Maximum rating<input aria-label={`${division} maximum rating`} type="number" min={1} max={7} step="0.001" placeholder="No maximum" value={rule.max_rating ?? ""} onChange={e => editRule(division, "max_rating", e.target.value)} /></label>
              <label className={styles.field}>Team composition<select aria-label={`${division} team composition`} value={rule.women_required ?? ""} onChange={e => editRule(division, "women_required", e.target.value)}><option value="">Any four players</option>{[0, 1, 2, 3, 4].map(n => <option key={n} value={n}>{n} women, {4 - n} men</option>)}</select></label>
            </div></fieldset>; })}
          </fieldset>
          <p className={styles.muted}>Clubs choose players later, separately for each meet. An eligibility exception needs organizer approval.</p>
        </>}
        {step === 3 && <>
          <p className={styles.muted}>A meet is one gathering of 2–4 clubs at a host club. Add the dates and locations for this season.</p>
          <div className={styles.note}><strong>Choose players later.</strong> This step only schedules meets. Each club will submit a fresh roster for each meet.</div>
          {!draft.meets.length && <p>No meets scheduled yet. Add your first meet below.</p>}
          {draft.meets.map((meet, index) => <fieldset disabled={disabled} key={index} className={`${styles.rule} ${styles.meetCard}`}><legend>Meet {index + 1}</legend>
            <div className={styles.grid}>
              <label className={styles.field}>Host club<select aria-label={`Meet ${index + 1} host`} value={meet.host_club_id} onChange={e => { const id = e.target.value; editMeet(index, { host_club_id: id, club_ids: meet.club_ids.includes(id) ? meet.club_ids : [...meet.club_ids.filter(c => c !== meet.host_club_id).slice(0, 3), id].filter(Boolean) }); }}><option value="">Choose host</option>{selectedClubs.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}</select></label>
              <label className={styles.field}>Date and time<input aria-label={`Meet ${index + 1} date and time`} type="datetime-local" min={draft.start_date ? `${draft.start_date}T00:00` : undefined} max={draft.end_date ? `${draft.end_date}T23:59` : undefined} value={meetLocalTime(meet.starts_at, draft.timezone)} onChange={e => { try { editMeet(index, { starts_at: meetUtcTime(e.target.value, draft.timezone) }); } catch (error) { setErrors([error instanceof Error ? error.message : "Choose a valid date and time."]); } }} /><small>{draft.timezone.replaceAll("_", " ")}</small></label>
              <label className={styles.field}>Duration (minutes)<input aria-label={`Meet ${index + 1} duration`} type="number" min={30} max={180} value={meet.duration_minutes ?? ""} onChange={e => editMeet(index, { duration_minutes: e.target.value ? Number(e.target.value) : null })} /></label>
              <label className={styles.field}>Courts available<input aria-label={`Meet ${index + 1} courts`} type="number" min={1} max={100} value={meet.courts ?? ""} onChange={e => editMeet(index, { courts: e.target.value ? Number(e.target.value) : null })} /></label>
            </div>
            <p><strong>Clubs attending this meet</strong> · {meet.club_ids.length} of 2–4</p><div className={styles.choices}>{selectedClubs.map(c => <label className={styles.choice} key={c.id} data-selected={meet.club_ids.includes(c.id)}><input aria-label={`Meet ${index + 1}: ${c.name}`} type="checkbox" checked={meet.club_ids.includes(c.id)} disabled={disabled || c.id === meet.host_club_id || (!meet.club_ids.includes(c.id) && meet.club_ids.length >= 4)} onChange={() => editMeet(index, { club_ids: meet.club_ids.includes(c.id) ? meet.club_ids.filter(id => id !== c.id) : [...meet.club_ids, c.id] })} /><span>{c.name}{c.id === meet.host_club_id && <small>Host · included</small>}</span></label>)}</div>
            <div className={styles.toolbar} style={{ marginTop: 18 }}><small className={styles.muted}>Roster deadline starts at the meet’s start time. Set an earlier one later if needed.</small><button type="button" onClick={() => edit({ meets: draft.meets.filter((_, i) => i !== index) })}>Remove meet {index + 1}</button></div>
          </fieldset>)}
          <div className={styles.toolbar} style={{ marginTop: 20 }}><button disabled={disabled || draft.meets.length >= 100} onClick={() => edit({ meets: [...draft.meets, { host_club_id: "", club_ids: [], starts_at: null, duration_minutes: 180, courts: 4 }] })}>Add a meet</button><span className={styles.muted}>{draft.meets.length} scheduled</span></div>
        </>}
        {step === 4 && <>
          <p className={styles.muted}>Check the setup before inviting clubs. The season rules and scheduled meets are fixed when invitations open.</p>
          <div className={styles.review}>
            <section><div className={styles.toolbar}><h3>Season details</h3><button disabled={disabled} onClick={() => goTo(0)}>Edit season details</button></div><dl><dt>Season</dt><dd>{draft.name}</dd><dt>Dates</dt><dd>{draft.start_date} to {draft.end_date}</dd><dt>Organizer</dt><dd>{club.name}</dd><dt>Timezone</dt><dd>{draft.timezone}</dd></dl></section>
            <section><div className={styles.toolbar}><h3>Clubs to invite</h3><button disabled={disabled} onClick={() => goTo(1)}>Edit clubs</button></div><p>{draft.club_ids.map(clubName).join(" · ")}</p></section>
            <section><div className={styles.toolbar}><h3>Divisions & eligibility</h3><button disabled={disabled} onClick={() => goTo(2)}>Edit divisions</button></div><div className={styles.tableWrap}><table className={styles.table}><thead><tr><th>Division</th><th>Minimum</th><th>Maximum</th><th>Team</th></tr></thead><tbody>{draft.divisions.map(d => { const r = draft.registration_rules[d] || emptyRule(); return <tr key={d}><td>{d}</td><td>{r.min_rating ?? "No minimum"}</td><td>{r.max_rating ?? "No maximum"}</td><td>{composition(r)}</td></tr>; })}</tbody></table></div></section>
            <section><div className={styles.toolbar}><h3>Meet schedule</h3><button disabled={disabled} onClick={() => goTo(3)}>Edit meets</button></div>{draft.meets.map((meet, index) => <p key={index}><strong>Meet {index + 1} · {when(meet.starts_at)}</strong><br />{clubName(meet.host_club_id)} hosts {meet.club_ids.map(clubName).join(", ")} · {meet.courts} courts · {meet.duration_minutes} minutes</p>)}</section>
          </div>
          <h3 style={{ marginTop: 24 }}>What happens next?</h3><SetupNextSteps />
          <p className={styles.muted}>Invitations appear in each club’s PCS workspace. No email is sent.</p>
          <label className={styles.check}><input aria-label="I have reviewed the season setup" type="checkbox" disabled={disabled} checked={reviewed} onChange={e => setReviewed(e.target.checked)} /><span>I’ve reviewed the clubs, eligibility rules, and meet dates.</span></label>
        </>}
      </>}
      {errors.length > 0 && <div className={styles.error} role="alert"><strong>{blocked ? "Reload before continuing" : "Check these details"}</strong><ul>{errors.map((error, index) => <li key={index}>{error}</li>)}</ul>{blocked && <p>Your entries remain visible for reference. <button disabled={busy} onClick={reloadSaved}>Reload saved setup</button></p>}</div>}
      {message && <p role="status">{message}</p>}
      <footer className={styles.footer}><button disabled={disabled || step === 0} onClick={() => goTo(step - 1)}>Back</button><small>{step < 4 ? "Your progress is saved when you continue." : "Players are chosen separately for each meet."}</small>
        {step < 4 ? <button className={styles.primary} disabled={disabled} onClick={continueSetup}>{busy ? "Saving…" : "Save and continue"}</button>
          : <button className={styles.primary} disabled={disabled || !reviewed} onClick={finishSetup}>{busy ? "Opening invitations…" : "Open club invitations"}</button>}
      </footer>
    </div>
  </section>;
}
