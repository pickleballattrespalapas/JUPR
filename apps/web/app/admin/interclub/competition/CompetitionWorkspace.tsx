"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";
import { readBrowserWorkspace } from "@/lib/adminWorkspace";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import { RegistrationSeason } from "@/lib/interclubRegistration";
import { CompetitionBatch, CompetitionContext, CompetitionDocument, CompetitionFormat, CompetitionPhase, MeetCompetition, competitionPath, competitionPlayers, competitionRequest, fromLocalInput, gameCount, phaseLabels } from "@/lib/interclubCompetition";
import ScoreEditor from "./ScoreEditor";
import PrintPacket from "./PrintPacket";
import Standings from "./Standings";
import ScheduleMeet from "./ScheduleMeet";
import styles from "./competition.module.css";

export default function CompetitionWorkspace({ initialSeasonId, initialMeetId }: { initialSeasonId: string; initialMeetId: string }) {
  const { session, accessToken, loading } = useAdminSession();
  const { clubId } = useAdminWorkspace();
  const allowed = session?.capabilities?.assignments.some(assignment => assignment.club_id === clubId && ["administrator", "club_owner", "super_admin", "operator"].includes(assignment.role));
  if (loading) return <p>Checking club access…</p>;
  if (!allowed || !accessToken) return <p>Sign in as a club administrator to open meet operations. <Link href="/admin/login">Sign in</Link></p>;
  return <CompetitionHome key={`${clubId}:${session?.user?.id || session?.user?.email}`} clubId={clubId} accessToken={accessToken} initialSeasonId={initialSeasonId} initialMeetId={initialMeetId} />;
}

export function CompetitionHome({ clubId, accessToken, initialSeasonId, initialMeetId }: { clubId: string; accessToken: string; initialSeasonId: string; initialMeetId: string }) {
  const api = getAdminApiBaseUrl();
  const [seasons, setSeasons] = useState<RegistrationSeason[]>([]), [seasonId, setSeasonId] = useState(initialSeasonId);
  const [data, setData] = useState<CompetitionContext | null>(null), [error, setError] = useState(""), [refresh, setRefresh] = useState(0), [loading, setLoading] = useState(true);
  const [meetId, setMeetId] = useState(initialMeetId), [locked, setLocked] = useState(false);
  const token = useRef(accessToken); token.current = accessToken;
  useEffect(() => {
    const controller = new AbortController();
    if (!api) { setError("Meet operations are unavailable. Please try again later."); setLoading(false); return; }
    void competitionRequest<{ seasons: RegistrationSeason[] }>(`${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub/competition`, token.current, controller.signal)
      .then(result => { if (!controller.signal.aborted) { setSeasons(result.seasons); setSeasonId(old => result.seasons.some(season => season.id === old) ? old : result.seasons[0]?.id || ""); if (!result.seasons.length) setLoading(false); } })
      .catch(cause => { if (!controller.signal.aborted) { setError(message(cause)); setLoading(false); } });
    return () => controller.abort();
  }, [api, clubId]);
  useEffect(() => {
    const controller = new AbortController();
    if (!seasonId || !api) return;
    setLoading(true); setError(""); setData(null);
    void competitionRequest<CompetitionContext>(competitionPath(api, clubId, seasonId), token.current, controller.signal)
      .then(result => { if (!controller.signal.aborted) { setData(result); setMeetId(old => result.meets.some(meet => meet.id === old) ? old : result.meets[0]?.id || ""); } })
      .catch(cause => { if (!controller.signal.aborted) setError(message(cause)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [api, clubId, seasonId, refresh]);
  const phase: CompetitionPhase = data?.meets.find(meet => meet.id === meetId)?.competition_phase || "regular";
  const clubName = (id: string) => data?.clubs.find(club => club.id === id)?.name || "Club";
  const when = (iso: string) => new Date(iso).toLocaleString(undefined, { timeZone: data?.season.details.timezone, dateStyle: "medium", timeStyle: "short" });
  return <main className={styles.page}>
    <p><Link href="/admin/interclub">← Interclub leagues</Link></p>
    <header><p className={styles.eyebrow}>Paper courts · One official score submission</p><h1>Meet operations</h1><p>Prepare and print the pairings, run the meet on paper, then bring all official scores back here.</p></header>
    <div className={styles.toolbar}><label>Season<select value={seasonId} disabled={locked || !seasons.length} onChange={event => { setSeasonId(event.target.value); setMeetId(""); }}>
      {!seasons.length && <option value="">No accepted seasons</option>}{seasons.map(season => <option key={season.id} value={season.id}>{season.details.name}</option>)}
    </select></label><button disabled={loading || locked} onClick={() => setRefresh(value => value + 1)}>Refresh season</button>
      {seasonId && <Link className={styles.button} href={`/admin/interclub/registrations?season=${encodeURIComponent(seasonId)}`}>Player pool &amp; meet rosters</Link>}
    </div>
    {error && <p role="alert" className={styles.error}>{error}</p>}
    {loading && <p role="status">Loading meet operations…</p>}
    {!loading && !seasons.length && !error && <p>Accept a season invitation first, then prepare the players for your meet.</p>}
    {data && <>
      <div className={styles.steps} aria-label="Meet workflow"><span>1. Prepare &amp; print</span><span>2. Record all scores</span><span>3. Submit for approval</span><span>4. Organizer approves</span></div>
      <div className={styles.toolbar}><label>Meet<select value={meetId} disabled={locked} onChange={event => setMeetId(event.target.value)}>{data.meets.map(meet => <option key={meet.id} value={meet.id}>{when(meet.starts_at)} · {clubName(meet.host_club_id)}{meet.competition_phase && meet.competition_phase !== "regular" ? ` · ${phaseLabels[meet.competition_phase]}` : ""}</option>)}</select></label>
        <label>Competition<select value={phase} disabled aria-label="Scheduled competition format">{Object.entries(phaseLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      </div>
      {!data.meets.length && <p>The organizer needs to schedule a meet before score sheets can be prepared.</p>}
      {data.is_organizer && <ScheduleMeet root={competitionPath(api!, clubId, seasonId)} clubId={clubId} accessToken={accessToken} context={data} disabled={locked} onScheduled={meet => { setMeetId(meet.id); setRefresh(value => value + 1); }} />}
      {meetId && <MeetOperations key={`${clubId}:${seasonId}:${meetId}:${phase}:${refresh}`} root={`${competitionPath(api!, clubId, seasonId)}/meets/${encodeURIComponent(meetId)}/${phase}`} clubId={clubId} accessToken={accessToken} phase={phase} context={data} clubName={clubName} onLock={setLocked} onSeasonChange={() => { setLocked(false); setRefresh(value => value + 1); }} />}
      <Standings data={data} clubName={clubName} />
    </>}
  </main>;
}

export function MeetOperations({ root, clubId, accessToken, phase, context, clubName, onLock, onSeasonChange }: { root: string; clubId: string; accessToken: string; phase: CompetitionPhase; context: CompetitionContext; clubName: (id: string) => string; onLock: (locked: boolean) => void; onSeasonChange: () => void }) {
  const [detail, setDetail] = useState<MeetCompetition | null>(null), [draft, setDraft] = useState<CompetitionDocument | null>(null);
  const [error, setError] = useState(""), [status, setStatus] = useState(""), [loading, setLoading] = useState(true), [busy, setBusy] = useState(false), [blocked, setBlocked] = useState(false), [refresh, setRefresh] = useState(0);
  const [format, setFormat] = useState<CompetitionFormat>(phase === "regular" ? "gender" : "mlp"), [division, setDivision] = useState(context.season.details.divisions[0] || ""), [clubA, setClubA] = useState(""), [clubB, setClubB] = useState("");
  const [review, setReview] = useState<"submit" | "approve" | null>(null), [reason, setReason] = useState(""), [startsAt, setStartsAt] = useState(""), [deadline, setDeadline] = useState("");
  const token = useRef(accessToken); token.current = accessToken;
  const pending = useRef(false), controller = useRef<AbortController | null>(null);
  const dirty = !!draft && JSON.stringify(draft) !== JSON.stringify(detail?.batch?.document);
  useEffect(() => { onLock(dirty || busy); return () => onLock(false); }, [dirty, busy, onLock]);
  useEffect(() => {
    if (!dirty) return;
    const protect = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", protect); return () => window.removeEventListener("beforeunload", protect);
  }, [dirty]);
  useEffect(() => {
    const request = new AbortController(); setLoading(true); setError(""); setBlocked(false);
    void competitionRequest<MeetCompetition>(root, token.current, request.signal).then(result => { if (!request.signal.aborted) { setDetail(result); setDraft(result.batch?.document || null); } })
      .catch(cause => { if (!request.signal.aborted) setError(message(cause)); }).finally(() => { if (!request.signal.aborted) setLoading(false); });
    return () => { request.abort(); controller.current?.abort(); };
  }, [root, refresh]);
  async function change(action: string, body: Record<string, unknown> = {}): Promise<boolean> {
    if (pending.current || blocked || !detail) return false;
    if (readBrowserWorkspace()?.clubId !== clubId) { setBlocked(true); setError("Your selected club changed in another tab. Reopen meet operations for the current club."); return false; }
    pending.current = true; setBusy(true); setError(""); setStatus("");
    const request = new AbortController(); controller.current = request;
    let received = false;
    try {
      const result = await competitionRequest<{ batch: CompetitionBatch }>(action === "save" ? root : `${root}/${action}`, token.current, request.signal, action === "save" ? "PUT" : "POST", { expected_revision: detail.batch?.revision || 0, ...body });
      received = true;
      if (request.signal.aborted) return false;
      if (!result.batch || result.batch.meet_id !== detail.meet.id || result.batch.phase !== phase) { setBlocked(true); throw new Error("Could not confirm the saved meet. Reload before making another change."); }
      setDetail(old => old ? { ...old, batch: result.batch } : old); setDraft(result.batch.document); setReview(null);
      setStatus(action === "save" ? "All draft scores saved. Submit when every score sheet is entered." : action === "generate" ? "Pairings prepared. Review the players and print the meet packet." : action === "submit" ? `Revision ${result.batch.revision} submitted for organizer approval.` : action === "approve" ? "Official scores approved. Check rating status below before treating updates as complete." : action === "retry-ratings" ? "Rating update status refreshed." : action === "reopen" ? "Correction draft opened. Save, submit and approve the corrected scores again." : action === "refresh-lineups" ? "Eligible lineups refreshed for the unfinished pairings." : "The unfinished pairings are ready for the rescheduled meet. Confirm the new eligible lineup before printing.");
      if (["approve", "retry-ratings", "reschedule", "refresh-lineups"].includes(action)) onSeasonChange();
      return true;
    } catch (cause) {
      if (!request.signal.aborted) { const code = (cause as { status?: number })?.status; if ((!received && !code) || code && ([401, 403, 409].includes(code) || code >= 500)) setBlocked(true); setError(message(cause)); }
      return false;
    } finally { pending.current = false; if (!request.signal.aborted) setBusy(false); }
  }
  function edit(document: CompetitionDocument) { setDraft(document); setReview(null); setStatus(""); }
  const disabled = busy || blocked, batch = detail?.batch, editable = detail?.can_manage && batch?.state === "draft" && !disabled;
  if (loading) return <p role="status">Loading this meet…</p>;
  if (!detail) return <div className={styles.error}><p role="alert">{error || "This meet could not be loaded."}</p><button onClick={() => setRefresh(value => value + 1)}>Retry loading meet</button></div>;
  const players = competitionPlayers(detail), count = draft ? gameCount(draft) : null;
  const canRefreshStartingLineups = phase === "regular" && new Date(detail.meet.starts_at).getTime() > Date.now() && !!draft?.encounters.every(encounter => encounter.pairings.every(pairing => pairing.games.every(game => game.status === "pending" && game.a === null && game.b === null || ["forfeit", "double_forfeit"].includes(game.status) && (!pairing.players_a.length || !pairing.players_b.length))));
  const qualification = context.standings?.qualification?.[division] || context.qualifying?.[division];
  const qualifyingClubs = phase === "final" ? qualification?.qualifiers : phase === "qualifier" ? qualification?.playoff_required : null;
  const candidates = Array.from(new Set(detail.teams.filter(team => team.division === division && (!qualifyingClubs || qualifyingClubs.includes(team.club_id))).map(team => team.club_id)));
  const alreadyScheduled = phase !== "regular" && !!batch?.document.encounters.some(encounter =>
    encounter.division === division && (phase === "final" ||
      [encounter.club_a, encounter.club_b].includes(clubA) && [encounter.club_a, encounter.club_b].includes(clubB)));
  return <section className={styles.section}>
    {error && <p className={styles.error} role="alert">{error}</p>}{status && <p className={styles.success} role="status">{status}</p>}
    {detail.lineups_hidden && <p className={styles.notice}>Opposing lineups become available at the roster deadline. You can still manage your club’s roster.</p>}
    {blocked && <div className={styles.notice}><p>Reload the latest saved meet before continuing. Your last change may already have been saved.</p><button disabled={busy} onClick={() => { setReview(null); setRefresh(value => value + 1); }}>Reload saved meet</button></div>}
    {(!batch || phase !== "regular" && batch.state === "draft" && draft?.encounters.every(encounter => encounter.pairings.every(pairing => pairing.games.every(game => game.status === "pending")))) && <div className={styles.card}><h2>{batch ? "Add another skill-level matchup" : `Prepare ${phaseLabels[phase].toLowerCase()} pairings`}</h2>
      <p>The schedule uses the approved meet rosters. Clubs may enter different skill levels at each meet. Each skill level needs two to four clubs.</p>
      {detail.can_manage && (phase === "regular" || detail.is_organizer) ? <form onSubmit={event => { event.preventDefault(); void change("generate", { format, ...(phase !== "regular" ? { division, club_a: clubA, club_b: clubB } : {}) }); }}>
        <fieldset className={styles.gameFields} disabled={disabled || dirty}><div className={styles.toolbar}>
          {phase === "regular" ? <label>Meet format<select value={format} onChange={event => setFormat(event.target.value as CompetitionFormat)}><option value="gender">Women’s and men’s doubles</option><option value="mixed">Two mixed doubles pairings</option></select></label> : <>
            <label>Skill level<select value={division} onChange={event => { setDivision(event.target.value); setClubA(""); setClubB(""); }}>{context.season.details.divisions.map(value => <option key={value}>{value}</option>)}</select></label>
            <label>Club A<select required value={clubA} onChange={event => setClubA(event.target.value)}><option value="">Choose club</option>{candidates.filter(id => id !== clubB).map(id => <option key={id} value={id}>{clubName(id)}</option>)}</select></label>
            <label>Club B<select required value={clubB} onChange={event => setClubB(event.target.value)}><option value="">Choose club</option>{candidates.filter(id => id !== clubA).map(id => <option key={id} value={id}>{clubName(id)}</option>)}</select></label>
          </>}
          <button type="submit" className={styles.primary} disabled={alreadyScheduled}>Generate pairings</button>
        </div></fieldset>
        {phase !== "regular" && <p>Each club fields two women and two men. The server checks current skill eligibility, prior regular-season appearances and championship qualification.</p>}
        {alreadyScheduled && <p>{phase === "qualifier" ? "This qualifying pair is already in the packet. Choose another pair." : `Skill level ${division} is already in this packet. Choose another skill level to add its matchup.`}</p>}
        {phase !== "regular" && candidates.length < 2 && <p className={styles.notice}>Two qualifying clubs need approved lineups for this meet and skill level. Check the standings below and prepare their meet rosters first.</p>}
      </form> : <p>The host or organizer will prepare this meet. Your club can review and print the packet here when it is ready.</p>}
      {!detail.teams.length && <p className={styles.warning}>No approved meet teams are available yet. Prepare the meet rosters first.</p>}
    </div>}
    {batch && draft && <>
      <div className={styles.card}>
        <div className={styles.toolbar}><div><p className={styles.eyebrow}>{phaseLabels[phase]} · Revision {batch.revision}</p><h2>{batch.state === "draft" ? "Meet score draft" : batch.state === "submitted" ? "Awaiting organizer approval" : "Official meet results"}</h2><p>{count?.entered} of {count?.total} game outcomes entered{dirty ? " · Unsaved changes" : " · Saved"}</p></div>
          <button onClick={() => window.print()} disabled={busy || dirty}>Print meet packet</button>
        </div>
        {dirty && <p className={styles.notice}>Save your changes before printing or switching meets. <button disabled={busy} onClick={() => { setDraft(batch.document); setReview(null); }}>Discard unsaved changes</button></p>}
        {batch.state === "approved" && <p className={batch.ratings_status === "failed" ? styles.error : styles.notice}><strong>Rating updates: {batch.ratings_status.replaceAll("_", " ")}</strong>{batch.ratings_error ? ` — ${batch.ratings_error}` : batch.ratings_status === "completed" ? ". Both league and represented-club updates are complete." : ". Standings approval does not mean all rating updates have completed."}</p>}
        <div className={styles.toolbar}>
          {editable && <><button className={styles.primary} disabled={!dirty} onClick={() => void change("save", { document: draft })}>Save all draft scores</button><button disabled={dirty || !count?.total || count.entered !== count.total} onClick={() => setReview("submit")}>Review and submit meet</button></>}
          {batch.state === "submitted" && detail.is_organizer && <button className={styles.primary} disabled={disabled} onClick={() => setReview("approve")}>Review official approval</button>}
          {batch.state === "approved" && detail.is_organizer && ["failed", "pending"].includes(batch.ratings_status) && <button disabled={disabled} onClick={() => void change("retry-ratings")}>Retry rating updates</button>}
          {(draft.weather === "rescheduled" || canRefreshStartingLineups) && editable && <button disabled={dirty || disabled} onClick={() => void change("refresh-lineups")}>{draft.weather === "rescheduled" ? "Refresh eligible replay lineups" : "Refresh approved lineups"}</button>}
        </div>
        {review && <div className={styles.review} role="region" aria-label={review === "submit" ? "Confirm meet submission" : "Confirm official approval"}>
          <h3>{review === "submit" ? "Submit the complete meet?" : `Approve revision ${batch.revision}?`}</h3>
          <p>{review === "submit" ? "Check both clubs’ signed sheets, actual players, stopped injury scores and any weather decisions. This sends the whole saved meet to the organizer." : "This exact saved revision becomes official for standings and starts rating updates for completed doubles games. Retirements, unplayed forfeits and rotating singles do not change ratings."}</p>
          <div className={styles.toolbar}><button className={styles.primary} disabled={disabled || dirty} onClick={() => void change(review)}>{review === "submit" ? "Submit all official scores" : "Approve this revision"}</button><button disabled={busy} onClick={() => setReview(null)}>Keep reviewing</button></div>
        </div>}
        {detail.is_organizer && batch.state !== "draft" && <details><summary>Correct submitted or official scores</summary><p>Corrections create a new draft and must be submitted and approved again. The earlier result remains in the audit history.</p><label>Reason for correction<textarea value={reason} onChange={event => setReason(event.target.value)} /></label><button disabled={disabled || !reason.trim()} onClick={() => void change("reopen", { reason: reason.trim() })}>Open correction draft</button></details>}
      </div>
      <details className={styles.card}><summary>Weather delay, cancellation or reschedule</summary>
        <p>{phase === "regular" ? "A temporary delay resumes from the stopped score. On a new date, replay every unfinished three-game pairing from the beginning; completed pairings stand. If no reschedule is possible, completed games decide the pairing, 1–1 is a draw, and a wholly unplayed matchup gives one standings point to each club." : "A temporary delay resumes from the stopped score. The organizer arranges completion of the full MLP matchup so the final or qualifying place has an on-court winner."}</p>
        <label>Weather decision<select disabled={!editable} value={draft.weather} onChange={event => edit({ ...draft, weather: event.target.value as CompetitionDocument["weather"] })}><option value="normal">No weather change</option><option value="delay">Temporary delay — resume where stopped</option>{draft.weather === "rescheduled" && <option value="rescheduled">Rescheduled — unfinished pairings replayed</option>}{phase === "regular" && <option value="finalized_partial">No reschedule — finalize available results</option>}</select></label>
        {phase === "regular" && detail.is_organizer && batch.state === "draft" && <form onSubmit={event => { event.preventDefault(); void change("reschedule", { starts_at: fromLocalInput(startsAt), roster_deadline: fromLocalInput(deadline), reason: reason.trim() }); }}>
          <h3>Replay unfinished pairings on a new date</h3><p>Save the stopped scores first. The new roster deadline locks current interclub ratings and permits a new eligible lineup from the approved season pool.</p>
          <fieldset disabled={disabled || dirty} className={styles.gameFields}><div className={styles.twoColumns}><label>New meet time (your device’s time)<input required type="datetime-local" value={startsAt} onChange={event => setStartsAt(event.target.value)} /></label><label>New roster deadline (your device’s time)<input required type="datetime-local" value={deadline} onChange={event => setDeadline(event.target.value)} /></label></div>
            <label>Reschedule reason<textarea required value={reason} onChange={event => setReason(event.target.value)} /></label><button type="submit" disabled={!startsAt || !deadline || !reason.trim()}>Reschedule unfinished pairings</button>
          </fieldset>
        </form>}
      </details>
      <ScoreEditor document={draft} detail={detail} players={players} clubName={clubName} disabled={!editable} onChange={edit} />
      {editable && <div className={styles.bottomBar}><span>{dirty ? "Unsaved score changes" : "All draft changes saved"}</span><button className={styles.primary} disabled={!dirty || disabled} onClick={() => void change("save", { document: draft })}>Save all draft scores</button></div>}
      <PrintPacket document={batch.document} meet={detail.meet} seasonName={context.season.details.name} timezone={context.season.details.timezone} revision={batch.revision} players={players} clubName={clubName} />
    </>}
  </section>;
}

function message(cause: unknown): string {
  return cause instanceof Error && !(cause instanceof TypeError) && !(cause instanceof SyntaxError) ? cause.message : "Unable to reach meet operations. Check your connection and reload.";
}
