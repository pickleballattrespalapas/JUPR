"use client";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import { useAdminSession } from "@/lib/useAdminSession";
import { getAdminPlayerEditorApiBaseUrl } from "@/lib/adminPlayerEditorApi";
import { apiError, RegistrationSeason } from "@/lib/interclubRegistration";
import { ClubChoice, PlanningSeason, newSeason, normalizeDraft, setupSteps } from "@/lib/interclubSetup";
import InterclubSetupWizard from "./InterclubSetupWizard";
import styles from "./setup.module.css";

export default function InterclubPage() {
  const { session, accessToken, loading } = useAdminSession();
  const { clubId } = useAdminWorkspace();
  const canManage = session?.capabilities?.assignments.some(a => a.club_id === clubId && ["administrator", "club_owner", "super_admin"].includes(a.role));
  if (loading) return <p>Checking club access…</p>;
  if (!canManage) return <p>Sign in as a club administrator to set up an interclub season. <Link href="/admin/login">Sign in</Link></p>;
  return <InterclubHome key={`${clubId}:${session?.user?.id || session?.user?.email}`} clubId={clubId} accessToken={accessToken} />;
}

function InterclubHome({ clubId, accessToken }: { clubId: string; accessToken: string }) {
  const api = getAdminPlayerEditorApiBaseUrl();
  const [choices, setChoices] = useState<ClubChoice[]>([]), [seasons, setSeasons] = useState<PlanningSeason[]>([]);
  const [registrations, setRegistrations] = useState<RegistrationSeason[]>([]), [active, setActive] = useState<PlanningSeason | null>(null);
  const [loaded, setLoaded] = useState(false), [error, setError] = useState(""), [reload, setReload] = useState(0);
  const token = useRef(accessToken); token.current = accessToken;
  useEffect(() => {
    const controller = new AbortController(); setLoaded(false); setError("");
    if (!api) { setError("Interclub setup is unavailable."); return; }
    const root = `${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub`;
    async function get(path: string) {
      const response = await fetch(`${root}${path}`, { headers: { Authorization: `Bearer ${token.current}` }, cache: "no-store", signal: controller.signal });
      const data = await response.json(); if (!response.ok) throw new Error(apiError(data, "Unable to load interclub seasons.")); return data;
    }
    async function clubs() {
      const all: ClubChoice[] = []; let offset: number | null = 0;
      while (offset !== null) { const data = await get(`/club-choices?offset=${offset}`); all.push(...data.clubs); offset = data.next_offset; }
      return all;
    }
    Promise.all([get("/setup"), get("/registrations"), clubs()]).then(([drafts, open, all]) => {
      if (!controller.signal.aborted) { setSeasons(drafts.seasons.map((s: PlanningSeason) => ({ ...s, draft: normalizeDraft(s.draft) }))); setRegistrations(open.seasons); setChoices(all); setLoaded(true); }
    }).catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [api, clubId, reload]);
  const club = choices.find(c => c.id === clubId) || { id: clubId, name: "Your club", slug: "" };
  const drafts = seasons.filter(s => !registrations.some(r => r.id === s.id));
  function saved(season: PlanningSeason) { setSeasons(old => [season, ...old.filter(s => s.id !== season.id)]); }
  return <section className={styles.page}>
    {active && api ? <InterclubSetupWizard key={active.id} api={api} club={club} accessToken={accessToken} initialSeason={active} choices={choices} onSaved={saved}
      onOpened={season => setRegistrations(old => [season, ...old.filter(s => s.id !== season.id)])} onClose={() => { setActive(null); setReload(n => n + 1); }} /> : <>
      <header className={styles.header}><div><p className={styles.eyebrow}>{club.name}</p><h1>Interclub leagues</h1><p className={styles.muted}>Set up a season, invite clubs, and prepare a roster for each meet.</p></div><button className={styles.primary} disabled={!loaded} onClick={() => setActive(newSeason())}>Set up a season</button></header>
      {error && <div className={styles.error} role="alert">{error} <button onClick={() => setReload(n => n + 1)}>Try again</button></div>}
      {!loaded && !error && <p>Loading your interclub seasons…</p>}
      {loaded && <>
        {!seasons.length && !registrations.length && <section className={styles.panel}><h2>Start with the season. Choose players later.</h2><p>The setup walks you through the dates, participating clubs, divisions, meet schedule, and invitations. Each club then chooses its available players separately for each meet.</p><ol className={styles.nextSteps}><li><strong>Set up</strong><span>Choose clubs, divisions and meet dates.</span></li><li><strong>Invite</strong><span>Review the setup and open club invitations.</span></li><li><strong>Prepare each meet</strong><span>Clubs accept and submit their own rosters.</span></li></ol></section>}
        {drafts.length > 0 && <section><h2>Continue setup</h2><div className={styles.cards}>{drafts.map(s => <article key={s.id} className={styles.card}><p className={styles.eyebrow}>Draft · Step {(s.draft.setup_step || 0) + 1} of 5</p><h3>{s.draft.name || "Untitled interclub season"}</h3><p className={styles.muted}>{s.draft.start_date && s.draft.end_date ? `${s.draft.start_date} to ${s.draft.end_date}` : "Season dates not set"}<br />{setupSteps[s.draft.setup_step || 0]}</p><button onClick={() => setActive(s)}>Continue setup</button></article>)}</div></section>}
        {registrations.length > 0 && <section><h2>Season workspaces</h2><p className={styles.muted}>Manage club responses and upcoming meet rosters here.</p><div className={styles.cards}>{registrations.map(s => <article key={s.id} className={styles.card}><p className={styles.eyebrow}>{s.organizer_club_id === clubId ? "Organizer" : s.participation?.status === "invited" ? "Invitation to respond to" : s.participation?.status === "accepted" ? "Participating club" : s.participation?.status === "declined" ? "Invitation declined" : "Invitation cancelled"}</p><h3>{s.details.name}</h3><p className={styles.muted}>{s.details.start_date} to {s.details.end_date}</p><Link className={styles.button} href={`/admin/interclub/registrations?season=${s.id}`}>{s.participation?.status === "invited" && s.organizer_club_id !== clubId ? "Review invitation" : "Manage season"}</Link></article>)}</div></section>}
        <p><Link href="/admin/interclub/registrations">All club invitations and meet rosters</Link></p>
      </>}
    </>}
  </section>;
}
