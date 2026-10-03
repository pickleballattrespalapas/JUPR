"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { getAdminApiBaseUrl } from "@/lib/adminAuthClient";
import { useAdminSession } from "@/lib/useAdminSession";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import type { AdminEventHistory } from "@/lib/eventSeasons";
import type { SeasonAwardPreview } from "@/lib/interclubAwards";
import { competitionRequest, phaseLabels, type CompetitionContext } from "@/lib/interclubCompetition";
import type { RegistrationDetail } from "@/lib/interclubRegistration";
import { buildSeasonGuide, meetProgress, seasonStages, type SeasonGuideData } from "@/lib/interclubSeasonGuide";
import styles from "./season.module.css";

export default function SeasonGuide({ seasonId }: { seasonId: string }) {
  const { clubId } = useAdminWorkspace();
  const { session, accessToken, loading } = useAdminSession();
  const allowed = session?.capabilities?.assignments.some(row => row.club_id === clubId && ["administrator", "club_owner", "super_admin"].includes(row.role));
  if (loading) return <p>Checking club access…</p>;
  if (!allowed || !accessToken) return <p>Sign in as a club administrator to open the season guide. <Link href="/admin/login">Sign in</Link></p>;
  if (!seasonId) return <p><Link href="/admin/interclub">Choose an interclub season.</Link></p>;
  return <GuideLoader key={`${clubId}:${seasonId}:${session?.user?.id}`} clubId={clubId} seasonId={seasonId} accessToken={accessToken} />;
}

export function GuideLoader({ clubId, seasonId, accessToken }: { clubId: string; seasonId: string; accessToken: string }) {
  const [data, setData] = useState<SeasonGuideData | null>(null), [error, setError] = useState("");
  const [refresh, setRefresh] = useState(0);
  const token = useRef(accessToken); token.current = accessToken;
  const api = getAdminApiBaseUrl();
  useEffect(() => {
    const controller = new AbortController();
    setData(null); setError("");
    async function load() {
      if (!api) throw new Error("The season guide is unavailable. Please try again later.");
      const root = `${api}/admin/clubs/${encodeURIComponent(clubId)}`;
      const get = <T,>(path: string) => competitionRequest<T>(`${root}${path}`, token.current, controller.signal);
      const registration = await get<RegistrationDetail>(`/interclub/registrations/${encodeURIComponent(seasonId)}`);
      const canOpen = registration.is_organizer || registration.own_participation?.status === "accepted";
      const competition = canOpen && registration.season.registration?.meet_planning_open ? await get<CompetitionContext>(`/interclub/competition/${encodeURIComponent(seasonId)}`) : null;
      const result: SeasonGuideData = { registration, competition, awards: null, history: null };
      if (registration.is_organizer && buildSeasonGuide(result).competitionDone) {
        result.awards = await get<SeasonAwardPreview>(`/interclub/${encodeURIComponent(seasonId)}/awards`);
        if (result.awards.ready && result.awards.current) result.history = await get<AdminEventHistory>(`/event-seasons?${new URLSearchParams({ kind: "interclub", event: seasonId })}`);
      }
      if (!controller.signal.aborted) setData(result);
    }
    void load().catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Could not load the season guide."); });
    return () => controller.abort();
  }, [api, clubId, seasonId, refresh]);
  return <main className={styles.page}>
    <div className={styles.toolbar}><Link href="/admin/interclub">← Interclub leagues</Link><button onClick={() => setRefresh(value => value + 1)}>Refresh progress</button></div>
    {error && <p role="alert" className={styles.error}>{error} Refresh progress to check the saved season before continuing.</p>}
    {!data && !error && <p role="status">Checking season progress…</p>}
    {data && <SeasonGuideView data={data} />}
  </main>;
}

export function SeasonGuideView({ data }: { data: SeasonGuideData }) {
  const guide = buildSeasonGuide(data);
  const { season } = data.registration;
  const date = (value: string) => new Date(value).toLocaleString(undefined, { timeZone: season.details.timezone, dateStyle: "medium", timeStyle: "short" });
  return <>
    <header><p className={styles.eyebrow}>Interclub · Season guide</p><h1>{season.details.name}</h1><p>Follow this season from setup to its place in History. Return here after each task to see what comes next.</p></header>
    <section className={styles.next} aria-labelledby="season-next-heading">
      <p className={styles.eyebrow}>{guide.complete ? "Season complete" : `Step ${guide.stage + 1} of ${seasonStages.length}`} · {seasonStages[guide.stage].title}</p>
      <h2 id="season-next-heading">{guide.title}</h2><p>{guide.why}</p>
      <ol>{guide.instructions.map(instruction => <li key={instruction}>{instruction}</li>)}</ol>
      <Link className={styles.primary} href={guide.action.href}>{guide.action.label} →</Link>
      {guide.complete && guide.organizer && <p><Link href={`/interclub/${encodeURIComponent(season.id)}/final-results`}>View this season’s final results & awards</Link></p>}
    </section>
    <section aria-labelledby="season-roadmap-heading"><h2 id="season-roadmap-heading">The life of this season</h2>
      {!guide.organizer && <p>Your club manages its players and meets. The organizer manages championships, awards, and the next season.</p>}
      <ol className={styles.stages}>{seasonStages.map((stage, index) => {
        const finished = index < guide.stage && (guide.organizer || index < 2 || guide.complete);
        const current = index === guide.stage;
        const available = index <= guide.stage && (guide.organizer || [1, 2, 4].includes(index));
        const status = finished ? "Complete" : current ? guide.linked ? "Next season linked" : index === 5 && guide.complete ? "Optional next step" : "You are here" : guide.organizer ? "Coming next" : "Season organizer";
        return <li key={stage.title} className={current ? styles.current : styles.card} aria-current={current ? "step" : undefined}>
          <p className={styles.eyebrow}>{index + 1} · {status}</p><h3>{stage.title}</h3><p>{stage.description}</p>
          {available && <Link href={guide.hrefs[index]}>{stage.link} →</Link>}
        </li>;
      })}</ol>
    </section>
    {!!guide.meets.length && <section aria-labelledby="season-meets-heading"><h2 id="season-meets-heading">{guide.organizer ? "Meet progress" : "Your club’s meet progress"}</h2>
      <ul className={styles.meets}>{guide.meets.map(meet => {
        const batch = guide.batches.get(meet.id);
        const host = data.registration.clubs.find(club => club.id === meet.host_club_id)?.name || "Host club";
        return <li key={meet.id}><div><strong>{phaseLabels[meet.competition_phase || "regular"]} · {host}</strong><p>{date(meet.starts_at)}</p><p>{meetProgress(batch)}</p></div>
          <Link href={`/admin/interclub/${batch ? "competition" : "registrations"}?${new URLSearchParams({ season: season.id, meet: meet.id, step: batch ? batch.state === "draft" ? "run" : "approve" : "availability" })}`}>Open meet →</Link></li>;
      })}</ul>
    </section>}
  </>;
}
