import Link from "@/components/PublicClubLink";
import PublicControls from "@/components/PublicControls";
import SectionLoadError from "@/components/SectionLoadError";
import { publicSiteFetch } from "@/lib/clubSiteServer";
import { roundRobinDate, roundRobinFormat, type RoundRobinHistory } from "@/lib/roundRobinHistory";
import styles from "./roundRobins.module.css";

type Props = {
  params: Promise<{ clubSlug: string }>;
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
};

export default async function RoundRobinsPage(props: Props) {
  const [{ clubSlug }, search] = await Promise.all([props.params, props.searchParams]);
  const first = (key: string) => { const raw = search?.[key]; return (Array.isArray(raw) ? raw[0] : raw) || ""; };
  const kind = ["round_robin", "moneyball"].includes(first("kind")) ? first("kind") : "all";
  const from = /^\d{4}-\d{2}-\d{2}$/.test(first("from")) ? first("from") : "";
  const to = /^\d{4}-\d{2}-\d{2}$/.test(first("to")) ? first("to") : "";
  const page = Math.min(4166, Math.max(0, Math.trunc(Number(first("page"))) || 0));
  const root = `/clubs/${encodeURIComponent(clubSlug)}/round-robins`;
  const query = new URLSearchParams({ kind, offset: String(page * 24), limit: "24" });
  if (from) query.set("from_date", from);
  if (to) query.set("to_date", to);
  const invalidRange = Boolean(from && to && from > to);
  let history: RoundRobinHistory | null = null;
  let failed = false;
  if (!invalidRange) {
    try { history = await publicSiteFetch<RoundRobinHistory>(`${root}?${query}`); }
    catch { failed = true; }
  }
  const pageHref = (next: number) => {
    const q = new URLSearchParams({ kind, page: String(next) });
    if (from) q.set("from", from);
    if (to) q.set("to", to);
    return `${root}?${q}`;
  };
  return <section>
    <h1>Round Robins</h1>
    <p className={styles.intro}>Browse past sessions and final standings, including Moneyball. Open a session to see the full results and round scores.</p>
    <PublicControls context="Round robin filters" resetKey={`${kind}:${from}:${to}`}>
      <form action={root} method="get" className={styles.filters}>
        <label>Session type<select name="kind" defaultValue={kind}>
          <option value="all">All sessions</option><option value="round_robin">Round robins</option><option value="moneyball">Moneyball</option>
        </select></label>
        <label>Started on or after<input type="date" name="from" defaultValue={from} /></label>
        <label>Started on or before<input type="date" name="to" defaultValue={to} /></label>
        <button type="submit" className={styles.button}>Apply filters</button>
        <Link href={root}>Clear filters</Link>
      </form>
    </PublicControls>
    {invalidRange ? <p role="alert">Choose an end date on or after the start date.</p> : failed || !history ?
      <SectionLoadError title="Round robin results are temporarily unavailable" message="Please try again to load past sessions." /> : <>
      {history.sessions.length ? <div className={styles.grid}>
        {history.sessions.map(session => <article className={styles.card} key={session.session_key}>
          <span className={styles.badge}>{session.kind === "moneyball" ? "Moneyball" : "Round robin"}</span>
          <time className={styles.date} dateTime={session.created_at || undefined}>{roundRobinDate(session.created_at)}</time>
          <h2><Link prefetch={false} href={`${root}/${encodeURIComponent(session.session_key)}`}>{session.title}</Link></h2>
          <p>{session.participant_count} players · {roundRobinFormat(session.play_format)} · {session.rating_mode === "rated" ? "Rated" : "Unrated"}</p>
          <Link prefetch={false} href={`${root}/${encodeURIComponent(session.session_key)}`}>{session.scoring_mode === "unscored" ? "View session" : "View final standings"} →</Link>
        </article>)}
      </div> : <article className={styles.card}><h2>No past sessions found</h2><p>{from || to || kind !== "all" ? "Try a wider date range or a different session type." : "Completed public round robins will appear here."}</p></article>}
      {page > 0 || history.has_more ? <nav aria-label="Round robin history pages" className={styles.pager}>
        {page > 0 ? <Link href={pageHref(page - 1)}>← Newer sessions</Link> : null}
        <span>Page {page + 1}</span>
        {history.has_more ? <Link href={pageHref(page + 1)}>Older sessions →</Link> : null}
      </nav> : null}
    </>}
  </section>;
}
