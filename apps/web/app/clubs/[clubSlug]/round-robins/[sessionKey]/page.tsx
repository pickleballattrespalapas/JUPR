import { notFound } from "next/navigation";
import Link from "@/components/PublicClubLink";
import SectionLoadError from "@/components/SectionLoadError";
import RoundRobinResults from "../RoundRobinResults";
import { publicSiteFetch } from "@/lib/clubSiteServer";
import type { RoundRobinResult } from "@/lib/roundRobinHistory";

export default async function RoundRobinResultsPage({ params }: {
  params: Promise<{ clubSlug: string; sessionKey: string }>;
}) {
  const { clubSlug, sessionKey } = await params;
  const root = `/clubs/${encodeURIComponent(clubSlug)}/round-robins`;
  let data: { session: RoundRobinResult } | null;
  try { data = await publicSiteFetch<{ session: RoundRobinResult }>(`${root}/${encodeURIComponent(sessionKey)}`); }
  catch { return <SectionLoadError title="Round robin results are temporarily unavailable" message="Please try again to load these standings." />; }
  if (!data) notFound();
  return <section>
    <p><Link href={root}>← All Round Robins</Link></p>
    <RoundRobinResults session={data.session} />
  </section>;
}
