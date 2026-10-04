import Link from "next/link";
import { notFound } from "next/navigation";
import { publicSiteFetch } from "@/lib/clubSiteServer";
import type { PublicLeague } from "@/lib/interclubPublic";
import type { SeasonTrophy } from "@/lib/interclubAwards";
import InterclubFinalResults from "@/components/InterclubFinalResults";

export const metadata = { title: "Interclub final results", robots: { index: false, follow: false } };
export default async function FinalResultsPage({ params }: { params: { seasonId: string } }) {
  if (!/^[0-9a-f-]{36}$/i.test(params.seasonId)) notFound();
  const [league, awards] = await Promise.all([
    publicSiteFetch<PublicLeague>(`/public/interclub/${params.seasonId}`),
    publicSiteFetch<{ trophies: SeasonTrophy[] }>(`/public/interclub/${params.seasonId}/awards`),
  ]);
  if (!league) notFound();
  return <><p><Link href={`/interclub/${params.seasonId}`}>← {league.document.name}</Link></p><h1>Final season results</h1><InterclubFinalResults league={league} awards={awards?.trophies || []} /></>;
}
