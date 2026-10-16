import { notFound } from "next/navigation";
import { publicSiteFetch } from "@/lib/clubSiteServer";
import type { SeasonTrophy } from "@/lib/interclubAwards";
import ClubTrophyCase from "@/components/ClubTrophyCase";

export default async function ClubTrophiesPage({ params }: { params: { clubSlug: string } }) {
  const data = await publicSiteFetch<{ club_name: string; trophies: SeasonTrophy[] }>(`/public/clubs/${encodeURIComponent(params.clubSlug)}/trophies`);
  if (!data) notFound();
  return <ClubTrophyCase clubName={data.club_name} trophies={data.trophies} />;
}
