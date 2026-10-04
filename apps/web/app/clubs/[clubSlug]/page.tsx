import { notFound } from "next/navigation";
import { getPublicSite, publicSiteFetch } from "@/lib/clubSiteServer";
import type { SeasonTrophy } from "@/lib/interclubAwards";
import type { TournamentGoldHighlight } from "@/lib/tournamentHighlights";
import ClubSiteContent from "@/components/ClubSiteContent";
export default async function ClubHome({ params }: { params: { clubSlug: string } }) {
  const site = await getPublicSite(params.clubSlug);
  if (!site) notFound();
  // Optional honors load independently; an unavailable service must not hide the website.
  const [awards, tournaments] = await Promise.all([
    site.document.page_visibility?.trophies === "private" ? null : publicSiteFetch<{ trophies: SeasonTrophy[] }>(
      `/public/clubs/${encodeURIComponent(site.slug)}/trophies`,
    ).catch(() => null),
    site.document.page_visibility?.tournaments === "private" ? null : publicSiteFetch<{ highlights: TournamentGoldHighlight[] }>(
      `/public/clubs/${encodeURIComponent(site.slug)}/tournament-highlights`,
    ).catch(() => null),
  ]);
  return <ClubSiteContent document={site.document} slug={site.slug} trophies={awards?.trophies ?? []} tournamentHighlights={tournaments?.highlights ?? []} />;
}
