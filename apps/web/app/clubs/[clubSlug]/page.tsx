import { notFound } from "next/navigation";
import { getPublicSite, publicSiteFetch } from "@/lib/clubSiteServer";
import type { TournamentGoldHighlight } from "@/lib/tournamentHighlights";
import ClubSiteContent from "@/components/ClubSiteContent";
export default async function ClubHome({ params }: { params: { clubSlug: string } }) {
  const site = await getPublicSite(params.clubSlug);
  if (!site) notFound();
  const tournaments = site.document.page_visibility?.tournaments === "private" ? null : await publicSiteFetch<{ highlights: TournamentGoldHighlight[] }>(
    `/public/clubs/${encodeURIComponent(site.slug)}/tournament-highlights`,
  ).catch(() => null);
  return <ClubSiteContent document={site.document} slug={site.slug} tournamentHighlights={tournaments?.highlights ?? []} />;
}
