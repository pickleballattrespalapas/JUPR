import { notFound } from "next/navigation";
import { getPublicSite, publicSiteFetch } from "@/lib/clubSiteServer";
import type { SeasonTrophy } from "@/lib/interclubAwards";
import ClubSiteContent from "@/components/ClubSiteContent";
export default async function ClubHome({ params }: { params: { clubSlug: string } }) {
  const site = await getPublicSite(params.clubSlug);
  if (!site) notFound();
  // Championships are optional; an unavailable trophy service must not hide the club website.
  const awards = site.document.page_visibility?.trophies === "private" ? null :
    await publicSiteFetch<{ trophies: SeasonTrophy[] }>(
      `/public/clubs/${encodeURIComponent(site.slug)}/trophies`,
    ).catch(() => null);
  return <ClubSiteContent document={site.document} slug={site.slug} trophies={awards?.trophies ?? []} />;
}
