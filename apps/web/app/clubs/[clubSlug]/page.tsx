import { notFound } from "next/navigation";
import { getPublicSite } from "@/lib/clubSiteServer";
import ClubSiteContent from "@/components/ClubSiteContent";
export default async function ClubHome({ params }: { params: { clubSlug: string } }) {
  const site = await getPublicSite(params.clubSlug);
  if (!site) notFound();
  return <ClubSiteContent document={site.document} slug={site.slug}/>;
}
