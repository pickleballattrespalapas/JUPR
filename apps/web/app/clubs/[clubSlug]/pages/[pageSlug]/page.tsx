import { notFound } from "next/navigation";
import { getPublicSite } from "@/lib/clubSiteServer";
import ClubSiteContent from "@/components/ClubSiteContent";
export default async function CustomClubPage({
  params,
}: {
  params: { clubSlug: string; pageSlug: string };
}) {
  const site = await getPublicSite(params.clubSlug);
  if (
    !site ||
    params.pageSlug === "home" ||
    !site.document.pages.some((p) => p.slug === params.pageSlug)
  )
    notFound();
  return (
    <ClubSiteContent
      document={site.document}
      slug={site.slug}
      pageSlug={params.pageSlug}
    />
  );
}
