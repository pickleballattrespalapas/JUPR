import { notFound } from "next/navigation";
import { headers } from "next/headers";
import type { Metadata } from "next";
import { getPublicSite } from "@/lib/clubSiteServer";
import { ClubDisplayProvider } from "@/components/ClubDisplay";
import RememberPublicClub from "@/components/RememberPublicClub";
import ClubSiteHeader from "@/components/ClubSiteHeader";
import { publicClubPage } from "@/lib/clubSite";
import { ClubPageNavigationProvider } from "@/components/PublicClubLink";
export async function generateMetadata({
  params,
}: {
  params: { clubSlug: string };
}): Promise<Metadata> {
  const site = await getPublicSite(params.clubSlug);
  if (!site)
    return {
      title: "Club unavailable",
      robots: { index: false, follow: false },
    };
  return {
    title: site.document.name,
    description: site.document.description.slice(0, 160),
    robots:
      site.document.visibility === "unlisted" ||
      !publicClubPage(
        site.document,
        site.slug,
        headers().get("x-pcs-club-path") || `/clubs/${site.slug}`,
      )
        ? { index: false, follow: false }
        : { index: true, follow: true },
  };
}
export default async function ClubLayout({
  params,
  children,
}: {
  params: { clubSlug: string };
  children: React.ReactNode;
}) {
  const site = await getPublicSite(params.clubSlug);
  if (!site) notFound();
  const doc = site.document;
  return (
    <ClubDisplayProvider display={doc.display}>
      <ClubPageNavigationProvider
        slug={site.slug}
        settings={{
          page_visibility: doc.page_visibility,
          pages: doc.pages.map(({ slug, in_navigation }) => ({ slug, in_navigation })),
        }}
      >
        <RememberPublicClub slug={site.slug} />
        <ClubSiteHeader document={doc} slug={site.slug} />
        {children}
      </ClubPageNavigationProvider>
    </ClubDisplayProvider>
  );
}
