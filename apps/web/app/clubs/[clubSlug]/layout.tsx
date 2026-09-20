import { notFound } from "next/navigation";
import { requestShareMetadata } from "@/lib/shareMetadata";
import { getPublicSite } from "@/lib/clubSiteServer";
import { ClubDisplayProvider } from "@/components/ClubDisplay";
import RememberPublicClub from "@/components/RememberPublicClub";
import ClubSiteHeader from "@/components/ClubSiteHeader";
import { ClubPageNavigationProvider } from "@/components/PublicClubLink";
export async function generateMetadata({ params }: { params: { clubSlug: string } }) {
  return requestShareMetadata(`/clubs/${params.clubSlug}`);
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
