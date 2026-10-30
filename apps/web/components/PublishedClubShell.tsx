import { notFound } from "next/navigation";
import { getPublicSiteShell } from "@/lib/clubSiteServer";
import SectionLoadError from "@/components/SectionLoadError";
import { ClubDisplayProvider } from "@/components/ClubDisplay";
import RememberPublicClub from "@/components/RememberPublicClub";
import ClubSiteHeader from "@/components/ClubSiteHeader";
import { ClubPageNavigationProvider } from "@/components/PublicClubLink";

export default async function PublishedClubShell(props: { params: Promise<{ clubSlug: string }>; children: React.ReactNode }) {
  const params = await props.params;

  const {
    children
  } = props;

  let site;
  try {
    site = await getPublicSiteShell(params.clubSlug);
  } catch {
    // Never expose children using stale publication/display settings. This
    // expected outage has a local retry instead of crashing the entire route.
    return <SectionLoadError title="Club page temporarily unavailable" message="We couldn’t confirm this club’s current page settings. Try again to open the page." />;
  }
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
