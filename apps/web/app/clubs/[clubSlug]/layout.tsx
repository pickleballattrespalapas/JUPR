import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { getPublicSite } from "@/lib/clubSiteServer";
import { ClubDisplayProvider } from "@/components/ClubDisplay";
import RememberPublicClub from "@/components/RememberPublicClub";
import ClubSiteHeader from "@/components/ClubSiteHeader";
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
      site.document.visibility === "unlisted"
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
      <RememberPublicClub slug={site.slug} />
      <ClubSiteHeader document={doc} slug={site.slug} />
      {children}
    </ClubDisplayProvider>
  );
}
