import { getPublicSite, getPublicSiteShell } from "@/lib/clubSiteServer";
import { publicClubPage } from "./clubSite";
import type { SiteDocument, SiteShellDocument } from "./clubSite";
import { plainText, record, type ReadPublic, type SharingClub } from "./shareMetadataCore";

// Staging uses ONLY the published public website document, never its draft.
export async function loadSharingClub(slug: string, path: string, _read: ReadPublic): Promise<SharingClub> {
  const needsContent = path.includes("/pages/") || path.replace(/\/$/, "") === `/clubs/${slug}`;
  const site = await (needsContent ? getPublicSite(slug) : getPublicSiteShell(slug)).catch(() => null);
  const doc = record(site?.document);
  if (!site || site.slug !== slug || !doc.name) return { name: "Club unavailable", slug, available: false };
  const document = doc as unknown as SiteShellDocument;
  const pages = needsContent && Array.isArray(document.pages) ? (doc as unknown as SiteDocument).pages : [];
  let pageSlug = '';
  try { pageSlug = decodeURIComponent(path.split('/pages/')[1] || ''); } catch { /* Invalid slug. */ }
  const page = pageSlug ? pages.find((p) => p.slug === pageSlug) : undefined;
  return {
    name: document.name, slug, description: document.description, location: document.location,
    image: document.logo_url, accent: document.accent,
    discoverable: document.visibility !== 'unlisted' && Boolean(publicClubPage(document, slug, path)),
    ...(page ? { page: {
      title: page.title,
      description: page.blocks.filter((b) => b.kind === 'text').map((b) => plainText(`${b.heading} ${b.text}`)).join(' '),
      image: page.blocks.find((b) => b.kind === 'image' && b.url)?.url
    } } : {})
  };
}
