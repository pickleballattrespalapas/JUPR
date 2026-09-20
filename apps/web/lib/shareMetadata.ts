import type { Metadata } from "next";
import { headers } from "next/headers";
import { loadSharingClub } from "./shareClub";
import { dateRange, plainText, privateShareLink, publicShareQuery, record, resolveClubShare, type PageShare, type ReadPublic } from "./shareMetadataCore";

export function sharingOrigin(value?: string | null): string {
  try {
    const url = new URL(value || "");
    if ((url.protocol === "https:" && /(^|\.)(pickleballclubsandwich\.com|juprleagues\.com|vercel\.app)$/.test(url.hostname)) ||
      (process.env.NODE_ENV === "development" && ["localhost", "127.0.0.1"].includes(url.hostname))) return url.origin;
  } catch { /* Use an environment-specific fallback, never an arbitrary host. */ }
  return process.env.NEXT_PUBLIC_JUPR_ENV === "staging"
    ? "https://jupr-git-staging-pickleballattrespalapas1.vercel.app"
    : "https://pickleballclubsandwich.com";
}
export const readSharingPublic: ReadPublic = async (path: string) => {
  const base = process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL;
  if (!base) return null;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 3500);
  try {
    // Only callers' fixed, encoded PUBLIC API paths are accepted. No browser credentials.
    if (!/^\/(clubs|public)\//.test(path) || path.includes('\\')) return null;
    const response = await fetch(`${base.replace(/\/$/, "")}${path}`, { method: "GET", cache: "no-store", signal: controller.signal });
    return response.ok ? record(await response.json()) : null;
  } catch { return null; }
  finally { clearTimeout(timeout); }
};

export async function resolvePageShare(path: string, rawQuery: string, privateLink = false): Promise<PageShare> {
  const query = new URLSearchParams(rawQuery);
  let parts: string[] = [];
  try { parts = path.split('/').filter(Boolean).map(decodeURIComponent); } catch { path = '/'; }
  if (!path.startsWith('/') || path.startsWith('//') || /[?#\\\u0000-\u001f]/.test(path)) path = '/';
  if (parts[0] === "clubs" && /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(parts[1] || "")) {
    const club = await loadSharingClub(parts[1], path, readSharingPublic);
    return resolveClubShare({ path, query, origin: sharingOrigin(), privateLink }, club, readSharingPublic);
  }
  const product = "Pickleball Club Sandwich";
  const labels: Record<string, [string, string]> = {
    clubs: ["Find a Pickleball Club", "Discover pickleball clubs, leagues, tournaments, and local play."],
    faq: ["Frequently Asked Questions", "Find answers about club play, results, and pickleball ratings."],
    "how-ratings-work": ["How Ratings Work", "Learn how club pickleball ratings reflect recorded match performance."],
    support: ["Contact & Support", "Get help with pickleball club pages, events, and player information."],
    privacy: ["Privacy Policy", "Read how personal information is handled."],
    terms: ["Terms of Use", "Read the terms for using Pickleball Club Sandwich."],
    "data-corrections": ["Request a Data Correction", "Find out how to correct player information and match results."],
    "site-map": ["Site Map", "Find club pages, events, play tools, and help."],
    interclub: ["Interclub Pickleball", "Follow interclub pickleball leagues, meets, standings, and results."],
  };
  const label = labels[parts[0]];
  const base: PageShare = {
    title: label ? `${label[0]} | ${product}` : parts.length ? `${plainText(parts[0].replace(/[-_]/g, ' '), 70)} | ${product}` : product,
    description: label?.[1] || "Find your pickleball club, follow events and results, and stay connected to local play.",
    siteName: product, path, query: publicShareQuery(query).toString(), noindex: false
  };
  if (privateLink || privateShareLink(path, query) || parts[0]?.startsWith('__')) return { ...base, title: `Private page | ${product}`, description: "A private account or staff link.", query: "", path: "/", noindex: true };
  if (parts[0] === "interclub" && /^[0-9a-f-]{36}$/i.test(parts[1] || "")) {
    const result = await readSharingPublic(`/public/interclub/${encodeURIComponent(parts[1])}`);
    const doc = record(result?.document);
    if (!doc.name) return { ...base, title: "Interclub league unavailable", noindex: true };
    const clubs = Array.isArray(doc.clubs) ? doc.clubs.map((c) => plainText(record(c).name, 80)).filter(Boolean).join(', ') : '';
    return { ...base, title: plainText(doc.name, 170), siteName: plainText(doc.name, 110), description: plainText(`${dateRange(doc.start_date, doc.end_date)} Interclub pickleball${clubs ? ` with ${clubs}` : ''}. Follow meet schedules, standings, and results.`), noindex: true };
  }
  return base;
}

export function sharingMetadata(share: PageShare, origin: string): Metadata {
  const url = new URL(share.path, origin);
  url.search = share.query;
  const imageUrl = new URL('/share-image', origin);
  imageUrl.searchParams.set('path', share.path);
  for (const [key, value] of new URLSearchParams(share.query)) imageUrl.searchParams.set(key, value);
  // Give changed text a new image identity, without putting any personal data in its URL.
  let revision = 0;
  for (const char of `${share.title}\n${share.description}`) revision = (Math.imul(revision, 31) + char.charCodeAt(0)) | 0;
  imageUrl.searchParams.set('v', (revision >>> 0).toString(36));
  const image = share.image || imageUrl.href;
  const images = [{ url: image, alt: share.title, ...(!share.image ? { width: 1200, height: 630, type: 'image/png' } : {}) }];
  const noindex = share.noindex;
  return {
    title: { absolute: share.title }, description: share.description, alternates: { canonical: url.href },
    robots: { index: !noindex, follow: !noindex },
    openGraph: { type: 'website', title: share.title, description: share.description, siteName: share.siteName, url: url.href, images },
    twitter: { card: 'summary_large_image', title: share.title, description: share.description, images: [image] }
  };
}
export async function requestShareMetadata(fallbackPath = '/', root = false): Promise<Metadata> {
  const h = headers();
  const path = h.get('x-pcs-club-path') || fallbackPath;
  // The club layout owns club metadata; avoid a duplicate metadata/API traversal.
  if (root && /^\/clubs\/[^/]+/.test(path)) return {};
  const share = await resolvePageShare(path, h.get('x-pcs-share-query') || '', h.get('x-pcs-share-private') === '1');
  return sharingMetadata(share, sharingOrigin(h.get('x-pcs-share-origin')));
}
