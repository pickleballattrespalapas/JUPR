import { plainText, type ReadPublic, type SharingClub } from "./shareMetadataCore";

// Production's existing public club endpoint; no staging website features required.
export async function loadSharingClub(slug: string, _path: string, read: ReadPublic): Promise<SharingClub> {
  const club = await read(`/clubs/${encodeURIComponent(slug)}`);
  if (!club || club.slug !== slug || !club.name || club.is_active === false) return { name: "Club unavailable", slug, available: false };
  return {
    name: plainText(club.name, 110), slug,
    description: plainText(club.tagline), location: plainText(club.location),
    image: typeof club.logo_url === 'string' ? club.logo_url : undefined,
    accent: typeof club.primary_color === 'string' ? club.primary_color : undefined
  };
}
