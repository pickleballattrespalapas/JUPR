export type SponsorTier = "presenting" | "premier" | "supporting";
export type TournamentSponsor = {
  id: string;
  name: string;
  tier: SponsorTier;
  level: string;
  website: string;
  public_description?: string;
  logo_url?: string;
  sort_order?: number;
};
export type SponsorDraft = TournamentSponsor & { notes: string; logo_path: string; is_visible: boolean };
export const sponsorTiers: SponsorTier[] = ["presenting", "premier", "supporting"];
export const sponsorTierLabels: Record<SponsorTier, string> = { presenting: "Premier / Presenting", premier: "Supporting sponsors", supporting: "Community sponsors" };
export function sponsorPlacement(tier: SponsorTier): string {
  return tier === "presenting" ? "Alongside the tournament title on every tournament page." : "At the bottom of every tournament page.";
}
export function normalizeSponsorWebsite(value: string): string {
  let text = value.trim();
  if (!text) return "";
  if (/\s|\\/.test(text)) throw new Error("Enter a valid sponsor website.");
  if (!/^[a-z][a-z0-9+.-]*:/i.test(text)) text = `https://${text}`;
  try {
    const url = new URL(text);
    if (!["http:", "https:"].includes(url.protocol) || !url.hostname.includes(".") || url.username || url.password) throw new Error();
    return url.href;
  } catch { throw new Error("Enter an HTTP or HTTPS sponsor website."); }
}
