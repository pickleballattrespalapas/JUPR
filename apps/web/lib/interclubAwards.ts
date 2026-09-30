import type { PublicLeague } from "./interclubPublic";

export type SeasonTrophy = {
  id: string; season_id: string; club_id: string; award_key: "participation" | "division_champion" | "club_cup_champion";
  division: string; title: string; recipient_type: "club" | "player"; recipient_name: string;
  season_name: string; earned_at: string; results_href: string;
  recipient_key: string;
};
export type AwardRecipient = Pick<SeasonTrophy, "id" | "club_id" | "award_key" | "division" | "title" | "recipient_type" | "recipient_name"> & { recipient_key?: string; entry_id?: string | null };
export type SeasonAwardPreview = {
  season_id: string; preview: PublicLeague; awards: AwardRecipient[]; problems: string[]; ready: boolean;
  revision: number; publication_revision: number; preview_fingerprint: string; current: boolean; issued_at: string | null;
};

export function recipientGroups(awards: AwardRecipient[]) {
  const groups = new Map<string, { id: string; name: string; club_id: string; titles: string[] }>();
  for (const award of awards.filter(row => row.recipient_type === "player")) {
    const key = award.recipient_key || award.entry_id || award.id;
    const group = groups.get(key) || { id: key, name: award.recipient_name, club_id: award.club_id, titles: [] };
    group.titles.push(award.title); groups.set(key, group);
  }
  return [...groups.values()].sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
}
