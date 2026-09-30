import type { PublicLeague } from "./interclubPublic";

export const interclubAwardLabels = {
  club_cup_champion: "League Champion", division_champion: "Division Champion", participation: "Season Participant",
  matchup_win: "Win Matchup", matchup_sweep: "Sweep Matchup", undefeated_meet: "Undefeated Day"
};
export type SeasonTrophy = {
  id: string; season_id: string; club_id: string; award_key: keyof typeof interclubAwardLabels;
  division: string; title: string; recipient_type: "club" | "player"; recipient_name: string;
  season_name: string; earned_at: string; results_href: string;
  recipient_key: string;
};
export type AwardRecipient = Pick<SeasonTrophy, "id" | "club_id" | "award_key" | "division" | "title" | "recipient_type" | "recipient_name"> & { recipient_key?: string; entry_id?: string | null };
export type SeasonAwardPreview = {
  season_id: string; preview: PublicLeague; awards: AwardRecipient[]; achievements?: AwardRecipient[]; problems: string[]; ready: boolean;
  revision: number; publication_revision: number; preview_fingerprint: string; current: boolean; issued_at: string | null;
};

export function recipientGroups(awards: AwardRecipient[]) {
  const groups = new Map<string, { id: string; name: string; club_id: string; titles: string[] }>();
  for (const award of awards.filter(row => row.recipient_type === "player")) {
    const key = award.recipient_key || award.entry_id || award.id;
    const group = groups.get(key) || { id: key, name: award.recipient_name, club_id: award.club_id, titles: [] };
    group.titles.push(award.title); groups.set(key, group);
  }
  return [...groups.values()].map(group => {
    const counts = new Map<string, number>();
    for (const title of group.titles) counts.set(title, (counts.get(title) || 0) + 1);
    return { ...group, titles: [...counts].map(([title, count]) => count > 1 ? `${title} × ${count}` : title) };
  }).sort((a, b) => a.name.localeCompare(b.name) || a.id.localeCompare(b.id));
}
