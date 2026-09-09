export type PartnerListing = {
  player_name?: string | null;
  player_entry_key?: string | null;
  board_entry_key?: string | null;
  event_day_label?: string | null;
  event_family?: string | null;
  division?: string | null;
  event_label?: string | null;
  skill?: string | number | null;
  age_bracket?: string | null;
  note?: string | null;
};

export type PartnerPlayerGroup = {
  playerKey: string;
  playerName: string;
  entries: PartnerListing[];
};

export function groupPartnerNotes(
  entries: PartnerListing[]
): { note: string; entries: PartnerListing[] }[] {
  const notes = new Map<string, { note: string; entries: PartnerListing[] }>();
  for (const entry of entries) {
    const note = String(entry.note || "").replace(/\s+/g, " ").trim();
    if (!note) continue;
    const existing = notes.get(note);
    if (existing) existing.entries.push(entry);
    else notes.set(note, { note, entries: [entry] });
  }
  return Array.from(notes.values());
}

export function groupPartnerEntries(
  entries: PartnerListing[]
): PartnerPlayerGroup[] {
  const groups = new Map<string, PartnerPlayerGroup>();
  entries.forEach((entry, index) => {
    const playerKey = String(
      entry.player_entry_key || entry.board_entry_key || `listing-${index}`
    ).trim();
    const playerName = String(entry.player_name || "Player").trim() || "Player";
    const existing = groups.get(playerKey);
    if (existing) {
      existing.entries.push(entry);
      return;
    }
    groups.set(playerKey, { playerKey, playerName, entries: [entry] });
  });
  return Array.from(groups.values()).sort((a, b) =>
    a.playerName.localeCompare(b.playerName)
  );
}
