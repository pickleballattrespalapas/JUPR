import type { PublicTournamentRosterEntry } from "./tournamentRegistrationApi";

/** Count registered people once per event, including across repeated roster rows. */
export function countIndividualEventEntries(entries: PublicTournamentRosterEntry[]): number {
  const keys = new Set<string>();
  entries.forEach((entry, rowIndex) => {
    entry.members.forEach((member, memberIndex) => {
      if (member.registration_pending) return;
      // Older API responses have no opaque member key. Keep those responses
      // usable without merging different players who happen to share a name.
      keys.add(member.event_entry_key || `legacy:${entry.public_entry_key || rowIndex}:${memberIndex}`);
    });
  });
  return keys.size;
}
