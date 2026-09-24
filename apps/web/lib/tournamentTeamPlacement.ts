import type { TeamCompetitionSnapshot } from "./tournamentTeamCompetitionApi";

const inactive = new Set(["CANCELLED", "CANCELED", "WITHDRAWN", "REMOVED"]);

// An individual event selection is durable signup; roster membership records
// whether the organizer has placed that registrant on a team yet.
export function playersNeedingTeam(snapshot: TeamCompetitionSnapshot | null, eventId: string): Array<Record<string, unknown>> {
  if (!snapshot || !eventId) return [];
  const selected = new Set(snapshot.selections.filter(row =>
    String(row.event_option_id || "") === eventId && row.is_active !== false &&
    !inactive.has(String(row.status || "").toUpperCase())
  ).map(row => String(row.registration_id || "")));
  const teams = snapshot.teams.filter(team => team.event_option_id === eventId && !inactive.has(team.status.toUpperCase()));
  const teamIds = new Set(teams.map(team => team.id));
  const members = snapshot.members.filter(member => teamIds.has(member.team_id) && ["INVITED", "ACCEPTED"].includes(member.status.toUpperCase()));
  return snapshot.registrations.filter(row => {
    const id = String(row.id || "");
    const email = String(row.email || "").trim().toLowerCase();
    return selected.has(id) && ["CONFIRMED", "ADMIN_CONFIRMED"].includes(String(row.status || "").toUpperCase()) &&
      !teams.some(team => team.captain_registration_id === id) &&
      !members.some(member => member.registration_id
        ? member.registration_id === id
        : Boolean(email) && String(member.invited_email || "").trim().toLowerCase() === email);
  });
}
