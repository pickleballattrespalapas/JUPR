import { notFound } from "next/navigation";
import type { PublicFourPlayerTeamSetupRecovery } from "@/lib/tournamentTeamCompetitionApi";
import TeamInvitationReview from "../clubs/[clubSlug]/tournament-team-invitation/TeamInvitationReview";
import TournamentRegistrationForm from "../clubs/[clubSlug]/tournament-registration/TournamentRegistrationForm";
import FourPlayerTeamSetupRecovery from "../clubs/[clubSlug]/tournament-registration/confirmation/FourPlayerTeamSetupRecovery";

// Fixture data only, enabled by the same explicit local/CI gate as the
// interaction harness. Never expose a test registration page in production.
export const dynamic = "force-dynamic";

export default async function TournamentSignupHarness(props: { searchParams: Promise<{ confirmation?: string; invitation?: string }> }) {
  if (process.env.JUPR_INTERACTION_TEST_HARNESS !== "1") notFound();
  const searchParams = await props.searchParams;
  if (searchParams.invitation) return <TeamInvitationReview clubSlug="fixture" />;
  const event = {
    id: "team-event", registration_day_id: "day1", label: "Mixed 5.0",
    event_family_label: "Team Tournament", division_name: "Mixed 5.0",
    event_type: "MIXED_DOUBLES", gender_restriction: "MIXED", skill_mode: "OPEN",
    competition_format: "FOUR_PLAYER_TEAM" as const, team_roster_size: 4,
    partner_required: true, selectable: true, price_usd: 40,
  };
  const team: NonNullable<PublicFourPlayerTeamSetupRecovery["events"][number]["team"]> | null =
    searchParams.confirmation === "partial" ? {
      id: "team1", name: "Fixture Team", status: "FORMING", is_captain: true, open_slots: ["WOMAN_2"],
      members: [
        { member_id: "captain", slot: "MAN_1", display_name: "Fixture Player", status: "ACCEPTED" },
        { member_id: "teammate1", slot: "MAN_2", display_name: "Teammate One", status: "INVITED" },
        { member_id: "teammate2", slot: "WOMAN_1", display_name: "Teammate Two", status: "INVITED" },
      ],
    } : null;
  return <main style={{ maxWidth: 850, margin: "auto", padding: "1rem" }}>
    <h1>Team tournament signup fixture</h1>
    {searchParams.confirmation ? <FourPlayerTeamSetupRecovery
      clubSlug="fixture" confirmationToken="fixture-only"
      initialRecovery={{ ok: true, club: {}, tournament: { id: "t1", name: "Team tournament" },
        captain: { registration_id: "solo", display_name: "Fixture Player", email: "fixture@example.invalid", gender: "Men", registration_status: "CONFIRMED" },
        events: [{ ...event, setup_state: team ? "COMPLETE" : "NEEDS_TEAM", team }],
      }}
    /> : <TournamentRegistrationForm clubSlug="fixture" tournamentId="t1" registrationSlug="fixture-tournament" registrationOpen
      days={[{ id: "day1", label: "Tournament day" }]} events={[event]} />}
  </main>;
}
