import PublicTournamentNav from "@/components/PublicTournamentNav";
import PublicTournamentSponsors from "@/components/PublicTournamentSponsors";
import { getClubTournamentRegistration } from "@/lib/tournamentRegistrationApi";
import EditLinkRequestForm from "../EditLinkRequestForm";

type Props = {
  params: { clubSlug: string };
  searchParams?: { tournament?: string; tournament_id?: string };
};

export default async function ManageTournamentRegistrationPage({ params, searchParams }: Props) {
  const { clubSlug } = params;
  const { data, error } = await getClubTournamentRegistration(clubSlug, {
    registrationSlug: searchParams?.tournament ?? null,
    tournamentId: searchParams?.tournament_id ?? null
  });
  const tournament = data?.tournament;
  return (
    <section>
      <PublicTournamentSponsors clubSlug={clubSlug} tournamentId={tournament?.id} placement="header" title={tournament?.name ?? "Edit my registration"} />
      <PublicTournamentNav clubSlug={clubSlug} tournamentId={tournament?.id ?? searchParams?.tournament_id} registrationSlug={data?.settings?.registration_slug ?? searchParams?.tournament} active="edit-registration" />
      <section style={{ border: "1px solid #bfdbfe", borderRadius: "14px", padding: "1.25rem", background: "white", maxWidth: "640px" }}>
        <h2 style={{ marginTop: 0 }}>Edit my registration</h2>
        {error || data?.setup_error ? (
          <p role="alert">Registration editing is temporarily unavailable. Please try again.</p>
        ) : tournament ? (
          <>
            <p>Enter the email address you registered with. We’ll email you a link to change your events, add a partner, or update your details.</p>
            <EditLinkRequestForm clubSlug={clubSlug} tournamentId={tournament.id} registrationSlug={data?.settings?.registration_slug} />
          </>
        ) : <p>We couldn’t find that tournament registration.</p>}
      </section>
      <PublicTournamentSponsors clubSlug={clubSlug} tournamentId={tournament?.id} placement="footer" />
    </section>
  );
}
