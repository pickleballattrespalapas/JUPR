import PublicTournamentSponsors from "@/components/PublicTournamentSponsors";
import Link from "next/link";
import { getClubTournamentRegistrationEdit } from "@/lib/tournamentRegistrationApi";
import EditTournamentRegistrationForm from "./EditTournamentRegistrationForm";

type EditTournamentRegistrationPageProps = {
  params: { clubSlug: string };
  searchParams?: { edit_token?: string; tournament?: string; tournament_id?: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

export default async function EditTournamentRegistrationPage({ params, searchParams }: EditTournamentRegistrationPageProps) {
  const { clubSlug } = params;
  const editToken = searchParams?.edit_token || "";
  const { data, error, status } = editToken
    ? await getClubTournamentRegistrationEdit(clubSlug, {
        editToken,
        registrationSlug: searchParams?.tournament ?? null,
        tournamentId: searchParams?.tournament_id ?? null
      })
    : { data: null, error: "missing_edit_link", status: null };

  const tournament = data?.tournament;
  const settings = data?.settings;

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Edit Tournament Registration
      </p>
      <h1 style={{ marginTop: 0, marginBottom: 0 }}>{tournament?.name ?? "Edit registration"}</h1>
      <PublicTournamentSponsors clubSlug={clubSlug} tournamentId={tournament?.id} placement="header" />
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Use the link we emailed you to update your registration.
      </p>

      {error ? (
        <article style={{ ...cardStyle, background: "#fef2f2", borderColor: "#fecaca" }}>
          <h2 style={{ marginTop: 0 }}>Edit link unavailable</h2>
          <p style={{ color: "#991b1b" }}>
            {!editToken
              ? "This edit link is incomplete or no longer valid. Request a new one."
              : status === 409
                ? "This registration can’t be changed here. Contact the organizer for help."
                : status != null && status >= 500
                  ? "Registration changes are temporarily unavailable. Please try again later."
                  : "This edit link is invalid or expired. Request a new one."}
          </p>
          <Link href={`/clubs/${clubSlug}/tournament-registration`}>Back to tournament registration</Link>
        </article>
      ) : null}

      {data?.setup_error ? <p style={{ color: "#b91c1c" }}>Tournament registration is temporarily unavailable.</p> : null}
      {data && !tournament ? <p>We couldn’t find that tournament registration.</p> : null}

      {data && tournament && !data.registration_open ? (
        <article style={{ ...cardStyle, marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0 }}>Registration changes are closed</h2>
          <p style={{ color: "#475569" }}>{data.registration_closed_reason || "Registration is not currently open."}</p>
          <Link href={`/clubs/${clubSlug}/tournament-registration`}>Back to tournament registration</Link>
        </article>
      ) : null}

      {data && tournament && data.registration_open ? (
        <EditTournamentRegistrationForm
          key={`${editToken}:${data.registration.id}:${data.registration.updated_at}`}
          clubSlug={clubSlug}
          tournamentId={tournament.id}
          registrationSlug={settings?.registration_slug ?? searchParams?.tournament ?? null}
          timeZone={settings?.timezone ?? null}
          editToken={editToken}
          registration={data.registration}
          selections={data.selections ?? []}
          days={data.days ?? []}
          events={data.events ?? []}
          players={data.players ?? []}
          commerce={data.commerce ?? null}
          commerceOrder={data.commerce_order ?? null}
        />
      ) : null}
      <PublicTournamentSponsors clubSlug={clubSlug} tournamentId={tournament?.id} placement="footer" />
    </section>
  );
}
