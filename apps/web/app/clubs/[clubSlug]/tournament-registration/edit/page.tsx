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
  const { data, error } = editToken
    ? await getClubTournamentRegistrationEdit(clubSlug, {
        editToken,
        registrationSlug: searchParams?.tournament ?? null,
        tournamentId: searchParams?.tournament_id ?? null
      })
    : { data: null, error: "Missing edit token." };

  const tournament = data?.tournament;
  const settings = data?.settings;

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Edit Tournament Registration
      </p>
      <h1 style={{ marginTop: 0 }}>{tournament?.name ?? "Edit registration"}</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Securely update an existing tournament registration. The edit link must be valid, unexpired, and tied to the original registration email.
      </p>

      {error ? (
        <article style={{ ...cardStyle, background: "#fef2f2", borderColor: "#fecaca" }}>
          <h2 style={{ marginTop: 0 }}>Edit link unavailable</h2>
          <p style={{ color: "#991b1b" }}>{error}</p>
          <Link href={`/clubs/${clubSlug}/tournament-registration`}>Back to tournament registration</Link>
        </article>
      ) : null}

      {data?.setup_error ? <p style={{ color: "#b91c1c" }}>{data.setup_error}</p> : null}
      {data && !tournament ? <p>No tournament registration is currently published.</p> : null}

      {data && tournament && !data.registration_open ? (
        <article style={{ ...cardStyle, marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0 }}>Registration editing is closed</h2>
          <p style={{ color: "#475569" }}>{data.registration_closed_reason || "Registration is not currently open."}</p>
          <Link href={`/clubs/${clubSlug}/tournament-registration`}>Back to tournament registration</Link>
        </article>
      ) : null}

      {data && tournament && data.registration_open ? (
        <EditTournamentRegistrationForm
          clubSlug={clubSlug}
          tournamentId={tournament.id}
          registrationSlug={settings?.registration_slug ?? searchParams?.tournament ?? null}
          editToken={editToken}
          registration={data.registration}
          selections={data.selections ?? []}
          days={data.days ?? []}
          events={data.events ?? []}
          players={data.players ?? []}
        />
      ) : null}
    </section>
  );
}
