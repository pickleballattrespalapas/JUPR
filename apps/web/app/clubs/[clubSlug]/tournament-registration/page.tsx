import Link from "next/link";
import PublicTournamentNav from "@/components/PublicTournamentNav";
import { getClubTournamentRegistration } from "@/lib/tournamentRegistrationApi";
import TournamentRegistrationForm from "./TournamentRegistrationForm";
import EditLinkRequestForm from "./EditLinkRequestForm";

type TournamentRegistrationPageProps = {
  params: { clubSlug: string };
  searchParams?: { tournament?: string; tournament_id?: string };
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

function dateLabel(value?: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  return date.toISOString().slice(0, 10);
}

export default async function TournamentRegistrationPage({
  params,
  searchParams
}: TournamentRegistrationPageProps) {
  const { clubSlug } = params;
  const { data, error } = await getClubTournamentRegistration(clubSlug, {
    registrationSlug: searchParams?.tournament ?? null,
    tournamentId: searchParams?.tournament_id ?? null
  });

  const tournament = data?.tournament;
  const settings = data?.settings;
  const selectableCount =
    data?.events?.filter((event) => event.selectable).length ?? 0;
  const tournamentQuery = settings?.registration_slug
    ? `tournament=${encodeURIComponent(settings.registration_slug)}`
    : tournament?.id
      ? `tournament_id=${encodeURIComponent(tournament.id)}`
      : "";
  const venueMapQuery = settings?.venue_address || settings?.location_name || "";

  return (
    <section>
      <p
        style={{
          margin: "0 0 0.5rem",
          color: "#2563eb",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          fontSize: "0.78rem"
        }}
      >
        Tournament Registration
      </p>
      <h1 style={{ marginTop: 0 }}>
        {tournament?.name ?? "Tournament registration"}
      </h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Choose your divisions and submit your entry. Tournament details,
        policies, roster, and partner information remain available from
        Tournament Home.
      </p>

      {error ? (
        <p style={{ color: "#b91c1c" }}>
          Tournament registration is temporarily unavailable. {error}
        </p>
      ) : null}
      {data?.setup_error ? (
        <p style={{ color: "#b91c1c" }}>{data.setup_error}</p>
      ) : null}
      {!error && data && !tournament ? (
        <p>No tournament registration is currently published.</p>
      ) : null}

      {data?.tournaments?.length ? (
        <div
          style={{
            display: "flex",
            gap: "0.5rem",
            flexWrap: "wrap",
            marginBottom: "1rem"
          }}
        >
          {data.tournaments.map((choice) => {
            const slug = choice.settings.registration_slug;
            const active = choice.tournament.id === tournament?.id;
            const href = slug
              ? `/clubs/${clubSlug}/tournament-registration?tournament=${encodeURIComponent(slug)}`
              : `/clubs/${clubSlug}/tournament-registration?tournament_id=${encodeURIComponent(choice.tournament.id)}`;
            return (
              <Link
                key={choice.tournament.id}
                href={href}
                style={{
                  border: "1px solid #cbd5e1",
                  borderRadius: "999px",
                  padding: "0.45rem 0.75rem",
                  background: active ? "#dbeafe" : "white",
                  color: "#0f172a",
                  textDecoration: "none",
                  fontWeight: active ? 800 : 600
                }}
              >
                {choice.tournament.name}
              </Link>
            );
          })}
        </div>
      ) : null}

      {tournament ? (
        <PublicTournamentNav
          clubSlug={clubSlug}
          tournamentName={tournament.name}
          tournamentId={tournament.id}
          registrationSlug={settings?.registration_slug ?? null}
          active="registration"
        />
      ) : null}

      {tournament ? (
        <article
          style={{
            ...cardStyle,
            marginBottom: "1rem",
            display: "flex",
            flexWrap: "wrap",
            gap: "1rem",
            alignItems: "center",
            justifyContent: "space-between",
            background: data?.registration_open ? "#eff6ff" : "#f8fafc",
            borderColor: data?.registration_open ? "#93c5fd" : "#cbd5e1"
          }}
        >
          <div>
            <h2 style={{ margin: 0 }}>
              {data?.registration_open ? "Ready to register?" : "Registration is closed"}
            </h2>
            <p style={{ color: "#475569", margin: "0.35rem 0 0" }}>
              {data?.registration_open
                ? `Complete the form below for ${selectableCount} open division${selectableCount === 1 ? "" : "s"}.`
                : data?.registration_closed_reason || "Registration is not currently open."}
            </p>
          </div>
          {data?.registration_open ? (
            <a
              href="#registration-form"
              style={{
                display: "inline-block",
                padding: "0.7rem 1rem",
                borderRadius: "999px",
                background: "#0f172a",
                color: "white",
                textDecoration: "none",
                fontWeight: 800
              }}
            >
              Register now
            </a>
          ) : null}
        </article>
      ) : null}

      {tournament ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
            gap: "0.75rem",
            marginBottom: "1rem"
          }}
        >
          <article style={cardStyle}>
            <strong>Start</strong>
            <br />
            {dateLabel(tournament.start_date) ?? "TBD"}
          </article>
          <article style={cardStyle}>
            <strong>Status</strong>
            <br />
            {settings?.registration_status ?? "draft"}
          </article>
          <article style={cardStyle}>
            <strong>Open divisions</strong>
            <br />
            {selectableCount}
          </article>
        </div>
      ) : null}

      {tournament && venueMapQuery ? (
        <article style={{ ...cardStyle, marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0 }}>Venue</h2>
          {settings?.location_name ? <p><strong>{settings.location_name}</strong></p> : null}
          {settings?.venue_address ? <p>{settings.venue_address}</p> : null}
          {settings?.timezone ? <p style={{ color: "#475569" }}>Tournament time zone: {settings.timezone}</p> : null}
          <p>
            <a
              href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(venueMapQuery)}`}
              target="_blank"
              rel="noreferrer"
            >
              Open map
            </a>
          </p>
          {settings?.venue_directions ? (
            <div>
              <h3>Arrival directions</h3>
              <p style={{ whiteSpace: "pre-wrap" }}>{settings.venue_directions}</p>
            </div>
          ) : null}
        </article>
      ) : null}

      {tournament && !data?.registration_open ? (
        <article style={{ ...cardStyle, marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0 }}>Registration is closed</h2>
          <p style={{ color: "#475569" }}>
            {data?.registration_closed_reason ||
              "Registration is not currently open."}{" "}
            Existing registrants can still request a secure edit link below.
          </p>
          <Link href="/support">Contact support</Link>
        </article>
      ) : null}

      {tournament ? (
        <div id="registration-form" style={{ scrollMarginTop: "1rem" }}>
          <TournamentRegistrationForm
            clubSlug={clubSlug}
            tournamentId={tournament.id}
            registrationSlug={settings?.registration_slug ?? null}
            registrationOpen={Boolean(data?.registration_open)}
            registrationClosedReason={data?.registration_closed_reason ?? null}
            days={data.days ?? []}
            events={data.events ?? []}
            commerce={data.commerce ?? null}
          />
        </div>
      ) : null}

      {tournament ? (
        <article
          id="manage-registration"
          style={{ ...cardStyle, marginBottom: "1rem", scrollMarginTop: "1rem" }}
        >
          <h2 style={{ marginTop: 0 }}>Manage an existing registration</h2>
          <p style={{ color: "#475569" }}>
            Request a secure link to edit an event, change partner details, or
            add another event. For privacy, the response is the same whether or
            not a matching registration exists.
          </p>
          <EditLinkRequestForm
            clubSlug={clubSlug}
            tournamentId={tournament.id}
            registrationSlug={settings?.registration_slug ?? null}
          />
        </article>
      ) : null}

      {tournament ? (
        <article
          style={{
            ...cardStyle,
            marginBottom: "1rem",
            display: "flex",
            flexWrap: "wrap",
            gap: "0.75rem",
            alignItems: "center",
            justifyContent: "space-between"
          }}
        >
          <div>
            <h2 style={{ margin: 0 }}>Need tournament details?</h2>
            <p style={{ color: "#475569", margin: "0.35rem 0 0" }}>
              Review the venue, rules, policies, public roster, and the Players
              Needing Partners page without losing your place in registration.
            </p>
          </div>
          <span style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <Link
              href={`/clubs/${clubSlug}/tournaments${tournamentQuery ? `?${tournamentQuery}` : ""}`}
              style={{ fontWeight: 800 }}
            >
              Tournament Home
            </Link>
            <Link
              href={`/clubs/${clubSlug}/tournament-roster${tournamentQuery ? `?${tournamentQuery}` : ""}`}
              style={{ fontWeight: 800 }}
            >
              Tournament Roster
            </Link>
            <Link
              href={`/clubs/${clubSlug}/tournament-team-results`}
              style={{ fontWeight: 800 }}
            >
              Four-player Team Results
            </Link>
          </span>
        </article>
      ) : null}
    </section>
  );
}
