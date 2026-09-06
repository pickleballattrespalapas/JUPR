import PublicTournamentSponsors from "@/components/PublicTournamentSponsors";
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
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeZone: "UTC"
  }).format(date);
}

function timeZoneLabel(value: string): string {
  const city = value.split("/").at(-1)?.replaceAll("_", " ") || "local";
  const labels: Record<string, string> = {
    Mazatlan: "Mazatlán",
    Cancun: "Cancún",
    "Mexico City": "Mexico City"
  };
  return labels[city] || city;
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
      <PublicTournamentSponsors clubSlug={clubSlug} tournamentId={tournament?.id} placement="header" title={tournament?.name ?? "Tournament registration"} />
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Choose your events and complete the form below. Need details? Visit
        Tournament Home anytime.
      </p>

      {error ? (
        <p style={{ color: "#b91c1c" }}>
          Tournament registration is temporarily unavailable. Please try again.
        </p>
      ) : null}
      {data?.setup_error ? (
        <p style={{ color: "#b91c1c" }}>
          Tournament registration is temporarily unavailable.
        </p>
      ) : null}
      {!error && data && !data.setup_error && !tournament ? (
        <p>There isn’t an open tournament registration right now.</p>
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
        <TournamentRegistrationForm
          clubSlug={clubSlug}
          tournamentId={tournament.id}
          registrationSlug={settings?.registration_slug ?? null}
          registrationOpen={Boolean(data?.registration_open)}
          registrationClosedReason={data?.registration_closed_reason ?? null}
          timeZone={settings?.timezone ?? null}
          days={data.days ?? []}
          events={data.events ?? []}
          commerce={data.commerce ?? null}
          overview={
            <>
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
                    {dateLabel(tournament.start_date) ?? "To be announced"}
                  </article>
                  <article style={cardStyle}>
                    <strong>Registration</strong>
                    <br />
                    {data?.registration_open ? "Open" : "Closed"}
                  </article>
                  <article style={cardStyle}>
                    <strong>Open divisions</strong>
                    <br />
                    {selectableCount}
                  </article>
                </div>

              {venueMapQuery ? (
                <article style={{ ...cardStyle, marginBottom: "1rem" }}>
                  <h2 style={{ marginTop: 0 }}>Venue</h2>
                  {settings?.location_name ? <p><strong>{settings.location_name}</strong></p> : null}
                  {settings?.venue_address ? <p>{settings.venue_address}</p> : null}
                  {settings?.timezone ? (
                    <p style={{ color: "#475569" }}>
                      All times are shown in {timeZoneLabel(settings.timezone)} time.
                    </p>
                  ) : null}
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
            </>
          }
        />
      ) : null}

      {tournament ? (
        <article
          id="manage-registration"
          style={{ ...cardStyle, marginBottom: "1rem", scrollMarginTop: "1rem" }}
        >
          <h2 style={{ marginTop: 0 }}>Manage an existing registration</h2>
          <p style={{ color: "#475569" }}>
            Enter the email you registered with and we’ll send you a private
            edit link.
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
              Need the venue, rules, or player list? Visit the tournament pages
              before you submit.
            </p>
          </div>
          <span style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <Link
              href={`/clubs/${clubSlug}/tournaments${tournamentQuery ? `?${tournamentQuery}` : ""}`}
              style={{ fontWeight: 800 }}
            >
              Tournament home
            </Link>
            <Link
              href={`/clubs/${clubSlug}/tournament-roster${tournamentQuery ? `?${tournamentQuery}` : ""}`}
              style={{ fontWeight: 800 }}
            >
              View roster
            </Link>
            <Link
              href={`/clubs/${clubSlug}/tournament-team-results`}
              style={{ fontWeight: 800 }}
            >
              Team results
            </Link>
          </span>
        </article>
      ) : null}
      <PublicTournamentSponsors clubSlug={clubSlug} tournamentId={tournament?.id} placement="footer" />
    </section>
  );
}
