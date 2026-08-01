import Link from "next/link";
import PublicTournamentNav, {
  publicTournamentHref
} from "@/components/PublicTournamentNav";
import {
  getClubTournamentRegistration,
  type PublicRegistrationEvent,
  type PublicTournamentChoice
} from "@/lib/tournamentRegistrationApi";

type Props = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white",
  minWidth: 0
};

function firstParam(
  searchParams: Props["searchParams"],
  key: string
): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function tournamentHref(clubSlug: string, choice: PublicTournamentChoice): string {
  const query = new URLSearchParams();
  if (choice.settings.registration_slug) {
    query.set("tournament", choice.settings.registration_slug);
  } else {
    query.set("tournament_id", choice.tournament.id);
  }
  return `/clubs/${clubSlug}/tournaments?${query.toString()}`;
}

function dateLabel(value?: string | null): string {
  if (!value) return "TBD";
  const parsed = new Date(`${String(value).slice(0, 10)}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return String(value).slice(0, 10);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC"
  }).format(parsed);
}

function eventLabel(event: PublicRegistrationEvent): string {
  return [event.event_family_label, event.division_name]
    .filter(Boolean)
    .join(" · ");
}

export default async function PublicTournamentsPage({
  params,
  searchParams
}: Props) {
  const registrationSlug = firstParam(searchParams, "tournament");
  const tournamentId = firstParam(searchParams, "tournament_id");
  const { data, error } = await getClubTournamentRegistration(params.clubSlug, {
    registrationSlug,
    tournamentId
  });

  const tournament = data?.tournament;
  const settings = data?.settings;
  const currentId = tournament?.id || null;
  const currentSlug = settings?.registration_slug || null;
  const selectableEvents = (data?.events || []).filter((event) => event.selectable);
  const eventsByDay = (data?.days || []).map((day) => ({
    day,
    events: (data?.events || []).filter(
      (event) => event.registration_day_id === day.id
    )
  }));

  return (
    <section>
      <p
        style={{
          margin: "0 0 0.5rem",
          color: "#2563eb",
          fontWeight: 800,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          fontSize: "0.78rem"
        }}
      >
        Tournaments
      </p>
      <h1 style={{ marginTop: 0 }}>Club tournaments</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Choose a tournament first, then use its Tournament Home, registration,
        roster, and Partner Board from one consistent public workspace.
      </p>

      {error ? (
        <article
          role="alert"
          style={{
            ...cardStyle,
            borderColor: "#fecaca",
            background: "#fef2f2",
            marginBottom: "1rem"
          }}
        >
          <h2 style={{ marginTop: 0 }}>Tournaments are temporarily unavailable</h2>
          <p style={{ color: "#7f1d1d" }}>{error}</p>
        </article>
      ) : null}

      {data?.tournaments?.length ? (
        <section style={{ marginBottom: "1.25rem" }}>
          <h2>Choose a tournament</h2>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: "0.75rem"
            }}
          >
            {data.tournaments.map((choice) => {
              const selected = choice.tournament.id === tournament?.id;
              return (
                <Link
                  key={choice.tournament.id}
                  href={tournamentHref(params.clubSlug, choice)}
                  aria-current={selected ? "page" : undefined}
                  style={{
                    ...cardStyle,
                    borderColor: selected ? "#60a5fa" : "#e2e8f0",
                    background: selected ? "#eff6ff" : "white",
                    color: "#0f172a",
                    textDecoration: "none"
                  }}
                >
                  <strong>{choice.tournament.name}</strong>
                  <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
                    {dateLabel(choice.tournament.start_date)}
                    {choice.tournament.end_date
                      ? ` – ${dateLabel(choice.tournament.end_date)}`
                      : ""}
                  </p>
                  <p style={{ margin: "0.35rem 0 0", color: "#64748b" }}>
                    Registration: {choice.settings.registration_status || "draft"}
                  </p>
                </Link>
              );
            })}
          </div>
        </section>
      ) : null}

      {!error && data && !tournament ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>No published tournament</h2>
          <p style={{ color: "#475569" }}>
            There is no public tournament workspace available for this club yet.
          </p>
        </article>
      ) : null}

      {tournament ? (
        <>
          <PublicTournamentNav
            clubSlug={params.clubSlug}
            tournamentName={tournament.name}
            tournamentId={currentId}
            registrationSlug={currentSlug}
            active="overview"
          />

          <article
            style={{
              ...cardStyle,
              marginBottom: "1rem",
              background: "#eff6ff",
              borderColor: "#bfdbfe"
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: "1rem",
                flexWrap: "wrap",
                alignItems: "flex-start"
              }}
            >
              <div>
                <h2 style={{ marginTop: 0 }}>{tournament.name}</h2>
                <p style={{ marginBottom: 0, color: "#475569" }}>
                  {dateLabel(tournament.start_date)} – {dateLabel(tournament.end_date)}
                </p>
              </div>
              <span
                style={{
                  border: "1px solid #93c5fd",
                  borderRadius: "999px",
                  padding: "0.25rem 0.6rem",
                  background: "white",
                  fontWeight: 800
                }}
              >
                {settings?.registration_status || "draft"}
              </span>
            </div>
          </article>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(175px, 1fr))",
              gap: "0.75rem",
              marginBottom: "1rem"
            }}
          >
            <article style={cardStyle}>
              <strong>Registration</strong>
              <br />
              {data?.registration_open ? "Open" : "Closed"}
            </article>
            <article style={cardStyle}>
              <strong>Open divisions</strong>
              <br />
              {selectableEvents.length}
            </article>
            <article style={cardStyle}>
              <strong>Registrations</strong>
              <br />
              {data?.roster_summary?.total_registrations ?? 0}
            </article>
            <article style={cardStyle}>
              <strong>Players needing partners</strong>
              <br />
              {data?.roster_summary?.players_needing_partners ?? 0}
            </article>
          </div>

          <section style={{ marginBottom: "1.25rem" }}>
            <h2>Tournament pages</h2>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
                gap: "0.75rem"
              }}
            >
              {[
                [
                  "Register",
                  data?.registration_open
                    ? "Choose divisions, submit registration, and manage extras."
                    : data?.registration_closed_reason ||
                      "Registration is not currently open.",
                  "registration" as const
                ],
                [
                  "Roster",
                  "Browse public-safe registrations by day, event, division, and status.",
                  "roster" as const
                ],
                [
                  "Partner Board",
                  "Find players who opted into public partner requests.",
                  "partner-board" as const
                ]
              ].map(([label, description, module]) => (
                <article key={module} style={cardStyle}>
                  <h3 style={{ marginTop: 0 }}>{label}</h3>
                  <p style={{ color: "#475569" }}>{description}</p>
                  <Link
                    href={publicTournamentHref(
                      params.clubSlug,
                      module,
                      currentId,
                      currentSlug
                    )}
                    style={{ fontWeight: 800 }}
                  >
                    Open {label}
                  </Link>
                </article>
              ))}
            </div>
          </section>

          <section>
            <h2>Events and days</h2>
            <div style={{ display: "grid", gap: "0.75rem" }}>
              {eventsByDay.map(({ day, events }) => (
                <article key={day.id} style={cardStyle}>
                  <h3 style={{ marginTop: 0 }}>
                    {day.label} · {dateLabel(day.event_date)}
                  </h3>
                  {events.length ? (
                    <ul style={{ marginBottom: 0 }}>
                      {events.map((event) => (
                        <li key={event.id}>
                          {eventLabel(event)}
                          {event.price_usd != null
                            ? ` · $${Number(event.price_usd).toFixed(2)}`
                            : ""}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p style={{ marginBottom: 0, color: "#64748b" }}>
                      No public events are assigned to this day.
                    </p>
                  )}
                </article>
              ))}
            </div>
          </section>

          <p style={{ marginTop: "1rem" }}>
            <Link href={`/clubs/${params.clubSlug}/tournament-team-results`}>
              View published four-player team results
            </Link>
          </p>
        </>
      ) : null}
    </section>
  );
}
