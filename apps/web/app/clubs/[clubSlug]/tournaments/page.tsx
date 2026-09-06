import PublicTournamentSponsors from "@/components/PublicTournamentSponsors";
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
  const family = String(event.event_family_label || "").trim();
  const division = String(event.division_name || "").trim();
  if (family && division.toLocaleLowerCase().startsWith(family.toLocaleLowerCase())) {
    return division;
  }
  return [family, division].filter(Boolean).join(" · ");
}

function priceLabel(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: Number.isInteger(value) ? 0 : 2,
    maximumFractionDigits: 2
  }).format(value);
}

function dayHeading(label: string, eventDate?: string | null): string {
  const formattedDate = dateLabel(eventDate);
  if (!eventDate) return label;
  const parsed = new Date(`${String(eventDate).slice(0, 10)}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return `${label} · ${formattedDate}`;
  const shortDate = new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC"
  }).format(parsed);
  const normalize = (value: string) =>
    value.toLocaleLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  return normalize(label).includes(normalize(shortDate))
    ? label
    : `${label} · ${formattedDate}`;
}

function registrationStatusLabel(isOpen?: boolean | null): string {
  if (isOpen === true) return "Registration open";
  if (isOpen === false) return "Registration closed";
  return "View registration";
}

function markdownish(text?: string | null) {
  if (!text) return null;
  return text
    .split("\n")
    .filter(Boolean)
    .map((line, index) => (
      <p key={`${index}:${line}`} style={{ margin: "0 0 0.5rem", color: "#475569" }}>
        {line.replace(/^#+\s*/, "")}
      </p>
    ));
}

export default async function PublicTournamentsPage({
  params,
  searchParams
}: Props) {
  const registrationSlug = firstParam(searchParams, "tournament");
  const tournamentId = firstParam(searchParams, "tournament_id");
  const explicitSelection = Boolean(registrationSlug || tournamentId);
  const { data, error, status } = await getClubTournamentRegistration(params.clubSlug, {
    registrationSlug,
    tournamentId
  });

  const apiTournament = data?.tournament ?? null;
  const settings = data?.settings;
  const selectionMatches = Boolean(
    explicitSelection &&
      apiTournament &&
      (!tournamentId || apiTournament.id === tournamentId) &&
      (!registrationSlug || settings?.registration_slug === registrationSlug)
  );
  const tournament = selectionMatches ? apiTournament : null;
  const currentId = tournament?.id || null;
  const currentSlug = tournament ? settings?.registration_slug || null : null;
  const selectableEvents = tournament
    ? (data?.events || []).filter((event) => event.selectable)
    : [];
  const eventsByDay = tournament
    ? (data?.days || []).map((day) => ({
        day,
        events: (data?.events || []).filter(
          (event) =>
            (event.scheduled_day_ids?.length
              ? event.scheduled_day_ids
              : [event.registration_day_id]
            ).includes(day.id)
        )
      }))
    : [];

  if (!explicitSelection) {
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
        <h1 style={{ marginTop: 0 }}>Choose a tournament</h1>
        <p style={{ color: "#334155", maxWidth: "820px" }}>
          Choose a tournament to register, see who&apos;s playing, find a partner,
          or follow results.
        </p>
        <p>
          <Link
            href={`/clubs/${params.clubSlug}/tournaments/past`}
            style={{ fontWeight: 800 }}
          >
            Past tournaments
          </Link>
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
            <p style={{ color: "#7f1d1d" }}>Please try again shortly.</p>
          </article>
        ) : null}

        {data?.tournaments?.length ? (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
              gap: "0.75rem"
            }}
          >
            {data.tournaments.map((choice) => (
              <article
                key={choice.tournament.id}
                style={{
                  ...cardStyle,
                  color: "#0f172a",
                  textDecoration: "none"
                }}
              >
                <PublicTournamentSponsors
                  clubSlug={params.clubSlug}
                  tournamentId={choice.tournament.id}
                  placement="header"
                  title={choice.tournament.name}
                  titleHref={tournamentHref(params.clubSlug, choice)}
                  headingLevel="h2"
                  compact
                />
                <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
                  {dateLabel(choice.tournament.start_date)}
                  {choice.tournament.end_date
                    ? ` – ${dateLabel(choice.tournament.end_date)}`
                    : ""}
                </p>
                <p style={{ margin: "0.35rem 0 0", color: "#64748b" }}>
                  {registrationStatusLabel(choice.registration_open)}
                </p>
                <Link href={tournamentHref(params.clubSlug, choice)} style={{ display: "inline-block", marginTop: "0.75rem", fontWeight: 700 }}>
                  View tournament
                </Link>
              </article>
            ))}
          </div>
        ) : !error ? (
          <article style={cardStyle}>
            <h2 style={{ marginTop: 0 }}>No tournaments yet</h2>
            <p style={{ color: "#475569" }}>
              This club hasn&apos;t published any tournaments yet.
            </p>
          </article>
        ) : null}
      </section>
    );
  }

  if (!tournament) {
    const unavailable = Boolean(error && status !== 404);
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
        <h1 style={{ marginTop: 0 }}>
          {unavailable ? "Tournament unavailable" : "Tournament not found"}
        </h1>
        <p style={{ color: "#334155" }}>
          {unavailable
            ? "We couldn’t load this tournament right now. Please try again shortly."
            : "We couldn’t find that tournament. It may no longer be public."}
        </p>
        <p>
          <Link href={`/clubs/${params.clubSlug}/tournaments`}>
            Return to tournament selection
          </Link>
        </p>
      </section>
    );
  }

  return (
    <section>
      <p style={{ margin: "0 0 0.75rem" }}>
        <Link href={`/clubs/${params.clubSlug}/tournaments`}>
          ← Choose another tournament
        </Link>
      </p>
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
        Tournament Home
      </p>
      <PublicTournamentSponsors clubSlug={params.clubSlug} tournamentId={tournament.id} placement="header" title={tournament.name} />
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Everything you need to register, find a partner, and follow the
        tournament.
      </p>

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
            <strong>Tournament dates</strong>
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
            {registrationStatusLabel(data?.registration_open)}
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
        <h2>What would you like to do?</h2>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "0.75rem"
          }}
        >
          {[
            {
              label: "Register",
              description: data?.registration_open
                ? "Sign up for your events and add any extras."
                : "Registration is not currently open.",
              cta: "Register",
              module: "registration" as const
            },
            {
              label: "Roster",
              description: "See who's playing in each event and division.",
              cta: "View roster",
              module: "roster" as const
            },
            {
              label: "Players Needing Partners",
              description: "Find a partner for your doubles event.",
              cta: "Find a partner",
              module: "partner-board" as const
            },
            {
              label: "Live & Results",
              description: "See live scores, standings, brackets, and medal winners.",
              cta: "View results",
              module: "results" as const
            }
          ].map((item) => (
            <article key={item.module} style={cardStyle}>
              <h3 style={{ marginTop: 0 }}>{item.label}</h3>
              <p style={{ color: "#475569" }}>{item.description}</p>
              <Link
                href={publicTournamentHref(
                  params.clubSlug,
                  item.module,
                  currentId,
                  currentSlug
                )}
                style={{ fontWeight: 800 }}
              >
                {item.cta}
              </Link>
            </article>
          ))}
        </div>
      </section>

      <section style={{ marginBottom: "1.25rem" }}>
        <h2>Schedule and events</h2>
        <div style={{ display: "grid", gap: "0.75rem" }}>
          {eventsByDay.map(({ day, events }) => (
            <article key={day.id} style={cardStyle}>
              <h3 style={{ marginTop: 0 }}>
                {dayHeading(day.label, day.event_date)}
              </h3>
              {events.length ? (
                <ul style={{ marginBottom: 0 }}>
                  {events.map((event) => (
                    <li key={event.id}>
                      {eventLabel(event)}
                      {event.price_usd != null
                        ? ` · ${priceLabel(Number(event.price_usd))}`
                        : ""}
                    </li>
                  ))}
                </ul>
              ) : (
                <p style={{ marginBottom: 0, color: "#64748b" }}>
                  No events are listed for this day yet.
                </p>
              )}
            </article>
          ))}
        </div>
      </section>

      {(settings?.location_name ||
        settings?.venue_address ||
        settings?.venue_directions ||
        settings?.sponsor_markdown ||
        settings?.rules_markdown ||
        settings?.refund_policy_markdown ||
        settings?.weather_policy_markdown) ? (
        <section style={{ marginBottom: "1.25rem" }}>
          <h2>Tournament information</h2>
          <div style={{ display: "grid", gap: "0.75rem" }}>
            {settings?.location_name || settings?.venue_address || settings?.venue_directions ? (
              <article style={cardStyle}>
                <h3 style={{ marginTop: 0 }}>Venue</h3>
                {settings?.location_name ? (
                  <p style={{ marginBottom: "0.35rem", fontWeight: 800 }}>
                    {settings.location_name}
                  </p>
                ) : null}
                {settings?.venue_address ? (
                  <address style={{ fontStyle: "normal", color: "#334155" }}>
                    {settings.venue_address}
                  </address>
                ) : null}
                {settings?.venue_address ? (
                  <p style={{ marginBottom: settings?.venue_directions ? "0.75rem" : 0 }}>
                    <a
                      href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(settings.venue_address)}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Open map
                    </a>
                  </p>
                ) : null}
                {settings?.venue_directions ? (
                  <div>
                    <strong>Arrival directions</strong>
                    {markdownish(settings.venue_directions)}
                  </div>
                ) : null}
              </article>
            ) : null}
            {settings?.sponsor_markdown ? (
              <article style={{ ...cardStyle, background: "#fffbeb", borderColor: "#fde68a" }}>
                <h3 style={{ marginTop: 0 }}>Tournament sponsors</h3>
                {markdownish(settings.sponsor_markdown)}
              </article>
            ) : null}
            {settings?.rules_markdown ? (
              <article style={cardStyle}>
                <h3 style={{ marginTop: 0 }}>Rules and registration notes</h3>
                {markdownish(settings.rules_markdown)}
              </article>
            ) : null}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
                gap: "0.75rem"
              }}
            >
              {settings?.refund_policy_markdown ? (
                <article style={cardStyle}>
                  <h3 style={{ marginTop: 0 }}>Refund policy</h3>
                  {markdownish(settings.refund_policy_markdown)}
                </article>
              ) : null}
              {settings?.weather_policy_markdown ? (
                <article style={cardStyle}>
                  <h3 style={{ marginTop: 0 }}>Weather policy</h3>
                  {markdownish(settings.weather_policy_markdown)}
                </article>
              ) : null}
            </div>
          </div>
        </section>
      ) : null}

      <p style={{ marginTop: "1rem" }}>
        <Link
          href={publicTournamentHref(
            params.clubSlug,
            "results",
            currentId,
            currentSlug
          )}
        >
          Singles &amp; doubles results
        </Link>
        {" · "}
        <Link href={`/clubs/${params.clubSlug}/tournament-team-results`}>
          Team results
        </Link>
      </p>
      <PublicTournamentSponsors clubSlug={params.clubSlug} tournamentId={tournament.id} placement="footer" />
    </section>
  );
}
