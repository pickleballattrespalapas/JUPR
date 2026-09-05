import Link from "next/link";
import { redirect } from "next/navigation";
import PublicTournamentModuleHeader from "@/components/PublicTournamentModuleHeader";
import {
  getClubTournamentRoster,
  type PublicTournamentRosterEntry,
  type PublicTournamentRosterMember
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

function dateTimeLabel(value?: string | null): string {
  if (!value) return "To be announced";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value).slice(0, 24);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: "UTC",
    timeZoneName: "short"
  }).format(parsed);
}

function selectedQuery(
  tournamentId?: string | null,
  registrationSlug?: string | null
): string {
  const query = new URLSearchParams();
  if (registrationSlug) query.set("tournament", registrationSlug);
  else if (tournamentId) query.set("tournament_id", tournamentId);
  return query.toString();
}

function orderedChoices(values: Array<string | null | undefined>): string[] {
  return Array.from(
    new Set(values.map((value) => String(value || "").trim()).filter(Boolean))
  ).sort((a, b) => a.localeCompare(b));
}

function groupLabel(entry: PublicTournamentRosterEntry): string {
  const family = String(entry.event_family || "Event").trim();
  const division = String(entry.division || "Division").trim();
  const event = division.toLocaleLowerCase().startsWith(family.toLocaleLowerCase())
    ? division
    : `${family} · ${division}`;
  return `${entry.event_day_label || "Day"} · ${event}`;
}

function statusLabel(entry: PublicTournamentRosterEntry): string {
  const value = String(entry.status || "").trim().toLowerCase();
  if (value === "registered") return "Registered";
  if (value === "waitlist") return "Waitlisted";
  if (value === "needs partner") return "Needs a partner";
  if (value === "pending partner request") return "Partner request pending";
  return "Under review";
}

function statusStyle(status: string) {
  if (status === "Registered") {
    return { background: "#dcfce7", borderColor: "#86efac", color: "#166534" };
  }
  if (status === "Needs a partner" || status === "Partner request pending") {
    return { background: "#fef3c7", borderColor: "#fde68a", color: "#92400e" };
  }
  if (status === "Waitlisted") {
    return { background: "#dbeafe", borderColor: "#93c5fd", color: "#1d4ed8" };
  }
  return { background: "#f1f5f9", borderColor: "#cbd5e1", color: "#475569" };
}

function memberLabel(member: PublicTournamentRosterMember): string {
  const details = [
    member.skill ? `Rating ${member.skill}` : null,
    member.age_bracket || null
  ].filter(Boolean);
  return details.length
    ? `${member.display_name} · ${details.join(" · ")}`
    : member.display_name;
}

export default async function TournamentRosterPage({
  params,
  searchParams
}: Props) {
  const registrationSlug = firstParam(searchParams, "tournament");
  const tournamentId = firstParam(searchParams, "tournament_id");
  if (!registrationSlug && !tournamentId) {
    redirect(`/clubs/${params.clubSlug}/tournaments`);
  }

  const selectedDay = firstParam(searchParams, "day");
  const selectedEvent = firstParam(searchParams, "event");
  const selectedDivision = firstParam(searchParams, "division");
  const selectedStatus = firstParam(searchParams, "status") || "all";
  const { data, error, status } = await getClubTournamentRoster(params.clubSlug, {
    registrationSlug,
    tournamentId
  });

  const tournament = data?.tournament || null;
  const settings = data?.settings || null;
  const selectionMatches = Boolean(
    tournament &&
      (!tournamentId || tournament.id === tournamentId) &&
      (!registrationSlug || settings?.registration_slug === registrationSlug)
  );

  if (!selectionMatches || !tournament) {
    const unavailable = Boolean(error && status !== 404);
    return (
      <section>
        <h1>{unavailable ? "Tournament roster unavailable" : "Tournament not found"}</h1>
        <p style={{ color: "#475569" }}>
          {unavailable
            ? "We couldn’t load the roster right now. Please try again shortly."
            : "We couldn’t find that tournament. It may no longer be public."}
        </p>
        <Link href={`/clubs/${params.clubSlug}/tournaments`}>
          Return to tournament selection
        </Link>
      </section>
    );
  }

  const roster = data?.roster;
  const entries = roster?.registrations_by_event || [];
  const filteredEntries = entries.filter((entry) => {
    const dayOk = !selectedDay || entry.event_day_label === selectedDay;
    const eventOk = !selectedEvent || entry.event_family === selectedEvent;
    const divisionOk = !selectedDivision || entry.division === selectedDivision;
    const statusOk = selectedStatus === "all" || statusLabel(entry) === selectedStatus;
    return dayOk && eventOk && divisionOk && statusOk;
  });
  const groupedEntries = filteredEntries.reduce<
    Record<string, PublicTournamentRosterEntry[]>
  >((groups, entry) => {
    const label = groupLabel(entry);
    groups[label] = [...(groups[label] || []), entry];
    return groups;
  }, {});

  const dayChoices = orderedChoices(entries.map((entry) => entry.event_day_label));
  const eventChoices = orderedChoices(entries.map((entry) => entry.event_family));
  const divisionChoices = orderedChoices(entries.map((entry) => entry.division));
  const statusChoices = orderedChoices(entries.map(statusLabel));
  const registeredEntries = entries.filter(
    (entry) => statusLabel(entry) === "Registered"
  ).length;
  const query = selectedQuery(tournament.id, settings?.registration_slug);
  const queryPrefix = query ? `?${query}` : "";

  return (
    <section>
      <PublicTournamentModuleHeader
        clubSlug={params.clubSlug}
        tournamentName={tournament.name}
        tournamentId={tournament.id}
        registrationSlug={settings?.registration_slug || null}
        active="roster"
        kicker="Tournament Roster"
        description="See who's playing in each event and division. Contact details are only visible to organizers."
      />

      {error ? (
        <article
          role="alert"
          style={{
            ...cardStyle,
            marginBottom: "1rem",
            borderColor: "#fecaca",
            background: "#fef2f2"
          }}
        >
          <h2 style={{ marginTop: 0 }}>Roster temporarily unavailable</h2>
          <p style={{ color: "#7f1d1d" }}>Please try again shortly.</p>
        </article>
      ) : null}

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
            <h2 style={{ marginTop: 0 }}>Roster overview</h2>
            <p style={{ marginBottom: 0, color: "#475569" }}>
              {dateLabel(tournament.start_date)} – {dateLabel(tournament.end_date)}
            </p>
          </div>
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))",
            gap: "0.75rem",
            marginTop: "1rem"
          }}
        >
          <div><strong>Sign-ups</strong><br />{data?.summary?.total_registrations ?? 0}</div>
          <div><strong>Players</strong><br />{data?.summary?.total_players ?? 0}</div>
          <div><strong>Confirmed entries</strong><br />{registeredEntries}</div>
          <div><strong>Looking for partners</strong><br />{data?.summary?.players_needing_partners ?? 0}</div>
          <div><strong>Registration deadline</strong><br />{dateTimeLabel(settings?.registration_close_at)}</div>
        </div>
      </article>

      {entries.length ? (
        <form
          method="get"
          action={`/clubs/${params.clubSlug}/tournament-roster`}
          aria-label="Tournament roster filters"
          style={{ ...cardStyle, display: "grid", gap: "0.75rem", marginBottom: "1rem" }}
        >
          {settings?.registration_slug ? (
            <input type="hidden" name="tournament" value={settings.registration_slug} />
          ) : (
            <input type="hidden" name="tournament_id" value={tournament.id} />
          )}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
              gap: "0.75rem"
            }}
          >
            <label>
              <strong>Day</strong><br />
              <select name="day" defaultValue={selectedDay || ""} style={{ width: "100%", padding: "0.55rem" }}>
                <option value="">All days</option>
                {dayChoices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}
              </select>
            </label>
            <label>
              <strong>Event</strong><br />
              <select name="event" defaultValue={selectedEvent || ""} style={{ width: "100%", padding: "0.55rem" }}>
                <option value="">All events</option>
                {eventChoices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}
              </select>
            </label>
            <label>
              <strong>Division</strong><br />
              <select name="division" defaultValue={selectedDivision || ""} style={{ width: "100%", padding: "0.55rem" }}>
                <option value="">All divisions</option>
                {divisionChoices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}
              </select>
            </label>
            <label>
              <strong>Status</strong><br />
              <select name="status" defaultValue={selectedStatus} style={{ width: "100%", padding: "0.55rem" }}>
                <option value="all">All statuses</option>
                {statusChoices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}
              </select>
            </label>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
            <button
              type="submit"
              style={{
                border: "1px solid #0f172a",
                borderRadius: "999px",
                padding: "0.5rem 0.85rem",
                background: "#0f172a",
                color: "white",
                fontWeight: 800
              }}
            >
              Apply filters
            </button>
            <Link href={`/clubs/${params.clubSlug}/tournament-roster${queryPrefix}`}>Clear filters</Link>
            <span aria-live="polite" style={{ color: "#475569" }}>
              Showing {filteredEntries.length} of {entries.length} entries.
            </span>
          </div>
        </form>
      ) : null}

      {filteredEntries.length ? (
        <div style={{ display: "grid", gap: "1rem" }}>
          {Object.entries(groupedEntries).map(([label, rows]) => (
            <article key={label} style={cardStyle}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  gap: "0.75rem",
                  flexWrap: "wrap",
                  alignItems: "center"
                }}
              >
                <h2 style={{ margin: 0 }}>{label}</h2>
                <span style={{ color: "#64748b" }}>
                  {rows.length === 1 ? "1 entry" : `${rows.length} entries`}
                </span>
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
                  gap: "0.75rem",
                  marginTop: "1rem"
                }}
              >
                {rows.map((entry, index) => {
                  const status = statusLabel(entry);
                  const tone = statusStyle(status);
                  return (
                    <div
                      key={entry.public_entry_key || `${label}-${index}`}
                      style={{
                        border: "1px solid #e2e8f0",
                        borderRadius: "12px",
                        padding: "0.8rem",
                        background: "#f8fafc"
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          gap: "0.5rem",
                          alignItems: "flex-start"
                        }}
                      >
                        <strong>{entry.entry_type || "Entry"}</strong>
                        <span
                          style={{
                            border: `1px solid ${tone.borderColor}`,
                            borderRadius: "999px",
                            padding: "0.15rem 0.45rem",
                            background: tone.background,
                            color: tone.color,
                            fontSize: "0.76rem",
                            fontWeight: 800
                          }}
                        >
                          {status}
                        </span>
                      </div>
                      <ul style={{ margin: "0.65rem 0 0", paddingLeft: "1.15rem" }}>
                        {(entry.members || []).map((member, memberIndex) => (
                          <li key={`${member.display_name}-${memberIndex}`}>
                            {memberLabel(member)}
                          </li>
                        ))}
                      </ul>
                    </div>
                  );
                })}
              </div>
            </article>
          ))}
        </div>
      ) : (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>
            {entries.length ? "No players match these filters" : "No players are listed yet"}
          </h2>
          <p style={{ color: "#475569" }}>
            {entries.length
              ? "Try changing or clearing your filters."
              : "Check back as players register."}
          </p>
          {entries.length ? (
            <Link href={`/clubs/${params.clubSlug}/tournament-roster${queryPrefix}`}>
              Clear filters
            </Link>
          ) : null}
        </article>
      )}
    </section>
  );
}
