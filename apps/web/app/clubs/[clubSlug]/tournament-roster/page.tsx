import Link from "next/link";
import { getClubTournamentRoster } from "@/lib/tournamentRegistrationApi";
import type { PublicTournamentRosterEntry, PublicTournamentRosterMember } from "@/lib/tournamentRegistrationApi";

type TournamentRosterPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

const thStyle = { textAlign: "left" as const, padding: "0.55rem", borderBottom: "1px solid #cbd5e1", color: "#475569", fontSize: "0.82rem", whiteSpace: "nowrap" as const };
const tdStyle = { padding: "0.55rem", borderBottom: "1px solid #e2e8f0", verticalAlign: "top" as const };

function firstParam(searchParams: TournamentRosterPageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "event";
}

function dateLabel(value?: string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value).slice(0, 10);
  return date.toISOString().slice(0, 10);
}

function queryFor({ tournamentId, registrationSlug, event, status }: { tournamentId?: string | null; registrationSlug?: string | null; event?: string | null; status?: string | null }): string {
  const query = new URLSearchParams();
  if (registrationSlug) query.set("tournament", registrationSlug);
  else if (tournamentId) query.set("tournament_id", tournamentId);
  if (event) query.set("event", event);
  if (status && status !== "all") query.set("status", status);
  const text = query.toString();
  return text ? `?${text}` : "";
}

function memberLabel(member: PublicTournamentRosterMember): string {
  const extras = [member.skill ? `Skill ${member.skill}` : null, member.age_bracket || null].filter(Boolean).join(" · ");
  return extras ? `${member.display_name} (${extras})` : member.display_name;
}

function groupKey(entry: PublicTournamentRosterEntry): string {
  return [entry.event_day_label || "Day", entry.event_family || "Event", entry.division || "Division"].join(" · ");
}

function eventKey(entry: PublicTournamentRosterEntry): string {
  return slugify(groupKey(entry));
}

function statusKey(entry: PublicTournamentRosterEntry): string {
  return String(entry.status || "Confirmed").trim() || "Confirmed";
}

function entryAnchor(entry: PublicTournamentRosterEntry, index: number): string {
  const source = entry.source_selection_ids?.join("-") || entry.source_registration_ids?.join("-") || String(index + 1);
  return `entry-${slugify(groupKey(entry))}-${slugify(source)}`;
}

export default async function TournamentRosterPage({ params, searchParams }: TournamentRosterPageProps) {
  const { clubSlug } = params;
  const selectedEvent = firstParam(searchParams, "event");
  const selectedStatus = firstParam(searchParams, "status") ?? "all";
  const registrationSlug = firstParam(searchParams, "tournament");
  const tournamentId = firstParam(searchParams, "tournament_id");
  const { data, error } = await getClubTournamentRoster(clubSlug, {
    registrationSlug,
    tournamentId
  });

  const tournament = data?.tournament;
  const settings = data?.settings;
  const roster = data?.roster;
  const entries = roster?.registrations_by_event ?? [];
  const eventChoices = Array.from(new Map(entries.map((entry) => [eventKey(entry), groupKey(entry)])).entries()).sort((a, b) => a[1].localeCompare(b[1]));
  const statusChoices = Array.from(new Set(entries.map(statusKey))).sort((a, b) => a.localeCompare(b));
  const filteredEntries = entries.filter((entry) => {
    const eventOk = !selectedEvent || eventKey(entry) === selectedEvent;
    const statusOk = selectedStatus === "all" || statusKey(entry) === selectedStatus;
    return eventOk && statusOk;
  });
  const grouped = filteredEntries.reduce<Record<string, PublicTournamentRosterEntry[]>>((acc, entry) => {
    const key = groupKey(entry);
    acc[key] = [...(acc[key] || []), entry];
    return acc;
  }, {});
  const baseQuery = queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug });
  const filteredQuery = queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug, event: selectedEvent, status: selectedStatus });
  const confirmedCount = entries.filter((entry) => statusKey(entry).toLowerCase() === "confirmed").length;
  const pendingCount = Math.max(0, entries.length - confirmedCount);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Roster
      </p>
      <h1 style={{ marginTop: 0 }}>{tournament?.name ?? "Tournament roster"}</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Public roster view for tournament registration. This is read-only and hides private contact fields; staff draw seeding and operations remain in admin tools.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Tournament roster is temporarily unavailable. {error}</p> : null}
      {data?.setup_error ? <p style={{ color: "#b91c1c" }}>{data.setup_error}</p> : null}
      {!error && data && !tournament ? <p>{data.empty_reason || "No tournament roster is currently published."}</p> : null}

      {data?.tournaments?.length ? (
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          {data.tournaments.map((choice) => {
            const slug = choice.settings.registration_slug;
            const active = choice.tournament.id === tournament?.id;
            const href = slug ? `/clubs/${clubSlug}/tournament-roster?tournament=${encodeURIComponent(slug)}` : `/clubs/${clubSlug}/tournament-roster?tournament_id=${encodeURIComponent(choice.tournament.id)}`;
            return (
              <Link key={choice.tournament.id} href={href} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                {choice.tournament.name}
              </Link>
            );
          })}
        </div>
      ) : null}

      {tournament ? (
        <>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
            <article style={cardStyle}><strong>Start</strong><br />{dateLabel(tournament.start_date) ?? "TBD"}</article>
            <article style={cardStyle}><strong>Registrations</strong><br />{data?.summary?.total_registrations ?? 0}</article>
            <article style={cardStyle}><strong>Players</strong><br />{data?.summary?.total_players ?? 0}</article>
            <article style={cardStyle}><strong>Needs partner</strong><br />{data?.summary?.players_needing_partners ?? 0}</article>
            <article style={cardStyle}><strong>Confirmed entries</strong><br />{confirmedCount}</article>
            <article style={cardStyle}><strong>Other statuses</strong><br />{pendingCount}</article>
          </div>

          <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <Link href={`/clubs/${clubSlug}/tournament-registration${baseQuery}`}>Open registration</Link>
            <Link href={`/clubs/${clubSlug}/tournament-partner-board${filteredQuery}`}>Partner board</Link>
          </p>
        </>
      ) : null}

      {entries.length ? (
        <div style={{ display: "grid", gap: "0.75rem", marginBottom: "1rem" }}>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <Link href={`/clubs/${clubSlug}/tournament-roster${queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug, status: selectedStatus })}`} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: !selectedEvent ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: !selectedEvent ? 800 : 600 }}>All events</Link>
            {eventChoices.map(([key, label]) => {
              const active = key === selectedEvent;
              return (
                <Link key={key} href={`/clubs/${clubSlug}/tournament-roster${queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug, event: key, status: selectedStatus })}#event-${key}`} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                  {label}
                </Link>
              );
            })}
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <Link href={`/clubs/${clubSlug}/tournament-roster${queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug, event: selectedEvent, status: "all" })}`} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: selectedStatus === "all" ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: selectedStatus === "all" ? 800 : 600 }}>All statuses</Link>
            {statusChoices.map((status) => {
              const active = status === selectedStatus;
              return (
                <Link key={status} href={`/clubs/${clubSlug}/tournament-roster${queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug, event: selectedEvent, status })}`} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: active ? "#dcfce7" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                  {status}
                </Link>
              );
            })}
          </div>
        </div>
      ) : null}

      {roster?.players_needing_partners?.length ? (
        <article style={{ ...cardStyle, marginBottom: "1rem", background: "#fffbeb", borderColor: "#fde68a" }}>
          <h2 style={{ marginTop: 0 }}>Players looking for partners</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
            {roster.players_needing_partners.map((entry) => (
              <div key={`${entry.selection_id}-${entry.player_name}`} style={{ border: "1px solid #fde68a", borderRadius: "12px", padding: "0.75rem", background: "white" }}>
                <strong>{entry.player_name}</strong>
                <div style={{ color: "#92400e" }}>{entry.event_family} · {entry.division}</div>
                {entry.note ? <p style={{ marginBottom: 0, color: "#475569" }}>{entry.note}</p> : null}
              </div>
            ))}
          </div>
        </article>
      ) : null}

      {filteredEntries.length ? (
        <div style={{ display: "grid", gap: "1rem" }}>
          {Object.entries(grouped).map(([label, rows]) => {
            const key = slugify(label);
            return (
              <article key={label} id={`event-${key}`} style={cardStyle}>
                <h2 style={{ marginTop: 0 }}>{label}</h2>
                <p style={{ color: "#64748b" }}>{rows.length} public entr{rows.length === 1 ? "y" : "ies"} in this filtered roster view.</p>
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "680px" }}>
                    <thead>
                      <tr>
                        <th style={thStyle}>Entry</th>
                        <th style={thStyle}>Players</th>
                        <th style={thStyle}>Status</th>
                        <th style={thStyle}>Link</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((entry, index) => {
                        const anchor = entryAnchor(entry, index);
                        return (
                          <tr key={`${entry.event_label}-${index}-${entry.source_selection_ids?.join("-")}`} id={anchor}>
                            <td style={tdStyle}>{entry.entry_type || "Registration"}</td>
                            <td style={tdStyle}>{entry.members.map(memberLabel).join(" / ")}</td>
                            <td style={tdStyle}>{statusKey(entry)}</td>
                            <td style={tdStyle}><Link href={`/clubs/${clubSlug}/tournament-roster${filteredQuery}#${anchor}`}>row link</Link></td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </article>
            );
          })}
        </div>
      ) : tournament && entries.length ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>No matching roster entries</h2>
          <p style={{ color: "#475569" }}>Try clearing the event or status filter.</p>
        </article>
      ) : tournament ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>No roster entries yet</h2>
          <p style={{ color: "#475569" }}>Registrations will appear here after players submit tournament entries.</p>
        </article>
      ) : null}
    </section>
  );
}
