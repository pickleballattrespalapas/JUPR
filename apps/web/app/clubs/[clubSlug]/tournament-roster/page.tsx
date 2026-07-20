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

function queryFor({ tournamentId, registrationSlug, day, event, division, status }: { tournamentId?: string | null; registrationSlug?: string | null; day?: string | null; event?: string | null; division?: string | null; status?: string | null }): string {
  const query = new URLSearchParams();
  if (registrationSlug) query.set("tournament", registrationSlug);
  else if (tournamentId) query.set("tournament_id", tournamentId);
  if (day) query.set("day", day);
  if (event) query.set("event", event);
  if (division) query.set("division", division);
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

function partnerBoardEventKey(entry: { event_day_label?: string | null; event_family?: string | null; division?: string | null }): string {
  return slugify([entry.event_day_label || "Day", entry.event_family || "Event", entry.division || "Division"].join(" · "));
}

function statusKey(entry: PublicTournamentRosterEntry): string {
  const status = String(entry.status || "").trim().toLowerCase();
  if (status === "registered") return "Registered";
  if (status === "waitlist") return "Waitlist";
  if (status === "needs partner") return "Needs Partner";
  if (status === "pending partner request") return "Pending Partner Request";
  return "Review";
}

function entryAnchor(entry: PublicTournamentRosterEntry, index: number): string {
  const source = entry.public_entry_key || String(index + 1);
  return `entry-${slugify(groupKey(entry))}-${slugify(source)}`;
}

function orderedChoices(values: Array<string | null | undefined>): string[] {
  return Array.from(new Set(values.map((value) => String(value || "").trim()).filter(Boolean))).sort((a, b) => a.localeCompare(b));
}

function dateTimeLabel(value?: string | null): string {
  if (!value) return "Not scheduled";
  const normalized = String(value).trim();
  const parsed = new Date(normalized);
  if (Number.isNaN(parsed.getTime())) return normalized.slice(0, 24);
  return parsed.toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

export default async function TournamentRosterPage({ params, searchParams }: TournamentRosterPageProps) {
  const { clubSlug } = params;
  const selectedDay = firstParam(searchParams, "day");
  const selectedEvent = firstParam(searchParams, "event");
  const selectedDivision = firstParam(searchParams, "division");
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
  const dayChoices = orderedChoices(entries.map((entry) => entry.event_day_label));
  const eventChoices = orderedChoices(entries.map((entry) => entry.event_family));
  const divisionChoices = orderedChoices(entries.map((entry) => entry.division));
  const statusOrder = ["Registered", "Needs Partner", "Pending Partner Request", "Waitlist", "Review"];
  const availableStatuses = new Set(entries.map(statusKey));
  const statusChoices = statusOrder.filter((status) => availableStatuses.has(status));
  const filteredEntries = entries.filter((entry) => {
    const dayOk = !selectedDay || entry.event_day_label === selectedDay;
    const eventOk = !selectedEvent || entry.event_family === selectedEvent;
    const divisionOk = !selectedDivision || entry.division === selectedDivision;
    const statusOk = selectedStatus === "all" || statusKey(entry) === selectedStatus;
    return dayOk && eventOk && divisionOk && statusOk;
  });
  const grouped = filteredEntries.reduce<Record<string, PublicTournamentRosterEntry[]>>((acc, entry) => {
    const key = groupKey(entry);
    acc[key] = [...(acc[key] || []), entry];
    return acc;
  }, {});
  const baseQuery = queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug });
  const filteredQuery = queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug, day: selectedDay, event: selectedEvent, division: selectedDivision, status: selectedStatus });
  const registeredCount = entries.filter((entry) => statusKey(entry) === "Registered").length;
  const pendingCount = Math.max(0, entries.length - registeredCount);

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Roster
      </p>
      <h1 style={{ marginTop: 0 }}>{tournament?.name ?? "Tournament roster"}</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Public roster view for tournament registration. This is read-only and hides private contact fields; staff draw seeding and operations remain in admin tools.
      </p>

      {error ? (
        <article role="alert" style={{ ...cardStyle, borderColor: "#fecaca", background: "#fef2f2", marginBottom: "1rem" }}>
          <h2 style={{ marginTop: 0, fontSize: "1.05rem" }}>Tournament roster is temporarily unavailable</h2>
          <p style={{ color: "#7f1d1d" }}>No private or incomplete roster data has been shown. Please try again shortly.</p>
          <Link href={`/clubs/${clubSlug}/tournament-roster${baseQuery}`}>Retry roster</Link>
        </article>
      ) : null}
      {data?.setup_error ? <p role="alert" style={{ color: "#b91c1c" }}>Tournament roster is not configured for this club yet.</p> : null}
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
            <article style={cardStyle}><strong>Registration opens</strong><br />{dateTimeLabel(settings?.registration_open_at)}</article>
            <article style={cardStyle}><strong>Registration closes</strong><br />{dateTimeLabel(settings?.registration_close_at)}</article>
            <article style={cardStyle}><strong>Registrations</strong><br />{data?.summary?.total_registrations ?? 0}</article>
            <article style={cardStyle}><strong>Players</strong><br />{data?.summary?.total_players ?? 0}</article>
            <article style={cardStyle}><strong>Needs partner</strong><br />{data?.summary?.players_needing_partners ?? 0}</article>
            <article style={cardStyle}><strong>Registered entries</strong><br />{registeredCount}</article>
            <article style={cardStyle}><strong>Other statuses</strong><br />{pendingCount}</article>
          </div>

          <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
            <Link href={`/clubs/${clubSlug}/tournament-registration${baseQuery}`}>Open registration</Link>
            <Link href={`/clubs/${clubSlug}/tournament-partner-board${baseQuery}`}>Partner board</Link>
          </p>
        </>
      ) : null}

      {entries.length ? (
        <form method="get" action={`/clubs/${clubSlug}/tournament-roster`} aria-label="Tournament roster filters" style={{ ...cardStyle, display: "grid", gap: "0.75rem", marginBottom: "1rem" }}>
          {settings?.registration_slug ? <input type="hidden" name="tournament" value={settings.registration_slug} /> : tournament?.id ? <input type="hidden" name="tournament_id" value={tournament.id} /> : null}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.75rem" }}>
            <label>Day<br /><select name="day" defaultValue={selectedDay ?? ""} style={{ width: "100%", padding: "0.5rem" }}><option value="">All days</option>{dayChoices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}</select></label>
            <label>Event<br /><select name="event" defaultValue={selectedEvent ?? ""} style={{ width: "100%", padding: "0.5rem" }}><option value="">All events</option>{eventChoices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}</select></label>
            <label>Division<br /><select name="division" defaultValue={selectedDivision ?? ""} style={{ width: "100%", padding: "0.5rem" }}><option value="">All divisions</option>{divisionChoices.map((choice) => <option key={choice} value={choice}>{choice}</option>)}</select></label>
            <label>Status<br /><select name="status" defaultValue={selectedStatus} style={{ width: "100%", padding: "0.5rem" }}><option value="all">All statuses</option>{statusChoices.map((status) => <option key={status} value={status}>{status}</option>)}</select></label>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexWrap: "wrap" }}>
            <button type="submit" style={{ border: "1px solid #0f172a", borderRadius: "999px", padding: "0.5rem 0.85rem", background: "#0f172a", color: "white", fontWeight: 800 }}>Apply filters</button>
            <Link href={`/clubs/${clubSlug}/tournament-roster${baseQuery}`}>Clear filters</Link>
            <span aria-live="polite" style={{ color: "#475569" }}>Showing {filteredEntries.length} of {entries.length} public entries.</span>
          </div>
        </form>
      ) : null}

      {roster?.players_needing_partners?.length ? (
        <article style={{ ...cardStyle, marginBottom: "1rem", background: "#fffbeb", borderColor: "#fde68a" }}>
          <h2 style={{ marginTop: 0 }}>Players looking for partners</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "0.75rem" }}>
            {roster.players_needing_partners.map((entry, index) => {
              const boardAnchor = entry.board_entry_key ? `partner-${slugify(entry.board_entry_key)}` : "";
              const boardQuery = queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug, event: partnerBoardEventKey(entry) });
              return (
              <div key={entry.board_entry_key || `${entry.player_name}-${index}`} style={{ border: "1px solid #fde68a", borderRadius: "12px", padding: "0.75rem", background: "white" }}>
                <strong>{entry.player_name}</strong>
                <div style={{ color: "#92400e" }}>{entry.event_day_label} · {entry.event_family} · {entry.division}</div>
                {entry.skill ? <div style={{ color: "#475569" }}>Skill {entry.skill}</div> : null}
                {entry.age_bracket ? <div style={{ color: "#475569" }}>Age {entry.age_bracket}</div> : null}
                {entry.note ? <p style={{ marginBottom: 0, color: "#475569" }}>{entry.note}</p> : null}
                <p style={{ marginBottom: 0 }}><Link href={`/clubs/${clubSlug}/tournament-partner-board${boardQuery}${boardAnchor ? `#${boardAnchor}` : ""}`}>View on partner board</Link></p>
              </div>
              );
            })}
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
                        <th scope="col" style={thStyle}>Entry</th>
                        <th scope="col" style={thStyle}>Players</th>
                        <th scope="col" style={thStyle}>Status</th>
                        <th scope="col" style={thStyle}>Link</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((entry, index) => {
                        const anchor = entryAnchor(entry, index);
                        const players = entry.members.map(memberLabel).join(" / ");
                        return (
                          <tr key={entry.public_entry_key || `${entry.event_label}-${index}`} id={anchor}>
                            <td style={tdStyle}>{entry.entry_type || "Registration"}</td>
                            <td style={tdStyle}>{players}</td>
                            <td style={tdStyle}>{statusKey(entry)}</td>
                            <td style={tdStyle}><Link aria-label={`Open ${players} roster entry`} href={`/clubs/${clubSlug}/tournament-roster${filteredQuery}#${anchor}`}>row link</Link></td>
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
