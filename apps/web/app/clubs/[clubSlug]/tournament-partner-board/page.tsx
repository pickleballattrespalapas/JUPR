import Link from "next/link";
import { getClubTournamentRegistrationEdit, getClubTournamentRoster } from "@/lib/tournamentRegistrationApi";
import type { PublicTournamentNeedsPartnerEntry } from "@/lib/tournamentRegistrationApi";
import PairingInterestPanel from "./PairingInterestPanel";

type TournamentPartnerBoardPageProps = {
  params: { clubSlug: string };
  searchParams?: Record<string, string | string[] | undefined>;
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

function firstParam(searchParams: TournamentPartnerBoardPageProps["searchParams"], key: string): string | null {
  const value = searchParams?.[key];
  if (Array.isArray(value)) return value[0] ?? null;
  return value ?? null;
}

function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "event";
}

function entryEventLabel(entry: PublicTournamentNeedsPartnerEntry): string {
  return [entry.event_day_label || "Day", entry.event_family || "Event", entry.division || "Division"].join(" · ");
}

function entryEventKey(entry: PublicTournamentNeedsPartnerEntry): string {
  return slugify(entryEventLabel(entry));
}

function entryAnchor(entry: PublicTournamentNeedsPartnerEntry): string {
  return `partner-${slugify(`${entry.selection_id || entry.player_id || entry.player_name || "player"}`)}`;
}

function queryFor({ tournamentId, registrationSlug, editToken, event }: { tournamentId?: string | null; registrationSlug?: string | null; editToken?: string | null; event?: string | null }): string {
  const query = new URLSearchParams();
  if (registrationSlug) query.set("tournament", registrationSlug);
  else if (tournamentId) query.set("tournament_id", tournamentId);
  if (editToken) query.set("edit_token", editToken);
  if (event) query.set("event", event);
  const text = query.toString();
  return text ? `?${text}` : "";
}

export default async function TournamentPartnerBoardPage({ params, searchParams }: TournamentPartnerBoardPageProps) {
  const { clubSlug } = params;
  const editToken = firstParam(searchParams, "edit_token") || "";
  const selectedEvent = firstParam(searchParams, "event");
  const registrationSlug = firstParam(searchParams, "tournament");
  const tournamentId = firstParam(searchParams, "tournament_id");
  const [{ data, error }, editResponse] = await Promise.all([
    getClubTournamentRoster(clubSlug, {
      registrationSlug,
      tournamentId
    }),
    editToken
      ? getClubTournamentRegistrationEdit(clubSlug, {
          editToken,
          registrationSlug,
          tournamentId
        })
      : Promise.resolve({ data: null, error: null })
  ]);

  const tournament = data?.tournament;
  const settings = data?.settings;
  const partnerEntries = data?.roster?.players_needing_partners ?? [];
  const eventChoices = Array.from(new Map(partnerEntries.map((entry) => [entryEventKey(entry), entryEventLabel(entry)])).entries()).sort((a, b) => a[1].localeCompare(b[1]));
  const visibleEntries = selectedEvent ? partnerEntries.filter((entry) => entryEventKey(entry) === selectedEvent) : partnerEntries;
  const query = queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug });
  const queryWithEdit = queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug, editToken: editToken || null, event: selectedEvent });
  const apiBase = process.env.JUPR_API_BASE_URL || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || null;

  return (
    <section>
      <p style={{ margin: "0 0 0.5rem", color: "#2563eb", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", fontSize: "0.78rem" }}>
        Tournament Partner Board
      </p>
      <h1 style={{ marginTop: 0 }}>{tournament?.name ?? "Partner board"}</h1>
      <p style={{ color: "#334155", maxWidth: "820px" }}>
        Public partner board for players who registered as needing a partner. Contact details are not exposed; sending pairing interest requires your secure registration edit link.
      </p>

      {error ? <p style={{ color: "#b91c1c" }}>Partner board is temporarily unavailable. {error}</p> : null}
      {editResponse.error ? <p style={{ color: "#b91c1c" }}>Your edit link could not be verified for pairing actions. {editResponse.error}</p> : null}
      {data?.setup_error ? <p style={{ color: "#b91c1c" }}>{data.setup_error}</p> : null}
      {!error && data && !tournament ? <p>{data.empty_reason || "No tournament partner board is currently published."}</p> : null}

      {data?.tournaments?.length ? (
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          {data.tournaments.map((choice) => {
            const slug = choice.settings.registration_slug;
            const active = choice.tournament.id === tournament?.id;
            const href = slug ? `/clubs/${clubSlug}/tournament-partner-board?tournament=${encodeURIComponent(slug)}` : `/clubs/${clubSlug}/tournament-partner-board?tournament_id=${encodeURIComponent(choice.tournament.id)}`;
            return (
              <Link key={choice.tournament.id} href={href} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.45rem 0.75rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                {choice.tournament.name}
              </Link>
            );
          })}
        </div>
      ) : null}

      {tournament ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.75rem", marginBottom: "1rem" }}>
          <article style={cardStyle}><strong>Players looking</strong><br />{partnerEntries.length}</article>
          <article style={cardStyle}><strong>Showing</strong><br />{visibleEntries.length}</article>
          <article style={cardStyle}><strong>Registrations</strong><br />{data?.summary?.total_registrations ?? 0}</article>
          <article style={cardStyle}><strong>Roster players</strong><br />{data?.summary?.total_players ?? 0}</article>
          <article style={cardStyle}><strong>Board enabled</strong><br />{settings?.partner_board_enabled ? "Yes" : "No"}</article>
        </div>
      ) : null}

      {tournament ? (
        <p style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
          <Link href={`/clubs/${clubSlug}/tournament-registration${query}`}>Open registration</Link>
          <Link href={`/clubs/${clubSlug}/tournament-roster${queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug, event: selectedEvent })}`}>Open roster</Link>
          {!editToken ? <Link href={`/clubs/${clubSlug}/tournament-registration${query}`}>Request edit link to send interest</Link> : null}
        </p>
      ) : null}

      {partnerEntries.length ? (
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          <Link href={`/clubs/${clubSlug}/tournament-partner-board${queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug, editToken: editToken || null })}`} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: !selectedEvent ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: !selectedEvent ? 800 : 600 }}>All events</Link>
          {eventChoices.map(([key, label]) => {
            const active = key === selectedEvent;
            return (
              <Link key={key} href={`/clubs/${clubSlug}/tournament-partner-board${queryFor({ tournamentId: tournament?.id, registrationSlug: settings?.registration_slug, editToken: editToken || null, event: key })}`} style={{ border: "1px solid #cbd5e1", borderRadius: "999px", padding: "0.35rem 0.65rem", background: active ? "#dbeafe" : "white", color: "#0f172a", textDecoration: "none", fontWeight: active ? 800 : 600 }}>
                {label}
              </Link>
            );
          })}
        </div>
      ) : null}

      {tournament && !settings?.partner_board_enabled ? (
        <article style={{ ...cardStyle, marginBottom: "1rem", background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Partner board is disabled</h2>
          <p style={{ color: "#475569" }}>The public partner board is not enabled for this tournament.</p>
        </article>
      ) : null}

      {visibleEntries.length ? (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem" }}>
          {visibleEntries.map((entry) => (
            <article key={`${entry.selection_id}-${entry.player_name}`} id={entryAnchor(entry)} style={cardStyle}>
              <h2 style={{ marginTop: 0, fontSize: "1.05rem" }}>{entry.player_name}</h2>
              <p style={{ color: "#475569" }}>{entryEventLabel(entry)}</p>
              <dl style={{ display: "grid", gap: "0.35rem", margin: 0 }}>
                {entry.skill ? <><dt style={{ fontWeight: 800 }}>Skill</dt><dd style={{ margin: 0 }}>{entry.skill}</dd></> : null}
                {entry.age_bracket ? <><dt style={{ fontWeight: 800 }}>Age bracket</dt><dd style={{ margin: 0 }}>{entry.age_bracket}</dd></> : null}
                {entry.note ? <><dt style={{ fontWeight: 800 }}>Note</dt><dd style={{ margin: 0 }}>{entry.note}</dd></> : null}
              </dl>
              <p style={{ marginBottom: 0 }}><Link href={`/clubs/${clubSlug}/tournament-partner-board${queryWithEdit}#${entryAnchor(entry)}`}>board link</Link></p>
              {editToken && editResponse.data && tournament ? (
                <PairingInterestPanel
                  apiBase={apiBase}
                  clubSlug={clubSlug}
                  tournamentId={tournament.id}
                  registrationSlug={settings?.registration_slug ?? null}
                  editToken={editToken}
                  requesterSelections={editResponse.data.selections ?? []}
                  boardEntries={[entry]}
                />
              ) : null}
            </article>
          ))}
        </div>
      ) : tournament && settings?.partner_board_enabled ? (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>{partnerEntries.length ? "No matching partner requests" : "No open partner requests"}</h2>
          <p style={{ color: "#475569" }}>{partnerEntries.length ? "Try clearing the event filter." : "There are no public partner-board entries for this tournament yet."}</p>
          {editToken ? <Link href={`/clubs/${clubSlug}/tournament-partner-board${queryWithEdit}`}>Refresh board</Link> : null}
        </article>
      ) : null}
    </section>
  );
}
