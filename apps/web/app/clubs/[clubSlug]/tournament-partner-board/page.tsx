import Link from "next/link";
import { redirect } from "next/navigation";
import PublicTournamentModuleHeader from "@/components/PublicTournamentModuleHeader";
import {
  getClubTournamentRegistrationEdit,
  getClubTournamentRoster,
  type PublicRegistrationDay,
  type PublicRegistrationEditSelection,
  type PublicRegistrationEvent,
  type PublicTournamentNeedsPartnerEntry
} from "@/lib/tournamentRegistrationApi";
import PairingInterestPanel from "./PairingInterestPanel";
import PartnerRequestReviewPanel from "./PartnerRequestReviewPanel";

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

function slugify(value: string): string {
  return (
    value
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "event"
  );
}

function eventLabel(entry: PublicTournamentNeedsPartnerEntry): string {
  return [
    entry.event_day_label || "Day",
    entry.event_family || "Event",
    entry.division || "Division"
  ].join(" · ");
}

function eventKey(entry: PublicTournamentNeedsPartnerEntry): string {
  return slugify(eventLabel(entry));
}

function entryAnchor(entry: PublicTournamentNeedsPartnerEntry): string {
  return `partner-${slugify(entry.board_entry_key || entry.player_name || "player")}`;
}

function requesterSelectionsForEntry(
  entry: PublicTournamentNeedsPartnerEntry,
  selections: PublicRegistrationEditSelection[],
  events: PublicRegistrationEvent[],
  days: PublicRegistrationDay[]
): PublicRegistrationEditSelection[] {
  const dayById = new Map(days.map((day) => [day.id, day]));
  const eventById = new Map(events.map((event) => [event.id, event]));
  const targetKey = eventKey(entry);
  return selections.filter((selection) => {
    const event = eventById.get(String(selection.event_option_id || ""));
    if (!event) return false;
    const day = dayById.get(event.registration_day_id);
    return (
      slugify(
        [
          day?.label || "Day",
          event.event_family_label || "Event",
          event.division_name || "Division"
        ].join(" · ")
      ) === targetKey
    );
  });
}

function boardQuery({
  tournamentId,
  registrationSlug,
  editToken,
  event,
  partnerRequestId
}: {
  tournamentId?: string | null;
  registrationSlug?: string | null;
  editToken?: string | null;
  event?: string | null;
  partnerRequestId?: string | null;
}): string {
  const query = new URLSearchParams();
  if (registrationSlug) query.set("tournament", registrationSlug);
  else if (tournamentId) query.set("tournament_id", tournamentId);
  if (editToken) query.set("edit_token", editToken);
  if (event) query.set("event", event);
  if (partnerRequestId) query.set("partner_request_id", partnerRequestId);
  const text = query.toString();
  return text ? `?${text}` : "";
}

export default async function TournamentPartnerBoardPage({
  params,
  searchParams
}: Props) {
  const registrationSlug = firstParam(searchParams, "tournament");
  const tournamentId = firstParam(searchParams, "tournament_id");
  if (!registrationSlug && !tournamentId) {
    redirect(`/clubs/${params.clubSlug}/tournaments`);
  }

  const editToken = firstParam(searchParams, "edit_token") || "";
  const selectedEvent = firstParam(searchParams, "event");
  const selectedPartnerRequestId = firstParam(
    searchParams,
    "partner_request_id"
  );
  const [{ data, error }, editResponse] = await Promise.all([
    getClubTournamentRoster(params.clubSlug, {
      registrationSlug,
      tournamentId
    }),
    editToken
      ? getClubTournamentRegistrationEdit(params.clubSlug, {
          editToken,
          registrationSlug,
          tournamentId
        })
      : Promise.resolve({ data: null, error: null })
  ]);

  const tournament = data?.tournament || null;
  const settings = data?.settings || null;
  const selectionMatches = Boolean(
    tournament &&
      (!tournamentId || tournament.id === tournamentId) &&
      (!registrationSlug || settings?.registration_slug === registrationSlug)
  );

  if (!selectionMatches || !tournament) {
    return (
      <section>
        <h1>Partner Board unavailable</h1>
        <p style={{ color: "#475569" }}>
          The selected tournament is unavailable or no longer published.
        </p>
        <Link href={`/clubs/${params.clubSlug}/tournaments`}>
          Return to tournament selection
        </Link>
      </section>
    );
  }

  const partnerEntries = data?.roster?.partner_board_entries || [];
  const eventChoices = Array.from(
    new Map(
      partnerEntries.map((entry) => [eventKey(entry), eventLabel(entry)])
    ).entries()
  ).sort((a, b) => a[1].localeCompare(b[1]));
  const visibleEntries = selectedEvent
    ? partnerEntries.filter((entry) => eventKey(entry) === selectedEvent)
    : partnerEntries;
  const selectedQuery = boardQuery({
    tournamentId: tournament.id,
    registrationSlug: settings?.registration_slug
  });
  const queryWithEdit = boardQuery({
    tournamentId: tournament.id,
    registrationSlug: settings?.registration_slug,
    editToken: editToken || null,
    event: selectedEvent,
    partnerRequestId: selectedPartnerRequestId
  });
  const apiBase =
    process.env.JUPR_API_BASE_URL ||
    process.env.NEXT_PUBLIC_JUPR_API_BASE_URL ||
    null;

  return (
    <section>
      <PublicTournamentModuleHeader
        clubSlug={params.clubSlug}
        tournamentName={tournament.name}
        tournamentId={tournament.id}
        registrationSlug={settings?.registration_slug || null}
        active="partner-board"
        kicker="Tournament Partner Board"
        description="Find players who opted into public partner requests. Contact details stay private, and pairing actions require a secure registration edit link."
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
          <h2 style={{ marginTop: 0 }}>Partner Board temporarily unavailable</h2>
          <p style={{ color: "#7f1d1d" }}>{error}</p>
        </article>
      ) : null}
      {editResponse.error ? (
        <article
          role="alert"
          style={{
            ...cardStyle,
            marginBottom: "1rem",
            borderColor: "#fecaca",
            background: "#fef2f2"
          }}
        >
          <h2 style={{ marginTop: 0 }}>Edit link could not be verified</h2>
          <p style={{ color: "#7f1d1d" }}>{editResponse.error}</p>
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
            <h2 style={{ marginTop: 0 }}>Open partner requests</h2>
            <p style={{ marginBottom: 0, color: "#475569" }}>
              Browse by event, then use your private registration edit link to
              send or accept interest.
            </p>
          </div>
          <span
            style={{
              border: `1px solid ${
                settings?.partner_board_enabled ? "#86efac" : "#cbd5e1"
              }`,
              borderRadius: "999px",
              padding: "0.25rem 0.6rem",
              background: settings?.partner_board_enabled ? "#dcfce7" : "#f1f5f9",
              color: settings?.partner_board_enabled ? "#166534" : "#475569",
              fontWeight: 800
            }}
          >
            {settings?.partner_board_enabled ? "Board open" : "Board disabled"}
          </span>
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(145px, 1fr))",
            gap: "0.75rem",
            marginTop: "1rem"
          }}
        >
          <div><strong>Open requests</strong><br />{partnerEntries.length}</div>
          <div><strong>Showing</strong><br />{visibleEntries.length}</div>
          <div><strong>Registrations</strong><br />{data?.summary?.total_registrations ?? 0}</div>
          <div><strong>Roster players</strong><br />{data?.summary?.total_players ?? 0}</div>
        </div>
      </article>

      {!editToken ? (
        <article
          style={{
            ...cardStyle,
            marginBottom: "1rem",
            display: "flex",
            justifyContent: "space-between",
            gap: "1rem",
            flexWrap: "wrap",
            alignItems: "center"
          }}
        >
          <div>
            <h2 style={{ margin: 0 }}>Want to contact or accept a player?</h2>
            <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
              Request your secure registration edit link. Public pages never
              expose email addresses or phone numbers.
            </p>
          </div>
          <Link
            href={`/clubs/${params.clubSlug}/tournament-registration${selectedQuery}`}
            style={{ fontWeight: 800 }}
          >
            Request edit link
          </Link>
        </article>
      ) : null}

      {editToken && editResponse.data ? (
        <PartnerRequestReviewPanel
          apiBase={apiBase}
          clubSlug={params.clubSlug}
          tournamentId={tournament.id}
          registrationSlug={settings?.registration_slug || null}
          editToken={editToken}
          focusRequestId={selectedPartnerRequestId}
        />
      ) : null}

      {partnerEntries.length ? (
        <nav
          aria-label="Partner Board event filters"
          style={{
            display: "flex",
            gap: "0.5rem",
            flexWrap: "wrap",
            marginBottom: "1rem"
          }}
        >
          <Link
            href={`/clubs/${params.clubSlug}/tournament-partner-board${boardQuery({
              tournamentId: tournament.id,
              registrationSlug: settings?.registration_slug,
              editToken: editToken || null
            })}`}
            style={{
              border: "1px solid #cbd5e1",
              borderRadius: "999px",
              padding: "0.4rem 0.7rem",
              background: !selectedEvent ? "#dbeafe" : "white",
              color: !selectedEvent ? "#1d4ed8" : "#0f172a",
              textDecoration: "none",
              fontWeight: !selectedEvent ? 800 : 650
            }}
          >
            All events
          </Link>
          {eventChoices.map(([key, label]) => {
            const active = selectedEvent === key;
            return (
              <Link
                key={key}
                href={`/clubs/${params.clubSlug}/tournament-partner-board${boardQuery({
                  tournamentId: tournament.id,
                  registrationSlug: settings?.registration_slug,
                  editToken: editToken || null,
                  event: key,
                  partnerRequestId: selectedPartnerRequestId
                })}`}
                style={{
                  border: "1px solid #cbd5e1",
                  borderRadius: "999px",
                  padding: "0.4rem 0.7rem",
                  background: active ? "#dbeafe" : "white",
                  color: active ? "#1d4ed8" : "#0f172a",
                  textDecoration: "none",
                  fontWeight: active ? 800 : 650
                }}
              >
                {label}
              </Link>
            );
          })}
        </nav>
      ) : null}

      {!settings?.partner_board_enabled ? (
        <article style={{ ...cardStyle, background: "#f8fafc" }}>
          <h2 style={{ marginTop: 0 }}>Partner Board is disabled</h2>
          <p style={{ color: "#475569" }}>
            The tournament administrator has not enabled public partner requests.
          </p>
        </article>
      ) : visibleEntries.length ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))",
            gap: "0.85rem"
          }}
        >
          {visibleEntries.map((entry) => (
            <article
              key={entry.board_entry_key || String(entry.player_name)}
              id={entryAnchor(entry)}
              style={{
                ...cardStyle,
                display: "grid",
                gap: "0.75rem",
                alignContent: "start"
              }}
            >
              <div>
                <h2 style={{ margin: "0 0 0.3rem", fontSize: "1.1rem" }}>
                  {entry.player_name}
                </h2>
                <p style={{ margin: 0, color: "#1d4ed8", fontWeight: 700 }}>
                  {eventLabel(entry)}
                </p>
              </div>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                  gap: "0.6rem"
                }}
              >
                <div>
                  <strong>Skill</strong><br />
                  {entry.skill || "Not listed"}
                </div>
                <div>
                  <strong>Age bracket</strong><br />
                  {entry.age_bracket || "Open"}
                </div>
              </div>
              {entry.note ? (
                <div>
                  <strong>Player note</strong>
                  <p style={{ margin: "0.25rem 0 0", color: "#475569" }}>
                    {entry.note}
                  </p>
                </div>
              ) : null}
              <Link
                href={`/clubs/${params.clubSlug}/tournament-partner-board${queryWithEdit}#${entryAnchor(
                  entry
                )}`}
                style={{ fontWeight: 800 }}
              >
                Link to this request
              </Link>
              {editToken && editResponse.data ? (
                <PairingInterestPanel
                  apiBase={apiBase}
                  clubSlug={params.clubSlug}
                  tournamentId={tournament.id}
                  registrationSlug={settings?.registration_slug || null}
                  editToken={editToken}
                  requesterSelections={requesterSelectionsForEntry(
                    entry,
                    editResponse.data.selections || [],
                    data?.events || [],
                    data?.days || []
                  )}
                  boardEntries={[entry]}
                />
              ) : null}
            </article>
          ))}
        </div>
      ) : (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>
            {partnerEntries.length
              ? "No requests match this event"
              : "No open partner requests"}
          </h2>
          <p style={{ color: "#475569" }}>
            {partnerEntries.length
              ? "Select All events to clear the current filter."
              : "No players have opted into the public Partner Board for this tournament."}
          </p>
          {selectedEvent ? (
            <Link
              href={`/clubs/${params.clubSlug}/tournament-partner-board${boardQuery({
                tournamentId: tournament.id,
                registrationSlug: settings?.registration_slug,
                editToken: editToken || null
              })}`}
            >
              Show all events
            </Link>
          ) : null}
        </article>
      )}
    </section>
  );
}
