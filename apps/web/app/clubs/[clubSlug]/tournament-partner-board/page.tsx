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
import { groupPartnerEntries } from "@/lib/tournamentPartnerBoard";

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

function eventName(familyValue?: string | null, divisionValue?: string | null): string {
  const family = String(familyValue || "Event").trim();
  const division = String(divisionValue || "Division").trim();
  const event = division.toLocaleLowerCase().startsWith(family.toLocaleLowerCase())
    ? division
    : `${family} · ${division}`;
  return event;
}

function eventLabel(entry: PublicTournamentNeedsPartnerEntry): string {
  return [entry.event_day_label || "Day", eventName(entry.event_family, entry.division)].join(" · ");
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
          eventName(event.event_family_label, event.division_name)
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
  const [{ data, error, status }, editResponse] = await Promise.all([
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
    const unavailable = Boolean(error && status !== 404);
    return (
      <section>
        <h1>{unavailable ? "Partner listings unavailable" : "Tournament not found"}</h1>
        <p style={{ color: "#475569" }}>
          {unavailable
            ? "We couldn’t load partner listings right now. Please try again shortly."
            : "We couldn’t find that tournament. It may no longer be public."}
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
  const playerGroups = groupPartnerEntries(partnerEntries);
  const visiblePlayerGroups = groupPartnerEntries(visibleEntries);
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
        kicker="Players Needing Partners"
        description="Looking for a doubles partner? Browse players below, then use your registration link to connect."
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
          <h2 style={{ marginTop: 0 }}>Players Needing Partners temporarily unavailable</h2>
          <p style={{ color: "#7f1d1d" }}>Please try again shortly.</p>
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
          <p style={{ color: "#7f1d1d" }}>
            Request a new edit link from the registration page.
          </p>
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
            <h2 style={{ marginTop: 0 }}>Players needing partners</h2>
            <p style={{ marginBottom: 0, color: "#475569" }}>
              Choose an event, then use your registration link to contact a
              player or reply to a request.
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
            {settings?.partner_board_enabled ? "Open" : "Unavailable"}
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
          <div><strong>Looking for partners</strong><br />{playerGroups.length}</div>
          <div><strong>Events</strong><br />{eventChoices.length}</div>
          <div><strong>Players shown</strong><br />{visiblePlayerGroups.length}</div>
          <div><strong>Tournament players</strong><br />{data?.summary?.total_players ?? 0}</div>
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
            <h2 style={{ margin: 0 }}>Want to connect with a player?</h2>
            <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
              Request your registration link. We&apos;ll keep your contact details
              private until you connect.
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
          aria-label="Players Needing Partners event filters"
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
          <h2 style={{ marginTop: 0 }}>Partner matching unavailable</h2>
          <p style={{ color: "#475569" }}>
            Partner matching isn&apos;t available for this tournament.
          </p>
        </article>
      ) : visiblePlayerGroups.length ? (
        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(auto-fit, minmax(min(100%, 320px), 1fr))",
            gap: "0.85rem"
          }}
        >
          {visiblePlayerGroups.map((group) => (
            <article
              key={group.playerKey}
              data-testid="partner-player-card"
              style={{
                ...cardStyle,
                display: "grid",
                gap: "0.75rem",
                alignContent: "start"
              }}
            >
              <header>
                <h2 style={{ margin: "0 0 0.2rem", fontSize: "1.15rem" }}>
                  {group.playerName}
                </h2>
                <p style={{ margin: 0, color: "#64748b" }}>
                  Looking in {group.entries.length} division{group.entries.length === 1 ? "" : "s"}
                </p>
              </header>
              <div style={{ display: "grid", gap: "0.65rem" }}>
                {group.entries.map((entry) => (
                  <section
                    key={entry.board_entry_key || eventKey(entry)}
                    id={entryAnchor(entry)}
                    data-testid="partner-division-listing"
                    style={{
                      border: "1px solid #e2e8f0",
                      borderRadius: "10px",
                      padding: "0.75rem",
                      background: "#f8fafc",
                      scrollMarginTop: "1rem"
                    }}
                  >
                    <h3 style={{ margin: "0 0 0.2rem", fontSize: "1rem" }}>
                      {entry.division || "Division"}
                    </h3>
                    <p style={{ margin: 0, color: "#1d4ed8", fontWeight: 700 }}>
                      {entry.event_day_label || "Day"} · {entry.event_family || "Event"}
                    </p>
                    <p style={{ margin: "0.35rem 0 0", color: "#475569" }}>
                      Rating {entry.skill || "not listed"} · {entry.age_bracket || "Any age"}
                    </p>
                    {entry.note ? (
                      <p style={{ margin: "0.45rem 0 0", color: "#475569" }}>
                        <strong>Note:</strong> {entry.note}
                      </p>
                    ) : null}
                    <p style={{ margin: "0.45rem 0 0" }}>
                      <Link
                        href={`/clubs/${params.clubSlug}/tournament-partner-board${queryWithEdit}#${entryAnchor(
                          entry
                        )}`}
                        style={{ fontWeight: 800 }}
                      >
                        Share this listing
                      </Link>
                    </p>
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
                  </section>
                ))}
              </div>
            </article>
          ))}
        </div>
      ) : (
        <article style={cardStyle}>
          <h2 style={{ marginTop: 0 }}>
            {partnerEntries.length
              ? "No players are listed for this event"
              : "No one is looking for a partner yet"}
          </h2>
          <p style={{ color: "#475569" }}>
            {partnerEntries.length
              ? "Select All events to clear the current filter."
              : "Check back as more players register."}
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
