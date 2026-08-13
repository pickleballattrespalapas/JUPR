"use client";

import Link from "next/link";
import { useMemo, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";

import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, type ActionSuccess } from "@/components/interaction";
import {
  FourPlayerTeamMember,
  getAdminTeamCompetitionSnapshot,
  mutateAdminTeamCompetition,
  TeamCompetitionEvent,
  TeamCompetitionSnapshot,
  TeamTournamentMatchup
} from "@/lib/tournamentTeamCompetitionApi";
import { useAuthenticatedAutoLoad } from "@/lib/useAuthenticatedAutoLoad";
import { useAdminSession } from "@/lib/useAdminSession";

import styles from "./TeamTournamentAdminPanel.module.css";

type Props = { clubId: string; initialTournamentId: string };
type EventFormat = "STANDARD" | "COMBINED_RATING_CAP" | "FOUR_PLAYER_TEAM";
type Tab = "setup" | "ratings" | "teams" | "schedule" | "matches" | "podium";
type TeamSlot = "MAN_1" | "MAN_2" | "WOMAN_1" | "WOMAN_2";
type RosterEntry = {
  registrationId: string;
  email: string;
  displayName: string;
};
type RosterDraft = Record<TeamSlot, RosterEntry>;
type ConfigDraft = {
  eligibilityMode: "STANDARD" | "COMBINED_RATING_CAP";
  combinedRatingCap: string;
  competitionFormat: "STANDARD" | "FOUR_PLAYER_TEAM";
  tiebreakMode: "SINGLES" | "SKINNY_RELAY";
  playoffFormat:
    | "NONE"
    | "TOP_2_FINAL"
    | "TOP_4_SEMIFINALS"
    | "TOP_4_SEMIFINALS_WITH_BRONZE";
  allowSubstitutes: boolean;
};

const TEAM_SLOTS: TeamSlot[] = ["MAN_1", "MAN_2", "WOMAN_1", "WOMAN_2"];
const tabLabels: Array<[Tab, string]> = [
  ["setup", "Setup"],
  ["ratings", "Rating review"],
  ["teams", "Teams"],
  ["schedule", "Schedule"],
  ["matches", "Lineups & scores"],
  ["podium", "Podium"]
];

function text(row: Record<string, unknown> | undefined, key: string): string {
  const value = row?.[key];
  return value == null ? "" : String(value);
}

function numberValue(
  row: Record<string, unknown> | undefined,
  key: string,
  fallback = 0
): number {
  const value = Number(row?.[key]);
  return Number.isFinite(value) ? value : fallback;
}

function registrationName(row: Record<string, unknown>): string {
  return (
    text(row, "display_name") ||
    [text(row, "first_name"), text(row, "last_name")]
      .filter(Boolean)
      .join(" ") ||
    text(row, "email") ||
    "Registration"
  );
}

function eventLabel(event: TeamCompetitionEvent): string {
  return (
    event.label ||
    [event.event_family_label, event.division_name].filter(Boolean).join(" — ") ||
    "Tournament event"
  );
}

function emptyRoster(): RosterDraft {
  return {
    MAN_1: { registrationId: "", email: "", displayName: "" },
    MAN_2: { registrationId: "", email: "", displayName: "" },
    WOMAN_1: { registrationId: "", email: "", displayName: "" },
    WOMAN_2: { registrationId: "", email: "", displayName: "" }
  };
}

function rosterPayload(draft: RosterDraft): Array<Record<string, unknown>> {
  return TEAM_SLOTS.map((slot) => ({
    slot,
    registration_id: draft[slot].registrationId || null,
    email: draft[slot].email.trim().toLowerCase(),
    display_name: draft[slot].displayName.trim(),
    gender: slot.startsWith("MAN_") ? "Men" : "Women"
  }));
}

function initialConfig(event: TeamCompetitionEvent): ConfigDraft {
  return {
    eligibilityMode:
      event.eligibility_mode === "COMBINED_RATING_CAP"
        ? "COMBINED_RATING_CAP"
        : "STANDARD",
    combinedRatingCap:
      event.combined_rating_cap == null ? "" : String(event.combined_rating_cap),
    competitionFormat:
      event.competition_format === "FOUR_PLAYER_TEAM"
        ? "FOUR_PLAYER_TEAM"
        : "STANDARD",
    tiebreakMode:
      event.team_tiebreak_mode === "SKINNY_RELAY"
        ? "SKINNY_RELAY"
        : "SINGLES",
    playoffFormat:
      event.team_playoff_format === "TOP_2_FINAL" ||
      event.team_playoff_format === "TOP_4_SEMIFINALS" ||
      event.team_playoff_format === "TOP_4_SEMIFINALS_WITH_BRONZE"
        ? event.team_playoff_format
        : "NONE",
    allowSubstitutes: Boolean(event.team_allow_substitutes)
  };
}

function eventFormat(draft: ConfigDraft | null): EventFormat {
  if (draft?.competitionFormat === "FOUR_PLAYER_TEAM") return "FOUR_PLAYER_TEAM";
  if (draft?.eligibilityMode === "COMBINED_RATING_CAP") return "COMBINED_RATING_CAP";
  return "STANDARD";
}

function friendlyWorkspaceWarning(warning: string): string {
  if (/tournament_team_operations unavailable|apierror/i.test(warning)) {
    return "Team scheduling and scoring data is temporarily unavailable. Event-format setup and rating review remain available.";
  }
  return warning;
}

function slotLabel(slot: TeamSlot): string {
  return {
    MAN_1: "Man 1",
    MAN_2: "Man 2",
    WOMAN_1: "Woman 1",
    WOMAN_2: "Woman 2"
  }[slot];
}

export default function TeamTournamentAdminPanel({
  clubId,
  initialTournamentId
}: Props) {
  const {
    accessToken,
    loading: sessionLoading,
    message: sessionMessage
  } = useAdminSession();
  const operationKeys = useRef(new Map<string, string>());
  const [tournamentId, setTournamentId] = useState(initialTournamentId);
  const [snapshot, setSnapshot] = useState<TeamCompetitionSnapshot | null>(null);
  const [tab, setTab] = useState<Tab>("setup");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [messageKind, setMessageKind] = useState<"notice" | "error" | "success">(
    "notice"
  );

  const [configEventId, setConfigEventId] = useState("");
  const [configDraft, setConfigDraft] = useState<ConfigDraft | null>(null);

  const [ratingEventId, setRatingEventId] = useState("");
  const [ratingRegistrationId, setRatingRegistrationId] = useState("");
  const [verifiedRating, setVerifiedRating] = useState("");
  const [ratingNote, setRatingNote] = useState("");
  const [reviewOverrides, setReviewOverrides] = useState<
    Record<string, { state: string; reason: string }>
  >({});

  const [teamEventId, setTeamEventId] = useState("");
  const [teamName, setTeamName] = useState("");
  const [captainRegistrationId, setCaptainRegistrationId] = useState("");
  const [createRoster, setCreateRoster] = useState<RosterDraft>(emptyRoster);
  const [inviteMemberId, setInviteMemberId] = useState("");
  const [inviteEmail, setInviteEmail] = useState("");
  const [manageTeamId, setManageTeamId] = useState("");
  const [rosterAction, setRosterAction] = useState<"REPLACE" | "WITHDRAW">(
    "REPLACE"
  );
  const [replacementRoster, setReplacementRoster] =
    useState<RosterDraft>(emptyRoster);
  const [rosterReason, setRosterReason] = useState("");

  const [drawId, setDrawId] = useState("");
  const [roundRobinTeamIds, setRoundRobinTeamIds] = useState<string[]>([]);

  const [lineupDrafts, setLineupDrafts] = useState<
    Record<string, { mixed: "STRAIGHT" | "CROSS"; tiebreakPlayerId: string }>
  >({});
  const [scoreDrafts, setScoreDrafts] = useState<
    Record<string, { scoreA: string; scoreB: string }>
  >({});
  const [reconcileDrafts, setReconcileDrafts] = useState<
    Record<string, { officialMatchId: string; reason: string }>
  >({});
  const [podiumReasons, setPodiumReasons] = useState<Record<string, string>>({});

  const registrationsById = useMemo(
    () =>
      new Map(
        (snapshot?.registrations || []).map((row) => [text(row, "id"), row])
      ),
    [snapshot?.registrations]
  );
  const eventsById = useMemo(
    () => new Map((snapshot?.event_options || []).map((row) => [row.id, row])),
    [snapshot?.event_options]
  );
  const teamsById = useMemo(
    () => new Map((snapshot?.teams || []).map((row) => [row.id, row])),
    [snapshot?.teams]
  );
  const matchupsById = useMemo(
    () => new Map((snapshot?.matchups || []).map((row) => [row.id, row])),
    [snapshot?.matchups]
  );
  const selectionsById = useMemo(
    () =>
      new Map((snapshot?.selections || []).map((row) => [text(row, "id"), row])),
    [snapshot?.selections]
  );

  const combinedEvents = (snapshot?.event_options || []).filter(
    (event) => event.eligibility_mode === "COMBINED_RATING_CAP"
  );
  const fourPlayerEvents = (snapshot?.event_options || []).filter(
    (event) => event.competition_format === "FOUR_PLAYER_TEAM"
  );
  const teamDraws = (snapshot?.draws || []).filter((draw) => {
    const event = eventsById.get(String(draw.event_option_id || ""));
    return (
      draw.draw_kind === "TEAM_PARENT" ||
      event?.competition_format === "FOUR_PLAYER_TEAM"
    );
  });
  const confirmedRegistrations = (snapshot?.registrations || []).filter((row) =>
    ["CONFIRMED", "ADMIN_CONFIRMED"].includes(text(row, "status").toUpperCase())
  );
  const selectedTeam = teamsById.get(manageTeamId);
  const selectedDraw = (snapshot?.draws || []).find((row) => row.id === drawId);

  function operationKey(scope: string): string {
    const existing = operationKeys.current.get(scope);
    if (existing) return existing;
    const generated =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    operationKeys.current.set(scope, generated);
    return generated;
  }

  async function loadSnapshot(nextTournamentId = tournamentId): Promise<void> {
    setTournamentId(nextTournamentId);
    if (nextTournamentId !== tournamentId) setSnapshot(null);
    if (!nextTournamentId || !accessToken) return;
    setBusy(true);
    const response = await getAdminTeamCompetitionSnapshot(
      clubId,
      nextTournamentId,
      accessToken
    );
    setBusy(false);
    if (response.error || !response.data) {
      setMessageKind("error");
      setMessage(response.error || "Unable to load the team tournament.");
      return;
    }
    setSnapshot(response.data);
    setMessageKind("success");
    setMessage(null);
    const firstEvent = response.data.event_options[0];
    if (firstEvent) {
      setConfigEventId(firstEvent.id);
      setConfigDraft(initialConfig(firstEvent));
    }
    const firstCombined = response.data.event_options.find(
      (event) => event.eligibility_mode === "COMBINED_RATING_CAP"
    );
    setRatingEventId(firstCombined?.id || "");
    const firstTeam = response.data.event_options.find(
      (event) => event.competition_format === "FOUR_PLAYER_TEAM"
    );
    setTeamEventId(firstTeam?.id || "");
    const firstDraw = response.data.draws.find(
      (row) =>
        row.draw_kind === "TEAM_PARENT" ||
        firstTeam?.id === String(row.event_option_id || "")
    );
    setDrawId(firstDraw?.id || "");
    setRoundRobinTeamIds([]);
  }

  useAuthenticatedAutoLoad(
    `${accessToken}\u0000${initialTournamentId}`,
    () => loadSnapshot(initialTournamentId),
    clubId
  );

  async function mutate(
    scope: string,
    path: string,
    payload: Record<string, unknown>,
    confirmationText: string,
    successMessage: string
  ): Promise<ActionSuccess> {
    if (!accessToken || !tournamentId) {
      throw new Error("Choose a tournament and sign in before making changes.");
    }
    setBusy(true);
    setMessage(null);
    const response = await mutateAdminTeamCompetition<Record<string, unknown>>(
      path,
      {
        ...payload,
        idempotency_key: operationKey(scope),
        confirmation_text: confirmationText,
        source: "next_team_tournament_workspace"
      },
      accessToken
    );
    setBusy(false);
    if (response.error) {
      setMessageKind("error");
      setMessage(response.error);
      throw new Error(response.error);
    }
    operationKeys.current.delete(scope);
    await loadSnapshot(tournamentId);
    setMessageKind("success");
    setMessage(successMessage);
    return actionSuccess("Tournament team action complete", successMessage);
  }

  function chooseConfigEvent(eventId: string): void {
    setConfigEventId(eventId);
    const event = eventsById.get(eventId);
    setConfigDraft(event ? initialConfig(event) : null);
  }

  function setRosterRegistration(
    setter: Dispatch<SetStateAction<RosterDraft>>,
    slot: TeamSlot,
    registrationId: string
  ): void {
    const registration = registrationsById.get(registrationId);
    setter((current) => ({
      ...current,
      [slot]: {
        registrationId,
        email: registration ? text(registration, "email") : current[slot].email,
        displayName: registration
          ? registrationName(registration)
          : current[slot].displayName
      }
    }));
  }

  function updateRosterEntry(
    setter: Dispatch<SetStateAction<RosterDraft>>,
    slot: TeamSlot,
    patch: Partial<RosterEntry>
  ): void {
    setter((current) => ({
      ...current,
      [slot]: { ...current[slot], ...patch }
    }));
  }

  function chooseManageTeam(teamId: string): void {
    setManageTeamId(teamId);
    const next = emptyRoster();
    for (const member of snapshot?.members || []) {
      if (member.team_id !== teamId || !TEAM_SLOTS.includes(member.slot)) continue;
      next[member.slot] = {
        registrationId: member.registration_id || "",
        email: member.invited_email || "",
        displayName:
          member.display_name || member.display_name_snapshot || "Player"
      };
    }
    setReplacementRoster(next);
    setRosterReason("");
  }

  function teamMembers(teamId: string): FourPlayerTeamMember[] {
    return (snapshot?.members || []).filter(
      (member) => member.team_id === teamId && member.status !== "REMOVED"
    );
  }

  function teamNameById(teamId: string | null | undefined): string {
    return teamId ? teamsById.get(teamId)?.name || "Team" : "TBD";
  }

  const basePath = `/admin/clubs/${encodeURIComponent(
    clubId
  )}/tournaments/admin/tournaments/${encodeURIComponent(tournamentId)}/team-competition`;

  if (sessionLoading) {
    return <p className={styles.notice}>Restoring admin access…</p>;
  }

  if (!accessToken) {
    return (
      <p className={styles.error} role="alert">
        {sessionMessage ||
          "Sign in through the admin access page before opening this workspace."}
      </p>
    );
  }

  return (
    <div className={styles.shell} data-testid="team-tournament-admin-workspace">
      {busy && !snapshot ? <p className={styles.notice}>Loading event setup…</p> : null}
      {message ? (
        <p className={styles[messageKind]} role={messageKind === "error" ? "alert" : "status"}>
          {message}
        </p>
      ) : null}
      {snapshot?.warnings.length ? (
        <div className={styles.card}>
          <strong>Workspace warnings</strong>
          <ul className={styles.warningList}>
            {snapshot.warnings
              .map((warning) => friendlyWorkspaceWarning(warning))
              .map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
          </ul>
        </div>
      ) : null}

      {snapshot ? (
        <>
          <div className={styles.tabs} role="tablist" aria-label="Team tournament tasks">
            {tabLabels.map(([value, label]) => (
              <button
                key={value}
                type="button"
                role="tab"
                aria-selected={tab === value}
                className={`${styles.tab} ${tab === value ? styles.tabActive : ""}`}
                onClick={() => setTab(value)}
              >
                {label}
              </button>
            ))}
          </div>

          {tab === "setup" ? (
            <section className={styles.section} aria-labelledby="team-setup-heading">
              <div className={styles.card}>
                <h2 id="team-setup-heading">Event format and eligibility</h2>
                <p className={styles.hint}>
                  Combined-rating doubles and four-player team play are separate
                  event formats. Existing entries or draws lock these choices.
                </p>
                <div className={styles.grid}>
                  <label className={styles.field}>
                    Event
                    <select
                      className={styles.select}
                      value={configEventId}
                      onChange={(event) => chooseConfigEvent(event.target.value)}
                    >
                      {snapshot.event_options.map((event) => (
                        <option key={event.id} value={event.id}>
                          {eventLabel(event)}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className={styles.field}>
                    Event format
                    <select
                      className={styles.select}
                      value={eventFormat(configDraft)}
                      onChange={(event) =>
                        setConfigDraft((current) => {
                          if (!current) return current;
                          const value = event.target.value as EventFormat;
                          if (value === "COMBINED_RATING_CAP") {
                            return {
                              ...current,
                              eligibilityMode: "COMBINED_RATING_CAP",
                              combinedRatingCap: current.combinedRatingCap || "8.0",
                              competitionFormat: "STANDARD"
                            };
                          }
                          if (value === "FOUR_PLAYER_TEAM") {
                            return {
                              ...current,
                              eligibilityMode: "STANDARD",
                              combinedRatingCap: "",
                              competitionFormat: "FOUR_PLAYER_TEAM"
                            };
                          }
                          return {
                            ...current,
                            eligibilityMode: "STANDARD",
                            combinedRatingCap: "",
                            competitionFormat: "STANDARD"
                          };
                        })
                      }
                    >
                      <option value="STANDARD">Standard singles or doubles</option>
                      <option value="COMBINED_RATING_CAP">Combined-rating doubles</option>
                      <option value="FOUR_PLAYER_TEAM">Four-player team · two men and two women</option>
                    </select>
                  </label>
                  {configDraft?.eligibilityMode === "COMBINED_RATING_CAP" ? (
                    <label className={styles.field}>
                      Maximum combined partner rating
                      <input
                        className={styles.input}
                        type="number"
                        min="0.01"
                        max="14"
                        step="0.01"
                        value={configDraft.combinedRatingCap}
                        onChange={(event) =>
                          setConfigDraft((current) =>
                            current
                              ? { ...current, combinedRatingCap: event.target.value }
                              : current
                          )
                        }
                      />
                      <small>Eligibility is strictly below this cap.</small>
                    </label>
                  ) : null}
                  {configDraft?.competitionFormat === "FOUR_PLAYER_TEAM" ? (
                    <>
                      <label className={styles.field}>
                        Tied after four games
                        <select
                          className={styles.select}
                          value={configDraft.tiebreakMode}
                          onChange={(event) =>
                            setConfigDraft((current) =>
                              current
                                ? {
                                    ...current,
                                    tiebreakMode: event.target
                                      .value as ConfigDraft["tiebreakMode"]
                                  }
                                : current
                            )
                          }
                        >
                          <option value="SINGLES">One singles game</option>
                          <option value="SKINNY_RELAY">Skinny-singles relay</option>
                        </select>
                      </label>
                      <label className={styles.field}>
                        Playoffs
                        <select
                          className={styles.select}
                          value={configDraft.playoffFormat}
                          onChange={(event) =>
                            setConfigDraft((current) =>
                              current
                                ? {
                                    ...current,
                                    playoffFormat: event.target
                                      .value as ConfigDraft["playoffFormat"]
                                  }
                                : current
                            )
                          }
                        >
                          <option value="NONE">No playoffs</option>
                          <option value="TOP_2_FINAL">Top two final</option>
                          <option value="TOP_4_SEMIFINALS">
                            Top four semifinals and final
                          </option>
                          <option value="TOP_4_SEMIFINALS_WITH_BRONZE">
                            Top four with bronze match
                          </option>
                        </select>
                      </label>
                      <label className={styles.check}>
                        <input
                          type="checkbox"
                          checked={configDraft.allowSubstitutes}
                          onChange={(event) =>
                            setConfigDraft((current) =>
                              current
                                ? {
                                    ...current,
                                    allowSubstitutes: event.target.checked
                                  }
                                : current
                            )
                          }
                        />
                        <span>
                          <strong>Allow substitutes</strong>
                          <br />
                          Roster replacements remain reasoned, version checked,
                          and blocked after play starts.
                        </span>
                      </label>
                    </>
                  ) : null}
                </div>
                {configDraft ? (
                  <p className={styles.notice}>
                    <strong>Review:</strong>{" "}
                    {eventFormat(configDraft) === "STANDARD"
                      ? "Standard singles or doubles"
                      : eventFormat(configDraft) === "COMBINED_RATING_CAP"
                        ? `Combined-rating doubles · below ${configDraft.combinedRatingCap || "—"}`
                        : `Four-player team · ${configDraft.allowSubstitutes ? "substitutes allowed" : "no substitutes"} · ${configDraft.tiebreakMode === "SKINNY_RELAY" ? "skinny-singles relay" : "one singles game"} · ${configDraft.playoffFormat.replaceAll("_", " ").toLowerCase()}`}
                  </p>
                ) : null}
                <div className={styles.actions}>
                  <ConfirmAction
                    triggerLabel="Save event rules"
                    title="Save tournament competition rules?"
                    description="This changes eligibility or match-format rules for the selected event."
                    confirmLabel="Save rules"
                    confirmationText="SAVE COMPETITION"
                    disabled={!configDraft || busy}
                    busy={busy}
                    onConfirm={(confirmation) => {
                      const event = eventsById.get(configEventId);
                      if (!event || !configDraft) {
                        throw new Error("Choose an event before saving.");
                      }
                      return mutate(
                        `config:${configEventId}`,
                        `${basePath}/events/${encodeURIComponent(configEventId)}/config`,
                        {
                          expected_updated_at: event.updated_at,
                          patch: {
                            eligibility_mode: configDraft.eligibilityMode,
                            combined_rating_cap:
                              configDraft.eligibilityMode ===
                              "COMBINED_RATING_CAP"
                                ? Number(configDraft.combinedRatingCap)
                                : null,
                            competition_format: configDraft.competitionFormat,
                            team_tiebreak_mode: configDraft.tiebreakMode,
                            team_playoff_format:
                              configDraft.competitionFormat ===
                              "FOUR_PLAYER_TEAM"
                                ? configDraft.playoffFormat
                                : "NONE",
                            team_allow_substitutes:
                              configDraft.competitionFormat ===
                                "FOUR_PLAYER_TEAM" &&
                              configDraft.allowSubstitutes
                          }
                        },
                        confirmation,
                        "Event competition rules saved."
                      );
                    }}
                  />
                </div>
              </div>
            </section>
          ) : null}

          {tab === "ratings" ? (
            <section className={styles.section} aria-labelledby="rating-heading">
              <div className={styles.card}>
                <h2 id="rating-heading">Organizer rating verification</h2>
                <p className={styles.hint}>
                  Use this when a linked JUPR rating is unavailable. Saving
                  refreshes every affected combined-rating review.
                </p>
                {!combinedEvents.length ? (
                  <p className={styles.notice}>
                    No event currently uses combined-rating eligibility.
                  </p>
                ) : (
                  <>
                    <div className={styles.threeGrid}>
                      <label className={styles.field}>
                        Event
                        <select
                          className={styles.select}
                          value={ratingEventId}
                          onChange={(event) => setRatingEventId(event.target.value)}
                        >
                          {combinedEvents.map((event) => (
                            <option key={event.id} value={event.id}>
                              {eventLabel(event)}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className={styles.field}>
                        Registration
                        <select
                          className={styles.select}
                          value={ratingRegistrationId}
                          onChange={(event) =>
                            setRatingRegistrationId(event.target.value)
                          }
                        >
                          <option value="">Choose player</option>
                          {confirmedRegistrations.map((registration) => (
                            <option
                              key={text(registration, "id")}
                              value={text(registration, "id")}
                            >
                              {registrationName(registration)} ·{" "}
                              {text(registration, "email")}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className={styles.field}>
                        Verified doubles rating
                        <input
                          className={styles.input}
                          type="number"
                          min="0"
                          max="7"
                          step="0.01"
                          value={verifiedRating}
                          onChange={(event) => setVerifiedRating(event.target.value)}
                        />
                      </label>
                    </div>
                    <label className={styles.field}>
                      Verification note
                      <textarea
                        className={styles.textarea}
                        value={ratingNote}
                        onChange={(event) => setRatingNote(event.target.value)}
                        placeholder="Source and date checked"
                      />
                    </label>
                    <ConfirmAction
                      triggerLabel="Verify rating"
                      title="Save this organizer-verified rating?"
                      description="The rating and note become part of the tournament audit record."
                      confirmLabel="Verify rating"
                      confirmationText="VERIFY RATING"
                      disabled={
                        busy ||
                        !ratingEventId ||
                        !ratingRegistrationId ||
                        !verifiedRating
                      }
                      busy={busy}
                      onConfirm={(confirmation) => {
                        const existing = snapshot.rating_verifications.find(
                          (row) =>
                            text(row, "event_option_id") === ratingEventId &&
                            text(row, "registration_id") === ratingRegistrationId &&
                            text(row, "status").toUpperCase() === "ACTIVE"
                        );
                        return mutate(
                          `rating:${ratingEventId}:${ratingRegistrationId}`,
                          `${basePath}/rating-verifications`,
                          {
                            event_option_id: ratingEventId,
                            registration_id: ratingRegistrationId,
                            rating: Number(verifiedRating),
                            note: ratingNote,
                            expected_version: existing
                              ? numberValue(existing, "version")
                              : null
                          },
                          confirmation,
                          "Organizer rating verified."
                        );
                      }}
                    />
                  </>
                )}
              </div>

              <div className={styles.card}>
                <h2>Combined-rating decisions</h2>
                <p className={styles.hint}>
                  The strict rule is calculated from the current rating evidence.
                  An override requires a written reason.
                </p>
                {snapshot.combined_rating_entries.filter(
                  (row) =>
                    text(row, "event_option_id") === ratingEventId &&
                    text(row, "review_phase") === "INITIAL"
                ).length ? (
                  <div className={styles.stack}>
                    {snapshot.combined_rating_entries
                      .filter(
                        (row) =>
                          text(row, "event_option_id") === ratingEventId &&
                          text(row, "review_phase") === "INITIAL"
                      )
                      .map((review) => {
                        const selectionId = text(review, "selection_id");
                        const override = reviewOverrides[selectionId] || {
                          state: "",
                          reason: ""
                        };
                        const selection = selectionsById.get(selectionId);
                        return (
                          <article className={styles.subcard} key={selectionId}>
                            <div className={styles.grid}>
                              <div>
                                <strong>{text(review, "registration_name")}</strong>
                                <p className={styles.muted}>
                                  With {text(review, "partner_name")}
                                </p>
                                <span className={styles.pill}>
                                  {text(review, "state").replaceAll("_", " ")}
                                </span>
                                <p>
                                  {text(review, "combined_rating")
                                    ? `${text(review, "player_rating")} + ${text(
                                        review,
                                        "partner_rating"
                                      )} = ${text(review, "combined_rating")}`
                                    : "Rating evidence is incomplete."}
                                </p>
                              </div>
                              <div className={styles.stack}>
                                <label className={styles.field}>
                                  Decision
                                  <select
                                    className={styles.select}
                                    value={override.state}
                                    onChange={(event) =>
                                      setReviewOverrides((current) => ({
                                        ...current,
                                        [selectionId]: {
                                          ...override,
                                          state: event.target.value
                                        }
                                      }))
                                    }
                                  >
                                    <option value="">Use calculated result</option>
                                    <option value="ELIGIBLE">Override: eligible</option>
                                    <option value="INELIGIBLE">
                                      Override: ineligible
                                    </option>
                                  </select>
                                </label>
                                <label className={styles.field}>
                                  Override reason
                                  <textarea
                                    className={styles.textarea}
                                    value={override.reason}
                                    disabled={!override.state}
                                    onChange={(event) =>
                                      setReviewOverrides((current) => ({
                                        ...current,
                                        [selectionId]: {
                                          ...override,
                                          reason: event.target.value
                                        }
                                      }))
                                    }
                                  />
                                </label>
                                <ConfirmAction
                                  triggerLabel="Save review"
                                  title="Save this rating review?"
                                  description="This records an admin review without closing registration eligibility."
                                  confirmLabel="Save review"
                                  confirmationText="SAVE RATING REVIEW"
                                  disabled={
                                    busy ||
                                    !selection?.updated_at ||
                                    Boolean(override.state && !override.reason.trim())
                                  }
                                  busy={busy}
                                  onConfirm={(confirmation) =>
                                    mutate(
                                      `review:${selectionId}:admin`,
                                      `${basePath}/rating-reviews`,
                                      {
                                        event_option_id: ratingEventId,
                                        selection_id: selectionId,
                                        review_phase: "ADMIN_REVIEW",
                                        override_state: override.state || null,
                                        override_reason: override.reason || null,
                                        expected_selection_updated_at:
                                          text(selection, "updated_at")
                                      },
                                      confirmation,
                                      "Rating review saved."
                                    )
                                  }
                                />
                              </div>
                            </div>
                          </article>
                        );
                      })}
                  </div>
                ) : (
                  <p className={styles.notice}>
                    No combined-rating entries need review for this event.
                  </p>
                )}
                {ratingEventId ? (
                  <ConfirmAction
                    triggerLabel="Finalize all at registration close"
                    title="Finalize combined-rating eligibility?"
                    description="This snapshots current ratings and blocks draw import for entries that are not eligible."
                    confirmLabel="Finalize reviews"
                    confirmationText="CLOSE RATING REVIEW"
                    disabled={
                      busy ||
                      !snapshot.combined_rating_entries.some(
                        (row) =>
                          text(row, "event_option_id") === ratingEventId &&
                          text(row, "review_phase") === "INITIAL"
                      )
                    }
                    busy={busy}
                    onConfirm={(confirmation) => {
                      const entries = snapshot.combined_rating_entries
                        .filter(
                          (row) =>
                            text(row, "event_option_id") === ratingEventId &&
                            text(row, "review_phase") === "INITIAL"
                        )
                        .map((review) => {
                          const selectionId = text(review, "selection_id");
                          const selection = selectionsById.get(selectionId);
                          const registration = registrationsById.get(
                            text(review, "registration_id")
                          );
                          const override = reviewOverrides[selectionId] || {
                            state: "",
                            reason: ""
                          };
                          return {
                            selection_id: selectionId,
                            expected_selection_updated_at: text(
                              selection,
                              "updated_at"
                            ),
                            registration_status:
                              text(registration, "status") || "CONFIRMED",
                            override_state: override.state || null,
                            override_reason: override.reason || null
                          };
                        });
                      if (
                        entries.some(
                          (entry) =>
                            entry.override_state && !entry.override_reason?.trim()
                        )
                      ) {
                        throw new Error(
                          "Every override needs a written reason before finalizing."
                        );
                      }
                      return mutate(
                        `review-close:${ratingEventId}`,
                        `${basePath}/rating-reviews/close`,
                        { event_option_id: ratingEventId, entries },
                        confirmation,
                        "Combined-rating eligibility finalized."
                      );
                    }}
                  />
                ) : null}
              </div>
            </section>
          ) : null}

          {tab === "teams" ? (
            <section className={styles.section} aria-labelledby="teams-heading">
              <div className={styles.card}>
                <h2 id="teams-heading">Create a four-player team</h2>
                {!fourPlayerEvents.length ? (
                  <p className={styles.notice}>
                    Configure an event as a four-player team format first.
                  </p>
                ) : (
                  <>
                    <div className={styles.grid}>
                      <label className={styles.field}>
                        Event
                        <select
                          className={styles.select}
                          value={teamEventId}
                          onChange={(event) => setTeamEventId(event.target.value)}
                        >
                          {fourPlayerEvents.map((event) => (
                            <option key={event.id} value={event.id}>
                              {eventLabel(event)}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label className={styles.field}>
                        Team name
                        <input
                          className={styles.input}
                          value={teamName}
                          onChange={(event) => setTeamName(event.target.value)}
                        />
                      </label>
                      <label className={styles.field}>
                        Captain registration
                        <select
                          className={styles.select}
                          value={captainRegistrationId}
                          onChange={(event) =>
                            setCaptainRegistrationId(event.target.value)
                          }
                        >
                          <option value="">Choose captain</option>
                          {confirmedRegistrations.map((registration) => (
                            <option
                              key={text(registration, "id")}
                              value={text(registration, "id")}
                            >
                              {registrationName(registration)}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <div className={styles.rosterGrid}>
                      {TEAM_SLOTS.map((slot) => (
                        <RosterSlotEditor
                          key={slot}
                          slot={slot}
                          draft={createRoster[slot]}
                          registrations={confirmedRegistrations}
                          onRegistration={(registrationId) =>
                            setRosterRegistration(
                              setCreateRoster,
                              slot,
                              registrationId
                            )
                          }
                          onChange={(patch) =>
                            updateRosterEntry(setCreateRoster, slot, patch)
                          }
                        />
                      ))}
                    </div>
                    <ConfirmAction
                      triggerLabel="Create team and send invitations"
                      title="Create this four-player team?"
                      description="The captain is accepted immediately. The other roster members receive dry-run or staging invitations according to the server email mode."
                      confirmLabel="Create team"
                      confirmationText="CREATE TEAM"
                      disabled={
                        busy ||
                        !teamEventId ||
                        !teamName.trim() ||
                        !captainRegistrationId ||
                        !TEAM_SLOTS.every(
                          (slot) =>
                            createRoster[slot].email.trim() &&
                            createRoster[slot].displayName.trim()
                        ) ||
                        !TEAM_SLOTS.some(
                          (slot) =>
                            createRoster[slot].registrationId ===
                            captainRegistrationId
                        )
                      }
                      busy={busy}
                      onConfirm={(confirmation) =>
                        mutate(
                          `team-create:${teamEventId}:${captainRegistrationId}`,
                          `${basePath}/teams`,
                          {
                            event_option_id: teamEventId,
                            team_name: teamName.trim(),
                            captain_registration_id: captainRegistrationId,
                            members: rosterPayload(createRoster)
                          },
                          confirmation,
                          "Team created and invitation delivery recorded."
                        )
                      }
                    />
                  </>
                )}
              </div>

              <div className={styles.card}>
                <h2>Invitation recovery</h2>
                <div className={styles.grid}>
                  <label className={styles.field}>
                    Roster member
                    <select
                      className={styles.select}
                      value={inviteMemberId}
                      onChange={(event) => {
                        const id = event.target.value;
                        setInviteMemberId(id);
                        const member = snapshot.members.find((row) => row.id === id);
                        setInviteEmail(member?.invited_email || "");
                      }}
                    >
                      <option value="">Choose invitation</option>
                      {snapshot.members
                        .filter((member) =>
                          ["INVITED", "DECLINED"].includes(
                            member.status.toUpperCase()
                          )
                        )
                        .map((member) => (
                          <option key={member.id} value={member.id}>
                            {teamNameById(member.team_id)} · {slotLabel(member.slot)} ·{" "}
                            {member.display_name || member.display_name_snapshot}
                          </option>
                        ))}
                    </select>
                  </label>
                  <label className={styles.field}>
                    Invitation email
                    <input
                      className={styles.input}
                      type="email"
                      value={inviteEmail}
                      onChange={(event) => setInviteEmail(event.target.value)}
                    />
                  </label>
                </div>
                <ConfirmAction
                  triggerLabel="Reissue invitation"
                  title="Reissue this team invitation?"
                  description="The old invitation becomes invalid. Delivery is recorded before the response returns."
                  confirmLabel="Reissue"
                  confirmationText="REISSUE INVITATION"
                  disabled={busy || !inviteMemberId || !inviteEmail.trim()}
                  busy={busy}
                  onConfirm={(confirmation) => {
                    const member = snapshot.members.find(
                      (row) => row.id === inviteMemberId
                    );
                    if (!member) throw new Error("Choose a roster invitation.");
                    return mutate(
                      `invite:${member.id}:${member.invitation_version}`,
                      `${basePath}/teams/${encodeURIComponent(
                        member.team_id
                      )}/invitations/reissue`,
                      {
                        member_id: member.id,
                        expected_invitation_version: member.invitation_version,
                        invited_email: inviteEmail.trim().toLowerCase()
                      },
                      confirmation,
                      "Invitation reissued."
                    );
                  }}
                />
              </div>

              <div className={styles.card}>
                <h2>Roster changes and withdrawal</h2>
                <div className={styles.grid}>
                  <label className={styles.field}>
                    Team
                    <select
                      className={styles.select}
                      value={manageTeamId}
                      onChange={(event) => chooseManageTeam(event.target.value)}
                    >
                      <option value="">Choose team</option>
                      {snapshot.teams
                        .filter(
                          (team) =>
                            !["WITHDRAWN", "CANCELLED"].includes(
                              team.status.toUpperCase()
                            )
                        )
                        .map((team) => (
                          <option key={team.id} value={team.id}>
                            {team.name} · {team.status.replaceAll("_", " ")}
                          </option>
                        ))}
                    </select>
                  </label>
                  <label className={styles.field}>
                    Change
                    <select
                      className={styles.select}
                      value={rosterAction}
                      onChange={(event) =>
                        setRosterAction(
                          event.target.value as "REPLACE" | "WITHDRAW"
                        )
                      }
                    >
                      <option value="REPLACE">Replace roster</option>
                      <option value="WITHDRAW">Withdraw team</option>
                    </select>
                  </label>
                </div>
                {rosterAction === "REPLACE" ? (
                  <div className={styles.rosterGrid}>
                    {TEAM_SLOTS.map((slot) => (
                      <RosterSlotEditor
                        key={slot}
                        slot={slot}
                        draft={replacementRoster[slot]}
                        registrations={confirmedRegistrations}
                        onRegistration={(registrationId) =>
                          setRosterRegistration(
                            setReplacementRoster,
                            slot,
                            registrationId
                          )
                        }
                        onChange={(patch) =>
                          updateRosterEntry(setReplacementRoster, slot, patch)
                        }
                      />
                    ))}
                  </div>
                ) : null}
                <label className={styles.field}>
                  Required reason
                  <textarea
                    className={styles.textarea}
                    value={rosterReason}
                    onChange={(event) => setRosterReason(event.target.value)}
                  />
                </label>
                <ConfirmAction
                  triggerLabel={
                    rosterAction === "WITHDRAW" ? "Withdraw team" : "Replace roster"
                  }
                  title={
                    rosterAction === "WITHDRAW"
                      ? "Withdraw this team?"
                      : "Replace this team roster?"
                  }
                  description="This change is version checked and retained in the tournament audit history."
                  confirmLabel={
                    rosterAction === "WITHDRAW" ? "Withdraw team" : "Replace roster"
                  }
                  confirmationText={
                    rosterAction === "WITHDRAW"
                      ? "WITHDRAW TEAM"
                      : "REPLACE ROSTER"
                  }
                  tone={rosterAction === "WITHDRAW" ? "danger" : "default"}
                  disabled={
                    busy ||
                    !selectedTeam ||
                    !rosterReason.trim() ||
                    (rosterAction === "REPLACE" &&
                      !TEAM_SLOTS.every(
                        (slot) =>
                          replacementRoster[slot].email.trim() &&
                          replacementRoster[slot].displayName.trim()
                      ))
                  }
                  busy={busy}
                  onConfirm={(confirmation) => {
                    if (!selectedTeam) throw new Error("Choose a team.");
                    return mutate(
                      `roster:${selectedTeam.id}:${selectedTeam.version}:${rosterAction}`,
                      `${basePath}/teams/${encodeURIComponent(
                        selectedTeam.id
                      )}/roster`,
                      {
                        expected_team_version: selectedTeam.version,
                        action: rosterAction,
                        members:
                          rosterAction === "REPLACE"
                            ? rosterPayload(replacementRoster)
                            : [],
                        reason: rosterReason.trim()
                      },
                      confirmation,
                      rosterAction === "WITHDRAW"
                        ? "Team withdrawn."
                        : "Team roster replaced."
                    );
                  }}
                />
              </div>
            </section>
          ) : null}

          {tab === "schedule" ? (
            <section className={styles.section} aria-labelledby="schedule-heading">
              <div className={styles.card}>
                <h2 id="schedule-heading">Round robin and playoffs</h2>
                {!teamDraws.length ? (
                  <p className={styles.notice}>
                    Create the event draw in{" "}
                    <Link href="/admin/tournaments/ops/draws">
                      Tournament Operations
                    </Link>{" "}
                    before building the team schedule.
                  </p>
                ) : (
                  <>
                    <label className={styles.field}>
                      Team draw
                      <select
                        className={styles.select}
                        value={drawId}
                        onChange={(event) => {
                          setDrawId(event.target.value);
                          setRoundRobinTeamIds([]);
                        }}
                      >
                        {teamDraws.map((draw) => (
                          <option key={draw.id} value={draw.id}>
                            {draw.name} · {draw.status}
                          </option>
                        ))}
                      </select>
                    </label>
                    <h3>Eligible teams</h3>
                    <div className={styles.checkGrid}>
                      {snapshot.teams
                        .filter(
                          (team) =>
                            team.event_option_id ===
                              String(selectedDraw?.event_option_id || "") &&
                            team.status === "CONFIRMED" &&
                            ["ELIGIBLE", "NOT_REQUIRED"].includes(
                              team.eligibility_state || ""
                            )
                        )
                        .map((team) => (
                          <label className={styles.check} key={team.id}>
                            <input
                              type="checkbox"
                              checked={roundRobinTeamIds.includes(team.id)}
                              onChange={(event) =>
                                setRoundRobinTeamIds((current) =>
                                  event.target.checked
                                    ? [...current, team.id]
                                    : current.filter((id) => id !== team.id)
                                )
                              }
                            />
                            {team.name}
                          </label>
                        ))}
                    </div>
                    <div className={styles.actions}>
                      <button
                        type="button"
                        className={styles.secondaryButton}
                        onClick={() =>
                          setRoundRobinTeamIds(
                            snapshot.teams
                              .filter(
                                (team) =>
                                  team.event_option_id ===
                                    String(selectedDraw?.event_option_id || "") &&
                                  team.status === "CONFIRMED" &&
                                  ["ELIGIBLE", "NOT_REQUIRED"].includes(
                                    team.eligibility_state || ""
                                  )
                              )
                              .map((team) => team.id)
                          )
                        }
                      >
                        Select all eligible
                      </button>
                      <ConfirmAction
                        triggerLabel="Build round robin"
                        title="Replace the team round-robin schedule?"
                        description="The selected teams play every opponent. Existing play blocks unsafe replacement."
                        confirmLabel="Build schedule"
                        confirmationText="BUILD TEAM SCHEDULE"
                        disabled={busy || !selectedDraw || roundRobinTeamIds.length < 2}
                        busy={busy}
                        onConfirm={(confirmation) => {
                          if (!selectedDraw) throw new Error("Choose a draw.");
                          return mutate(
                            `round-robin:${selectedDraw.id}:${selectedDraw.updated_at}`,
                            `${basePath}/draws/${encodeURIComponent(
                              selectedDraw.id
                            )}/round-robin`,
                            {
                              event_option_id: selectedDraw.event_option_id,
                              team_ids: roundRobinTeamIds,
                              expected_draw_updated_at: selectedDraw.updated_at
                            },
                            confirmation,
                            "Round-robin schedule built."
                          );
                        }}
                      />
                      {(() => {
                        const event = eventsById.get(
                          String(selectedDraw?.event_option_id || "")
                        );
                        if (!selectedDraw || event?.team_playoff_format === "NONE") {
                          return null;
                        }
                        return (
                          <ConfirmAction
                            triggerLabel="Build playoffs"
                            title="Build the configured playoff bracket?"
                            description="Final round-robin standings seed the configured playoff format."
                            confirmLabel="Build playoffs"
                            confirmationText="BUILD TEAM PLAYOFFS"
                            disabled={busy}
                            busy={busy}
                            onConfirm={(confirmation) =>
                              mutate(
                                `playoffs:${selectedDraw.id}:${selectedDraw.updated_at}`,
                                `${basePath}/draws/${encodeURIComponent(
                                  selectedDraw.id
                                )}/playoffs`,
                                {
                                  playoff_format: event?.team_playoff_format,
                                  expected_draw_updated_at: selectedDraw.updated_at
                                },
                                confirmation,
                                "Playoff bracket built."
                              )
                            }
                          />
                        );
                      })()}
                    </div>
                  </>
                )}
              </div>
            </section>
          ) : null}

          {tab === "matches" ? (
            <section className={styles.section} aria-labelledby="matches-heading">
              <div className={styles.card}>
                <h2 id="matches-heading">Blind lineups</h2>
                <p className={styles.hint}>
                  Each team locks mixed pairings before either lineup is revealed.
                  Women’s doubles, men’s doubles, mixed one, and mixed two remain
                  in the fixed order.
                </p>
                <div className={styles.stack}>
                  {snapshot.matchups.map((matchup) => (
                    <MatchupLineups
                      key={matchup.id}
                      matchup={matchup}
                      event={eventsById.get(matchup.event_option_id)}
                      teamNameById={teamNameById}
                      teamMembers={teamMembers}
                      existingLineups={snapshot.lineups}
                      drafts={lineupDrafts}
                      setDrafts={setLineupDrafts}
                      busy={busy}
                      onLock={(teamId, draft, confirmation) => {
                        const existing = snapshot.lineups.find(
                          (row) =>
                            text(row, "matchup_id") === matchup.id &&
                            text(row, "team_id") === teamId
                        );
                        return mutate(
                          `lineup:${matchup.id}:${teamId}:${matchup.version}:${
                            existing ? numberValue(existing, "version") : 0
                          }`,
                          `${basePath}/matchups/${encodeURIComponent(
                            matchup.id
                          )}/lineups`,
                          {
                            team_id: teamId,
                            mixed_pairing: draft.mixed,
                            singles_tiebreak_player_id: draft.tiebreakPlayerId
                              ? Number(draft.tiebreakPlayerId)
                              : null,
                            expected_matchup_version: matchup.version,
                            expected_lineup_version: existing
                              ? numberValue(existing, "version")
                              : null
                          },
                          confirmation,
                          "Team lineup locked."
                        );
                      }}
                    />
                  ))}
                  {!snapshot.matchups.length ? (
                    <p className={styles.notice}>Build a team schedule first.</p>
                  ) : null}
                </div>
              </div>

              <div className={styles.card}>
                <h2>Game scores and official match recovery</h2>
                <p className={styles.hint}>
                  Score each child game here. Publish rating-eligible child
                  matches in{" "}
                  <Link href="/admin/tournaments/ops/publish">
                    Official Publish
                  </Link>
                  , then reconcile only if an official row differs.
                </p>
                <div className={styles.tableWrap}>
                  <table className={styles.table}>
                    <thead>
                      <tr>
                        <th>Match</th>
                        <th>Game</th>
                        <th>Score</th>
                        <th>Official state</th>
                        <th>Recovery</th>
                      </tr>
                    </thead>
                    <tbody>
                      {snapshot.games.map((game) => {
                        const matchup = matchupsById.get(game.matchup_id);
                        const score = scoreDrafts[game.id] || {
                          scoreA:
                            game.score_a == null ? "" : String(game.score_a),
                          scoreB:
                            game.score_b == null ? "" : String(game.score_b)
                        };
                        const reconcile = reconcileDrafts[game.id] || {
                          officialMatchId: "",
                          reason: ""
                        };
                        const officialChoices = snapshot.canonical_matches.filter(
                          (row) =>
                            text(row, "tournament_game_id") ===
                            String(game.tournament_game_id || "")
                        );
                        const selectedOfficial = officialChoices.find(
                          (row) => text(row, "id") === reconcile.officialMatchId
                        );
                        return (
                          <tr key={game.id}>
                            <td>
                              {teamNameById(matchup?.team_a_id)} vs{" "}
                              {teamNameById(matchup?.team_b_id)}
                              <br />
                              <span className={styles.muted}>
                                {matchup?.stage} · round {matchup?.round_number}
                              </span>
                            </td>
                            <td>
                              <strong>{game.game_code.replaceAll("_", " ")}</strong>
                              <br />
                              <span className={styles.muted}>
                                {game.match_format}
                              </span>
                            </td>
                            <td>
                              <div className={styles.actions}>
                                <input
                                  className={styles.input}
                                  style={{ width: "5rem" }}
                                  aria-label={`${game.game_code} team A score`}
                                  type="number"
                                  min="0"
                                  value={score.scoreA}
                                  onChange={(event) =>
                                    setScoreDrafts((current) => ({
                                      ...current,
                                      [game.id]: {
                                        ...score,
                                        scoreA: event.target.value
                                      }
                                    }))
                                  }
                                />
                                <input
                                  className={styles.input}
                                  style={{ width: "5rem" }}
                                  aria-label={`${game.game_code} team B score`}
                                  type="number"
                                  min="0"
                                  value={score.scoreB}
                                  onChange={(event) =>
                                    setScoreDrafts((current) => ({
                                      ...current,
                                      [game.id]: {
                                        ...score,
                                        scoreB: event.target.value
                                      }
                                    }))
                                  }
                                />
                              </div>
                              <ConfirmAction
                                triggerLabel="Save score"
                                title="Finalize this team game score?"
                                description="The score is version checked and updates the parent team matchup."
                                confirmLabel="Save score"
                                confirmationText="SAVE TEAM SCORE"
                                disabled={
                                  busy ||
                                  !matchup ||
                                  score.scoreA === "" ||
                                  score.scoreB === "" ||
                                  score.scoreA === score.scoreB
                                }
                                busy={busy}
                                onConfirm={(confirmation) => {
                                  if (!matchup) throw new Error("Matchup is missing.");
                                  return mutate(
                                    `score:${game.id}:${game.version}:${matchup.version}`,
                                    `${basePath}/games/${encodeURIComponent(
                                      game.id
                                    )}/score`,
                                    {
                                      score_a: Number(score.scoreA),
                                      score_b: Number(score.scoreB),
                                      expected_game_version: game.version,
                                      expected_matchup_version: matchup.version
                                    },
                                    confirmation,
                                    "Team game score saved."
                                  );
                                }}
                              />
                            </td>
                            <td>
                              <span className={styles.pill}>
                                {snapshot.game_publish_state[game.id] ||
                                  "NOT READY"}
                              </span>
                            </td>
                            <td>
                              {officialChoices.length && matchup ? (
                                <div className={styles.stack}>
                                  <label className={styles.field}>
                                    Official match
                                    <select
                                      className={styles.select}
                                      value={reconcile.officialMatchId}
                                      onChange={(event) =>
                                        setReconcileDrafts((current) => ({
                                          ...current,
                                          [game.id]: {
                                            ...reconcile,
                                            officialMatchId: event.target.value
                                          }
                                        }))
                                      }
                                    >
                                      <option value="">Choose official row</option>
                                      {officialChoices.map((row) => (
                                        <option
                                          key={text(row, "id")}
                                          value={text(row, "id")}
                                        >
                                          {text(row, "player1_name") ||
                                            "Published match"}{" "}
                                          {text(row, "player1_score")}–
                                          {text(row, "player2_score")}
                                        </option>
                                      ))}
                                    </select>
                                  </label>
                                  <label className={styles.field}>
                                    Correction reason
                                    <textarea
                                      className={styles.textarea}
                                      value={reconcile.reason}
                                      onChange={(event) =>
                                        setReconcileDrafts((current) => ({
                                          ...current,
                                          [game.id]: {
                                            ...reconcile,
                                            reason: event.target.value
                                          }
                                        }))
                                      }
                                    />
                                  </label>
                                  <ConfirmAction
                                    triggerLabel="Reconcile official match"
                                    title="Reconcile this official match?"
                                    description="This recovery action locks both rows, records the reason, and preserves rating history."
                                    confirmLabel="Reconcile"
                                    confirmationText="RECONCILE TEAM SCORE"
                                    disabled={
                                      busy ||
                                      !selectedOfficial ||
                                      !reconcile.reason.trim()
                                    }
                                    busy={busy}
                                    onConfirm={(confirmation) =>
                                      mutate(
                                        `reconcile:${game.id}:${game.version}:${
                                          matchup.version
                                        }:${text(selectedOfficial, "row_version")}`,
                                        `${basePath}/games/${encodeURIComponent(
                                          game.id
                                        )}/reconcile`,
                                        {
                                          official_match_id: text(
                                            selectedOfficial,
                                            "id"
                                          ),
                                          expected_official_row_version:
                                            numberValue(
                                              selectedOfficial,
                                              "row_version",
                                              numberValue(
                                                selectedOfficial,
                                                "version",
                                                1
                                              )
                                            ),
                                          expected_game_version: game.version,
                                          expected_matchup_version:
                                            matchup.version,
                                          reason: reconcile.reason.trim()
                                        },
                                        confirmation,
                                        "Official match reconciled."
                                      )
                                    }
                                  />
                                </div>
                              ) : (
                                <span className={styles.muted}>
                                  No official child match to reconcile.
                                </span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          ) : null}

          {tab === "podium" ? (
            <section className={styles.section} aria-labelledby="podium-heading">
              <div className={styles.card}>
                <h2 id="podium-heading">Calculated podium</h2>
                <p className={styles.hint}>
                  Placements are derived from completed standings and playoffs;
                  they cannot be entered by hand.
                </p>
                <div className={styles.stack}>
                  {teamDraws.map((draw) => {
                    const calculated =
                      snapshot.calculated_podium_by_draw[draw.id] || null;
                    const existing = snapshot.podium.filter(
                      (row) => text(row, "draw_id") === draw.id
                    );
                    const reason = podiumReasons[draw.id] || "";
                    return (
                      <article className={styles.subcard} key={draw.id}>
                        <h3>{draw.name}</h3>
                        {calculated ? (
                          <ol>
                            {calculated.map((row) => (
                              <li key={row.placement}>
                                {teamNameById(row.team_id)}
                              </li>
                            ))}
                          </ol>
                        ) : (
                          <p className={styles.notice}>
                            Finish required match results before saving a podium.
                          </p>
                        )}
                        {existing.some((row) => text(row, "published_at")) ? (
                          <p>
                            <span className={styles.pill}>Published</span>
                          </p>
                        ) : null}
                        <label className={styles.field}>
                          Correction reason
                          <textarea
                            className={styles.textarea}
                            value={reason}
                            onChange={(event) =>
                              setPodiumReasons((current) => ({
                                ...current,
                                [draw.id]: event.target.value
                              }))
                            }
                            placeholder="Required when changing a published podium"
                          />
                        </label>
                        <div className={styles.actions}>
                          <ConfirmAction
                            triggerLabel="Save podium draft"
                            title="Save the calculated podium?"
                            description="This stores the current calculated placements without making the draw public."
                            confirmLabel="Save draft"
                            confirmationText="SAVE TEAM PODIUM"
                            disabled={busy || !calculated}
                            busy={busy}
                            onConfirm={(confirmation) =>
                              mutate(
                                `podium:${draw.id}:${draw.updated_at}:draft`,
                                `${basePath}/draws/${encodeURIComponent(
                                  draw.id
                                )}/podium`,
                                {
                                  expected_draw_updated_at: draw.updated_at,
                                  publish: false,
                                  reason: reason.trim(),
                                  podium: calculated
                                },
                                confirmation,
                                "Podium draft saved."
                              )
                            }
                          />
                          <ConfirmAction
                            triggerLabel="Publish podium and results"
                            title="Publish this team podium?"
                            description="The calculated placements and draw results become visible on the club results page."
                            confirmLabel="Publish podium"
                            confirmationText="PUBLISH TEAM PODIUM"
                            disabled={busy || !calculated}
                            busy={busy}
                            onConfirm={(confirmation) =>
                              mutate(
                                `podium:${draw.id}:${draw.updated_at}:publish`,
                                `${basePath}/draws/${encodeURIComponent(
                                  draw.id
                                )}/podium`,
                                {
                                  expected_draw_updated_at: draw.updated_at,
                                  publish: true,
                                  reason: reason.trim(),
                                  podium: calculated
                                },
                                confirmation,
                                "Podium and results published."
                              )
                            }
                          />
                        </div>
                      </article>
                    );
                  })}
                  {!teamDraws.length ? (
                    <p className={styles.notice}>
                      Create and complete a team draw before publishing results.
                    </p>
                  ) : null}
                </div>
              </div>
            </section>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

function RosterSlotEditor({
  slot,
  draft,
  registrations,
  onRegistration,
  onChange
}: {
  slot: TeamSlot;
  draft: RosterEntry;
  registrations: Array<Record<string, unknown>>;
  onRegistration: (registrationId: string) => void;
  onChange: (patch: Partial<RosterEntry>) => void;
}) {
  return (
    <div className={styles.slot}>
      <h4>{slotLabel(slot)}</h4>
      <label className={styles.field}>
        Existing registration (optional)
        <select
          className={styles.select}
          value={draft.registrationId}
          onChange={(event) => onRegistration(event.target.value)}
        >
          <option value="">Invite by email</option>
          {registrations.map((registration) => (
            <option
              key={text(registration, "id")}
              value={text(registration, "id")}
            >
              {registrationName(registration)}
            </option>
          ))}
        </select>
      </label>
      <label className={styles.field}>
        Name
        <input
          className={styles.input}
          value={draft.displayName}
          onChange={(event) => onChange({ displayName: event.target.value })}
        />
      </label>
      <label className={styles.field}>
        Email
        <input
          className={styles.input}
          type="email"
          value={draft.email}
          onChange={(event) => onChange({ email: event.target.value })}
        />
      </label>
    </div>
  );
}

function MatchupLineups({
  matchup,
  event,
  teamNameById,
  teamMembers,
  existingLineups,
  drafts,
  setDrafts,
  busy,
  onLock
}: {
  matchup: TeamTournamentMatchup;
  event?: TeamCompetitionEvent;
  teamNameById: (teamId: string | null | undefined) => string;
  teamMembers: (teamId: string) => FourPlayerTeamMember[];
  existingLineups: Array<Record<string, unknown>>;
  drafts: Record<
    string,
    { mixed: "STRAIGHT" | "CROSS"; tiebreakPlayerId: string }
  >;
  setDrafts: Dispatch<
    SetStateAction<
      Record<
        string,
        { mixed: "STRAIGHT" | "CROSS"; tiebreakPlayerId: string }
      >
    >
  >;
  busy: boolean;
  onLock: (
    teamId: string,
    draft: { mixed: "STRAIGHT" | "CROSS"; tiebreakPlayerId: string },
    confirmation: string
  ) => Promise<ActionSuccess>;
}) {
  const teamIds = [matchup.team_a_id, matchup.team_b_id].filter(
    (value): value is string => Boolean(value)
  );
  return (
    <article className={styles.subcard}>
      <h3>
        {teamNameById(matchup.team_a_id)} vs {teamNameById(matchup.team_b_id)}
      </h3>
      <p className={styles.muted}>
        {matchup.stage} · round {matchup.round_number} · match {matchup.slot_number}
      </p>
      <div className={styles.grid}>
        {teamIds.map((teamId) => {
          const key = `${matchup.id}:${teamId}`;
          const existing = existingLineups.find(
            (row) =>
              text(row, "matchup_id") === matchup.id &&
              text(row, "team_id") === teamId
          );
          const draft = drafts[key] || {
            mixed:
              text(existing, "mixed_pairing") === "CROSS"
                ? ("CROSS" as const)
                : ("STRAIGHT" as const),
            tiebreakPlayerId: text(existing, "singles_tiebreak_player_id")
          };
          const candidates = teamMembers(teamId).filter(
            (member) => member.status === "ACCEPTED" && member.player_id != null
          );
          return (
            <div className={styles.slot} key={teamId}>
              <h4>{teamNameById(teamId)}</h4>
              <label className={styles.field}>
                Mixed doubles pairings
                <select
                  className={styles.select}
                  value={draft.mixed}
                  onChange={(change) =>
                    setDrafts((current) => ({
                      ...current,
                      [key]: {
                        ...draft,
                        mixed: change.target.value as "STRAIGHT" | "CROSS"
                      }
                    }))
                  }
                >
                  <option value="STRAIGHT">Man 1 + Woman 1</option>
                  <option value="CROSS">Man 1 + Woman 2</option>
                </select>
              </label>
              {event?.team_tiebreak_mode === "SINGLES" ? (
                <label className={styles.field}>
                  Singles tiebreak player
                  <select
                    className={styles.select}
                    value={draft.tiebreakPlayerId}
                    onChange={(change) =>
                      setDrafts((current) => ({
                        ...current,
                        [key]: {
                          ...draft,
                          tiebreakPlayerId: change.target.value
                        }
                      }))
                    }
                  >
                    <option value="">Choose player</option>
                    {candidates.map((member) => (
                      <option key={member.id} value={String(member.player_id)}>
                        {member.display_name || member.display_name_snapshot}
                      </option>
                    ))}
                  </select>
                </label>
              ) : (
                <p className={styles.hint}>
                  Skinny-singles relay uses all four roster players.
                </p>
              )}
              <p>
                <span className={styles.pill}>
                  {existing ? text(existing, "status") : "NOT LOCKED"}
                </span>
              </p>
              <ConfirmAction
                triggerLabel={existing ? "Update locked lineup" : "Lock lineup"}
                title={`Lock ${teamNameById(teamId)} lineup?`}
                description="The opposing lineup remains hidden until both teams have locked."
                confirmLabel="Lock lineup"
                confirmationText="LOCK TEAM LINEUP"
                disabled={
                  busy ||
                  (event?.team_tiebreak_mode === "SINGLES" &&
                    !draft.tiebreakPlayerId)
                }
                busy={busy}
                onConfirm={(confirmation) => onLock(teamId, draft, confirmation)}
              />
            </div>
          );
        })}
      </div>
    </article>
  );
}
