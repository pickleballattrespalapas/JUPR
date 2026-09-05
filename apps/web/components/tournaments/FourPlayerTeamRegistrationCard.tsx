"use client";

import type { PublicRegistrationEvent } from "@/lib/tournamentRegistrationApi";
import { publicTournamentEventLabel } from "@/lib/tournamentRegistrationEligibility";
import styles from "./TournamentTeamCompetition.module.css";

export const TEAM_SLOTS = [
  "MAN_1",
  "MAN_2",
  "WOMAN_1",
  "WOMAN_2"
] as const;

export type TeamSlot = (typeof TEAM_SLOTS)[number];

export type TeamRegistrationDraft = {
  teamName: string;
  captainSlot: TeamSlot | "";
  teammates: Record<TeamSlot, { displayName: string; email: string }>;
  idempotencyKey: string;
};

const slotLabels: Record<TeamSlot, string> = {
  MAN_1: "Men’s spot 1",
  MAN_2: "Men’s spot 2",
  WOMAN_1: "Women’s spot 1",
  WOMAN_2: "Women’s spot 2"
};

export function teamSlotLabel(slot: string): string {
  return slotLabels[slot as TeamSlot] || "Roster spot";
}

export function defaultCaptainSlot(gender: string): TeamSlot | "" {
  const normalized = String(gender || "").toLowerCase();
  if (normalized.startsWith("men") || normalized === "man" || normalized === "male") {
    return "MAN_1";
  }
  if (
    normalized.startsWith("women") ||
    normalized === "woman" ||
    normalized === "female"
  ) {
    return "WOMAN_1";
  }
  return "";
}

export function newTeamRegistrationDraft(
  captainGender: string
): TeamRegistrationDraft {
  return {
    teamName: "",
    captainSlot: defaultCaptainSlot(captainGender),
    teammates: {
      MAN_1: { displayName: "", email: "" },
      MAN_2: { displayName: "", email: "" },
      WOMAN_1: { displayName: "", email: "" },
      WOMAN_2: { displayName: "", email: "" }
    },
    idempotencyKey: crypto.randomUUID()
  };
}

export function validateTeamRegistrationDraft(
  draft: TeamRegistrationDraft | undefined,
  captainEmail: string,
  captainGender: string
): string | null {
  if (!draft?.teamName.trim()) return "Enter a team name.";
  if (!draft.captainSlot) return "Choose your roster spot.";
  const expectedCaptainSlot = defaultCaptainSlot(captainGender);
  if (!expectedCaptainSlot) {
    return "Choose Men or Women in your player details before setting up this team.";
  }
  if (
    (expectedCaptainSlot.startsWith("MAN_") &&
      !draft.captainSlot.startsWith("MAN_")) ||
    (expectedCaptainSlot.startsWith("WOMAN_") &&
      !draft.captainSlot.startsWith("WOMAN_"))
  ) {
    return "Choose the men’s or women’s spot that matches your selection above.";
  }
  const emails = [captainEmail.trim().toLowerCase()];
  for (const slot of TEAM_SLOTS) {
    if (slot === draft.captainSlot) continue;
    const teammate = draft.teammates[slot];
    if (!teammate.displayName.trim() || !teammate.email.trim()) {
      return `Enter a name and email for ${slotLabels[slot]}.`;
    }
    emails.push(teammate.email.trim().toLowerCase());
  }
  if (new Set(emails).size !== 4) {
    return "Every player needs a different email address.";
  }
  return null;
}

type Props = {
  event: PublicRegistrationEvent;
  captainName: string;
  captainEmail: string;
  captainGender: string;
  value: TeamRegistrationDraft;
  onChange: (value: TeamRegistrationDraft) => void;
};

export default function FourPlayerTeamRegistrationCard({
  event,
  captainName,
  captainEmail,
  value,
  onChange
}: Props) {
  function updateTeammate(
    slot: TeamSlot,
    patch: Partial<{ displayName: string; email: string }>
  ) {
    onChange({
      ...value,
      teammates: {
        ...value.teammates,
        [slot]: { ...value.teammates[slot], ...patch }
      }
    });
  }

  return (
    <section className={styles.card} data-testid="four-player-team-roster">
      <div className={styles.header}>
        <div>
          <h4>Four-player team roster</h4>
          <p className={styles.hint}>
            {publicTournamentEventLabel(event.event_family_label, event.division_name)}
          </p>
        </div>
        <strong>2 men · 2 women</strong>
      </div>

      <label className={styles.field}>
        Team name
        <input
          className={styles.input}
          value={value.teamName}
          onChange={(change) =>
            onChange({ ...value, teamName: change.target.value })
          }
          autoComplete="organization"
        />
      </label>

      <label className={styles.field}>
        Your roster spot
        <select
          className={styles.input}
          value={value.captainSlot}
          onChange={(change) =>
            onChange({
              ...value,
              captainSlot: change.target.value as TeamSlot | ""
            })
          }
        >
          <option value="">Choose a spot</option>
          {TEAM_SLOTS.map((slot) => (
            <option key={slot} value={slot}>
              {slotLabels[slot]}
            </option>
          ))}
        </select>
      </label>

      <div className={styles.grid}>
        {TEAM_SLOTS.map((slot) => {
          const captain = slot === value.captainSlot;
          const teammate = value.teammates[slot];
          return (
            <article className={styles.slot} key={slot}>
              <h5>{slotLabels[slot]}</h5>
              {captain ? (
                <p>
                  <strong>{captainName || "Team captain"}</strong>
                  <br />
                  {captainEmail}
                </p>
              ) : (
                <>
                  <label className={styles.field}>
                    Player name
                    <input
                      className={styles.input}
                      value={teammate.displayName}
                      onChange={(change) =>
                        updateTeammate(slot, {
                          displayName: change.target.value
                        })
                      }
                    />
                  </label>
                  <label className={styles.field}>
                    Player email
                    <input
                      className={styles.input}
                      type="email"
                      value={teammate.email}
                      onChange={(change) =>
                        updateTeammate(slot, { email: change.target.value })
                      }
                    />
                  </label>
                </>
              )}
            </article>
          );
        })}
      </div>

      <ul className={styles.rules}>
        <li>We’ll invite each teammate at the email you enter. They should register using that address.</li>
        <li>Match order is women&apos;s, men&apos;s, mixed 1, mixed 2, then a tiebreaker if needed.</li>
        <li>
          Substitutes: {event.team_allow_substitutes ? "allowed with organizer approval" : "not allowed after play starts"}.
        </li>
      </ul>
    </section>
  );
}
