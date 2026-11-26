"use client";

import { useState } from "react";
import FourPlayerTeamRegistrationCard, {
  TEAM_SLOTS,
  TeamRegistrationDraft,
  invitedTeamMembers,
  newTeamMembers,
  validateTeamInvites,
  newTeamRegistrationDraft,
  teamSlotLabel,
  validateTeamRegistrationDraft
} from "@/components/tournaments/FourPlayerTeamRegistrationCard";
import {
  PublicFourPlayerTeamSetupRecovery,
  createPublicFourPlayerTeam,
  recoverPublicFourPlayerTeamSetup
} from "@/lib/tournamentTeamCompetitionApi";
import { publicTournamentEventLabel } from "@/lib/tournamentRegistrationEligibility";

type Props = {
  clubSlug: string;
  confirmationToken: string;
  initialRecovery: PublicFourPlayerTeamSetupRecovery;
  needsAttention?: boolean;
};

const cardStyle = {
  border: "1px solid #e2e8f0",
  borderRadius: "14px",
  padding: "1rem",
  background: "white"
};

function teamStatusLabel(status?: string | null): string {
  switch (String(status || "").toUpperCase()) {
    case "CONFIRMED":
      return "Confirmed";
    case "FORMING":
      return "Team forming";
    case "WAITLIST":
      return "On the waitlist";
    case "REVIEW_REQUIRED":
      return "Organizer review needed";
    case "INELIGIBLE":
      return "Not eligible";
    case "WITHDRAWN":
      return "Withdrawn";
    case "CANCELLED":
    case "CANCELED":
      return "Canceled";
    default:
      return "Team saved";
  }
}

function memberStatusLabel(status?: string | null): string {
  switch (String(status || "").toUpperCase()) {
    case "ACCEPTED":
      return "Confirmed";
    case "INVITED":
      return "Invitation pending";
    case "DECLINED":
      return "Declined";
    case "REMOVED":
      return "Removed";
    case "CANCELLED":
    case "CANCELED":
      return "Canceled";
    default:
      return "Status unavailable";
  }
}

function initialDrafts(
  recovery: PublicFourPlayerTeamSetupRecovery
): Record<string, TeamRegistrationDraft> {
  return Object.fromEntries(
    recovery.events.map((event) => [
      event.id,
      newTeamRegistrationDraft(recovery.captain.gender || "")
    ])
  );
}

export default function FourPlayerTeamSetupRecovery({
  clubSlug,
  confirmationToken,
  initialRecovery,
  needsAttention = false
}: Props) {
  const [recovery, setRecovery] = useState(initialRecovery);
  const [drafts, setDrafts] = useState(() => initialDrafts(initialRecovery));
  const [pendingEventId, setPendingEventId] = useState("");
  const [message, setMessage] = useState<string | null>(
    needsAttention && recovery.events.some((event) => event.setup_state === "SETUP_REQUIRED")
      ? "Your registration is saved. Enter the teammates you have to finish your team signup."
      : null
  );

  async function refreshRecovery(): Promise<PublicFourPlayerTeamSetupRecovery | null> {
    const response = await recoverPublicFourPlayerTeamSetup(
      clubSlug,
      confirmationToken
    );
    if (response.error || !response.data) {
      setMessage(
        "We couldn’t refresh your team. Try again or contact the organizer."
      );
      return null;
    }
    const updated = response.data;
    setDrafts((current) => {
      const next = { ...current };
      for (const event of updated.events) {
        next[event.id] ??= newTeamRegistrationDraft(updated.captain.gender || "");
      }
      return next;
    });
    setRecovery(updated);
    return updated;
  }

  async function submitTeam(eventId: string) {
    const event = recovery.events.find((row) => row.id === eventId);
    const draft = drafts[eventId];
    if (!event || !draft) return;
    const team = event.team;
    const slots = team?.open_slots || [];
    const captain = recovery.captain;
    const validation = team
      ? validateTeamInvites(draft, slots, captain.email)
      : validateTeamRegistrationDraft(draft, captain.email, captain.gender || "");
    if (validation) {
      setMessage(`${event.division_name}: ${validation}`);
      return;
    }
    if (
      String(recovery.captain.registration_status || "").toUpperCase() !==
      "CONFIRMED"
    ) {
      setMessage(
        "The organizer needs to review your registration before you can finish the team."
      );
      return;
    }

    setPendingEventId(eventId);
    setMessage(null);
    const response = await createPublicFourPlayerTeam(clubSlug, {
      tournament_id: recovery.tournament.id,
      event_option_id: event.id,
      team_name: team ? team.name : draft.teamName.trim(),
      fill_team_id: team?.id,
      captain_registration_id: recovery.captain.registration_id,
      confirmation_token: confirmationToken,
      members: team ? invitedTeamMembers(draft, slots) : newTeamMembers(draft, {
        registration_id: captain.registration_id,
        email: captain.email,
        display_name: captain.display_name,
        gender: captain.gender
      }),
      idempotency_key: draft.idempotencyKey,
      website: ""
    });

    // Always consult server state.  A team may have committed even if email
    // delivery or the browser response failed after the database transaction.
    const refreshed = await refreshRecovery();
    setPendingEventId("");
    const refreshedEvent = refreshed?.events.find((row) => row.id === eventId);
    if (!refreshedEvent) return;
    if (refreshedEvent.team && (!team || !response.error)) {
      setDrafts((current) => ({ ...current, [eventId]: newTeamRegistrationDraft(captain.gender || "") }));
      setMessage("Team saved.");
      return;
    }
    setMessage(response.error || "We couldn’t save your team yet. Check the roster and try again.");
  }

  if (!recovery.events.length) return null;

  return (
    <section style={{ marginTop: "1rem" }} data-testid="team-setup-recovery">
      <h2>Your four-player team</h2>
      <p style={{ color: "#475569" }}>
        Check your team below. If you registered on your own, the organizer can place you on a team.
      </p>
      {message ? (
        <p
          role="status"
          style={{
            color: "#92400e",
            background: "#fffbeb",
            borderRadius: "10px",
            padding: "0.75rem"
          }}
        >
          {message}
        </p>
      ) : null}
      <div style={{ display: "grid", gap: "1rem" }}>
        {recovery.events.map((event) => {
          if (event.setup_state === "STAFF_RECOVERY_REQUIRED") {
            return (
              <article key={event.id} style={cardStyle}>
                <h3 style={{ marginTop: 0 }}>
                  {publicTournamentEventLabel(event.event_family_label, event.division_name)}
                </h3>
                <p style={{ color: "#991b1b" }}>
                  The organizer needs to finish setting up this team. You don’t
                  need to register again.
                </p>
              </article>
            );
          }
          const team = event.team;
          const showEditor = !team || Boolean(team.is_captain && team.status === "FORMING" && team.open_slots?.length);
          return (
            <article key={event.id} style={cardStyle}>
              <h3 style={{ marginTop: 0 }}>{publicTournamentEventLabel(event.event_family_label, event.division_name)}</h3>
              {team ? (
                <>
                  <p><strong>{team.name}</strong> · {teamStatusLabel(team.status)}</p>
                  <ul>
                    {TEAM_SLOTS.map((slot) => {
                      const member = team.members.find((player) =>
                        player.slot === slot && ["ACCEPTED", "INVITED"].includes(player.status)
                      );
                      return <li key={slot}>{teamSlotLabel(slot)}: {member
                        ? `${member.display_name} · ${memberStatusLabel(member.status)}`
                        : "Need a player"}</li>;
                    })}
                  </ul>
                </>
              ) : (
                <p data-testid="team-placement-pending">
                  {needsAttention && event.setup_state === "SETUP_REQUIRED"
                    ? "Your individual registration is saved. Finish entering your team below."
                    : "Your individual registration is saved. You’re listed as needing a team until you join one."}
                </p>
              )}
              {showEditor && <details open={(!team && needsAttention) || undefined}>
                <summary style={{ cursor: "pointer", fontWeight: 700 }}>{team ? "Invite players to open spots" : "I have a team to add"}</summary>
                <FourPlayerTeamRegistrationCard
                  event={{ ...event, selectable: true }}
                  openSlots={team?.open_slots}
                  captainName={recovery.captain.display_name}
                  captainEmail={recovery.captain.email}
                  captainGender={recovery.captain.gender || ""}
                  value={drafts[event.id]}
                  onChange={(next) =>
                    setDrafts((current) => ({
                      ...current,
                      [event.id]: { ...next, idempotencyKey: crypto.randomUUID() }
                    }))
                  }
                />
                <button
                  type="button"
                  disabled={Boolean(pendingEventId)}
                  onClick={() => void submitTeam(event.id)}
                  style={{
                    marginTop: "0.75rem",
                    padding: "0.7rem 1rem",
                    borderRadius: "10px",
                    border: "1px solid #0f172a",
                    background: "#0f172a",
                    color: "white",
                    fontWeight: 800,
                    cursor: "pointer"
                  }}
                >
                  {pendingEventId === event.id
                    ? "Saving…"
                    : team ? "Invite players" : "Save team"}
                </button>
              </details>}
            </article>
          );
        })}
      </div>
    </section>
  );
}
