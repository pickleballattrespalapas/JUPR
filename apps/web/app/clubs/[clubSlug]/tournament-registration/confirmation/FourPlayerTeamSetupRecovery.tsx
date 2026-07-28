"use client";

import { useState } from "react";
import FourPlayerTeamRegistrationCard, {
  TEAM_SLOTS,
  TeamRegistrationDraft,
  newTeamRegistrationDraft,
  validateTeamRegistrationDraft
} from "@/components/tournaments/FourPlayerTeamRegistrationCard";
import {
  PublicFourPlayerTeamSetupRecovery,
  createPublicFourPlayerTeam,
  recoverPublicFourPlayerTeamSetup
} from "@/lib/tournamentTeamCompetitionApi";
import type { PublicRegistrationEvent } from "@/lib/tournamentRegistrationApi";

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

function initialDrafts(
  recovery: PublicFourPlayerTeamSetupRecovery
): Record<string, TeamRegistrationDraft> {
  return Object.fromEntries(
    recovery.events
      .filter((event) => event.setup_state === "SETUP_REQUIRED")
      .map((event) => [
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
    needsAttention
      ? "Your registration is safe. Review the durable team setup status below."
      : null
  );

  async function refreshRecovery(): Promise<PublicFourPlayerTeamSetupRecovery | null> {
    const response = await recoverPublicFourPlayerTeamSetup(
      clubSlug,
      confirmationToken
    );
    if (response.error || !response.data) {
      setMessage(
        response.error ||
          "Unable to refresh the durable team setup. Contact tournament staff."
      );
      return null;
    }
    setRecovery(response.data);
    setDrafts((current) => {
      const next = { ...current };
      response.data!.events.forEach((event) => {
        if (event.setup_state === "SETUP_REQUIRED" && !next[event.id]) {
          next[event.id] = newTeamRegistrationDraft(
            response.data!.captain.gender || ""
          );
        }
      });
      return next;
    });
    return response.data;
  }

  async function submitTeam(eventId: string) {
    const event = recovery.events.find((row) => row.id === eventId);
    const draft = drafts[eventId];
    if (!event || !draft) return;
    const validation = validateTeamRegistrationDraft(
      draft,
      recovery.captain.email,
      recovery.captain.gender || ""
    );
    if (validation) {
      setMessage(`${event.division_name}: ${validation}`);
      return;
    }
    if (
      String(recovery.captain.registration_status || "").toUpperCase() !==
      "CONFIRMED"
    ) {
      setMessage(
        "This registration is no longer confirmed. Contact tournament staff before creating a team."
      );
      return;
    }

    setPendingEventId(eventId);
    setMessage(null);
    const response = await createPublicFourPlayerTeam(clubSlug, {
      tournament_id: recovery.tournament.id,
      event_option_id: event.id,
      team_name: draft.teamName.trim(),
      captain_registration_id: recovery.captain.registration_id,
      confirmation_token: confirmationToken,
      members: TEAM_SLOTS.map((slot) =>
        slot === draft.captainSlot
          ? {
              slot,
              registration_id: recovery.captain.registration_id,
              email: recovery.captain.email,
              display_name: recovery.captain.display_name,
              gender: recovery.captain.gender
            }
          : {
              slot,
              email: draft.teammates[slot].email.trim().toLowerCase(),
              display_name: draft.teammates[slot].displayName.trim(),
              gender: slot.startsWith("MAN_") ? "Men" : "Women"
            }
      ),
      idempotency_key: draft.idempotencyKey,
      website: ""
    });

    // Always consult server state.  A team may have committed even if email
    // delivery or the browser response failed after the database transaction.
    const refreshed = await refreshRecovery();
    setPendingEventId("");
    const refreshedEvent = refreshed?.events.find((row) => row.id === eventId);
    if (refreshedEvent?.setup_state === "COMPLETE") {
      setMessage(
        response.error
          ? "The team was already saved. Its durable setup was recovered after the interrupted response."
          : "Team setup saved."
      );
      return;
    }
    setMessage(
      response.error ||
        "The team is not yet saved. Review the roster and try again."
    );
  }

  if (!recovery.events.length) return null;

  return (
    <section style={{ marginTop: "1rem" }} data-testid="team-setup-recovery">
      <h2>Four-player team setup</h2>
      <p style={{ color: "#475569" }}>
        This status comes from your saved registration, team records, and
        idempotent operation history. A refresh will not create a second
        registration or team.
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
          if (event.setup_state === "COMPLETE" && event.team) {
            return (
              <article key={event.id} style={cardStyle}>
                <h3 style={{ marginTop: 0 }}>
                  {event.event_family_label} — {event.division_name}
                </h3>
                <p>
                  <strong>{event.team.name}</strong> · {event.team.status}
                </p>
                <ul>
                  {event.team.members.map((member) => (
                    <li key={member.member_id}>
                      {member.slot}: {member.display_name} · {member.status}
                    </li>
                  ))}
                </ul>
              </article>
            );
          }
          if (event.setup_state === "STAFF_RECOVERY_REQUIRED") {
            return (
              <article key={event.id} style={cardStyle}>
                <h3 style={{ marginTop: 0 }}>
                  {event.event_family_label} — {event.division_name}
                </h3>
                <p style={{ color: "#991b1b" }}>
                  The operation ledger needs staff reconciliation before this
                  team can be submitted again. No second registration is
                  required.
                </p>
              </article>
            );
          }
          const draft =
            drafts[event.id] ||
            newTeamRegistrationDraft(recovery.captain.gender || "");
          const registrationEvent: PublicRegistrationEvent = {
            ...event,
            selectable: true
          };
          return (
            <article key={event.id} style={cardStyle}>
              <FourPlayerTeamRegistrationCard
                event={registrationEvent}
                captainName={recovery.captain.display_name}
                captainEmail={recovery.captain.email}
                captainGender={recovery.captain.gender || ""}
                value={draft}
                onChange={(next) =>
                  setDrafts((current) => ({
                    ...current,
                    [event.id]: next
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
                  ? "Recovering…"
                  : "Save team setup"}
              </button>
            </article>
          );
        })}
      </div>
    </section>
  );
}
