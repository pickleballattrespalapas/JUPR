"use client";

import { useEffect, useRef, useState } from "react";
import {
  resolvePublicTeamInvitation,
  respondPublicTeamInvitation
} from "@/lib/tournamentTeamCompetitionApi";

type Props = {
  clubSlug: string;
};

type Invitation = {
  tournament?: { name?: string };
  team?: { name?: string };
  invitation?: {
    invited_name?: string;
    slot?: string;
    status?: string;
  };
  registration?: { id?: string; display_name?: string };
};

const button = {
  border: "1px solid #0f172a",
  borderRadius: "10px",
  padding: "0.7rem 1rem",
  fontWeight: 800,
  cursor: "pointer"
};

function rosterSpotLabel(slot?: string): string {
  const labels: Record<string, string> = {
    MAN_1: "Men’s spot 1",
    MAN_2: "Men’s spot 2",
    WOMAN_1: "Women’s spot 1",
    WOMAN_2: "Women’s spot 2"
  };
  return labels[String(slot || "").trim().toUpperCase()] || "Team spot";
}

export default function TeamInvitationReview({ clubSlug }: Props) {
  const token = useRef("");
  const idempotencyKeys = useRef(new Map<string, string>());
  const [invitation, setInvitation] = useState<Invitation | null>(null);
  const [message, setMessage] = useState("Checking your invitation…");
  const [pending, setPending] = useState(false);

  useEffect(() => {
    const raw = window.location.hash.slice(1);
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${window.location.search}`
    );
    if (!raw) {
      setMessage("This invitation link is incomplete. Open the original link from your invitation email.");
      return;
    }
    try {
      token.current = decodeURIComponent(
        raw.startsWith("token=") ? raw.slice("token=".length) : raw
      );
    } catch {
      setMessage("This invitation link is incomplete. Open the original link from your invitation email.");
      return;
    }
    void (async () => {
      const response = await resolvePublicTeamInvitation(
        clubSlug,
        token.current
      );
      if (response.error || !response.data) {
        token.current = "";
        setMessage("This invitation is unavailable. It may have expired or already been answered.");
        return;
      }
      setInvitation(response.data as Invitation);
      setMessage("");
    })();
  }, [clubSlug]);

  async function respond(action: "ACCEPT" | "DECLINE") {
    const registrationId = invitation?.registration?.id;
    if (!token.current || !registrationId) {
      setMessage("This invitation has expired or has already been answered.");
      return;
    }
    const key =
      idempotencyKeys.current.get(action) || crypto.randomUUID();
    idempotencyKeys.current.set(action, key);
    setPending(true);
    setMessage("");
    const response = await respondPublicTeamInvitation(clubSlug, {
      token: token.current,
      action,
      registration_id: registrationId,
      idempotency_key: key,
      website: ""
    });
    setPending(false);
    if (response.error) {
      setMessage("We couldn’t update this invitation. Please try again.");
      return;
    }
    token.current = "";
    idempotencyKeys.current.delete(action);
    setInvitation(null);
    setMessage(
      action === "ACCEPT"
        ? "You joined the team. The organizer can now confirm the roster."
        : "You declined the invitation."
    );
  }

  return (
    <section
      style={{
        border: "1px solid #e2e8f0",
        borderRadius: "14px",
        padding: "1rem",
        background: "white",
        maxWidth: "42rem"
      }}
    >
      {invitation ? (
        <>
          <h2>{invitation.team?.name || "Team invitation"}</h2>
          <p>
            {invitation.registration?.display_name ||
              invitation.invitation?.invited_name ||
              "Player"}
            , you were invited to{" "}
            <strong>{invitation.tournament?.name || "this tournament"}</strong>.
          </p>
          <p>
            Roster spot:{" "}
            {rosterSpotLabel(invitation.invitation?.slot)}
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.65rem" }}>
            <button
              type="button"
              disabled={pending}
              onClick={() => void respond("ACCEPT")}
              style={{ ...button, background: "#0f172a", color: "white" }}
            >
              Accept invitation
            </button>
            <button
              type="button"
              disabled={pending}
              onClick={() => void respond("DECLINE")}
              style={{ ...button, background: "white", color: "#0f172a" }}
            >
              Decline
            </button>
          </div>
        </>
      ) : null}
      {message ? <p role="status">{message}</p> : null}
    </section>
  );
}
