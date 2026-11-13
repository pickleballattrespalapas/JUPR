"use client";

import { FormEvent, useRef, useState } from "react";
import { requestClubTournamentRegistrationEditLink } from "@/lib/tournamentRegistrationApi";
import { InteractionDialog } from "@/components/interaction";

type EditLinkRequestFormProps = {
  clubSlug: string;
  tournamentId: string;
  registrationSlug?: string | null;
  initialEmail?: string;
};

const inputStyle = { width: "100%", padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };

export default function EditLinkRequestForm({ clubSlug, tournamentId, registrationSlug, initialEmail = "" }: EditLinkRequestFormProps) {
  const [email, setEmail] = useState(initialEmail);
  const [pending, setPending] = useState(false);
  const [requestedEmail, setRequestedEmail] = useState<string | null>(null);
  const [noticeOpen, setNoticeOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const idempotencyKeyRef = useRef("");
  const requestInFlightRef = useRef(false);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (requestInFlightRef.current) return;
    setRequestedEmail(null);
    setNoticeOpen(false);
    setError(null);
    const formData = new FormData(event.currentTarget);
    const website = String(formData.get("website") ?? "").trim();
    const cleanEmail = email.trim();
    if (!cleanEmail || !cleanEmail.includes("@")) {
      setError("Enter the email address used for your registration.");
      return;
    }
    const idempotencyKey = idempotencyKeyRef.current || `edit-link:${globalThis.crypto.randomUUID()}`;
    idempotencyKeyRef.current = idempotencyKey;
    requestInFlightRef.current = true;
    setPending(true);
    try {
      const response = await requestClubTournamentRegistrationEditLink(clubSlug, {
        tournament_id: tournamentId,
        registration_slug: registrationSlug || null,
        email: cleanEmail,
        website,
        idempotency_key: idempotencyKey
      });
      if (response.error || !response.data?.accepted) {
        setError("We couldn’t confirm your request. Check your inbox before trying again. If no link arrives, contact tournament staff for help.");
        return;
      }
      idempotencyKeyRef.current = "";
      setRequestedEmail(cleanEmail);
      setNoticeOpen(true);
    } catch {
      setError("We couldn’t confirm your request. Check your inbox before trying again. If no link arrives, contact tournament staff for help.");
    } finally {
      requestInFlightRef.current = false;
      setPending(false);
    }
  }

  return (
    <form onSubmit={onSubmit} style={{ display: "grid", gap: "0.75rem" }} data-testid="registration-edit-link-form">
      <input type="text" name="website" autoComplete="off" tabIndex={-1} style={{ position: "absolute", left: "-10000px" }} aria-hidden="true" />
      <label>
        Registration email<br />
        <input name="email" value={email} disabled={pending} onChange={(event) => { idempotencyKeyRef.current = ""; setEmail(event.target.value); }} type="email" autoComplete="email" placeholder="you@example.com" required style={inputStyle} />
      </label>
      <button type="submit" disabled={pending} style={{ minHeight: "48px", padding: "0.75rem 1rem", borderRadius: "10px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontSize: "1.125rem", fontWeight: 800 }}>
        {pending ? "Sending…" : "Send edit link"}
      </button>
      {requestedEmail ? <p role="status" style={{ color: "#166534", margin: 0 }}>Request received for <strong>{requestedEmail}</strong>. If this address matches your registration, check your email for the edit link.</p> : null}
      {error ? <p role="alert" style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
      {noticeOpen ? (
        <InteractionDialog open phase="ready" title="Edit-link request received" onRequestClose={() => setNoticeOpen(false)} actions={(
          <button type="button" onClick={() => setNoticeOpen(false)} style={{ minHeight: "48px", padding: "0.75rem 1.25rem", fontSize: "1.125rem", fontWeight: 800 }}>OK, I’ll check my email</button>
        )}>
          <div style={{ fontSize: "1.125rem", lineHeight: 1.6 }}>
            <p>Your edit link has been requested for:</p>
            <p style={{ fontSize: "1.25rem", overflowWrap: "anywhere" }}><strong>{requestedEmail}</strong></p>
            <p>If this address matches your registration, look for an email with a secure link to make your changes.</p>
            <p>If no link arrives within a few minutes, check your spam folder and confirm that you used the email address on your registration. If it still hasn’t arrived, contact tournament staff for help.</p>
          </div>
        </InteractionDialog>
      ) : null}
    </form>
  );
}
