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
  const [sentEmail, setSentEmail] = useState<string | null>(null);
  const [noticeOpen, setNoticeOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const idempotencyKeyRef = useRef("");

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending) return;
    setSentEmail(null);
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
    setPending(true);
    const response = await requestClubTournamentRegistrationEditLink(clubSlug, {
      tournament_id: tournamentId,
      registration_slug: registrationSlug || null,
      email: cleanEmail,
      website,
      idempotency_key: idempotencyKey
    });
    setPending(false);
    if (response.error) {
      setError("We couldn’t send your edit link right now. Please try again.");
      return;
    }
    idempotencyKeyRef.current = "";
    setSentEmail(cleanEmail);
    setNoticeOpen(true);
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
      {sentEmail ? <p role="status" style={{ color: "#166534", margin: 0 }}>Check your email at <strong>{sentEmail}</strong> for your registration edit link.</p> : null}
      {error ? <p role="alert" style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
      {noticeOpen ? (
        <InteractionDialog open phase="ready" title="Check your email for your edit link" onRequestClose={() => setNoticeOpen(false)} actions={(
          <button type="button" onClick={() => setNoticeOpen(false)} style={{ minHeight: "48px", padding: "0.75rem 1.25rem", fontSize: "1.125rem", fontWeight: 800 }}>OK, I’ll check my email</button>
        )}>
          <div style={{ fontSize: "1.125rem", lineHeight: 1.6 }}>
            <p>Your edit link has been requested for:</p>
            <p style={{ fontSize: "1.25rem", overflowWrap: "anywhere" }}><strong>{sentEmail}</strong></p>
            <p><strong>Open your email and click the edit registration link to make your changes.</strong></p>
            <p>If this address matches a registration, the link will arrive shortly. Check your spam or junk folder if you don’t see it.</p>
          </div>
        </InteractionDialog>
      ) : null}
    </form>
  );
}
