"use client";

import { FormEvent, useRef, useState } from "react";
import { requestClubTournamentRegistrationEditLink } from "@/lib/tournamentRegistrationApi";

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
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const idempotencyKeyRef = useRef("");

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage(null);
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
    setMessage("If that email matches a registration, we’ll send the edit link there.");
  }

  return (
    <form onSubmit={onSubmit} style={{ display: "grid", gap: "0.75rem" }} data-testid="registration-edit-link-form">
      <input type="text" name="website" autoComplete="off" tabIndex={-1} style={{ position: "absolute", left: "-10000px" }} aria-hidden="true" />
      <label>
        Registration email<br />
        <input name="email" value={email} onChange={(event) => { idempotencyKeyRef.current = ""; setEmail(event.target.value); }} type="email" autoComplete="email" placeholder="you@example.com" required style={inputStyle} />
      </label>
      <button type="submit" disabled={pending} style={{ padding: "0.65rem 0.9rem", borderRadius: "10px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 }}>
        {pending ? "Sending…" : "Send edit link"}
      </button>
      {message ? <p role="status" style={{ color: "#166534", margin: 0 }}>{message}</p> : null}
      {error ? <p role="alert" style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
    </form>
  );
}
