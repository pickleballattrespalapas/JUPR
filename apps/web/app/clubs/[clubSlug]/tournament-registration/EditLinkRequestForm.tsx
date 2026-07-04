"use client";

import { FormEvent, useState } from "react";
import { requestClubTournamentRegistrationEditLink } from "@/lib/tournamentRegistrationApi";

type EditLinkRequestFormProps = {
  clubSlug: string;
  tournamentId: string;
  registrationSlug?: string | null;
};

const inputStyle = { width: "100%", padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };

export default function EditLinkRequestForm({ clubSlug, tournamentId, registrationSlug }: EditLinkRequestFormProps) {
  const [email, setEmail] = useState("");
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

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
    setPending(true);
    const response = await requestClubTournamentRegistrationEditLink(clubSlug, {
      tournament_id: tournamentId,
      registration_slug: registrationSlug || null,
      email: cleanEmail,
      website
    });
    setPending(false);
    if (response.error) {
      setError(response.error);
      return;
    }
    setMessage(response.data?.message || "If a matching registration exists, an edit link will be sent to that email address.");
  }

  return (
    <form onSubmit={onSubmit} style={{ display: "grid", gap: "0.75rem" }}>
      <input type="text" name="website" autoComplete="off" tabIndex={-1} style={{ position: "absolute", left: "-10000px" }} aria-hidden="true" />
      <label>
        Registration email<br />
        <input value={email} onChange={(event) => setEmail(event.target.value)} type="email" placeholder="you@example.com" style={inputStyle} />
      </label>
      <button type="submit" disabled={pending} style={{ padding: "0.65rem 0.9rem", borderRadius: "10px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 }}>
        {pending ? "Sending…" : "Send secure edit link"}
      </button>
      {message ? <p style={{ color: "#166534", margin: 0 }}>{message}</p> : null}
      {error ? <p style={{ color: "#b91c1c", margin: 0 }}>{error}</p> : null}
    </form>
  );
}
