"use client";

import { useState } from "react";

export type GeneratorSubmissionStatus = {
  status: "pending" | "processing" | "approved" | "rejected";
  rating_mode?: "rated" | "unrated";
  approved_mode?: "rated" | "unrated" | null;
  rejection_reason?: string | null;
};

export function generatorResultLabel(submission?: GeneratorSubmissionStatus | null) {
  if (submission?.status === "approved") return `Approved · ${submission.approved_mode === "rated" ? "Rated" : "Unrated"}`;
  if (submission?.status === "pending" || submission?.status === "processing") return "Awaiting admin approval";
  if (submission?.status === "rejected") return "Submission rejected";
  return "Not added to club records";
}

export default function GeneratorSubmission({ submission, canSubmit, defaultDate, onSubmit, ratingMode }: {
  submission?: GeneratorSubmissionStatus | null;
  canSubmit: boolean;
  ratingMode: "rated" | "unrated";
  defaultDate?: string;
  onSubmit: (organizerName: string, matchDate: string) => Promise<void>;
}) {
  const [organizerName, setOrganizerName] = useState("");
  const [matchDate, setMatchDate] = useState(() => (defaultDate || new Date().toISOString()).slice(0, 10));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const field = { padding: "0.6rem", border: "1px solid #cbd5e1", borderRadius: 8, font: "inherit", maxWidth: "100%" };
  return <article style={{ border: "1px solid #cbd5e1", borderRadius: 14, padding: "1rem", background: "#f8fafc" }}>
    <h2 style={{ marginTop: 0 }}>Club results · {ratingMode === "rated" ? "Rated" : "Unrated"}</h2>
    {submission ? <div role="status">
      <strong>{generatorResultLabel(submission)}</strong>
      <p>{submission.status === "approved"
        ? `These games are included in player stats and available for weekly recaps.${submission.approved_mode === "rated" ? " Ratings have been updated." : " Ratings are unchanged."}`
        : submission.status === "rejected" ? "These games have not been added to club records."
        : "Your results have been sent to the club administrator. They will appear in club records once approved."}</p>
      {submission.rejection_reason ? <p>{submission.rejection_reason}</p> : null}
    </div> : <>
      <p>Send the saved scores to the club administrator for approval. This session was set to {ratingMode} before play. Approved games count toward player stats and weekly recaps.{ratingMode === "rated" ? " They will also update ratings." : " Ratings stay unchanged."}</p>
      {canSubmit ? <form onSubmit={async event => {
        event.preventDefault();
        if (busy) return;
        setBusy(true); setError("");
        try { await onSubmit(organizerName.trim(), matchDate); }
        catch (failure) { setError(failure instanceof Error ? failure.message : "Could not confirm submission. Try again."); }
        finally { setBusy(false); }
      }} style={{ display: "flex", gap: "0.8rem", flexWrap: "wrap", alignItems: "end" }}>
        <label style={{ display: "grid", gap: 4 }}>Organizer’s name<input required maxLength={160} value={organizerName} onChange={e => setOrganizerName(e.target.value)} disabled={busy} style={field} /></label>
        <label style={{ display: "grid", gap: 4 }}>Date played<input required type="date" value={matchDate} onChange={e => setMatchDate(e.target.value)} disabled={busy} style={field} /></label>
        <button disabled={busy || !organizerName.trim() || !matchDate} type="submit" style={{ ...field, background: "#0f172a", color: "white", fontWeight: 700 }}>{busy ? "Submitting…" : "Submit for approval"}</button>
      </form> : <p>Finish the session, then use the organizer link to submit the scored games. No account is required.</p>}
    </>}
    {error ? <p role="alert" style={{ color: "#b91c1c" }}>{error}</p> : null}
  </article>;
}
