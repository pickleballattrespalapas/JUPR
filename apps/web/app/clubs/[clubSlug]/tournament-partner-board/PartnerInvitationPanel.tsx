"use client";

import { useState } from "react";
import { FormDialog } from "@/components/interaction/FormDialog";
import { InteractionActionError, actionSuccess } from "@/components/interaction/types";
import { partnerInvitationRequest } from "@/lib/tournamentPartnerInvitations";
import type { PublicTournamentNeedsPartnerEntry } from "@/lib/tournamentRegistrationApi";

const inputStyle = { width: "100%", boxSizing: "border-box" as const, padding: "0.75rem", border: "1px solid #94a3b8", borderRadius: "8px", font: "inherit" };

export default function PartnerInvitationPanel({ apiBase, clubSlug, tournamentId, registrationSlug, entry, editToken, senderName = "", senderEmail = "" }: {
  apiBase: string | null; clubSlug: string; tournamentId: string; registrationSlug?: string | null;
  entry: PublicTournamentNeedsPartnerEntry; editToken?: string; senderName?: string; senderEmail?: string;
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(senderName);
  const [email, setEmail] = useState(senderEmail);
  const [message, setMessage] = useState("");
  const [website, setWebsite] = useState("");
  const [requestKey, setRequestKey] = useState("");
  const [sent, setSent] = useState(false);
  function start() {
    setRequestKey(crypto.randomUUID());
    setMessage(`Hi ${entry.player_name || "there"}, would you like to partner with me for ${entry.division || "this division"}?`);
    setOpen(true);
  }
  function changed(change: () => void) { change(); setRequestKey(crypto.randomUUID()); }
  return <>
    <button type="button" onClick={start} disabled={sent || !entry.board_entry_key}
      style={{ marginTop: "0.8rem", padding: "0.8rem 1rem", minHeight: "48px", borderRadius: "10px", border: 0, background: "#174f43", color: "white", fontSize: "1.05rem", fontWeight: 750, cursor: "pointer" }}>
      {sent ? "Request sent" : "Request to partner"}
    </button>
    <FormDialog open={open} mode="create" title={`Request to partner with ${entry.player_name || "this player"}`}
      description={`${entry.event_day_label || ""} · ${entry.division || "Division"}`}
      dirty={false} submitLabel="Send request" workingLabel="Sending…" onCancel={() => setOpen(false)}
      onSubmit={async () => {
        try {
          const result = await partnerInvitationRequest<{ status: string; notification_status: Record<string, string> }>(clubSlug, "", {
            tournament_id: tournamentId, registration_slug: registrationSlug || null, board_entry_key: entry.board_entry_key,
            name: name.trim(), email: email.trim(), message: message.trim(), request_key: requestKey, website,
            edit_token: editToken || null
          }, apiBase);
          if (Object.values(result.notification_status).some(status => status === "failed")) {
            throw new Error("Your request is saved, but the email couldn’t be sent. Click Send request to try delivery again.");
          }
          setSent(true);
          return actionSuccess(`Request sent to ${entry.player_name}`, "We’ve emailed them your message. We’ll email you when they respond and pair your registrations when they accept, or guide you through registration if needed.");
        } catch (error) {
          throw new InteractionActionError(error instanceof Error ? error.message : "We couldn’t send your request. Please try again.");
        }
      }}>
      <div style={{ display: "grid", gap: "1rem", fontSize: "1.05rem" }}>
        {!editToken ? <>
          <label>Your full name<input required autoComplete="name" maxLength={160} value={name} onChange={event => changed(() => setName(event.target.value))} style={inputStyle} /></label>
          <label>Your email<input required type="email" autoComplete="email" maxLength={320} value={email} onChange={event => changed(() => setEmail(event.target.value))} style={inputStyle} /></label>
        </> : <p style={{ margin: 0 }}>From <strong>{senderName}</strong></p>}
        <label>Your message<textarea required rows={5} maxLength={2000} value={message} onChange={event => changed(() => setMessage(event.target.value))} style={inputStyle} /></label>
        <label aria-hidden="true" style={{ position: "absolute", left: "-10000px" }}>Website<input tabIndex={-1} autoComplete="off" value={website} onChange={event => setWebsite(event.target.value)} /></label>
        <p style={{ margin: 0, color: "#475569" }}>Their email stays private. Your email will be shared only with this player so they can reply.</p>
        <p style={{ margin: 0, color: "#475569" }}>Click Send request to email them your message. We’ll email you when they respond.</p>
      </div>
    </FormDialog>
  </>;
}
