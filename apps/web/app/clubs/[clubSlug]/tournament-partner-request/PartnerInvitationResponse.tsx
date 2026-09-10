"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { partnerInvitationRequest, type PartnerInvitation } from "@/lib/tournamentPartnerInvitations";

const button = { display: "inline-block", padding: "1rem 1.3rem", border: "1px solid #174f43", borderRadius: "10px", background: "#174f43", color: "white", fontSize: "1.15rem", fontWeight: 750, cursor: "pointer", textDecoration: "none" };
const secondary = { ...button, background: "white", color: "#174f43" };

export default function PartnerInvitationResponse({ clubSlug }: { clubSlug: string }) {
  const [token, setToken] = useState("");
  const [invitation, setInvitation] = useState<PartnerInvitation | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const [feedback, setFeedback] = useState("");
  useEffect(() => {
    let active = true;
    const capability = new URLSearchParams(window.location.hash.slice(1)).get("token") || "";
    setToken(capability);
    if (!capability) { setError("This link is incomplete. Open the button in your partner request email."); return; }
    partnerInvitationRequest<PartnerInvitation>(clubSlug, "review", { token: capability })
      .then(data => { if (active) setInvitation(data); })
      .catch(reason => { if (active) setError(reason.message); });
    return () => { active = false; };
  }, [clubSlug]);
  async function respond(action: string) {
    if (pending) return;
    setPending(true); setError(""); setFeedback("");
    try {
      const result = await partnerInvitationRequest<PartnerInvitation>(clubSlug, "respond", { token, action });
      setInvitation(result);
      setFeedback(action === "verify" ? `Your request has been sent to ${result.target_name}.` : "");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Please try again.");
      // Refresh stale state without repeating an acceptance or email delivery.
      try { setInvitation(await partnerInvitationRequest<PartnerInvitation>(clubSlug, "review", { token })); } catch { /* Keep the original error. */ }
    } finally { setPending(false); }
  }
  const status = invitation?.status;
  const sender = invitation?.role === "requester";
  const titles: Record<string, string> = {
    UNVERIFIED: "Send your partner request", PENDING: sender ? "Your partner request" : "Confirm your partnership",
    RESERVED: "Your partnership is reserved", COMPLETED: "You’re partnered up!",
    DECLINED: "Partner request declined", CANCELLED: "This request is no longer available", EXPIRED: "This request has expired"
  };
  const noticesFailed = Object.values(invitation?.notification_status || {}).some(value => value === "failed");
  return <section style={{ maxWidth: "680px", margin: "2rem auto", padding: "1.5rem", background: "white", border: "1px solid #cbd5e1", borderRadius: "16px", fontSize: "1.1rem", lineHeight: 1.6 }}>
    <p style={{ color: "#174f43", fontWeight: 750, marginTop: 0 }}>PCS · Partner request</p>
    <h1 style={{ fontSize: "2rem", lineHeight: 1.2 }}>{status ? titles[status] || "Partner request" : error ? "Link unavailable" : "Opening your request…"}</h1>
    {error ? <p role="alert" style={{ color: "#b91c1c" }}>{error}</p> : null}
    {feedback ? <p role="status" style={{ background: "#dcfce7", padding: "1rem", borderRadius: "10px" }}>{feedback}</p> : null}
    {invitation ? <>
      <p><strong>{invitation.tournament_name}</strong><br />{invitation.division_name}</p>
      <p><strong>{invitation.requester_name}</strong> and <strong>{invitation.target_name}</strong></p>
      {status === "PENDING" || status === "UNVERIFIED" ? <blockquote style={{ whiteSpace: "pre-wrap", margin: "1.5rem 0", borderLeft: "4px solid #b8d3cb", paddingLeft: "1rem", overflowWrap: "anywhere" }}>{invitation.message}</blockquote> : null}
      {status === "UNVERIFIED" ? <p>Confirm your email below to send this request to {invitation.target_name}.</p> : null}
      {status === "PENDING" ? <p>{sender ? `We’ll email you when ${invitation.target_name} responds.` : "Confirm below and PCS will pair your registrations for this division. If your partner still needs to register, we’ll reserve the partnership and send them a registration link."}</p> : null}
      {status === "RESERVED" ? <p>{sender ? "Complete your registration for this division. PCS will add your partner automatically when you save." : "Your partner has been sent a registration link. PCS will finish pairing you automatically when they register."} Reserved until {new Date(invitation.expires_at).toLocaleDateString()}.</p> : null}
      {status === "COMPLETED" ? <p>Both registrations are updated, your team is on the roster, and you’re no longer listed as needing partners in this division.</p> : null}
      {status === "CANCELLED" ? <p>The request was cancelled or one player already has another partner. You can look for another player on the Partner Board.</p> : null}
      {status === "EXPIRED" ? <p>Open the Partner Board to send a new request.</p> : null}
      <div style={{ display: "flex", gap: "0.8rem", flexWrap: "wrap", margin: "1.5rem 0" }}>
        {invitation.actions.includes("verify") ? <button type="button" style={button} disabled={pending} onClick={() => respond("verify")}>{pending ? "Sending…" : "Send my partner request"}</button> : null}
        {invitation.actions.includes("accept") ? <button type="button" style={button} disabled={pending} onClick={() => respond("accept")}>{pending ? "Confirming…" : "Confirm partnership"}</button> : null}
        {invitation.actions.includes("decline") ? <button type="button" style={secondary} disabled={pending} onClick={() => respond("decline")}>Decline request</button> : null}
        {status === "RESERVED" && sender && invitation.registration_url ? <Link style={button} href={invitation.registration_url}>Complete registration</Link> : null}
        {status === "RESERVED" && sender ? <button type="button" style={secondary} disabled={pending} onClick={() => respond("complete")}>I’ve registered — finish pairing</button> : null}
        {status === "COMPLETED" ? <Link href={invitation.roster_url} style={button}>View our team</Link> : null}
      </div>
      {noticesFailed ? <div role="alert"><p>Your change is saved, but a confirmation email couldn’t be sent.</p><button type="button" style={secondary} disabled={pending} onClick={() => respond("retry_email")}>Retry email</button></div> : null}
      <p><Link href={invitation.board_url}>Back to Partner Board</Link></p>
      {invitation.actions.includes("cancel") ? <button type="button" style={{ border: 0, background: "transparent", color: "#64748b", padding: "0.75rem 0", font: "inherit", textDecoration: "underline", cursor: "pointer" }} disabled={pending} onClick={() => respond("cancel")}>{status === "RESERVED" ? "Cancel this partnership reservation" : "Cancel my request"}</button> : null}
    </> : !error ? <p role="status">Loading…</p> : <Link href={`/clubs/${clubSlug}/tournaments`}>Find your tournament</Link>}
  </section>;
}
