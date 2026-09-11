"use client";

import { useEffect, useState } from "react";
import { partnerInvitationRequest, type PartnerInvitation } from "@/lib/tournamentPartnerInvitations";

export function usePartnerInvitationRegistration(clubSlug: string) {
  const [token, setToken] = useState("");
  const [invitation, setInvitation] = useState<PartnerInvitation | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const value = new URLSearchParams(window.location.hash.slice(1)).get("partner_invitation") || "";
    if (!value) return;
    setToken(value);
    partnerInvitationRequest<PartnerInvitation>(clubSlug, "review", { token: value }).then(result => {
      if (result.role !== "requester" || !result.registration_prefill) throw new Error("This partner registration link is no longer available. Open your partner request email to check its status.");
      if (active) setInvitation(result);
    }).catch(reason => { if (active) setError(reason.message); });
    return () => { active = false; };
  }, [clubSlug]);
  return { token, invitation, error };
}

export function PartnerInvitationRegistrationNotice({ invitation, error }: { invitation: PartnerInvitation | null; error: string }) {
  if (error) return <p role="alert" style={{ color: "#b91c1c" }}>{error}</p>;
  if (!invitation) return null;
  return <aside style={{ padding: "1rem", background: "#ecfdf5", border: "1px solid #a7f3d0", borderRadius: "12px", marginBottom: "1rem" }}>
    <strong>{invitation.division_name} · Partner: {invitation.target_name}</strong>
    <p style={{ marginBottom: 0 }}>{invitation.status === "RESERVED"
      ? "Your partner accepted! You’re listed together on the roster as pending registration. Your division and partner are already selected. Complete this form to confirm your team."
      : "Your partner request has been sent. Complete your registration now and PCS will pair you automatically if they accept."}</p>
  </aside>;
}
