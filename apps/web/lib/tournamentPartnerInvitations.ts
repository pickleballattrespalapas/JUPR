export type PartnerInvitation = {
  ok: boolean;
  status: string;
  role: "requester" | "target";
  actions: string[];
  requester_name: string;
  target_name: string;
  tournament_name: string;
  division_name: string;
  message: string;
  expires_at: string;
  board_url: string;
  roster_url: string;
  registration_url?: string;
  registration_prefill?: { name: string; email: string; event_option_id: string };
  notification_status?: Record<string, string>;
};

export async function partnerInvitationRequest<T>(clubSlug: string, action: "" | "review" | "respond", payload: unknown, apiBase?: string | null): Promise<T> {
  const base = (apiBase || process.env.NEXT_PUBLIC_JUPR_API_BASE_URL || "").replace(/\/$/, "");
  if (!base) throw new Error("Partner requests are temporarily unavailable. Please try again shortly.");
  const response = await fetch(`${base}/clubs/${encodeURIComponent(clubSlug)}/tournament-registration/partner-invitations${action ? `/${action}` : ""}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload), cache: "no-store", referrerPolicy: "no-referrer"
  });
  const data = await response.json().catch(() => null);
  if (!response.ok || !data?.ok) {
    throw new Error(typeof data?.detail === "string" ? data.detail : "We couldn’t complete this request. Please try again.");
  }
  return data as T;
}

export function invitationReturnPath(clubSlug: string, token: string): string {
  return `/clubs/${encodeURIComponent(clubSlug)}/tournament-partner-request#token=${encodeURIComponent(token)}`;
}
