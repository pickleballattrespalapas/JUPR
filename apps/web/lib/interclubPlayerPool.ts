export type PoolMember = {
  id: string; season_id: string; club_id: string; name: string; email: string; divisions: string[];
  notes: string; manage_url?: string; status: "active" | "withdrawn"; player_id: string | null; revision: number;
};
export type SeasonPool = {
  signup: { share_id: string | null; revision: number; open: boolean; url: string | null };
  members: PoolMember[]; email_mode: string;
};
export type AvailabilityResponse = {
  id: string; member_id: string; name: string; email: string; player_id: string | null;
  divisions: string[]; notes: string; member_status: "active" | "withdrawn";
  status: "invited" | "available" | "maybe" | "unavailable"; revision: number;
  invited_at: string; responded_at: string | null; response_url: string;
};
export type MeetAvailabilityData = {
  settings: { revision: number; open: boolean; deadline: string | null };
  responses: AvailabilityResponse[]; email_mode: string;
};
export type InvitationCandidate = { id: string; name: string; email: string; available: boolean; unavailable_reason: string };
export type InvitationAudience = {
  candidates: InvitationCandidate[]; delivery_mode: string;
  defaults: { subject: string; message: string };
};
export type InvitationPreview = {
  recipient_count: number; recipients: { id: string; name: string; email: string }[];
  preview: { subject: string; html: string; text: string }; preview_fingerprint: string;
  delivery_mode: string; send_available: boolean; send_unavailable_reason?: string;
};
export type InvitationDelivery = { index: number; name?: string; email?: string; status: string; detail: string; links?: { name?: string; url: string }[] };
export type InvitationBatch = { operation_key: string; recipients: InvitationDelivery[]; pending_count: number; delivery_mode: string };
export const availabilityLabel: Record<AvailabilityResponse["status"], string> = {
  invited: "Not replied", available: "Available", maybe: "Maybe", unavailable: "Unavailable",
};
