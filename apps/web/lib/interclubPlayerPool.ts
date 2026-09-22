import type { SeasonRegistrationWindow } from "./interclubRegistrationWindow";
export type PoolMember = {
  id: string; season_id: string; club_id: string; name: string; email: string; divisions: string[];
  notes: string; manage_url?: string; status: "active" | "withdrawn"; player_id: string | null; revision: number;
  approval_status?: "pending" | "approved" | "rejected"; late_join?: boolean; approval_reason?: string | null;
  late_request_reason?: string | null; late_requested_at?: string | null;
  rating?: number | null; league_rating?: number | null; gender?: string | null; eligible_divisions?: string[];
};
export type SeasonPool = {
  registration?: SeasonRegistrationWindow;
  can_request_late?: boolean;
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
  registration?: SeasonRegistrationWindow;
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

export type PoolPlayerChoice = { id: string; name: string; rating: number | null; league_rating?: number | null; gender: string | null; eligible_divisions: string[] };
export type PoolPlayerChoices = { players: PoolPlayerChoice[]; next_offset: number | null };
export type BulkPoolMember = { name?: string; email?: string | null; player_id?: string | null; divisions?: string[]; notes?: string };
export type BulkPoolPreviewRow = BulkPoolMember & {
  index: number; name: string; email: string; player_id: string | null;
  status: "matched" | "new" | "ambiguous" | "duplicate"; candidates: PoolPlayerChoice[];
  rating: number | null; league_rating?: number | null; gender: string | null; eligible_divisions: string[];
};
export type BulkPoolPreview = { rows: BulkPoolPreviewRow[]; ready_count: number; duplicate_count: number; ambiguous_count: number };
export type BulkPoolResult = { added_count: number; skipped_count: number; pool: SeasonPool };
export type NewClubPlayer = { name: string; starting_jupr: number; gender?: "male" | "female" | null; email?: string | null };
export type NewPoolPlayerInput = { new_player: NewClubPlayer; divisions?: string[]; reason?: string };
export type CreatePoolPlayerRequest = NewPoolPlayerInput & { request_id: string };
export type ExistingLatePlayerRequest = { player_id: number; divisions?: string[]; reason?: string };
export type LatePlayerInput = ExistingLatePlayerRequest | NewPoolPlayerInput;
export type LatePlayerRequest = ExistingLatePlayerRequest | CreatePoolPlayerRequest;
export type LatePlayerRequestResult = { member: PoolMember; pool: SeasonPool };

export function poolRating(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "Not rated";
}
export function poolGender(value: string | null | undefined): string {
  const gender = value?.trim().toLowerCase();
  return gender === "f" || gender === "female" || gender === "woman" ? "Women" : gender === "m" || gender === "male" || gender === "man" ? "Men" : "Not specified";
}
export function sortedPoolDivisions(divisions: string[]): string[] {
  return [...divisions].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
}

/** Accept one name per line, optionally followed by an email in CSV, tab or angle-bracket form. */
export function parsePoolPlayerList(value: string): { members: BulkPoolMember[]; errors: string[] } {
  const members: BulkPoolMember[] = [], errors: string[] = [];
  value.split(/\r?\n/).forEach((raw, index) => {
    const line = raw.trim();
    if (!line || /^(?:full\s+)?name\s*[,\t]\s*email(?:\s+address)?$/i.test(line)) return;
    const emails = line.match(/[^\s,;<>"\t]+@[^\s,;<>"\t]+/g) || [];
    if (emails.length > 1) { errors.push(`Line ${index + 1}: use one player and one email per line.`); return; }
    const email = emails[0];
    if (line.includes("@") && (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email))) { errors.push(`Line ${index + 1}: check the email address.`); return; }
    const name = (email ? line.replace(email, "") : line).replace(/^[\s,;<>"\t]+|[\s,;<>"\t]+$/g, "").replace(/""/g, '"').trim();
    if (!name) { errors.push(`Line ${index + 1}: include the player’s name.`); return; }
    members.push({ name, ...(email ? { email } : {}) });
  });
  return { members, errors };
}
