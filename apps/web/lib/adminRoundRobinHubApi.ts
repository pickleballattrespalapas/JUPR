export type HubCourtConfig = { id: string; label: string; dedicated_session_key: string | null; open: boolean };
export type HubPlayer = { id: string; name: string; display_name?: string; identity: string };
export type HubMatch = {
  session_key: string; match_id: string; round_number: number; side_a: HubPlayer[]; side_b: HubPlayer[];
  label: string; wait_seconds: number; allowed_court_ids: string[]; blockers: string[]; status: string;
  court_id?: string | null; reserved_court_id?: string | null; score_a?: number | null; score_b?: number | null;
};
export type HubCourt = HubCourtConfig & { current: HubMatch | null; next: HubMatch | null };
export type HubGroupPlayer = {
  participant_id: string; name: string; display_name?: string; identity: string; profile_id: number | null;
  paused: boolean; wait_seconds: number; on_court: boolean; reserved: boolean; active: boolean;
  active_from_round?: number; inactive_from_round?: number | null; inactive_rounds?: number[];
  can_rejoin?: boolean;
  on_bye?: boolean; can_remove?: boolean; remove_unavailable_reason?: string | null;
};
export type HubGroup = {
  session_key: string; title: string; current_round_number: number; round_status: string;
  completed_matches: number; total_matches: number; can_advance: boolean; event_format: string; scoring_mode: string;
  total_rounds?: number; advance_label?: string; rating_mode?: string; status?: string; dedicated_court_ids?: string[];
  players?: HubGroupPlayer[]; can_add_player?: boolean; add_player_unavailable_reason?: string | null;
  can_remove_player?: boolean; remove_player_unavailable_reason?: string | null;
};
export type HubWaitingPlayer = {
  identity: string; name: string; display_name?: string; paused: boolean; wait_seconds: number; session_keys: string[];
  participant_ids: string[] | Record<string, string>;
};
export type HubIdentityParticipant = {
  name: string; display_name: string; session_key: string; group_title: string; participant_id: string;
  identity: string; profile_id: number | null;
};
export type HubIdentityReview = {
  id: string; name: string; status: "pending" | "same" | "different" | "mixed";
  can_merge: boolean; assigned: boolean; participants: HubIdentityParticipant[];
};
export type HubWorkspace = {
  courts: HubCourt[]; groups: HubGroup[]; eligible_queue: HubMatch[]; blocked_queue: HubMatch[];
  reserved_queue: HubMatch[]; players: HubWaitingPlayer[];
  identity_review?: HubIdentityReview[]; identity_pending_count?: number;
};
export type HubSummary = { id: string; title: string; status: string; version: string | number; updated_at?: string };
export type HubSessionSummary = {
  session_key: string; title: string; status: string; rating_mode?: string; event_format?: string;
  round_robin_hub_id?: string | null; current_round_number?: number; scoring_mode?: string;
};
export type HubListResponse = { ok: boolean; hubs: HubSummary[]; sessions: HubSessionSummary[] };
export type HubDetailResponse = { ok: boolean; hub: HubSummary; workspace: HubWorkspace };
export type HubCommandAction = "add_group" | "remove_group" | "configure_courts" | "assign_next_court" |
  "assign_game" | "reserve_game_for_court" | "cancel_reservation" | "move_game" | "release_game" |
  "score_and_release" | "mark_played" | "pause_player" | "resume_player" | "advance_round" | "finish_hub" | "auto_fill_courts" |
  "resolve_identities" | "add_player" | "remove_player";
export type HubCommand = {
  action: HubCommandAction; payload: Record<string, unknown>; expected_version: string | number; idempotency_key: string;
};
export type CreateHubInput = { title: string; courts: HubCourtConfig[]; idempotency_key: string };

export class RoundRobinHubApiError extends Error {
  constructor(message: string, readonly status: number) { super(message); this.name = "RoundRobinHubApiError"; }
}

function errorText(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && "message" in detail) return String(detail.message);
  if (Array.isArray(detail)) return detail.map(item => typeof item?.msg === "string" ? item.msg : "Check the submitted values.").join(" ");
  return "The hub could not complete that action. Refresh and try again.";
}

export async function requestRoundRobinHub<T>(
  apiBase: string, clubId: string, accessToken: string, suffix = "", body?: unknown
): Promise<T> {
  const response = await fetch(`${apiBase.replace(/\/$/, "")}/admin/clubs/${encodeURIComponent(clubId)}/round-robin-hubs${suffix}`, {
    method: body === undefined ? "GET" : "POST", cache: "no-store",
    headers: { Authorization: `Bearer ${accessToken}`, ...(body === undefined ? {} : { "Content-Type": "application/json" }) },
    ...(body === undefined ? {} : { body: JSON.stringify(body) })
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) throw new RoundRobinHubApiError(errorText(payload?.detail), response.status);
  if (!payload?.ok) throw new RoundRobinHubApiError("We could not confirm the action. Retry the same action to check its result.", 502);
  return payload as T;
}

export function hubOperationKey(): string {
  return `rr-hub-${typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`}`;
}

export function hubWaitLabel(seconds: number): string {
  const minutes = Math.floor(Math.max(0, Number(seconds) || 0) / 60);
  return minutes < 1 ? "Under 1 min" : `${minutes} min`;
}

export function hubPlayerLabel(player: { name: string; display_name?: string }): string {
  return player.display_name || player.name;
}

export function hubMatchLabel(match: HubMatch): string {
  return `${match.side_a.map(player => player.name).join(" / ")} vs ${match.side_b.map(player => player.name).join(" / ")}`;
}

export function isUncertainHubError(error: unknown): boolean {
  return !(error instanceof RoundRobinHubApiError) || error.status >= 500 || error.status === 408;
}
