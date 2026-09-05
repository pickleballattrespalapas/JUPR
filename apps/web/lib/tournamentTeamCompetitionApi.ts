export type TeamCompetitionEvent = {
  id: string;
  label?: string | null;
  event_family_label?: string | null;
  division_name?: string | null;
  eligibility_mode?: string | null;
  combined_rating_cap?: number | null;
  rating_source_policy?: string | null;
  rating_review_timing?: string | null;
  competition_format?: string | null;
  team_roster_size?: number | null;
  team_gender_rule?: string | null;
  team_tiebreak_mode?: string | null;
  team_playoff_format?: string | null;
  team_allow_substitutes?: boolean | null;
  updated_at?: string | null;
};

export type FourPlayerTeamMember = {
  id: string;
  team_id: string;
  slot: "MAN_1" | "MAN_2" | "WOMAN_1" | "WOMAN_2";
  registration_id?: string | null;
  player_id?: number | null;
  invited_email?: string | null;
  display_name?: string | null;
  display_name_snapshot?: string | null;
  status: string;
  invitation_version?: number | null;
};

export type FourPlayerTeam = {
  id: string;
  event_option_id: string;
  draw_id?: string | null;
  name: string;
  status: string;
  eligibility_state?: string | null;
  version: number;
};

export type TeamTournamentDraw = {
  id: string;
  event_option_id?: string | null;
  name: string;
  status: string;
  draw_kind?: string | null;
  updated_at?: string | null;
};

export type TeamTournamentMatchup = {
  id: string;
  draw_id: string;
  event_option_id: string;
  stage: string;
  round_number: number;
  slot_number: number;
  playoff_game_code?: string | null;
  team_a_id?: string | null;
  team_b_id?: string | null;
  status: string;
  version: number;
  team_a_game_wins?: number | null;
  team_b_game_wins?: number | null;
};

export type TeamTournamentGame = {
  id: string;
  matchup_id: string;
  game_code: string;
  game_order: number;
  match_format: string;
  counts_for_rating: boolean;
  status: string;
  score_a?: number | null;
  score_b?: number | null;
  version: number;
  tournament_game_id?: string | null;
};

export type TeamCompetitionSnapshot = {
  tournament: Record<string, unknown> & { id: string; name: string };
  event_options: TeamCompetitionEvent[];
  draws: TeamTournamentDraw[];
  registrations: Array<Record<string, unknown>>;
  selections: Array<Record<string, unknown>>;
  players: Array<Record<string, unknown>>;
  teams: FourPlayerTeam[];
  members: FourPlayerTeamMember[];
  rating_verifications: Array<Record<string, unknown>>;
  rating_reviews: Array<Record<string, unknown>>;
  combined_rating_entries: Array<Record<string, unknown>>;
  matchups: TeamTournamentMatchup[];
  lineups: Array<Record<string, unknown>>;
  games: TeamTournamentGame[];
  canonical_matches: Array<Record<string, unknown>>;
  podium: Array<Record<string, unknown>>;
  standings_by_draw: Record<string, Array<Record<string, unknown>>>;
  calculated_podium_by_draw: Record<
    string,
    Array<{ placement: number; team_id: string }> | null
  >;
  game_publish_state: Record<string, string>;
  warnings: string[];
};

export type PublicTeamTournamentIndex = {
  club: Record<string, unknown>;
  tournaments: Array<{
    id: string;
    name: string;
    start_date?: string | null;
    end_date?: string | null;
  }>;
  draws: Array<{
    id: string;
    name: string;
    tournament_id: string;
    tournament_name: string;
    event_family_label?: string | null;
    division_name?: string | null;
    team_count: number;
  }>;
};

export type PublicTeamTournamentResults = {
  club: Record<string, unknown>;
  tournament: {
    id: string;
    name: string;
    start_date?: string | null;
    end_date?: string | null;
  };
  draw: {
    id: string;
    name: string;
    event_family_label?: string | null;
    division_name?: string | null;
    team_playoff_format?: string | null;
  };
  teams: Array<
    FourPlayerTeam & {
      members: FourPlayerTeamMember[];
    }
  >;
  standings: Array<Record<string, unknown>>;
  bracket: TeamTournamentMatchup[];
  podium: Array<{ placement: number; team_id: string; team_name: string }>;
};

export type PublicFourPlayerTeamSetupRecovery = {
  club: Record<string, unknown>;
  ok: boolean;
  tournament: { id: string; name: string };
  captain: {
    registration_id: string;
    display_name: string;
    email: string;
    gender?: string | null;
    registration_status?: string | null;
  };
  events: Array<{
    id: string;
    registration_day_id: string;
    label: string;
    event_family_label: string;
    division_name: string;
    event_type?: string | null;
    competition_format: "FOUR_PLAYER_TEAM";
    team_allow_substitutes?: boolean | null;
    setup_state:
      | "COMPLETE"
      | "SETUP_REQUIRED"
      | "STAFF_RECOVERY_REQUIRED";
    operation_status?: string | null;
    team?: {
      id: string;
      name: string;
      status: string;
      eligibility_state?: string | null;
      version?: number | null;
      members: Array<{
        member_id: string;
        slot: "MAN_1" | "MAN_2" | "WOMAN_1" | "WOMAN_2";
        display_name: string;
        status: string;
        invitation_version?: number | null;
      }>;
    } | null;
  }>;
};

export type ApiResult<T> = {
  data: T | null;
  error: string | null;
  status?: number | null;
};

function baseUrl(): string | null {
  return (
    process.env.JUPR_API_BASE_URL ||
    process.env.NEXT_PUBLIC_JUPR_API_BASE_URL ||
    null
  );
}

async function adminErrorMessage(response: Response): Promise<string> {
  const fallback = `API error (${response.status}).`;
  try {
    const payload = (await response.json()) as { detail?: unknown };
    return payload.detail ? String(payload.detail) : fallback;
  } catch {
    return fallback;
  }
}

async function fetchAdminJson<T>(
  path: string,
  init?: RequestInit
): Promise<ApiResult<T>> {
  const apiBase = baseUrl();
  if (!apiBase) {
    return {
      data: null,
      error: "Tournament team features are unavailable.",
      status: null
    };
  }
  try {
    const response = await fetch(`${apiBase.replace(/\/$/, "")}${path}`, init);
    if (!response.ok) {
      return {
        data: null,
        error: await adminErrorMessage(response),
        status: response.status
      };
    }
    return {
      data: (await response.json()) as T,
      error: null,
      status: response.status
    };
  } catch (error) {
    return {
      data: null,
      error:
        error instanceof Error ? error.message : "Unable to reach the API.",
      status: null
    };
  }
}

type PublicTeamRequest =
  | "results-index"
  | "results-detail"
  | "create-team"
  | "recover-team"
  | "resolve-invitation"
  | "respond-invitation";

function safeTeamValidation(detail: string): string | null {
  const message = detail.trim();
  if (!message || message.length > 350 || /[<>\u0000-\u001f]/.test(message)) return null;
  if (/\b(?:database|supabase|credential|runtime|stack|traceback|exception|rpc|sql|fingerprint|idempotency|operation bound)\b|JUPR_|\b\w+_id\b/i.test(message)) {
    return null;
  }
  if (!/\b(?:team|captain|invitation|player|registration|roster|slot|email|accept|decline|event)\b/i.test(message)) {
    return null;
  }
  return message;
}

function publicTeamError(
  status: number | null | undefined,
  detail: string,
  request: PublicTeamRequest
): string {
  const normalized = detail.trim().toLowerCase();
  const invitationRequest =
    request === "resolve-invitation" || request === "respond-invitation";

  if (invitationRequest) {
    if (normalized.includes("feature not found")) {
      return "Team invitations aren’t available for this tournament.";
    }
    if (normalized.includes("expired")) return "This invitation has expired.";
    if (/already (?:been )?(?:resolved|answered)/.test(normalized)) {
      return "This invitation has already been answered.";
    }
    if (/no longer current|replaced by a newer invitation/.test(normalized)) {
      return "This invitation has been replaced. Open the newest invitation email.";
    }
    if (/different email|identity does not match|single confirmed registration/.test(normalized)) {
      return "This invitation doesn’t match the registered player. Contact the organizer.";
    }
    if (/different (?:club|tournament|team)|does not match this team/.test(normalized)) {
      return "This invitation doesn’t match this tournament. Open the original invitation link.";
    }
    if (normalized.includes("invalid team invitation")) {
      return "This invitation link isn’t valid.";
    }
    if (normalized.includes("no longer available") || normalized.includes("not found")) {
      return "This invitation is no longer available.";
    }
    if (normalized.includes("must be accept or decline")) {
      return "Choose Accept or Decline.";
    }
    if (status === 401) return "This invitation link isn’t valid.";
    if (status === 403) {
      if (/disabled|staging|write|wave|acceptance/.test(normalized)) {
        return "Team invitations can’t be changed right now.";
      }
      return "You don’t have permission to use this invitation.";
    }
    if (status === 409) {
      return "This invitation is no longer current. Open the newest invitation email.";
    }
    if (status === 429) return "Too many attempts were made. Wait a moment and try again.";
    if (status != null && status < 500) {
      return safeTeamValidation(detail) || "This invitation can’t be used.";
    }
    return "Team invitations are temporarily unavailable. Please try again.";
  }

  if (request === "results-index") {
    if (normalized.includes("feature not found")) {
      return "Team tournament results are unavailable right now.";
    }
    return status === 404
      ? "Team tournament results haven’t been published yet."
      : "Team tournament results are unavailable right now.";
  }
  if (request === "results-detail") {
    if (normalized.includes("feature not found")) {
      return "These team results are unavailable right now.";
    }
    if (/not published/.test(normalized)) {
      return "These team results haven’t been published yet.";
    }
    return status === 404
      ? "We couldn’t find those team results."
      : "These team results are unavailable right now.";
  }

  if (/confirmation (?:link|token).*expired|confirmation link has expired/.test(normalized)) {
    return "Your registration confirmation link has expired.";
  }
  if (/invalid registration confirmation|valid captain confirmation token/.test(normalized)) {
    return "Your registration confirmation link isn’t valid.";
  }
  if (normalized.includes("registration confirmation is no longer available")) {
    return "This registration confirmation is no longer available.";
  }
  if (normalized.includes("registration confirmation is for a different club")) {
    return "This registration confirmation belongs to a different club.";
  }
  if (status === 401) return "Your registration confirmation link isn’t valid.";
  if (status === 403) {
    if (/disabled|staging|write|wave|acceptance/.test(normalized)) {
      return "Four-player team registration isn’t available right now.";
    }
    return "You don’t have permission to change this team.";
  }
  if (status === 409) {
    return "This team changed since you opened it. Refresh the page and try again.";
  }
  if (status === 429) return "Too many attempts were made. Wait a moment and try again.";
  if (status != null && status < 500) {
    return (
      safeTeamValidation(detail) ||
      (request === "recover-team"
        ? "We couldn’t open this team setup. Check the confirmation link and try again."
        : "Check the team information and try again.")
    );
  }
  return request === "recover-team"
    ? "We couldn’t open this team setup. Please try again or contact the organizer."
    : "We couldn’t save this team. Please try again or contact the organizer.";
}

async function fetchPublicJson<T>(
  path: string,
  init: RequestInit | undefined,
  request: PublicTeamRequest
): Promise<ApiResult<T>> {
  const result = await fetchAdminJson<T>(path, init);
  return result.error
    ? {
        ...result,
        error: publicTeamError(result.status, result.error, request)
      }
    : result;
}

function adminHeaders(accessToken: string): HeadersInit {
  return {
    "Content-Type": "application/json",
    Authorization: `Bearer ${accessToken}`
  };
}

export async function listAdminTeamTournaments(
  clubId: string,
  accessToken: string
): Promise<
  ApiResult<{
    tournaments: Array<{ id: string; name: string; status?: string }>;
  }>
> {
  return fetchAdminJson(
    `/admin/clubs/${encodeURIComponent(
      clubId
    )}/tournaments/admin/tournaments`,
    { headers: adminHeaders(accessToken), cache: "no-store" }
  );
}

export async function getAdminTeamCompetitionSnapshot(
  clubId: string,
  tournamentId: string,
  accessToken: string
): Promise<ApiResult<TeamCompetitionSnapshot>> {
  return fetchAdminJson(
    `/admin/clubs/${encodeURIComponent(
      clubId
    )}/tournaments/admin/tournaments/${encodeURIComponent(
      tournamentId
    )}/team-competition`,
    { headers: adminHeaders(accessToken), cache: "no-store" }
  );
}

export async function mutateAdminTeamCompetition<T>(
  path: string,
  payload: Record<string, unknown>,
  accessToken: string
): Promise<ApiResult<T>> {
  return fetchAdminJson(path, {
    method: "POST",
    headers: adminHeaders(accessToken),
    body: JSON.stringify(payload)
  });
}

export async function getPublicTeamTournamentIndex(
  clubSlug: string
): Promise<ApiResult<PublicTeamTournamentIndex>> {
  return fetchPublicJson(
    `/clubs/${encodeURIComponent(clubSlug)}/tournament-team-results`,
    { next: { revalidate: 30 } },
    "results-index"
  );
}

export async function getPublicTeamTournamentResults(
  clubSlug: string,
  tournamentId: string,
  drawId: string
): Promise<ApiResult<PublicTeamTournamentResults>> {
  return fetchPublicJson(
    `/clubs/${encodeURIComponent(
      clubSlug
    )}/tournament-team-results/${encodeURIComponent(
      tournamentId
    )}/${encodeURIComponent(drawId)}`,
    { next: { revalidate: 15 } },
    "results-detail"
  );
}

export async function createPublicFourPlayerTeam(
  clubSlug: string,
  payload: Record<string, unknown>
): Promise<ApiResult<Record<string, unknown>>> {
  return fetchPublicJson(
    `/clubs/${encodeURIComponent(
      clubSlug
    )}/tournament-registration/four-player-team`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    },
    "create-team"
  );
}

export async function recoverPublicFourPlayerTeamSetup(
  clubSlug: string,
  confirmationToken: string
): Promise<ApiResult<PublicFourPlayerTeamSetupRecovery>> {
  return fetchPublicJson(
    `/clubs/${encodeURIComponent(
      clubSlug
    )}/tournament-registration/four-player-team/recover`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        confirmation_token: confirmationToken,
        website: ""
      }),
      cache: "no-store"
    },
    "recover-team"
  );
}

export async function resolvePublicTeamInvitation(
  clubSlug: string,
  token: string
): Promise<ApiResult<Record<string, unknown>>> {
  return fetchPublicJson(
    `/clubs/${encodeURIComponent(clubSlug)}/tournament-team-invitation/resolve`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token })
    },
    "resolve-invitation"
  );
}

export async function respondPublicTeamInvitation(
  clubSlug: string,
  payload: Record<string, unknown>
): Promise<ApiResult<Record<string, unknown>>> {
  return fetchPublicJson(
    `/clubs/${encodeURIComponent(clubSlug)}/tournament-team-invitation/respond`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    },
    "respond-invitation"
  );
}
