export type SignupClub = { id: string; name: string };
export type SignupSeason = { id: string; name: string; start_date: string; end_date: string; timezone: string; divisions: string[] };
export type SignupMeet = { id: string; starts_at: string; host_club_id: string | null; host_club_name?: string | null };
export type SeasonSignupDetails = { club: SignupClub; season: SignupSeason; signup: { open: boolean }; meets: SignupMeet[] };
export type SeasonMember = { id: string; name: string; email: string; divisions: string[]; notes: string; status: "active" | "withdrawn"; revision: number };
export type PlayerResponseDetails = {
  kind: "season" | "meet";
  club: SignupClub;
  season: SignupSeason;
  member: SeasonMember;
  meet?: SignupMeet;
  availability?: { status: "invited" | "available" | "maybe" | "unavailable"; revision: number; deadline: string | null; open: boolean };
  can_respond: boolean;
  can_withdraw?: boolean;
};

export class PlayerSignupError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

export function playerSignupApi(): string | null {
  return process.env.NEXT_PUBLIC_JUPR_API_BASE_URL?.trim().replace(/\/+$/, "") || null;
}

export async function playerSignupRequest<T>(path: string, signal: AbortSignal, body?: unknown): Promise<T> {
  const api = playerSignupApi();
  if (!api) throw new PlayerSignupError("The signup service is unavailable. Please try again later.", 503);
  const response = await fetch(`${api}${path}`, {
    ...(body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
    signal, cache: "no-store", credentials: "omit", referrerPolicy: "no-referrer",
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const message = response.status === 404 ? "This link is no longer available. Ask your club administrator for a new link."
      : response.status === 409 ? "These details have changed or responses have closed. Reload the latest details before continuing."
      : response.status === 429 ? "Too many attempts. Please wait a few minutes and try again."
      : typeof data?.detail === "string" ? data.detail : "We could not complete this request. Please try again.";
    throw new PlayerSignupError(message, response.status);
  }
  if (!data) throw new PlayerSignupError("We could not read the response. Please reload and try again.", 503);
  return data as T;
}

export function signupErrorMessage(error: unknown): string {
  return error instanceof PlayerSignupError ? error.message : "We could not connect. Check your connection and try again.";
}

export function seasonDates(season: SignupSeason): string {
  const format = (value: string) => new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(new Date(`${value}T12:00:00Z`));
  return `${format(season.start_date)} – ${format(season.end_date)}`;
}

export function signupMeetTime(value: string, timezone: string): string {
  return new Intl.DateTimeFormat("en-US", { dateStyle: "full", timeStyle: "short", timeZone: timezone }).format(new Date(value));
}
