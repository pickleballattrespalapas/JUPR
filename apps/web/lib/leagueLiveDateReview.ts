// League Live's operating date follows the club, not the browser/UTC day.
export const LEAGUE_LIVE_TIME_ZONE = "America/Mazatlan";

export function leagueLiveTimeZone(clubId: string): string {
  return clubId === "tres_palapas" ? LEAGUE_LIVE_TIME_ZONE : Intl.DateTimeFormat().resolvedOptions().timeZone;
}

export function leagueLiveToday(now = new Date(), timeZone = LEAGUE_LIVE_TIME_ZONE): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone, year: "numeric", month: "2-digit", day: "2-digit",
  }).formatToParts(now);
  const part = (type: string) => parts.find(value => value.type === type)?.value || "";
  return `${part("year")}-${part("month")}-${part("day")}`;
}

export function leagueLiveDateReview(
  matchDate: string, sessionId: string, roundNumber: number, now = new Date(), timeZone = LEAGUE_LIVE_TIME_ZONE,
): { today: string; timeZone: string; needsReview: boolean; reviewKey: string } {
  const today = leagueLiveToday(now, timeZone);
  return {
    today,
    timeZone,
    needsReview: Boolean(matchDate) && matchDate !== today,
    reviewKey: JSON.stringify([sessionId, roundNumber, matchDate, today]),
  };
}
