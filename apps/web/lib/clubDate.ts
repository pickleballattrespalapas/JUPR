export const DEFAULT_CLUB_TIME_ZONE = "America/Mazatlan";

function zonedIso(value: Date, timeZone: string): string {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).formatToParts(value);
  const part = (type: Intl.DateTimeFormatPartTypes) => parts.find((item) => item.type === type)?.value || "";
  return `${part("year")}-${part("month")}-${part("day")}`;
}

function utcCalendar(isoDate: string): Date {
  const [year, month, day] = isoDate.split("-").map(Number);
  return new Date(Date.UTC(year, month - 1, day, 12));
}

function utcCalendarIso(value: Date): string {
  const year = value.getUTCFullYear();
  const month = String(value.getUTCMonth() + 1).padStart(2, "0");
  const day = String(value.getUTCDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

export function clubTodayIso(now = new Date(), timeZone = DEFAULT_CLUB_TIME_ZONE): string {
  return zonedIso(now, timeZone);
}

export function clubDaysAgoIso(days: number, now = new Date(), timeZone = DEFAULT_CLUB_TIME_ZONE): string {
  const calendar = utcCalendar(clubTodayIso(now, timeZone));
  calendar.setUTCDate(calendar.getUTCDate() - Math.max(0, Math.trunc(days)));
  return utcCalendarIso(calendar);
}

export function clubWeekStartIso(now = new Date(), timeZone = DEFAULT_CLUB_TIME_ZONE): string {
  const calendar = utcCalendar(clubTodayIso(now, timeZone));
  const daysSinceMonday = (calendar.getUTCDay() + 6) % 7;
  calendar.setUTCDate(calendar.getUTCDate() - daysSinceMonday);
  return utcCalendarIso(calendar);
}
