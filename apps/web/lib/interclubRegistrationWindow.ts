export type SeasonRegistrationWindow = {
  opens_at: string | null; closes_at: string | null; revision: number;
  status: "unconfigured" | "scheduled" | "open" | "closed";
  can_register: boolean; meet_planning_open: boolean;
};

/** Clock-driven phase changes do not increment the commissioner's date revision. */
export function latestRegistrationWindow(left: SeasonRegistrationWindow | null | undefined, right: SeasonRegistrationWindow | null | undefined): SeasonRegistrationWindow | undefined {
  if (!left) return right || undefined;
  if (!right) return left;
  if (left.revision !== right.revision) return left.revision > right.revision ? left : right;
  const phaseOrder = { unconfigured: 0, scheduled: 1, open: 2, closed: 3 };
  return phaseOrder[left.status] >= phaseOrder[right.status] ? left : right;
}

export function registrationPhase(window: SeasonRegistrationWindow | null | undefined, now = Date.now()): SeasonRegistrationWindow["status"] {
  const opens = Date.parse(window?.opens_at || ""), closes = Date.parse(window?.closes_at || "");
  if (!Number.isFinite(opens) || !Number.isFinite(closes) || closes <= opens) return "unconfigured";
  return now < opens ? "scheduled" : now >= closes ? "closed" : "open";
}
export function registrationCanAccept(window: SeasonRegistrationWindow | null | undefined, now = Date.now()): boolean {
  return window?.can_register === true && window.status === "open" && registrationPhase(window, now) === "open";
}
export function registrationMeetPlanning(window: SeasonRegistrationWindow | null | undefined, now = Date.now()): boolean {
  return window?.meet_planning_open === true && window.status === "closed" && registrationPhase(window, now) === "closed";
}
export function registrationWindowMessage(window: SeasonRegistrationWindow | null | undefined, now = Date.now()): string {
  const phase = registrationPhase(window, now);
  return phase === "unconfigured" ? "The commissioner has not set registration dates yet."
    : phase === "scheduled" ? "Season registration has not opened yet."
    : phase === "closed" ? "Season registration has closed."
    : registrationCanAccept(window, now) ? "Season registration is open."
    : "Checking the latest season registration dates…";
}
export function registrationWindowDates(window: SeasonRegistrationWindow | null | undefined, timezone: string): string | null {
  if (!window?.opens_at || !window.closes_at || registrationPhase(window) === "unconfigured") return null;
  const format = (value: string) => new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short", timeZone: timezone }).format(new Date(value));
  return `${format(window.opens_at)} – ${format(window.closes_at)} (${timezone})`;
}
