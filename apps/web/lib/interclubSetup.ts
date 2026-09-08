import type { DivisionRule } from "./interclubRegistration";

export type ClubChoice = { id: string; name: string; slug: string };
export type PlanningMeet = { host_club_id: string; club_ids: string[]; starts_at: string | null; duration_minutes: number | null; courts: number | null };
export type PlanningDraft = { name: string; start_date: string | null; end_date: string | null; timezone: string;
  divisions: string[]; club_ids: string[]; meets: PlanningMeet[]; registration_rules: Record<string, DivisionRule>; setup_step: number };
export type PlanningSeason = { id: string; revision: number; draft: PlanningDraft; updated_at?: string };
export const setupSteps = ["Season details", "Participating clubs", "Divisions & eligibility", "Meet schedule", "Review & invite"];
export const divisionChoices = ["2.5", "3.0", "3.5", "4.0", "4.5", "5.0", "Open"];
export const emptyRule = (): DivisionRule => ({ min_rating: null, max_rating: null, women_required: null });
export const newSeason = (): PlanningSeason => ({ id: crypto.randomUUID(), revision: 0, draft: {
  name: "", start_date: null, end_date: null, timezone: "America/Mazatlan", divisions: ["3.5", "4.0"], club_ids: [], meets: [], registration_rules: {}, setup_step: 0
} });
export function normalizeDraft(draft: PlanningDraft): PlanningDraft {
  return { ...draft, start_date: draft.start_date || null, end_date: draft.end_date || null,
    setup_step: draft.setup_step || 0, registration_rules: Object.fromEntries(draft.divisions.map(d => [d, draft.registration_rules?.[d] || emptyRule()])) };
}

export function meetLocalTime(value: string | null, timeZone: string): string {
  if (!value || !Number.isFinite(Date.parse(value))) return "";
  const parts = new Intl.DateTimeFormat("en-CA", { timeZone, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).formatToParts(new Date(value));
  const get = (type: string) => parts.find(p => p.type === type)?.value;
  return `${get("year")}-${get("month")}-${get("day")}T${get("hour")}:${get("minute")}`;
}

// Derive possible offsets from the selected timezone, independently of the
// browser's location. Reject clock-change gaps and ambiguous local times.
export function meetUtcTime(local: string, timeZone: string): string | null {
  if (!local) return null;
  const wall = Date.parse(`${local}:00Z`);
  if (!Number.isFinite(wall)) throw new Error("Choose a valid meet date and time.");
  const offsets = new Set([-36, -12, 0, 12, 36].map(hours => {
    const probe = wall + hours * 3600000;
    return Date.parse(`${meetLocalTime(new Date(probe).toISOString(), timeZone)}:00Z`) - probe;
  }));
  const candidates = [...offsets].map(offset => new Date(wall - offset).toISOString()).filter(value => meetLocalTime(value, timeZone) === local);
  if (candidates.length !== 1) throw new Error("That time falls during a clock change. Choose another meet time.");
  return candidates[0];
}

function validDate(value: string | null): boolean {
  return Boolean(value && /^\d{4}-\d{2}-\d{2}$/.test(value) && Number.isFinite(Date.parse(`${value}T12:00:00Z`)) && new Date(`${value}T12:00:00Z`).toISOString().startsWith(value));
}
export function stepIssues(draft: PlanningDraft, step: number, now = Date.now()): string[] {
  const issues: string[] = [];
  if (step === 0) {
    if (!draft.name.trim()) issues.push("Enter a season name.");
    if (!validDate(draft.start_date)) issues.push("Choose the season start date.");
    if (!validDate(draft.end_date)) issues.push("Choose the season end date.");
    if (draft.start_date && draft.end_date && draft.end_date < draft.start_date) issues.push("The end date must follow the start date.");
  }
  if (step === 1 && draft.club_ids.length < 2) issues.push("Select at least two participating clubs. Each needs a PCS club account.");
  if (step === 2) {
    if (!draft.divisions.length) issues.push("Choose at least one division.");
    for (const division of draft.divisions) {
      const rule = draft.registration_rules[division] || emptyRule();
      for (const [name, value] of [["Minimum", rule.min_rating], ["Maximum", rule.max_rating]] as const) {
        if (value !== null && (!Number.isFinite(value) || value < 1 || value > 7)) issues.push(`${division}: ${name.toLowerCase()} rating must be between 1 and 7.`);
      }
      if (rule.min_rating !== null && rule.max_rating !== null && rule.min_rating > rule.max_rating) issues.push(`${division}: minimum rating cannot exceed maximum rating.`);
    }
  }
  if (step === 3) {
    if (!draft.meets.length) issues.push("Add at least one meet so clubs can prepare their rosters.");
    draft.meets.forEach((meet, index) => {
      const label = `Meet ${index + 1}`;
      if (!meet.host_club_id || !meet.club_ids.includes(meet.host_club_id)) issues.push(`${label}: choose a host club.`);
      if (meet.club_ids.length < 2 || meet.club_ids.length > 4) issues.push(`${label}: select 2–4 clubs, including the host.`);
      if (meet.club_ids.some(id => !draft.club_ids.includes(id))) issues.push(`${label}: choose clubs included in this season.`);
      const start = Date.parse(meet.starts_at || "");
      if (!Number.isFinite(start)) issues.push(`${label}: choose a date and time.`);
      else {
        const day = meetLocalTime(meet.starts_at, draft.timezone).slice(0, 10);
        const endDay = meetLocalTime(new Date(start + (meet.duration_minutes || 0) * 60000).toISOString(), draft.timezone).slice(0, 10);
        if (!draft.start_date || !draft.end_date || day < draft.start_date || endDay > draft.end_date) issues.push(`${label}: the meet must fall within the season dates.`);
        if (draft.meets.slice(0, index).some(other => start < Date.parse(other.starts_at || "") + (other.duration_minutes || 0) * 60000 && start + (meet.duration_minutes || 0) * 60000 > Date.parse(other.starts_at || "") && meet.club_ids.some(id => other.club_ids.includes(id)))) issues.push(`${label}: a club is already scheduled at another meet during this time.`);
      }
      if (!Number.isInteger(meet.duration_minutes) || meet.duration_minutes! < 30 || meet.duration_minutes! > 180) issues.push(`${label}: allow 30–180 minutes.`);
      if (!Number.isInteger(meet.courts) || meet.courts! < 1 || meet.courts! > 100) issues.push(`${label}: choose 1–100 courts.`);
    });
    if (draft.meets.length && !draft.meets.some(meet => Date.parse(meet.starts_at || "") > now)) issues.push("Schedule at least one upcoming meet before opening invitations.");
  }
  return issues;
}
export function firstIncompleteStep(draft: PlanningDraft): number {
  return [0, 1, 2, 3].find(step => stepIssues(draft, step).length > 0) ?? 4;
}
