"use client";

import { useEffect, useRef, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, type ActionCompletion } from "@/components/interaction";
import type {
  AdminLeagueManagerDetailResponse,
  AdminLeagueManagerSchedulePreviewResponse,
  AdminLeagueManagerSchedulePreviewRow
} from "@/lib/adminLeagueManagerApi";

type SettingsPatch = Record<string, unknown>;
type LeagueFormat = "ladder" | "round_robin" | "rotating_partner" | "fixed_team" | "flex_challenge";
type SessionMode = "scheduled_rounds" | "live_court_board" | "self_scheduled";
type SeriesPreset = "one_game" | "two_games" | "best_of_3" | "three_games" | "best_of_5" | "custom_fixed" | "custom_best_of";
type LadderPodSize = "2" | "3" | "4" | "5" | "6" | "7" | "8";
type FormState = {
  description: string;
  divisions: string;
  summary: string;
  leagueFormat: LeagueFormat;
  sessionMode: SessionMode;
  startDate: string;
  weeks: string;
  useEndDate: boolean;
  endDate: string;
  weekday: string;
  timeStart: string;
  timeEnd: string;
  timezone: string;
  blackoutDates: string;
  sessionCapacity: string;
  totalCourts: string;
  courtIdentifiers: string;
  maxUsedCourts: string;
  ladderPodSize: LadderPodSize;
  ladderMoveUp: string;
  ladderMoveDown: string;
  seriesPreset: SeriesPreset;
  customSeriesGames: string;
  standingsTiebreak: "wins_then_point_differential" | "wins_then_total_points" | "points_then_point_differential";
  correctionWindow: "until_next_round" | "same_day" | "seven_days";
  scoreSubmissionPolicy: "admin_only" | "captain_or_admin" | "rostered_player_or_admin";
  playoffFormat: "none" | "single_elimination" | "double_elimination";
  minGames: string;
  kFactor: string;
};

type Props = {
  detail: AdminLeagueManagerDetailResponse;
  saving: boolean;
  canWrite: boolean;
  onSave: (patch: SettingsPatch, confirmationText: string) => Promise<boolean>;
  onPreview: (scheduleConfig: Record<string, unknown>) => Promise<AdminLeagueManagerSchedulePreviewResponse | null>;
};

const inputStyle = { width: "100%", padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "#f8fafc" };
const detailsStyle = { border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.75rem", background: "white" };
const gridStyle = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem", alignItems: "end" };
const timezoneOptions = ["America/Mazatlan", "America/Chicago", "America/Denver", "America/Los_Angeles", "America/New_York", "UTC"];
const singlesPodSizes = ["2", "3", "4", "5", "6", "7", "8"] as const;
const doublesPodSizes = ["4", "5", "6", "7", "8"] as const;
const setupSteps = ["Structure", "Schedule", "Courts & live play", "Match & standings", "Awards & eligibility", "Review"] as const;
const leagueFormatLabels: Record<LeagueFormat, string> = {
  ladder: "Ladder league",
  round_robin: "Season round robin",
  rotating_partner: "Rotating-partner individual league",
  fixed_team: "Fixed-team league",
  flex_challenge: "Flex challenge league"
};
const sessionModeLabels: Record<SessionMode, string> = {
  scheduled_rounds: "Scheduled rounds",
  live_court_board: "Live court board",
  self_scheduled: "Self-scheduled flex play"
};
const seriesLabels: Record<SeriesPreset, string> = {
  one_game: "1 game",
  two_games: "2 games — count each game",
  best_of_3: "2 out of 3",
  three_games: "3 games — count each game",
  best_of_5: "3 out of 5",
  custom_fixed: "Custom number of games",
  custom_best_of: "Custom best-of format"
};

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function textValue(value: unknown, fallback = ""): string { return value == null ? fallback : String(value); }
function numberText(value: unknown, fallback: number): string { return value == null || value === "" ? String(fallback) : String(value); }
function listText(value: unknown): string { return Array.isArray(value) ? value.map((item) => String(item)).join(", ") : ""; }
function optionValue<T extends string>(value: unknown, options: readonly T[], fallback: T): T { return options.includes(String(value) as T) ? String(value) as T : fallback; }
function uniqueList(raw: string): string[] { return Array.from(new Set(raw.replace(/\n/g, ",").split(",").map((item) => item.trim()).filter(Boolean))); }
function wholeNumber(value: string, label: string, minimum: number, maximum: number, optional = false): number | null {
  if (optional && !value.trim()) return null;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) throw new Error(`${label} must be a whole number from ${minimum} to ${maximum}.`);
  return parsed;
}
function isIsoDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}
function downloadTextFile(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/calendar;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
function isTeamLeague(detail: AdminLeagueManagerDetailResponse): boolean {
  return String(detail.league.league_type || "").trim().toLowerCase() === "team";
}
function ladderPodSizes(detail: AdminLeagueManagerDetailResponse): readonly LadderPodSize[] {
  return detail.league.match_format === "singles" ? singlesPodSizes : doublesPodSizes;
}
function ladderPodDefault(detail: AdminLeagueManagerDetailResponse): LadderPodSize {
  return detail.league.match_format === "singles" ? "2" : "4";
}
function normalizeLadderPodSize(value: unknown, detail: AdminLeagueManagerDetailResponse): LadderPodSize {
  const options = ladderPodSizes(detail);
  const legacyValue = String(value) === "6+" ? "6" : value;
  return optionValue(legacyValue, options, ladderPodDefault(detail));
}
function structurePreset(value: unknown): { preset: SeriesPreset; games: string } {
  const structure = asRecord(value);
  const kind = textValue(structure.kind, "fixed_games");
  const games = Number(structure.games || 1);
  if (kind === "best_of" && games === 3) return { preset: "best_of_3", games: "3" };
  if (kind === "best_of" && games === 5) return { preset: "best_of_5", games: "5" };
  if (kind === "best_of") return { preset: "custom_best_of", games: String(games) };
  if (games === 1) return { preset: "one_game", games: "1" };
  if (games === 2) return { preset: "two_games", games: "2" };
  if (games === 3) return { preset: "three_games", games: "3" };
  return { preset: "custom_fixed", games: String(games) };
}
function matchStructure(form: FormState): Record<string, unknown> {
  const known: Record<Exclude<SeriesPreset, "custom_fixed" | "custom_best_of">, { kind: "fixed_games" | "best_of"; games: number }> = {
    one_game: { kind: "fixed_games", games: 1 },
    two_games: { kind: "fixed_games", games: 2 },
    best_of_3: { kind: "best_of", games: 3 },
    three_games: { kind: "fixed_games", games: 3 },
    best_of_5: { kind: "best_of", games: 5 }
  };
  const selected = form.seriesPreset === "custom_fixed" || form.seriesPreset === "custom_best_of"
    ? { kind: form.seriesPreset === "custom_best_of" ? "best_of" as const : "fixed_games" as const, games: wholeNumber(form.customSeriesGames, "Custom game count", form.seriesPreset === "custom_best_of" ? 3 : 1, 9) as number }
    : known[form.seriesPreset];
  if (selected.kind === "best_of" && selected.games % 2 === 0) throw new Error("A best-of format needs an odd number of games.");
  return { kind: selected.kind, games: selected.games, result_counting: "each_game", completion: selected.kind === "best_of" ? "clinch" : "all_games" };
}

function formFromDetail(detail: AdminLeagueManagerDetailResponse): FormState {
  const league = detail.league;
  const schedule = asRecord(league.schedule_config);
  const courts = asRecord(league.court_board_defaults);
  const rules = asRecord(league.rules_config);
  const overview = asRecord(rules.overview);
  const competition = asRecord(rules.competition);
  const operation = asRecord(rules.operation);
  const series = structurePreset(competition.match_structure);
  const team = isTeamLeague(detail);
  return {
    description: textValue(league.description),
    divisions: listText(overview.divisions),
    summary: textValue(overview.summary),
    leagueFormat: team ? "fixed_team" : optionValue(overview.league_format, ["ladder", "round_robin", "rotating_partner", "fixed_team", "flex_challenge"] as const, "ladder"),
    sessionMode: optionValue(operation.session_mode, ["scheduled_rounds", "live_court_board", "self_scheduled"] as const, courts.rotation_mode === "queue" ? "live_court_board" : "scheduled_rounds"),
    startDate: textValue(schedule.start_date),
    weeks: schedule.weeks == null ? "" : String(schedule.weeks),
    useEndDate: Boolean(schedule.end_date),
    endDate: textValue(schedule.end_date),
    weekday: numberText(schedule.weekday, 0),
    timeStart: textValue(schedule.time_start, "18:00"),
    timeEnd: textValue(schedule.time_end, "20:00"),
    timezone: textValue(schedule.timezone) || "America/Mazatlan",
    blackoutDates: listText(schedule.blackout_dates),
    sessionCapacity: schedule.session_capacity == null ? "" : String(schedule.session_capacity),
    totalCourts: numberText(courts.total_courts, 0),
    courtIdentifiers: listText(courts.court_identifiers),
    maxUsedCourts: numberText(courts.max_used_courts, 0),
    ladderPodSize: normalizeLadderPodSize(courts.players_per_court, detail),
    ladderMoveUp: numberText(operation.move_up_count, 1),
    ladderMoveDown: numberText(operation.move_down_count, 1),
    seriesPreset: series.preset,
    customSeriesGames: series.games,
    standingsTiebreak: optionValue(competition.standings_tiebreak, ["wins_then_point_differential", "wins_then_total_points", "points_then_point_differential"] as const, "wins_then_point_differential"),
    correctionWindow: optionValue(competition.correction_window, ["until_next_round", "same_day", "seven_days"] as const, "until_next_round"),
    scoreSubmissionPolicy: optionValue(competition.score_submission_policy, ["admin_only", "captain_or_admin", "rostered_player_or_admin"] as const, "admin_only"),
    playoffFormat: optionValue(competition.playoff_format, ["none", "single_elimination", "double_elimination"] as const, "none"),
    minGames: numberText(league.min_games, 0),
    kFactor: numberText(league.k_factor, 32)
  };
}

function buildScheduleConfig(form: FormState, detail: AdminLeagueManagerDetailResponse): Record<string, unknown> {
  const weeks = wholeNumber(form.weeks, "Weeks", 1, 260, true);
  const weekday = wholeNumber(form.weekday, "Weekday", 0, 6);
  const capacity = wholeNumber(form.sessionCapacity, "Session capacity", 0, 1000, true);
  if (form.startDate && !isIsoDate(form.startDate)) throw new Error("Start date must use YYYY-MM-DD format.");
  if (form.sessionMode !== "self_scheduled" && form.startDate && weeks === null && !form.useEndDate) throw new Error("Choose a week count or an end date so the schedule has a defined length.");
  if (form.useEndDate && (!form.endDate || !isIsoDate(form.endDate))) throw new Error("Choose a valid end date or turn off Use end date.");
  if (form.startDate && form.useEndDate && form.endDate < form.startDate) throw new Error("End date cannot be before start date.");
  if (form.timeStart && form.timeEnd && form.timeEnd <= form.timeStart) throw new Error("End time must be after start time.");
  const blackoutDates = uniqueList(form.blackoutDates);
  const invalidBlackout = blackoutDates.find((value) => !isIsoDate(value));
  if (invalidBlackout) throw new Error(`Blackout date ${invalidBlackout} must use YYYY-MM-DD format.`);
  return { ...asRecord(detail.league.schedule_config), start_date: form.startDate, weeks: weeks || null, end_date: form.useEndDate ? form.endDate : "", weekday, time_start: form.timeStart, time_end: form.timeEnd, timezone: form.timezone, blackout_dates: blackoutDates, session_capacity: capacity || null };
}

function buildDraftPatch(form: FormState, detail: AdminLeagueManagerDetailResponse): SettingsPatch {
  const minGames = wholeNumber(form.minGames, "Minimum games", 0, 1000) as number;
  const kFactor = wholeNumber(form.kFactor, "K-factor", 1, 128) as number;
  const totalCourts = wholeNumber(form.totalCourts, "Total courts", 0, 100) as number;
  const maxUsedCourts = wholeNumber(form.maxUsedCourts, "Max used courts", 0, 100) as number;
  const moveUp = wholeNumber(form.ladderMoveUp, "Move up count", 0, 20) as number;
  const moveDown = wholeNumber(form.ladderMoveDown, "Move down count", 0, 20) as number;
  const leagueFormat = isTeamLeague(detail) ? "fixed_team" : form.leagueFormat;
  if (leagueFormat === "ladder" && form.sessionMode === "self_scheduled") throw new Error("Ladder leagues need scheduled rounds or a live court board.");
  if (leagueFormat === "flex_challenge" && form.sessionMode !== "self_scheduled") throw new Error("Flex challenge leagues use self-scheduled play.");
  const ladderPodSize = Number(form.ladderPodSize);
  if (leagueFormat === "ladder" && (moveUp >= ladderPodSize || moveDown >= ladderPodSize)) {
    throw new Error(`Move counts must be less than the ${ladderPodSize}-player pod size.`);
  }
  if (totalCourts && maxUsedCourts > totalCourts) throw new Error("Max used courts cannot exceed total courts.");
  const rules = asRecord(detail.league.rules_config);
  const overview = asRecord(rules.overview);
  const competition = asRecord(rules.competition);
  const operation = asRecord(rules.operation);
  const courtDefaults = { ...asRecord(detail.league.court_board_defaults) };
  delete courtDefaults.game_format_points;
  delete courtDefaults.game_format_time;
  return {
    description: form.description,
    min_games: minGames,
    k_factor: kFactor,
    schedule_config: buildScheduleConfig(form, detail),
    court_board_defaults: { ...courtDefaults, total_courts: totalCourts, court_identifiers: uniqueList(form.courtIdentifiers), max_used_courts: maxUsedCourts, players_per_court: form.ladderPodSize, rotation_mode: form.sessionMode === "live_court_board" ? "queue" : "fixed" },
    rules_config: {
      ...rules,
      overview: { ...overview, league_format: leagueFormat, divisions: uniqueList(form.divisions), summary: form.summary },
      competition: { ...competition, scoring_profile: "standard_pickleball", match_structure: matchStructure(form), standings_tiebreak: form.standingsTiebreak, correction_window: form.correctionWindow, score_submission_policy: form.scoreSubmissionPolicy, playoff_format: form.playoffFormat },
      operation: { ...operation, session_mode: form.sessionMode, move_up_count: leagueFormat === "ladder" ? moveUp : 0, move_down_count: leagueFormat === "ladder" ? moveDown : 0 }
    }
  };
}

function previewFromDetail(detail: AdminLeagueManagerDetailResponse): AdminLeagueManagerSchedulePreviewResponse {
  return { ok: true, mode: "league_manager_saved_schedule_preview", league_name: detail.league.league_name, schedule_config: asRecord(detail.league.schedule_config), schedule_preview: detail.schedule_preview || [], schedule_ics: detail.schedule_ics, schedule_ics_filename: detail.schedule_ics_filename };
}

function SchedulePreview({ preview }: { preview: AdminLeagueManagerSchedulePreviewResponse | null }) {
  if (!preview) return <p style={{ color: "#64748b" }}>Preview the draft to calculate sessions without saving.</p>;
  if (!preview.schedule_preview.length) return <p style={{ color: "#64748b" }}>The current schedule does not produce any sessions yet.</p>;
  return <div style={{ marginTop: "0.75rem", overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr>{["Session", "Date", "Start", "End"].map((title) => <th key={title} style={{ textAlign: "left", padding: "0.4rem", borderBottom: "1px solid #cbd5e1" }}>{title}</th>)}</tr></thead><tbody>{preview.schedule_preview.map((row: AdminLeagueManagerSchedulePreviewRow) => <tr key={`${row.session}-${row.date}`}><td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{row.session}</td><td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{row.date}</td><td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{row.start || "—"}</td><td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{row.end || "—"}</td></tr>)}</tbody></table>{preview.schedule_ics ? <p><button type="button" onClick={() => downloadTextFile(preview.schedule_ics_filename || "league-schedule.ics", preview.schedule_ics || "")} style={ghostButtonStyle}>Download preview ICS</button></p> : null}</div>;
}

function SetupReview({ detail, form, team }: { detail: AdminLeagueManagerDetailResponse; form: FormState; team: boolean }) {
  const matchFormat = team ? "Team doubles" : detail.league.match_format === "singles" ? "Individual singles" : "Individual doubles";
  const matchStructureLabel = form.seriesPreset === "custom_fixed" ? `${form.customSeriesGames || "—"} games — count each game` : form.seriesPreset === "custom_best_of" ? `Best of ${form.customSeriesGames || "—"}` : seriesLabels[form.seriesPreset];
  const scheduleLength = form.useEndDate ? `Ends ${form.endDate || "not set"}` : form.weeks ? `${form.weeks} week${form.weeks === "1" ? "" : "s"}` : "Length not set";
  const rows = [
    ["Created play format", matchFormat],
    ["League format", leagueFormatLabels[team ? "fixed_team" : form.leagueFormat]],
    ["Session operation", sessionModeLabels[form.sessionMode]],
    ["Schedule", form.sessionMode === "self_scheduled" ? "Self-scheduled" : `${form.startDate || "Start date not set"} · ${scheduleLength}`],
    ["Courts", `${form.totalCourts || "0"} available · ${form.maxUsedCourts || "0"} max in use`],
    ["Match structure", matchStructureLabel],
    ["Awards eligibility", `${form.minGames} minimum games`],
    ["Postseason", form.playoffFormat === "none" ? "No playoffs" : form.playoffFormat.replace(/_/g, " ")]
  ];

  return <section aria-label="League setup review" style={{ ...detailsStyle, marginTop: "0.75rem" }}>
    <h3 style={{ marginTop: 0 }}>Review league setup</h3>
    <p style={{ color: "#475569" }}>Confirm the saved settings below, then save the draft. Activation remains a separate lifecycle action.</p>
    <dl style={{ ...gridStyle, margin: 0 }}>
      {rows.map(([label, value]) => <div key={label} style={{ padding: "0.65rem", border: "1px solid #e2e8f0", borderRadius: "8px" }}><dt style={{ color: "#64748b", fontSize: "0.9rem" }}>{label}</dt><dd style={{ margin: "0.2rem 0 0", fontWeight: 800, textTransform: label === "Postseason" ? "capitalize" : undefined }}>{value}</dd></div>)}
    </dl>
  </section>;
}

export function GuidedLeagueSettingsEditor({ detail, saving, canWrite, onSave, onPreview }: Props) {
  const [form, setForm] = useState<FormState>(() => formFromDetail(detail));
  const [setupStep, setSetupStep] = useState(0);
  const configurationRef = useRef<HTMLFieldSetElement>(null);
  const reviewRef = useRef<HTMLDivElement>(null);
  const [localMessage, setLocalMessage] = useState<string | null>(null);
  const [localMessageIsError, setLocalMessageIsError] = useState(false);
  const [preview, setPreview] = useState<AdminLeagueManagerSchedulePreviewResponse | null>(() => previewFromDetail(detail));
  const status = detail.league.status;
  const isDraft = status === "draft";
  const isClosed = status === "ended" || status === "archived";
  const team = isTeamLeague(detail);
  const podSizes = ladderPodSizes(detail);
  const podMovementMaximum = Math.max(1, Number(form.ladderPodSize) - 1);
  const hasChanges = JSON.stringify(form) !== JSON.stringify(formFromDetail(detail));
  const timezones = timezoneOptions.includes(form.timezone) ? timezoneOptions : [form.timezone, ...timezoneOptions];

  useEffect(() => {
    setForm(formFromDetail(detail));
    setLocalMessage(null);
    setLocalMessageIsError(false);
    setPreview(previewFromDetail(detail));
    setSetupStep(0);
  }, [detail]);

  function updateField<K extends keyof FormState>(key: K, value: FormState[K]) { setForm((current) => ({ ...current, [key]: value })); }
  function updateScheduleField<K extends keyof FormState>(key: K, value: FormState[K]) { updateField(key, value); setPreview(null); }
  function selectSetupStep(nextStep: number) {
    const bounded = Math.max(0, Math.min(setupSteps.length - 1, nextStep));
    setSetupStep(bounded);
    requestAnimationFrame(() => {
      if (bounded >= 5) {
        reviewRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
        return;
      }
      const panels = Array.from(configurationRef.current?.querySelectorAll<HTMLDetailsElement>("details") || []);
      panels.forEach((panel, index) => { panel.open = index === bounded; });
      panels[bounded]?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }
  function changeLeagueFormat(value: LeagueFormat) {
    setForm((current) => ({ ...current, leagueFormat: value, sessionMode: value === "flex_challenge" ? "self_scheduled" : current.sessionMode === "self_scheduled" ? "scheduled_rounds" : current.sessionMode }));
  }
  async function previewSchedule() {
    setLocalMessage(null);
    setLocalMessageIsError(false);
    try {
      const payload = await onPreview(buildScheduleConfig(form, detail));
      if (payload) { setPreview(payload); setLocalMessage(`Previewed ${payload.schedule_preview.length} session(s) without saving.`); }
    } catch (error) {
      setLocalMessageIsError(true);
      setLocalMessage(error instanceof Error ? error.message : "Unable to preview schedule.");
    }
  }
  async function saveSettings(confirmationText: string): Promise<ActionCompletion> {
    setLocalMessage(null);
    setLocalMessageIsError(false);
    try {
      const saved = await onSave(isDraft ? buildDraftPatch(form, detail) : { description: form.description }, confirmationText);
      if (!saved) throw new Error("The league settings were not saved. Review the page message and try again.");
      const message = isDraft ? "The structured league draft was saved." : "The league description was saved.";
      setLocalMessage(message);
      return actionSuccess(isDraft ? "League draft saved" : "League description saved", message);
    } catch (error) {
      setLocalMessageIsError(true);
      setLocalMessage(error instanceof Error ? error.message : "Unable to validate league settings.");
      throw error;
    }
  }
  function resetForm() { setForm(formFromDetail(detail)); setPreview(previewFromDetail(detail)); setLocalMessage(null); setLocalMessageIsError(false); }

  return <article style={cardStyle}>
    <h2 style={{ marginTop: 0 }}>{isDraft ? "League setup wizard" : "League setup summary"}</h2>
    <p style={{ color: isClosed ? "#92400e" : "#475569" }}>{isDraft ? "Complete this setup before activation. League type and match modality were chosen when the draft was created and are shown as locked structural choices." : isClosed ? `This league is ${status}; its complete saved configuration is shown read-only.` : `This league is ${status}; its complete saved configuration is shown below, and only its description can be edited.`}</p>
    <label><strong>Description</strong><br /><textarea value={form.description} onChange={(event) => updateField("description", event.target.value)} disabled={isClosed} maxLength={2000} rows={3} style={inputStyle} /></label>
    {isDraft ? <section aria-label="League setup progress" style={{ ...detailsStyle, marginTop: "0.9rem" }}><strong>Step {setupStep + 1} of {setupSteps.length}: {setupSteps[setupStep]}</strong><div style={{ display: "flex", gap: "0.45rem", flexWrap: "wrap", marginTop: "0.7rem" }}>{setupSteps.map((label, index) => <button key={label} type="button" onClick={() => selectSetupStep(index)} aria-current={setupStep === index ? "step" : undefined} style={{ ...ghostButtonStyle, borderColor: setupStep === index ? "#2563eb" : "#cbd5e1", color: setupStep === index ? "#1d4ed8" : "#0f172a", background: setupStep === index ? "#dbeafe" : "white" }}>{index + 1}. {label}</button>)}</div><p style={{ display: "flex", gap: "0.5rem", marginBottom: 0 }}><button type="button" onClick={() => selectSetupStep(setupStep - 1)} disabled={setupStep === 0 || saving} style={ghostButtonStyle}>Back</button><button type="button" onClick={() => selectSetupStep(setupStep + 1)} disabled={setupStep === setupSteps.length - 1 || saving} style={buttonStyle}>Continue</button></p></section> : null}
    <fieldset ref={configurationRef} disabled={!isDraft} aria-label="League configuration" style={{ border: 0, padding: 0, margin: "0.75rem 0 0", minWidth: 0, opacity: isDraft ? 1 : 0.78 }}>
      {!isDraft ? <legend style={{ padding: 0, marginBottom: "0.6rem", fontWeight: 800, color: "#475569" }}>Saved configuration · read-only</legend> : null}
      <div style={{ display: "grid", gap: "0.75rem" }}>
        <details open={!isDraft || setupStep === 0} style={detailsStyle}><summary style={{ cursor: "pointer", fontWeight: 800 }}>League structure</summary><div style={{ ...gridStyle, marginTop: "0.75rem" }}><label><strong>League format</strong><br /><select value={team ? "fixed_team" : form.leagueFormat} onChange={(event) => changeLeagueFormat(event.target.value as LeagueFormat)} disabled={team} style={inputStyle}><option value="ladder">Ladder league</option><option value="round_robin">Season round robin</option><option value="rotating_partner">Rotating-partner individual league</option><option value="flex_challenge">Flex challenge league</option>{team ? <option value="fixed_team">Fixed-team league</option> : null}</select></label><label><strong>Created play format</strong><br /><div style={{ ...inputStyle, background: "#f8fafc" }}>{team ? "Team doubles" : detail.league.match_format === "singles" ? "Individual singles" : "Individual doubles"}</div></label><label><strong>Session operation</strong><br /><select value={form.sessionMode} onChange={(event) => updateField("sessionMode", event.target.value as SessionMode)} disabled={form.leagueFormat === "flex_challenge"} style={inputStyle}><option value="scheduled_rounds">Scheduled rounds</option><option value="live_court_board">Live court board</option><option value="self_scheduled">Self-scheduled flex play</option></select></label></div><label><strong>Divisions</strong><br /><input value={form.divisions} onChange={(event) => updateField("divisions", event.target.value)} placeholder="Open, Advanced" style={inputStyle} /></label><label><strong>Public summary</strong><br /><textarea value={form.summary} onChange={(event) => updateField("summary", event.target.value)} maxLength={2000} rows={3} style={inputStyle} /></label></details>

        <details open={!isDraft || setupStep === 1} style={detailsStyle}><summary style={{ cursor: "pointer", fontWeight: 800 }}>Schedule</summary><div style={{ ...gridStyle, marginTop: "0.75rem" }}><label><strong>Start date</strong><br /><input type="date" value={form.startDate} onChange={(event) => updateScheduleField("startDate", event.target.value)} style={inputStyle} /></label><label><strong>Weeks (or use an end date)</strong><br /><input type="number" min={1} max={260} value={form.weeks} onChange={(event) => updateScheduleField("weeks", event.target.value)} style={inputStyle} /></label><label><strong>Weekday</strong><br /><select value={form.weekday} onChange={(event) => updateScheduleField("weekday", event.target.value)} style={inputStyle}><option value="0">Monday</option><option value="1">Tuesday</option><option value="2">Wednesday</option><option value="3">Thursday</option><option value="4">Friday</option><option value="5">Saturday</option><option value="6">Sunday</option></select></label><label><strong>Timezone</strong><br /><select value={form.timezone} onChange={(event) => updateScheduleField("timezone", event.target.value)} style={inputStyle}>{timezones.map((timezone) => <option key={timezone} value={timezone}>{timezone}</option>)}</select></label><label><strong>Start time</strong><br /><input type="time" value={form.timeStart} onChange={(event) => updateScheduleField("timeStart", event.target.value)} style={inputStyle} /></label><label><strong>End time</strong><br /><input type="time" value={form.timeEnd} onChange={(event) => updateScheduleField("timeEnd", event.target.value)} style={inputStyle} /></label><label><strong>Session capacity</strong><br /><input type="number" min={0} max={1000} value={form.sessionCapacity} onChange={(event) => updateScheduleField("sessionCapacity", event.target.value)} style={inputStyle} /></label><label style={{ alignSelf: "center" }}><input type="checkbox" checked={form.useEndDate} onChange={(event) => updateScheduleField("useEndDate", event.target.checked)} /> <strong>Use end date</strong></label>{form.useEndDate ? <label><strong>End date</strong><br /><input type="date" value={form.endDate} onChange={(event) => updateScheduleField("endDate", event.target.value)} style={inputStyle} /></label> : null}</div><label><strong>Blackout dates</strong><br /><textarea value={form.blackoutDates} onChange={(event) => updateScheduleField("blackoutDates", event.target.value)} placeholder="2026-08-03, 2026-09-07" rows={2} style={inputStyle} /></label><p><button type="button" onClick={previewSchedule} disabled={saving || !canWrite} style={ghostButtonStyle}>{saving ? "Working…" : "Preview schedule without saving"}</button></p><SchedulePreview preview={preview} /></details>

        <details open={!isDraft || setupStep === 2} style={detailsStyle}><summary style={{ cursor: "pointer", fontWeight: 800 }}>Courts &amp; live operation</summary><div style={{ ...gridStyle, marginTop: "0.75rem" }}><label><strong>Total courts available</strong><br /><input type="number" min={0} max={100} value={form.totalCourts} onChange={(event) => updateField("totalCourts", event.target.value)} style={inputStyle} /></label><label><strong>Maximum courts this league may use</strong><br /><input type="number" min={0} max={100} value={form.maxUsedCourts} onChange={(event) => updateField("maxUsedCourts", event.target.value)} style={inputStyle} /></label>{form.leagueFormat === "ladder" ? <label><strong>Ladder pod size</strong><br /><select value={form.ladderPodSize} onChange={(event) => updateField("ladderPodSize", event.target.value as LadderPodSize)} style={inputStyle}>{podSizes.map((size) => <option key={size} value={size}>{size} players</option>)}</select></label> : null}{form.leagueFormat === "ladder" ? <label><strong>Move up each round</strong><br /><input type="number" min={0} max={podMovementMaximum} value={form.ladderMoveUp} onChange={(event) => updateField("ladderMoveUp", event.target.value)} style={inputStyle} /></label> : null}{form.leagueFormat === "ladder" ? <label><strong>Move down each round</strong><br /><input type="number" min={0} max={podMovementMaximum} value={form.ladderMoveDown} onChange={(event) => updateField("ladderMoveDown", event.target.value)} style={inputStyle} /></label> : null}</div><label><strong>Optional custom court labels</strong><br /><input value={form.courtIdentifiers} onChange={(event) => updateField("courtIdentifiers", event.target.value)} placeholder="1, 2, Championship" style={inputStyle} /></label><p style={{ color: "#475569", marginBottom: 0 }}>{detail.league.match_format === "singles" ? "Singles pods can run from 2 through 8 players; doubles pods run from 4 through 8 players." : "Doubles pods can run from 4 through 8 players."} Movement counts are limited by the selected pod size. Normal pickleball scoring is fixed; point caps and time caps are intentionally not league settings.</p></details>

        <details open={!isDraft || setupStep === 3} style={detailsStyle}><summary style={{ cursor: "pointer", fontWeight: 800 }}>Match, standings &amp; correction rules</summary><div style={{ ...gridStyle, marginTop: "0.75rem" }}><label><strong>Match structure</strong><br /><select value={form.seriesPreset} onChange={(event) => updateField("seriesPreset", event.target.value as SeriesPreset)} style={inputStyle}><option value="one_game">1 game</option><option value="two_games">2 games — count each game</option><option value="best_of_3">2 out of 3</option><option value="three_games">3 games — count each game</option><option value="best_of_5">3 out of 5</option><option value="custom_fixed">Custom number of games</option><option value="custom_best_of">Custom best-of format</option></select></label>{form.seriesPreset === "custom_fixed" || form.seriesPreset === "custom_best_of" ? <label><strong>{form.seriesPreset === "custom_best_of" ? "Best-of games (odd)" : "Number of games"}</strong><br /><input type="number" min={form.seriesPreset === "custom_best_of" ? 3 : 1} max={9} value={form.customSeriesGames} onChange={(event) => updateField("customSeriesGames", event.target.value)} style={inputStyle} /></label> : null}<label><strong>Standings tie-break</strong><br /><select value={form.standingsTiebreak} onChange={(event) => updateField("standingsTiebreak", event.target.value as FormState["standingsTiebreak"])} style={inputStyle}><option value="wins_then_point_differential">Wins → point differential</option><option value="wins_then_total_points">Wins → total points</option><option value="points_then_point_differential">Total points → point differential</option></select></label><label><strong>Score correction window</strong><br /><select value={form.correctionWindow} onChange={(event) => updateField("correctionWindow", event.target.value as FormState["correctionWindow"])} style={inputStyle}><option value="until_next_round">Until the next round</option><option value="same_day">Same day</option><option value="seven_days">Within 7 days</option></select></label><label><strong>Score submission</strong><br /><select value={form.scoreSubmissionPolicy} onChange={(event) => updateField("scoreSubmissionPolicy", event.target.value as FormState["scoreSubmissionPolicy"])} style={inputStyle}><option value="admin_only">Admin only</option><option value="captain_or_admin">Captain or admin</option><option value="rostered_player_or_admin">Rostered player or admin</option></select></label><label><strong>Postseason</strong><br /><select value={form.playoffFormat} onChange={(event) => updateField("playoffFormat", event.target.value as FormState["playoffFormat"])} style={inputStyle}><option value="none">No playoffs</option><option value="single_elimination">Single-elimination playoffs</option><option value="double_elimination">Double-elimination playoffs</option></select></label></div><p style={{ color: "#475569", marginBottom: 0 }}>{form.seriesPreset === "best_of_3" || form.seriesPreset === "best_of_5" || form.seriesPreset === "custom_best_of" ? "Best-of series finish when one side clinches the required wins; every played game still counts as an official league game." : "Fixed two- and three-game series create a separate league result for every game."} All leagues use standard pickleball scoring.</p></details>

        <details open={!isDraft || setupStep === 4} style={detailsStyle}><summary style={{ cursor: "pointer", fontWeight: 800 }}>Ratings, awards &amp; eligibility</summary><div style={{ ...gridStyle, marginTop: "0.75rem" }}><label><strong>Minimum games for awards</strong><br /><input type="number" min={0} max={1000} value={form.minGames} onChange={(event) => updateField("minGames", event.target.value)} style={inputStyle} /></label><label><strong>K-factor</strong><br /><input type="number" min={1} max={128} value={form.kFactor} onChange={(event) => updateField("kFactor", event.target.value)} style={inputStyle} /></label></div><p style={{ color: "#475569", marginBottom: 0 }}>Awards-race cards use selected award categories and their minimum criteria. Rating standings remain a separate reference, not the default awards leaderboard.</p></details>
      </div>
    </fieldset>
    {isDraft && setupStep === 5 ? <div ref={reviewRef}><SetupReview detail={detail} form={form} team={team} /></div> : null}
    {!isClosed ? <div style={{ ...gridStyle, marginTop: "0.9rem" }}><ConfirmAction triggerLabel={saving ? "Saving…" : isDraft ? "Save structured draft" : "Save description"} title={isDraft ? "Save this structured league draft?" : "Save this league description?"} description={isDraft ? "This saves the reviewed league format, schedule, courts, match structure, eligibility, and operating rules for this draft." : "This updates the description for the selected active league."} confirmLabel={isDraft ? "Yes, save draft" : "Yes, save description"} confirmationText="SAVE LEAGUE" disabled={!canWrite || !hasChanges} busy={saving} onConfirm={saveSettings} /><p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", margin: 0 }}><button type="button" onClick={resetForm} disabled={saving || !hasChanges} style={ghostButtonStyle}>Reset loaded values</button></p></div> : null}
    {localMessage ? <p role="status" style={{ color: localMessageIsError ? "#b91c1c" : "#166534" }}>{localMessage}</p> : null}
  </article>;
}
