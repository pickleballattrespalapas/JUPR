"use client";

import { useEffect, useState } from "react";
import { ConfirmAction } from "@/components/ConfirmAction";
import { actionSuccess, type ActionCompletion } from "@/components/interaction";
import type {
  AdminLeagueManagerDetailResponse,
  AdminLeagueManagerSchedulePreviewResponse,
  AdminLeagueManagerSchedulePreviewRow
} from "@/lib/adminLeagueManagerApi";

type SettingsPatch = Record<string, unknown>;
type AwardKey = "highest_rating" | "most_improved" | "best_win_pct" | "most_wins";
type AwardCategoryForm = { enabled: boolean; minGames: string; depth: "1" | "3" };
type FormState = {
  description: string;
  leagueType: string;
  divisions: string;
  summary: string;
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
  playersPerCourt: "4" | "5" | "6+";
  rotationMode: "fixed" | "queue";
  gameFormatPoints: string;
  gameFormatTime: string;
  scoringRules: string;
  matchFormat: string;
  tieBreakRules: string;
  disputeWindow: string;
  disputePolicy: string;
  minGames: string;
  kFactor: string;
  awardDepth: "1" | "3";
  awardCategories: Record<AwardKey, AwardCategoryForm>;
};

type Props = {
  detail: AdminLeagueManagerDetailResponse;
  saving: boolean;
  canWrite: boolean;
  onSave: (patch: SettingsPatch, confirmationText: string) => Promise<boolean>;
  onPreview: (scheduleConfig: Record<string, unknown>) => Promise<AdminLeagueManagerSchedulePreviewResponse | null>;
};

const awardDefinitions: Array<{ key: AwardKey; label: string }> = [
  { key: "highest_rating", label: "Highest Rating" },
  { key: "most_improved", label: "Most Improved" },
  { key: "best_win_pct", label: "Best Win %" },
  { key: "most_wins", label: "Most Wins" }
];
const inputStyle = { width: "100%", padding: "0.5rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
const buttonStyle = { padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800 };
const ghostButtonStyle = { ...buttonStyle, background: "white", color: "#0f172a" };
const cardStyle = { border: "1px solid #e2e8f0", borderRadius: "14px", padding: "1rem", background: "#f8fafc" };
const detailsStyle = { border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.75rem", background: "white" };
const gridStyle = { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem", alignItems: "end" };

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function textValue(value: unknown, fallback = ""): string { return value == null ? fallback : String(value); }
function numberText(value: unknown, fallback: number): string { return value == null || value === "" ? String(fallback) : String(value); }
function listText(value: unknown): string { return Array.isArray(value) ? value.map((item) => String(item)).join(", ") : ""; }
function depthValue(value: unknown, fallback: "1" | "3" = "1"): "1" | "3" { return Number(value) === 3 ? "3" : fallback; }
function optionValue<T extends string>(value: unknown, options: readonly T[], fallback: T): T { return options.includes(String(value) as T) ? String(value) as T : fallback; }
function uniqueList(raw: string): string[] { return Array.from(new Set(raw.replace(/\n/g, ",").split(",").map((item) => item.trim()).filter(Boolean))); }
function isIsoDate(value: string): boolean {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.getTime()) && parsed.toISOString().slice(0, 10) === value;
}
function wholeNumber(value: string, label: string, minimum: number, maximum: number, optional = false): number | null {
  if (optional && !value.trim()) return null;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) throw new Error(`${label} must be a whole number from ${minimum} to ${maximum}.`);
  return parsed;
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

function categoryForm(config: Record<string, unknown>, minGames: string, defaultDepth: "1" | "3"): AwardCategoryForm {
  return {
    enabled: typeof config.enabled === "boolean" ? config.enabled : true,
    minGames: numberText(config.min_games, Number(minGames) || 0),
    depth: depthValue(config.depth, defaultDepth)
  };
}

function formFromDetail(detail: AdminLeagueManagerDetailResponse): FormState {
  const league = detail.league;
  const schedule = asRecord(league.schedule_config);
  const courts = asRecord(league.court_board_defaults);
  const rules = asRecord(league.rules_config);
  const overview = asRecord(rules.overview);
  const competition = asRecord(rules.competition);
  const awards = asRecord(league.awards_config);
  const categories = asRecord(awards.categories);
  const minGames = numberText(league.min_games, 0);
  const awardDepth = depthValue(awards.default_depth);
  return {
    description: textValue(league.description),
    leagueType: textValue(overview.league_type),
    divisions: listText(overview.divisions),
    summary: textValue(overview.summary),
    startDate: textValue(schedule.start_date),
    weeks: schedule.weeks == null ? "" : String(schedule.weeks),
    useEndDate: Boolean(schedule.end_date),
    endDate: textValue(schedule.end_date),
    weekday: numberText(schedule.weekday, 0),
    timeStart: textValue(schedule.time_start, "18:00"),
    timeEnd: textValue(schedule.time_end, "20:00"),
    timezone: textValue(schedule.timezone, "UTC"),
    blackoutDates: listText(schedule.blackout_dates),
    sessionCapacity: schedule.session_capacity == null ? "" : String(schedule.session_capacity),
    totalCourts: numberText(courts.total_courts, 0),
    courtIdentifiers: listText(courts.court_identifiers),
    maxUsedCourts: numberText(courts.max_used_courts, 0),
    playersPerCourt: optionValue(courts.players_per_court, ["4", "5", "6+"] as const, "4"),
    rotationMode: optionValue(courts.rotation_mode, ["fixed", "queue"] as const, "fixed"),
    gameFormatPoints: numberText(courts.game_format_points, 11),
    gameFormatTime: numberText(courts.game_format_time, 15),
    scoringRules: textValue(competition.scoring_rules),
    matchFormat: textValue(competition.match_format),
    tieBreakRules: textValue(competition.tie_break_rules),
    disputeWindow: textValue(competition.dispute_window),
    disputePolicy: textValue(competition.dispute_policy),
    minGames,
    kFactor: numberText(league.k_factor, 32),
    awardDepth,
    awardCategories: {
      highest_rating: categoryForm(asRecord(categories.highest_rating), minGames, awardDepth),
      most_improved: categoryForm(asRecord(categories.most_improved), minGames, awardDepth),
      best_win_pct: categoryForm(asRecord(categories.best_win_pct), minGames, awardDepth),
      most_wins: categoryForm(asRecord(categories.most_wins), minGames, awardDepth)
    }
  };
}

function previewFromDetail(detail: AdminLeagueManagerDetailResponse): AdminLeagueManagerSchedulePreviewResponse {
  return {
    ok: true,
    mode: "league_manager_saved_schedule_preview",
    league_name: detail.league.league_name,
    schedule_config: asRecord(detail.league.schedule_config),
    schedule_preview: detail.schedule_preview || [],
    schedule_ics: detail.schedule_ics,
    schedule_ics_filename: detail.schedule_ics_filename
  };
}

function buildScheduleConfig(form: FormState, detail: AdminLeagueManagerDetailResponse): Record<string, unknown> {
  const weeks = wholeNumber(form.weeks, "Weeks", 1, 260, true);
  const weekday = wholeNumber(form.weekday, "Weekday", 0, 6);
  const capacity = wholeNumber(form.sessionCapacity, "Session capacity", 0, 1000, true);
  if (form.startDate && !isIsoDate(form.startDate)) throw new Error("Start date must use YYYY-MM-DD format.");
  if (form.startDate && weeks === null && !form.useEndDate) throw new Error("Choose a week count or an end date so the schedule has a defined length.");
  if (form.useEndDate && (!form.endDate || !isIsoDate(form.endDate))) throw new Error("Choose a valid end date or turn off Use end date.");
  if (form.startDate && form.useEndDate && form.endDate < form.startDate) throw new Error("End date cannot be before start date.");
  if (form.timeStart && form.timeEnd && form.timeEnd <= form.timeStart) throw new Error("End time must be after start time.");
  const blackoutDates = uniqueList(form.blackoutDates);
  const invalidBlackout = blackoutDates.find((value) => !isIsoDate(value));
  if (invalidBlackout) throw new Error(`Blackout date ${invalidBlackout} must use YYYY-MM-DD format.`);
  return {
    ...asRecord(detail.league.schedule_config),
    start_date: form.startDate,
    weeks: weeks || null,
    end_date: form.useEndDate ? form.endDate : "",
    weekday,
    time_start: form.timeStart,
    time_end: form.timeEnd,
    timezone: form.timezone.trim() || "UTC",
    blackout_dates: blackoutDates,
    session_capacity: capacity || null
  };
}

function buildDraftPatch(form: FormState, detail: AdminLeagueManagerDetailResponse): SettingsPatch {
  const minGames = wholeNumber(form.minGames, "Minimum games", 0, 1000) as number;
  const kFactor = wholeNumber(form.kFactor, "K-factor", 1, 128) as number;
  const totalCourts = wholeNumber(form.totalCourts, "Total courts", 0, 100) as number;
  const maxUsedCourts = wholeNumber(form.maxUsedCourts, "Max used courts", 0, 100) as number;
  const gameFormatPoints = wholeNumber(form.gameFormatPoints, "Game format points", 1, 99) as number;
  const gameFormatTime = wholeNumber(form.gameFormatTime, "Game format time", 1, 240) as number;
  if (totalCourts && maxUsedCourts > totalCourts) throw new Error("Max used courts cannot exceed total courts.");
  const rules = asRecord(detail.league.rules_config);
  const overview = asRecord(rules.overview);
  const competition = asRecord(rules.competition);
  const awards = asRecord(detail.league.awards_config);
  const categories = asRecord(awards.categories);
  const awardCategories: Record<string, unknown> = { ...categories };
  for (const { key } of awardDefinitions) {
    const category = form.awardCategories[key];
    awardCategories[key] = {
      ...asRecord(categories[key]),
      enabled: category.enabled,
      min_games: wholeNumber(category.minGames, `${key} minimum games`, 0, 1000),
      depth: wholeNumber(category.depth, `${key} award depth`, 1, 3)
    };
  }
  return {
    description: form.description,
    min_games: minGames,
    k_factor: kFactor,
    schedule_config: buildScheduleConfig(form, detail),
    court_board_defaults: {
      ...asRecord(detail.league.court_board_defaults),
      total_courts: totalCourts,
      court_identifiers: uniqueList(form.courtIdentifiers),
      max_used_courts: maxUsedCourts,
      players_per_court: form.playersPerCourt,
      rotation_mode: form.rotationMode,
      game_format_points: gameFormatPoints,
      game_format_time: gameFormatTime
    },
    rules_config: {
      ...rules,
      overview: {
        ...overview,
        league_type: form.leagueType,
        divisions: uniqueList(form.divisions),
        summary: form.summary
      },
      competition: {
        ...competition,
        scoring_rules: form.scoringRules,
        match_format: form.matchFormat,
        tie_break_rules: form.tieBreakRules,
        dispute_window: form.disputeWindow,
        dispute_policy: form.disputePolicy
      }
    },
    awards_config: {
      ...awards,
      default_min_games: minGames,
      default_depth: wholeNumber(form.awardDepth, "Award depth", 1, 3),
      categories: awardCategories
    }
  };
}

function SchedulePreview({ preview }: { preview: AdminLeagueManagerSchedulePreviewResponse | null }) {
  if (!preview) return <p style={{ color: "#64748b" }}>Preview the draft to calculate sessions without saving.</p>;
  if (!preview.schedule_preview.length) return <p style={{ color: "#64748b" }}>The current schedule does not produce any sessions yet.</p>;
  return <div style={{ marginTop: "0.75rem" }}><div style={{ overflowX: "auto" }}><table style={{ width: "100%", borderCollapse: "collapse" }}><thead><tr><th style={{ textAlign: "left", padding: "0.4rem", borderBottom: "1px solid #cbd5e1" }}>Session</th><th style={{ textAlign: "left", padding: "0.4rem", borderBottom: "1px solid #cbd5e1" }}>Date</th><th style={{ textAlign: "left", padding: "0.4rem", borderBottom: "1px solid #cbd5e1" }}>Start</th><th style={{ textAlign: "left", padding: "0.4rem", borderBottom: "1px solid #cbd5e1" }}>End</th></tr></thead><tbody>{preview.schedule_preview.map((row: AdminLeagueManagerSchedulePreviewRow) => <tr key={`${row.session}-${row.date}`}><td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{row.session}</td><td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{row.date}</td><td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{row.start || "—"}</td><td style={{ padding: "0.4rem", borderBottom: "1px solid #e2e8f0" }}>{row.end || "—"}</td></tr>)}</tbody></table></div>{preview.schedule_ics ? <p><button type="button" onClick={() => downloadTextFile(preview.schedule_ics_filename || "league-schedule.ics", preview.schedule_ics || "")} style={ghostButtonStyle}>Download preview ICS</button></p> : null}</div>;
}

export function GuidedLeagueSettingsEditor({ detail, saving, canWrite, onSave, onPreview }: Props) {
  const [form, setForm] = useState<FormState>(() => formFromDetail(detail));
  const [localMessage, setLocalMessage] = useState<string | null>(null);
  const [localMessageIsError, setLocalMessageIsError] = useState(false);
  const [preview, setPreview] = useState<AdminLeagueManagerSchedulePreviewResponse | null>(() => previewFromDetail(detail));
  const status = detail.league.status;
  const isDraft = status === "draft";
  const isClosed = status === "ended" || status === "archived";
  const hasChanges = JSON.stringify(form) !== JSON.stringify(formFromDetail(detail));

  useEffect(() => {
    setForm(formFromDetail(detail));
    setLocalMessage(null);
    setLocalMessageIsError(false);
    setPreview(previewFromDetail(detail));
  }, [detail]);

  function updateField<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }
  function updateScheduleField<K extends keyof FormState>(key: K, value: FormState[K]) {
    updateField(key, value);
    setPreview(null);
  }
  function updateAwardCategory(key: AwardKey, patch: Partial<AwardCategoryForm>) {
    setForm((current) => ({
      ...current,
      awardCategories: {
        ...current.awardCategories,
        [key]: { ...current.awardCategories[key], ...patch }
      }
    }));
  }

  async function previewSchedule() {
    setLocalMessage(null);
    setLocalMessageIsError(false);
    try {
      const scheduleConfig = buildScheduleConfig(form, detail);
      const payload = await onPreview(scheduleConfig);
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
      const patch = isDraft ? buildDraftPatch(form, detail) : { description: form.description };
      const saved = await onSave(patch, confirmationText);
      if (!saved) throw new Error("The league settings were not saved. Review the page message and try again.");
      const successMessage = isDraft ? "The structured league draft was saved." : "The league description was saved.";
      setLocalMessage(successMessage);
      return actionSuccess(isDraft ? "League draft saved" : "League description saved", successMessage);
    } catch (error) {
      setLocalMessageIsError(true);
      setLocalMessage(error instanceof Error ? error.message : "Unable to validate league settings.");
      throw error;
    }
  }

  function resetForm() {
    setForm(formFromDetail(detail));
    setPreview(previewFromDetail(detail));
    setLocalMessage(null);
    setLocalMessageIsError(false);
  }

  return <article style={cardStyle}>
    <h2 style={{ marginTop: 0 }}>Guided settings editor</h2>
    <p style={{ color: isClosed ? "#92400e" : "#475569" }}>{isDraft ? "Draft leagues allow the full overview, schedule, courts, competition, ratings, and awards configuration. Unknown compatible extension keys are preserved." : isClosed ? `This league is ${status}; its complete saved configuration is shown read-only.` : `This league is ${status}; its complete saved configuration is shown below, and only its description can be edited.`}</p>
    <label><strong>Description</strong><br /><textarea value={form.description} onChange={(event) => updateField("description", event.target.value)} disabled={isClosed} maxLength={2000} rows={3} style={inputStyle} /></label>

    <fieldset disabled={!isDraft} aria-label="League configuration" style={{ border: 0, padding: 0, margin: "0.75rem 0 0", minWidth: 0, opacity: isDraft ? 1 : 0.78 }}>
      {!isDraft ? <legend style={{ padding: 0, marginBottom: "0.6rem", fontWeight: 800, color: "#475569" }}>Saved configuration · read-only</legend> : null}
      <div style={{ display: "grid", gap: "0.75rem" }}>
      <details open style={detailsStyle}><summary style={{ cursor: "pointer", fontWeight: 800 }}>Overview</summary><div style={{ ...gridStyle, marginTop: "0.75rem" }}><label><strong>League type</strong><br /><input value={form.leagueType} onChange={(event) => updateField("leagueType", event.target.value)} maxLength={80} style={inputStyle} /></label><label><strong>Divisions</strong><br /><input value={form.divisions} onChange={(event) => updateField("divisions", event.target.value)} placeholder="Open, Advanced" style={inputStyle} /></label></div><label><strong>Summary</strong><br /><textarea value={form.summary} onChange={(event) => updateField("summary", event.target.value)} maxLength={2000} rows={3} style={inputStyle} /></label></details>

      <details open style={detailsStyle}><summary style={{ cursor: "pointer", fontWeight: 800 }}>Schedule</summary><div style={{ ...gridStyle, marginTop: "0.75rem" }}><label><strong>Start date</strong><br /><input type="date" value={form.startDate} onChange={(event) => updateScheduleField("startDate", event.target.value)} style={inputStyle} /></label><label><strong>Weeks (or use an end date)</strong><br /><input type="number" min={1} max={260} value={form.weeks} onChange={(event) => updateScheduleField("weeks", event.target.value)} style={inputStyle} /></label><label><strong>Weekday</strong><br /><select value={form.weekday} onChange={(event) => updateScheduleField("weekday", event.target.value)} style={inputStyle}><option value="0">Monday</option><option value="1">Tuesday</option><option value="2">Wednesday</option><option value="3">Thursday</option><option value="4">Friday</option><option value="5">Saturday</option><option value="6">Sunday</option></select></label><label><strong>Timezone</strong><br /><input value={form.timezone} onChange={(event) => updateScheduleField("timezone", event.target.value)} maxLength={80} placeholder="America/Chicago" style={inputStyle} /></label><label><strong>Start time</strong><br /><input type="time" value={form.timeStart} onChange={(event) => updateScheduleField("timeStart", event.target.value)} style={inputStyle} /></label><label><strong>End time</strong><br /><input type="time" value={form.timeEnd} onChange={(event) => updateScheduleField("timeEnd", event.target.value)} style={inputStyle} /></label><label><strong>Session capacity</strong><br /><input type="number" min={0} max={1000} value={form.sessionCapacity} onChange={(event) => updateScheduleField("sessionCapacity", event.target.value)} style={inputStyle} /></label><label style={{ alignSelf: "center" }}><input type="checkbox" checked={form.useEndDate} onChange={(event) => updateScheduleField("useEndDate", event.target.checked)} /> <strong>Use end date</strong></label>{form.useEndDate ? <label><strong>End date</strong><br /><input type="date" value={form.endDate} onChange={(event) => updateScheduleField("endDate", event.target.value)} style={inputStyle} /></label> : null}</div><label><strong>Blackout dates</strong><br /><textarea value={form.blackoutDates} onChange={(event) => updateScheduleField("blackoutDates", event.target.value)} placeholder="2026-08-03, 2026-09-07" rows={2} style={inputStyle} /></label><p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}><button type="button" onClick={previewSchedule} disabled={saving || !canWrite} style={ghostButtonStyle}>{saving ? "Working…" : "Preview schedule without saving"}</button></p><SchedulePreview preview={preview} /></details>

      <details style={detailsStyle}><summary style={{ cursor: "pointer", fontWeight: 800 }}>Courts &amp; court board defaults</summary><div style={{ ...gridStyle, marginTop: "0.75rem" }}><label><strong>Total courts</strong><br /><input type="number" min={0} max={100} value={form.totalCourts} onChange={(event) => updateField("totalCourts", event.target.value)} style={inputStyle} /></label><label><strong>Court identifiers</strong><br /><input value={form.courtIdentifiers} onChange={(event) => updateField("courtIdentifiers", event.target.value)} placeholder="1, 2, Championship" style={inputStyle} /></label><label><strong>Max used courts</strong><br /><input type="number" min={0} max={100} value={form.maxUsedCourts} onChange={(event) => updateField("maxUsedCourts", event.target.value)} style={inputStyle} /></label><label><strong>Players per court</strong><br /><select value={form.playersPerCourt} onChange={(event) => updateField("playersPerCourt", event.target.value as FormState["playersPerCourt"])} style={inputStyle}><option value="4">4</option><option value="5">5</option><option value="6+">6+</option></select></label><label><strong>Rotation mode</strong><br /><select value={form.rotationMode} onChange={(event) => updateField("rotationMode", event.target.value as FormState["rotationMode"])} style={inputStyle}><option value="fixed">Fixed</option><option value="queue">Queue</option></select></label><label><strong>Game points cap</strong><br /><input type="number" min={1} max={99} value={form.gameFormatPoints} onChange={(event) => updateField("gameFormatPoints", event.target.value)} style={inputStyle} /></label><label><strong>Game time cap (minutes)</strong><br /><input type="number" min={1} max={240} value={form.gameFormatTime} onChange={(event) => updateField("gameFormatTime", event.target.value)} style={inputStyle} /></label></div></details>

      <details style={detailsStyle}><summary style={{ cursor: "pointer", fontWeight: 800 }}>Competition format &amp; rules</summary><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "0.75rem", marginTop: "0.75rem" }}><label><strong>Scoring rules</strong><br /><textarea value={form.scoringRules} onChange={(event) => updateField("scoringRules", event.target.value)} maxLength={2000} rows={4} style={inputStyle} /></label><label><strong>Match format</strong><br /><textarea value={form.matchFormat} onChange={(event) => updateField("matchFormat", event.target.value)} maxLength={2000} rows={4} style={inputStyle} /></label><label><strong>Tie-break rules</strong><br /><textarea value={form.tieBreakRules} onChange={(event) => updateField("tieBreakRules", event.target.value)} maxLength={2000} rows={4} style={inputStyle} /></label></div><div style={gridStyle}><label><strong>Dispute window</strong><br /><input value={form.disputeWindow} onChange={(event) => updateField("disputeWindow", event.target.value)} maxLength={200} style={inputStyle} /></label><label><strong>Who can submit disputes</strong><br /><input value={form.disputePolicy} onChange={(event) => updateField("disputePolicy", event.target.value)} maxLength={500} style={inputStyle} /></label></div></details>

      <details style={detailsStyle}><summary style={{ cursor: "pointer", fontWeight: 800 }}>Ratings &amp; eligibility</summary><div style={{ ...gridStyle, marginTop: "0.75rem" }}><label><strong>Minimum games</strong><br /><input type="number" min={0} max={1000} value={form.minGames} onChange={(event) => updateField("minGames", event.target.value)} style={inputStyle} /></label><label><strong>K-factor</strong><br /><input type="number" min={1} max={128} value={form.kFactor} onChange={(event) => updateField("kFactor", event.target.value)} style={inputStyle} /></label></div></details>

      <details style={detailsStyle}><summary style={{ cursor: "pointer", fontWeight: 800 }}>Awards &amp; trophies</summary><div style={{ ...gridStyle, marginTop: "0.75rem" }}><label><strong>Default award depth</strong><br /><select value={form.awardDepth} onChange={(event) => updateField("awardDepth", event.target.value as "1" | "3")} style={inputStyle}><option value="1">Top 1</option><option value="3">Top 3</option></select></label></div><div style={{ display: "grid", gap: "0.75rem", marginTop: "0.75rem" }}>{awardDefinitions.map(({ key, label }) => { const category = form.awardCategories[key]; return <fieldset key={key} style={{ border: "1px solid #e2e8f0", borderRadius: "10px", padding: "0.75rem" }}><legend style={{ fontWeight: 800 }}>{label}</legend><div style={gridStyle}><label><input type="checkbox" checked={category.enabled} onChange={(event) => updateAwardCategory(key, { enabled: event.target.checked })} /> Enabled</label><label><strong>Minimum games</strong><br /><input type="number" min={0} max={1000} value={category.minGames} onChange={(event) => updateAwardCategory(key, { minGames: event.target.value })} style={inputStyle} /></label><label><strong>Award depth</strong><br /><select value={category.depth} onChange={(event) => updateAwardCategory(key, { depth: event.target.value as "1" | "3" })} style={inputStyle}><option value="1">Top 1</option><option value="3">Top 3</option></select></label></div></fieldset>; })}</div></details>
      </div>
    </fieldset>

    {!isClosed ? <div style={{ ...gridStyle, marginTop: "0.9rem" }}><ConfirmAction triggerLabel={saving ? "Saving…" : isDraft ? "Save structured draft" : "Save description"} title={isDraft ? "Save this structured league draft?" : "Save this league description?"} description={isDraft ? "This saves the reviewed overview, schedule, court, competition, rating, and award settings for this draft." : "This updates the description for the selected active league."} confirmLabel={isDraft ? "Yes, save draft" : "Yes, save description"} confirmationText="SAVE LEAGUE" disabled={!canWrite || !hasChanges} busy={saving} onConfirm={saveSettings} /><p style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", margin: 0 }}><button type="button" onClick={resetForm} disabled={saving || !hasChanges} style={ghostButtonStyle}>Reset loaded values</button></p></div> : null}
    {localMessage ? <p role="status" style={{ color: localMessageIsError ? "#b91c1c" : "#166534" }}>{localMessage}</p> : null}
  </article>;
}
