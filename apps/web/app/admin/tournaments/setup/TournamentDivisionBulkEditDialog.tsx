"use client";

import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from "react";
import { FormDialog, InteractionActionError, type ActionCompletion } from "@/components/interaction";
import {
  COMPETITION_FORMATS,
  SCORING_OPTIONS,
  cleanString,
  effectiveParticipantType,
  dayLabel,
  dayReference,
  eventDayReferences,
  eventDivisionName,
  eventFamilyDefaults,
  eventFamilyName,
  numberInputValue,
  recordBoolean,
  setEventDayReferences,
  setRecordNumber,
  type BuilderRow,
  type SetupRecord
} from "../../tournament-setup/tournamentSetupBuilder";
import {
  skillAnchor,
  skillEligibilityMode,
  skillEligibilitySummary,
  type SkillEligibilityMode
} from "@/lib/tournamentSkillEligibility";

type Props = {
  open: boolean;
  divisions: BuilderRow[];
  eventFamilies: BuilderRow[];
  days: BuilderRow[];
  disabled?: boolean;
  onCancel: () => void;
  onConfirm: (rows: Array<{ key: string; value: SetupRecord }>) => Promise<ActionCompletion>;
  onAcknowledge?: () => void;
};

type EditableKey = "capacity" | "fee" | "waitlist" | "partnerBoard" | "draw" | "scoring" | "eligibility" | "schedule";
type FieldSelection = Record<EditableKey, boolean>;

type BulkValues = {
  capacity: string;
  fee: string;
  waitlist: boolean;
  partnerBoard: boolean;
  draw: string;
  scoring: string;
  eligibility: SkillEligibilityMode;
  minimum: string;
  maximum: string;
  combinedCap: string;
  scheduleMode: "INHERIT_EVENT" | "CUSTOM";
  scheduledDayIds: string[];
};

const inputStyle: CSSProperties = { width: "100%", minWidth: 0, boxSizing: "border-box", padding: "0.55rem", border: "1px solid #cbd5e1", borderRadius: "8px", font: "inherit" };
function optionLabel(value: string): string {
  return value.toLowerCase().split("_").map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
}

function uniqueValue<T>(values: T[]): T | undefined {
  if (!values.length) return undefined;
  return values.every((value) => Object.is(value, values[0])) ? values[0] : undefined;
}

function sameStringArray(values: string[][]): string[] | undefined {
  if (!values.length) return undefined;
  const normalized = values.map((value) => [...value].sort().join("\u0000"));
  return normalized.every((value) => value === normalized[0]) ? [...values[0]] : undefined;
}

function eligibilityMode(value: SetupRecord): SkillEligibilityMode {
  return skillEligibilityMode(value);
}


function applyBulkValues(value: SetupRecord, selected: FieldSelection, values: BulkValues): SetupRecord {
  let next = { ...value };
  if (selected.capacity) next = setRecordNumber(next, "capacity_teams", values.capacity);
  if (selected.fee) next = setRecordNumber(next, "price_usd", values.fee);
  if (selected.waitlist) next.waitlist_enabled = values.waitlist;
  if (selected.partnerBoard) next.partner_board_enabled = values.partnerBoard;
  if (selected.draw) { next.event_format_override = values.draw; next.division_format = values.draw; }
  if (selected.scoring) { next.scoring_override = values.scoring; next.division_scoring = values.scoring; }
  if (selected.eligibility) {
    next.eligibility_mode = values.eligibility;
    next.skill_mode = values.eligibility;
    next.combined_rating_cap = values.eligibility === "COMBINED_RATING_CAP" ? Number(values.combinedCap) : null;
    next.skill_min_rating = ["MINIMUM", "CUSTOM"].includes(values.eligibility) && values.minimum !== "" ? Number(values.minimum) : null;
    next.skill_max_rating = values.eligibility === "CUSTOM" && values.maximum !== "" ? Number(values.maximum) : null;
    if (values.eligibility === "OPEN") next.skill_label = "Open";
    if (values.eligibility === "MINIMUM" && values.minimum !== "") next.skill_label = `${Number(values.minimum).toFixed(2).replace(/0+$/, "").replace(/\.$/, "")}+`;
    if (values.eligibility === "STANDARD") {
      const anchor = skillAnchor(next.skill_label) ?? 3.5;
      next.skill_label = anchor.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
    }
  }
  if (selected.schedule) {
    next.schedule_mode = values.scheduleMode;
    next = setEventDayReferences(next, values.scheduledDayIds);
  }
  return next;
}

function changedFields(before: SetupRecord, after: SetupRecord): string[] {
  const fields: Array<[string, unknown, unknown]> = [
    ["Capacity", before.capacity_teams, after.capacity_teams],
    ["Entry fee", before.price_usd, after.price_usd],
    ["Waitlist", before.waitlist_enabled, after.waitlist_enabled],
    ["Partner Board", before.partner_board_enabled, after.partner_board_enabled],
    ["Draw override", before.event_format_override ?? before.division_format, after.event_format_override ?? after.division_format],
    ["Scoring override", before.scoring_override ?? before.division_scoring, after.scoring_override ?? after.division_scoring],
    ["Eligibility", before.eligibility_mode, after.eligibility_mode],
    ["Minimum rating", before.skill_min_rating, after.skill_min_rating],
    ["Maximum rating", before.skill_max_rating, after.skill_max_rating],
    ["Combined cap", before.combined_rating_cap, after.combined_rating_cap],
    ["Schedule", eventDayReferences(before).join("|"), eventDayReferences(after).join("|")]
  ];
  return fields.filter(([, previous, next]) => JSON.stringify(previous ?? null) !== JSON.stringify(next ?? null)).map(([label]) => label);
}

export default function TournamentDivisionBulkEditDialog({ open, divisions, eventFamilies, days, disabled = false, onCancel, onConfirm, onAcknowledge }: Props) {
  const [selected, setSelected] = useState<FieldSelection>({ capacity: false, fee: false, waitlist: false, partnerBoard: false, draw: false, scoring: false, eligibility: false, schedule: false });
  const [values, setValues] = useState<BulkValues>({ capacity: "", fee: "", waitlist: true, partnerBoard: true, draw: "", scoring: "", eligibility: "STANDARD", minimum: "3.5", maximum: "4.0", combinedCap: "8.0", scheduleMode: "INHERIT_EVENT", scheduledDayIds: [] });

  const participantTypes = useMemo(() => divisions.map((row) => effectiveParticipantType(row.value, eventFamilies)), [divisions, eventFamilies]);
  const allTeamFormats = participantTypes.every((value) => value !== "SINGLES");
  const allCombinedEligible = useMemo(
    () =>
      allTeamFormats &&
      divisions.every((row) => {
        const family = eventFamilyDefaults(eventFamilies, eventFamilyName(row.value));
        const format = cleanString(
          row.value.competition_format ?? family?.competition_format
        ).toUpperCase();
        return format !== "FOUR_PLAYER_TEAM";
      }),
    [allTeamFormats, divisions, eventFamilies]
  );
  const familyNames = useMemo(() => [...new Set(divisions.map((row) => eventFamilyName(row.value).toLowerCase()))], [divisions]);
  const sameFamily = familyNames.length === 1;
  const familyDefaults = sameFamily ? eventFamilyDefaults(eventFamilies, eventFamilyName(divisions[0]?.value || {})) : null;
  const familyDays = useMemo(() => eventDayReferences(familyDefaults || {}), [familyDefaults]);
  const dayLabels = useMemo(() => new Map(days.map((day) => [dayReference(day.value), dayLabel(day.value) || dayReference(day.value)])), [days]);

  useEffect(() => {
    if (!open || !divisions.length) return;
    const commonCapacity = uniqueValue(divisions.map((row) => numberInputValue(row.value.capacity_teams)));
    const commonFee = uniqueValue(divisions.map((row) => numberInputValue(row.value.price_usd)));
    const commonWaitlist = uniqueValue(divisions.map((row) => recordBoolean(row.value.waitlist_enabled, true)));
    const commonPartnerBoard = uniqueValue(divisions.map((row) => recordBoolean(row.value.partner_board_enabled, true)));
    const commonDraw = uniqueValue(divisions.map((row) => cleanString(row.value.event_format_override ?? row.value.division_format)));
    const commonScoring = uniqueValue(divisions.map((row) => cleanString(row.value.scoring_override ?? row.value.division_scoring)));
    const commonEligibility = uniqueValue(divisions.map((row) => eligibilityMode(row.value)));
    const commonMinimum = uniqueValue(divisions.map((row) => numberInputValue(row.value.skill_min_rating)));
    const commonMaximum = uniqueValue(divisions.map((row) => numberInputValue(row.value.skill_max_rating)));
    const commonCap = uniqueValue(divisions.map((row) => numberInputValue(row.value.combined_rating_cap)));
    const commonScheduleMode = uniqueValue(divisions.map((row) => cleanString(row.value.schedule_mode).toUpperCase() === "CUSTOM" ? "CUSTOM" : "INHERIT_EVENT"));
    const commonSchedule = sameStringArray(divisions.map((row) => eventDayReferences(row.value)));
    setSelected({ capacity: false, fee: false, waitlist: false, partnerBoard: false, draw: false, scoring: false, eligibility: false, schedule: false });
    setValues({ capacity: commonCapacity ?? "", fee: commonFee ?? "", waitlist: commonWaitlist ?? true, partnerBoard: commonPartnerBoard ?? true, draw: commonDraw ?? "", scoring: commonScoring ?? "", eligibility: commonEligibility ?? "STANDARD", minimum: commonMinimum === undefined ? "3.5" : commonMinimum, maximum: commonMaximum === undefined ? "4.0" : commonMaximum, combinedCap: commonCap || "8.0", scheduleMode: (commonScheduleMode as "INHERIT_EVENT" | "CUSTOM" | undefined) ?? "INHERIT_EVENT", scheduledDayIds: commonSchedule ?? (sameFamily ? familyDays : []) });
  }, [open, divisions, sameFamily, familyDays]);

  const proposedRows = useMemo(() => divisions.map((row) => ({ key: row.key, value: applyBulkValues(row.value, selected, values) })), [divisions, selected, values]);
  const previewRows = useMemo(() => proposedRows.map((row, index) => ({ key: row.key, name: eventDivisionName(divisions[index]?.value || row.value), changes: changedFields(divisions[index]?.value || {}, row.value) })), [proposedRows, divisions]);

  if (!open) return null;

  function toggleField(key: EditableKey, enabled: boolean) { setSelected((current) => ({ ...current, [key]: enabled })); }

  function invalid(message: string, field = "Bulk edit"): never {
    throw new InteractionActionError(message, { kind: "validation", fieldErrors: { [field]: message } });
  }

  async function submit(): Promise<ActionCompletion> {
    if (!divisions.length) invalid("Select at least one division.");
    if (!Object.values(selected).some(Boolean)) invalid("Choose at least one shared setting to change.");
    if (selected.capacity) { const capacity = Number(values.capacity); if (!Number.isInteger(capacity) || capacity < 1) invalid("Capacity must be a whole number of at least 1.", "Capacity"); }
    if (selected.fee) { const fee = Number(values.fee); if (!Number.isFinite(fee) || fee < 0) invalid("Entry fee cannot be negative.", "Entry fee"); }
    if (selected.eligibility) {
      if (values.eligibility === "COMBINED_RATING_CAP") {
        const cap = Number(values.combinedCap);
        if (!allCombinedEligible) invalid("Combined team rating is available only for standard doubles/team divisions.", "Eligibility");
        if (!Number.isFinite(cap) || cap <= 0 || cap > 14) invalid("Combined rating cap must be greater than 0 and no more than 14.", "Eligibility");
      }
      if (values.eligibility === "MINIMUM") {
        const minimum = Number(values.minimum);
        if (!Number.isFinite(minimum) || minimum < 1 || minimum > 7) invalid("Minimum rating must be between 1.0 and 7.0.", "Eligibility");
      }
      if (values.eligibility === "CUSTOM") {
        const minimum = values.minimum === "" ? null : Number(values.minimum);
        const maximum = values.maximum === "" ? null : Number(values.maximum);
        if (minimum == null && maximum == null) invalid("Custom eligibility requires a minimum, a maximum, or both.", "Eligibility");
        if (minimum != null && (!Number.isFinite(minimum) || minimum < 1 || minimum > 7)) invalid("Custom minimum rating must be between 1.0 and 7.0.", "Eligibility");
        if (maximum != null && (!Number.isFinite(maximum) || maximum <= 1 || maximum > 7.5)) invalid("Custom maximum rating must be greater than 1.0 and no more than 7.5.", "Eligibility");
        if (minimum != null && maximum != null && maximum <= minimum) invalid("Custom maximum must be greater than the minimum.", "Eligibility");
      }
    }
    if (selected.schedule) {
      if (!sameFamily) invalid("Tournament days can be bulk edited only when all selected divisions share the same parent Event.", "Tournament days");
      if (!values.scheduledDayIds.length) invalid("Choose at least one tournament day.", "Tournament days");
    }
    if (!previewRows.some((row) => row.changes.length)) invalid("The selected settings do not change any division.");
    return onConfirm(proposedRows);
  }

  const mixedSummary = (items: unknown[]) => uniqueValue(items) === undefined ? "Multiple values" : "Same value";

  return (
    <FormDialog open={open} mode="bulk" size="xwide" title={`Bulk edit ${divisions.length} divisions`} description="Only shared compatible settings are available. Check a setting to change it; unchecked settings remain untouched. Review every affected division before the atomic private-draft save." dirty={Object.values(selected).some(Boolean)} submitLabel={`Save ${divisions.length} divisions together`} submitDisabled={disabled || !Object.values(selected).some(Boolean)} workingLabel="Saving divisions…" onSubmit={submit} onCancel={onCancel} onAcknowledge={onAcknowledge}>
        <div style={{ display: "grid", gap: "0.7rem" }}>
          <BulkField label="Capacity" checked={selected.capacity} disabled={disabled} summary={mixedSummary(divisions.map((row) => row.value.capacity_teams))} onChange={(checked) => toggleField("capacity", checked)}><input type="number" min="1" step="1" value={values.capacity} disabled={!selected.capacity || disabled} style={inputStyle} onChange={(event) => setValues((current) => ({ ...current, capacity: event.target.value }))} /></BulkField>
          <BulkField label="Entry fee (USD)" checked={selected.fee} disabled={disabled} summary={mixedSummary(divisions.map((row) => row.value.price_usd))} onChange={(checked) => toggleField("fee", checked)}><input type="number" min="0" step="0.01" value={values.fee} disabled={!selected.fee || disabled} style={inputStyle} onChange={(event) => setValues((current) => ({ ...current, fee: event.target.value }))} /></BulkField>
          <BulkField label="Waitlist" checked={selected.waitlist} disabled={disabled} summary={mixedSummary(divisions.map((row) => recordBoolean(row.value.waitlist_enabled, true)))} onChange={(checked) => toggleField("waitlist", checked)}><select value={values.waitlist ? "ENABLED" : "DISABLED"} disabled={!selected.waitlist || disabled} style={inputStyle} onChange={(event) => setValues((current) => ({ ...current, waitlist: event.target.value === "ENABLED" }))}><option value="ENABLED">Enabled</option><option value="DISABLED">Disabled</option></select></BulkField>
          {allTeamFormats ? <BulkField label="Partner Board" checked={selected.partnerBoard} disabled={disabled} summary={mixedSummary(divisions.map((row) => recordBoolean(row.value.partner_board_enabled, true)))} onChange={(checked) => toggleField("partnerBoard", checked)}><select value={values.partnerBoard ? "ENABLED" : "DISABLED"} disabled={!selected.partnerBoard || disabled} style={inputStyle} onChange={(event) => setValues((current) => ({ ...current, partnerBoard: event.target.value === "ENABLED" }))}><option value="ENABLED">Enabled</option><option value="DISABLED">Disabled</option></select></BulkField> : null}
          <BulkField label="Draw-format override" checked={selected.draw} disabled={disabled} summary={mixedSummary(divisions.map((row) => cleanString(row.value.event_format_override ?? row.value.division_format)))} onChange={(checked) => toggleField("draw", checked)}><select value={values.draw} disabled={!selected.draw || disabled} style={inputStyle} onChange={(event) => setValues((current) => ({ ...current, draw: event.target.value }))}><option value="">Use each Event default</option>{COMPETITION_FORMATS.map((format) => <option key={format} value={format}>{optionLabel(format)}</option>)}</select></BulkField>
          <BulkField label="Scoring override" checked={selected.scoring} disabled={disabled} summary={mixedSummary(divisions.map((row) => cleanString(row.value.scoring_override ?? row.value.division_scoring)))} onChange={(checked) => toggleField("scoring", checked)}><select value={values.scoring} disabled={!selected.scoring || disabled} style={inputStyle} onChange={(event) => setValues((current) => ({ ...current, scoring: event.target.value }))}><option value="">Use each Event default</option>{SCORING_OPTIONS.map((format) => <option key={format} value={format}>{optionLabel(format)}</option>)}</select></BulkField>
          <BulkField label="Eligibility" checked={selected.eligibility} disabled={disabled} summary={mixedSummary(divisions.map((row) => skillEligibilitySummary(row.value)))} onChange={(checked) => toggleField("eligibility", checked)}><div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.5rem" }}><select value={values.eligibility} disabled={!selected.eligibility || disabled} style={inputStyle} onChange={(event) => setValues((current) => ({ ...current, eligibility: event.target.value as SkillEligibilityMode }))}><option value="STANDARD">Standard skill ceiling</option><option value="MINIMUM">Minimum skill / Skill+</option><option value="OPEN">Open — no rating restriction</option><option value="CUSTOM">Custom rating boundaries</option><option value="COMBINED_RATING_CAP" disabled={!allCombinedEligible}>Combined team rating cap</option></select>{["MINIMUM", "CUSTOM"].includes(values.eligibility) ? <input aria-label="Minimum rating" type="number" min="1" max="7" step="0.01" value={values.minimum} disabled={!selected.eligibility || disabled} style={inputStyle} onChange={(event) => setValues((current) => ({ ...current, minimum: event.target.value }))} /> : null}{values.eligibility === "CUSTOM" ? <input aria-label="Maximum rating exclusive" type="number" min="1.01" max="7.5" step="0.01" value={values.maximum} disabled={!selected.eligibility || disabled} style={inputStyle} onChange={(event) => setValues((current) => ({ ...current, maximum: event.target.value }))} /> : null}{values.eligibility === "COMBINED_RATING_CAP" ? <input aria-label="Combined rating cap" type="number" min="0.1" max="14" step="0.1" value={values.combinedCap} disabled={!selected.eligibility || disabled} style={inputStyle} onChange={(event) => setValues((current) => ({ ...current, combinedCap: event.target.value }))} /> : null}</div></BulkField>
          {sameFamily ? <BulkField label="Tournament days" checked={selected.schedule} disabled={disabled} summary={mixedSummary(divisions.map((row) => eventDayReferences(row.value).sort().join("|")))} onChange={(checked) => toggleField("schedule", checked)}><div><select value={values.scheduleMode} disabled={!selected.schedule || disabled} style={inputStyle} onChange={(event) => setValues((current) => ({ ...current, scheduleMode: event.target.value === "CUSTOM" ? "CUSTOM" : "INHERIT_EVENT", scheduledDayIds: event.target.value === "INHERIT_EVENT" ? familyDays : current.scheduledDayIds }))}><option value="INHERIT_EVENT">Use every day selected for the parent Event</option><option value="CUSTOM">Choose a permitted subset</option></select>{selected.schedule && values.scheduleMode === "CUSTOM" ? <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.35rem", marginTop: "0.45rem" }}>{familyDays.map((dayId) => <label key={dayId} style={{ display: "flex", gap: "0.4rem", alignItems: "center" }}><input type="checkbox" checked={values.scheduledDayIds.includes(dayId)} disabled={disabled || (values.scheduledDayIds.length === 1 && values.scheduledDayIds.includes(dayId))} onChange={(event) => setValues((current) => ({ ...current, scheduledDayIds: event.target.checked ? [...current.scheduledDayIds, dayId] : current.scheduledDayIds.filter((value) => value !== dayId) }))} />{dayLabels.get(dayId) || dayId}</label>)}</div> : null}</div></BulkField> : <p style={{ margin: 0, padding: "0.65rem", borderRadius: "10px", background: "#fff7ed", color: "#9a3412" }}>Tournament days are hidden because the selected divisions do not share one parent Event.</p>}
        </div>
        <article style={{ marginTop: "0.9rem", padding: "0.8rem", border: "1px solid #e2e8f0", borderRadius: "12px", background: "#f8fafc" }}><h3 style={{ marginTop: 0 }}>Per-division preview</h3><div style={{ display: "grid", gap: "0.45rem" }}>{previewRows.map((row) => <div key={row.key} style={{ padding: "0.55rem", border: "1px solid #e2e8f0", borderRadius: "8px", background: "white" }}><strong>{row.name || "Untitled division"}</strong><br /><small style={{ color: row.changes.length ? "#166534" : "#64748b" }}>{row.changes.length ? row.changes.join(" · ") : "No change"}</small></div>)}</div></article>
    </FormDialog>
  );
}

type BulkFieldProps = { label: string; checked: boolean; disabled: boolean; summary: string; onChange: (checked: boolean) => void; children: ReactNode; };
function BulkField({ label, checked, disabled, summary, onChange, children }: BulkFieldProps) {
  return <article style={{ display: "grid", gridTemplateColumns: "minmax(190px, 0.8fr) minmax(240px, 1.4fr)", gap: "0.7rem", alignItems: "center", padding: "0.7rem", border: `1px solid ${checked ? "#93c5fd" : "#e2e8f0"}`, borderRadius: "12px", background: checked ? "#eff6ff" : "#f8fafc" }}><label style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start", fontWeight: 800 }}><input type="checkbox" checked={checked} disabled={disabled} onChange={(event) => onChange(event.target.checked)} /><span>{label}<br /><small style={{ color: "#64748b", fontWeight: 400 }}>{summary}</small></span></label>{children}</article>;
}
