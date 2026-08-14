"use client";

import { useMemo, useRef, useState, type CSSProperties } from "react";
import {
  FormDialog,
  InteractionActionError,
  useOpenDialogInitializer,
  type ActionCompletion
} from "@/components/interaction";
import {
  SKILL_LABEL_OPTIONS,
  cleanString,
  dayLabel,
  dayReference,
  eventDayReferences,
  eventFamilyName,
  numberInputValue,
  setEventDayReferences,
  setRecordNumber,
  setRecordString,
  type BuilderRow,
  type SetupRecord
} from "../../tournament-setup/tournamentSetupBuilder";
import TournamentAgePolicyEditor, {
  DIVISION_AGE_POLICY_FIELDS,
  EVENT_AGE_POLICY_FIELDS,
  agePolicySummary,
  readAgePolicy,
  validateAgePolicy,
  writeAgePolicy
} from "./TournamentAgePolicyEditor";
import {
  skillAnchor,
  skillEligibilityLabel,
  skillEligibilityMode,
  skillEligibilitySummary,
  validateTournamentSkillEligibility,
  type SkillEligibilityMode
} from "@/lib/tournamentSkillEligibility";

type Props = {
  open: boolean;
  mode?: "add" | "edit";
  initialValue: SetupRecord;
  eventFamilies: BuilderRow[];
  days: BuilderRow[];
  onCancel: () => void;
  onConfirm: (value: SetupRecord) => Promise<ActionCompletion>;
  onAcknowledge?: () => void;
};

const inputStyle: CSSProperties = {
  width: "100%",
  minWidth: 0,
  boxSizing: "border-box",
  padding: "0.55rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

function applyEligibilityMode(
  value: SetupRecord,
  nextMode: SkillEligibilityMode
): SetupRecord {
  const anchor = skillAnchor(value.skill_label) ?? 3.5;
  if (nextMode === "COMBINED_RATING_CAP") {
    const cap = Number(value.combined_rating_cap);
    return {
      ...value,
      eligibility_mode: "COMBINED_RATING_CAP",
      skill_mode: "COMBINED_RATING_CAP",
      skill_label: `Below ${Number.isFinite(cap) && cap > 0 ? cap : 8} combined`,
      skill_min_rating: null,
      skill_max_rating: null,
      combined_rating_cap: Number.isFinite(cap) && cap > 0 ? cap : 8
    };
  }
  if (nextMode === "MINIMUM") {
    const minimum = Number(value.skill_min_rating);
    const threshold = Number.isFinite(minimum) && minimum > 0 ? minimum : anchor;
    return {
      ...value,
      eligibility_mode: "MINIMUM",
      skill_mode: "MINIMUM",
      skill_label: `${threshold.toFixed(2).replace(/0+$/, "").replace(/\.$/, "")}+`,
      skill_min_rating: threshold,
      skill_max_rating: null,
      combined_rating_cap: null
    };
  }
  if (nextMode === "OPEN") {
    return {
      ...value,
      eligibility_mode: "OPEN",
      skill_mode: "OPEN",
      skill_label: "Open",
      skill_min_rating: null,
      skill_max_rating: null,
      combined_rating_cap: null
    };
  }
  if (nextMode === "CUSTOM") {
    return {
      ...value,
      eligibility_mode: "CUSTOM",
      skill_mode: "CUSTOM",
      skill_label: cleanString(value.skill_label) || "Custom rating",
      skill_min_rating: value.skill_min_rating ?? null,
      skill_max_rating: value.skill_max_rating ?? null,
      combined_rating_cap: null
    };
  }
  return {
    ...value,
    eligibility_mode: "STANDARD",
    skill_mode: "STANDARD",
    skill_label: anchor.toFixed(2).replace(/0+$/, "").replace(/\.$/, ""),
    skill_min_rating: null,
    skill_max_rating: null,
    combined_rating_cap: null
  };
}

function applyFamily(value: SetupRecord, family: SetupRecord): SetupRecord {
  const familyName = eventFamilyName(family);
  const schedule = eventDayReferences(family);
  const eventAgePolicy = readAgePolicy(family, EVENT_AGE_POLICY_FIELDS);
  const inheritAge = cleanString(value.age_policy_source).toUpperCase() !== "OVERRIDE";
  let next = setEventDayReferences(
    {
      ...value,
      event_family_label: familyName,
      event_family: familyName,
      participant_type: cleanString(family.participant_type) || value.participant_type,
      event_type: cleanString(family.participant_type) || value.event_type,
      gender_restriction:
        cleanString(family.gender_restriction) || value.gender_restriction,
      competition_format: cleanString(family.competition_format) || "STANDARD",
      team_roster_size: family.team_roster_size ?? value.team_roster_size ?? 2,
      team_gender_rule: cleanString(family.team_gender_rule) || value.team_gender_rule || "NONE",
      team_tiebreak_mode: cleanString(family.team_tiebreak_mode) || value.team_tiebreak_mode || "SINGLES",
      team_playoff_format: cleanString(family.team_playoff_format) || value.team_playoff_format || "NONE",
      team_allow_substitutes: family.team_allow_substitutes ?? value.team_allow_substitutes ?? false,
      capacity_teams:
        family.default_capacity_teams ?? value.capacity_teams ?? 16,
      price_usd: family.default_price_usd ?? value.price_usd ?? 0,
      waitlist_enabled:
        family.default_waitlist ?? value.waitlist_enabled ?? true,
      partner_board_enabled:
        family.default_partner_board ?? value.partner_board_enabled ?? true,
      schedule_mode: "INHERIT_EVENT",
      age_policy_source: inheritAge ? "INHERIT_EVENT" : "OVERRIDE"
    },
    schedule
  );
  if (inheritAge) {
    next = writeAgePolicy(next, DIVISION_AGE_POLICY_FIELDS, eventAgePolicy);
  }
  return next;
}

export default function TournamentSetupDivisionDialog({
  open,
  mode: dialogMode = "add",
  initialValue,
  eventFamilies,
  days,
  onCancel,
  onConfirm,
  onAcknowledge
}: Props) {
  const [draft, setDraft] = useState<SetupRecord>(initialValue);
  const [baseline, setBaseline] = useState("");
  const familyRef = useRef<HTMLSelectElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const skillRef = useRef<HTMLInputElement>(null);
  const eligibilityRef = useRef<HTMLSelectElement>(null);
  const capacityRef = useRef<HTMLInputElement>(null);
  const priceRef = useRef<HTMLInputElement>(null);
  const agePolicyRef = useRef<HTMLFieldSetElement>(null);
  const scheduleRef = useRef<HTMLFieldSetElement>(null);
  const invalidFieldRef = useRef<HTMLElement | null>(null);

  useOpenDialogInitializer(open, () => {
    if (!open) return;
    const familyName = eventFamilyName(initialValue);
    const family =
      eventFamilies.find(
        (row) => eventFamilyName(row.value).toLowerCase() === familyName.toLowerCase()
      )?.value || eventFamilies[0]?.value;
    const prepared =
      dialogMode === "add" && family
        ? applyFamily({ ...initialValue }, family)
        : { ...initialValue };
    setDraft(prepared);
    setBaseline(JSON.stringify(prepared));
    invalidFieldRef.current = null;
  });

  const family = useMemo(() => {
    const name = eventFamilyName(draft);
    return eventFamilies.find(
      (row) => eventFamilyName(row.value).toLowerCase() === name.toLowerCase()
    )?.value;
  }, [draft, eventFamilies]);
  const availableDayIds = eventDayReferences(family || {});
  const selectedDayIds = eventDayReferences(draft);
  const scheduleMode = cleanString(draft.schedule_mode) || "INHERIT_EVENT";
  const agePolicySource = cleanString(draft.age_policy_source).toUpperCase() === "OVERRIDE"
    ? "OVERRIDE"
    : "INHERIT_EVENT";
  const inheritedAgePolicy = readAgePolicy(family || {}, EVENT_AGE_POLICY_FIELDS);
  const divisionAgePolicy = readAgePolicy(draft, DIVISION_AGE_POLICY_FIELDS);
  const dayById = new Map(
    days.map((day) => [dayReference(day.value), dayLabel(day.value) || dayReference(day.value)])
  );

  if (!open) return null;

  function validationError(message: string, field: string, target: HTMLElement | null): never {
    invalidFieldRef.current = target;
    throw new InteractionActionError(message, { kind: "validation", fieldErrors: { [field]: message } });
  }

  async function submit(): Promise<ActionCompletion> {
    if (!eventFamilyName(draft)) {
      validationError("Choose an event before adding the division.", "Parent event", familyRef.current);
    }
    if (!cleanString(draft.division_name ?? draft.label)) {
      validationError("Division name is required.", "Division name", nameRef.current);
    }
    if (!cleanString(draft.skill_label) && skillEligibilityMode(draft) !== "OPEN") {
      validationError("Skill division is required.", "Skill division", skillRef.current);
    }
    if (!selectedDayIds.length) {
      validationError("Choose at least one tournament day for this division.", "Tournament days", scheduleRef.current);
    }
    const capacity = Number(draft.capacity_teams);
    if (!Number.isInteger(capacity) || capacity < 1) {
      validationError("Capacity must be a whole number of at least 1.", "Capacity", capacityRef.current);
    }
    const price = Number(draft.price_usd);
    if (!Number.isFinite(price) || price < 0) {
      validationError("Entry fee cannot be negative.", "Entry fee", priceRef.current);
    }
    const skillIssue = validateTournamentSkillEligibility(draft);
    if (skillIssue) {
      validationError(skillIssue, "Division eligibility", eligibilityRef.current);
    }
    const participantType = cleanString(family?.participant_type).toUpperCase();
    const competitionFormat = cleanString(family?.competition_format).toUpperCase();
    if (
      skillEligibilityMode(draft) === "COMBINED_RATING_CAP" &&
      (participantType === "SINGLES" || competitionFormat === "FOUR_PLAYER_TEAM")
    ) {
      validationError("Combined team-rating cap is available only for standard doubles/team divisions.", "Division eligibility", eligibilityRef.current);
    }
    if (agePolicySource === "OVERRIDE") {
      const ageIssues = validateAgePolicy(divisionAgePolicy, cleanString(family?.participant_type));
      if (ageIssues.length) {
        validationError(ageIssues[0], "Age policy", agePolicyRef.current);
      }
    }
    return onConfirm(draft);
  }

  return (
    <FormDialog
      open={open}
      mode={dialogMode === "add" ? "create" : "edit"}
      size="wide"
      title={dialogMode === "add" ? "Add division" : `Edit ${cleanString(draft.division_name ?? draft.label) || "division"}`}
      description="Divisions implement the event policy for a specific skill or age group. Saving returns a compact, read-only card; published data remains unchanged until Review."
      dirty={Boolean(baseline) && JSON.stringify(draft) !== baseline}
      submitLabel={dialogMode === "add" ? "Add division" : "Save division"}
      workingLabel="Saving division…"
      initialFocusRef={nameRef}
      getFirstInvalidField={() => invalidFieldRef.current}
      onSubmit={submit}
      onCancel={onCancel}
      onAcknowledge={onAcknowledge}
    >
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
            gap: "0.75rem"
          }}
        >
          <label>
            <strong>Parent event</strong><br />
            <select
              ref={familyRef}
              value={eventFamilyName(draft)}
              style={inputStyle}
              onChange={(event) => {
                const selected = eventFamilies.find(
                  (row) => eventFamilyName(row.value) === event.target.value
                )?.value;
                if (selected) setDraft((current) => applyFamily(current, selected));
              }}
            >
              <option value="">Choose an event</option>
              {eventFamilies.map((row) => {
                const name = eventFamilyName(row.value);
                return name ? <option key={row.key} value={name}>{name}</option> : null;
              })}
            </select>
          </label>
          <label>
            <strong>Division name</strong><br />
            <input
              ref={nameRef}
              value={cleanString(draft.division_name ?? draft.label)}
              placeholder="3.5 · 50+"
              style={inputStyle}
              onChange={(event) =>
                setDraft((current) =>
                  setRecordString(current, ["division_name", "label"], event.target.value)
                )
              }
            />
          </label>
          <label>
            <strong>Skill division</strong><br />
            <input
              ref={skillRef}
              list="new-division-skills"
              value={draft.skill_label == null ? "" : String(draft.skill_label)}
              style={inputStyle}
              onChange={(event) =>
                setDraft((current) => setRecordString(current, ["skill_label"], event.target.value))
              }
            />
            <datalist id="new-division-skills">
              {SKILL_LABEL_OPTIONS.map((option) => <option key={option} value={option} />)}
            </datalist>
          </label>
          <label>
            <strong>Division eligibility</strong><br />
            <select
              ref={eligibilityRef}
              value={skillEligibilityMode(draft)}
              style={inputStyle}
              onChange={(event) =>
                setDraft((current) => applyEligibilityMode(current, event.target.value as SkillEligibilityMode))
              }
            >
              <option value="STANDARD">Standard skill ceiling</option>
              <option value="MINIMUM">Minimum skill / Skill+</option>
              <option value="OPEN">Open — no rating restriction</option>
              <option
                value="COMBINED_RATING_CAP"
                disabled={
                  cleanString(family?.participant_type).toUpperCase() === "SINGLES" ||
                  cleanString(family?.competition_format).toUpperCase() === "FOUR_PLAYER_TEAM"
                }
              >
                Combined team-rating cap
              </option>
              <option value="CUSTOM">Custom rating boundaries</option>
            </select>
            <small>{skillEligibilitySummary(draft)}</small>
          </label>
          {skillEligibilityMode(draft) === "MINIMUM" ? (
            <label>
              <strong>Minimum rating</strong><br />
              <input
                type="number"
                inputMode="decimal"
                min="1"
                max="7"
                step="0.01"
                value={numberInputValue(draft.skill_min_rating ?? skillAnchor(draft.skill_label))}
                style={inputStyle}
                onChange={(event) => {
                  const value = event.target.value;
                  setDraft((current) => {
                    const next = setRecordNumber(current, "skill_min_rating", value);
                    const parsed = Number(value);
                    return {
                      ...next,
                      skill_label: Number.isFinite(parsed) && value !== ""
                        ? `${parsed.toFixed(2).replace(/0+$/, "").replace(/\.$/, "")}+`
                        : current.skill_label
                    };
                  });
                }}
              />
            </label>
          ) : null}
          {skillEligibilityMode(draft) === "CUSTOM" ? (
            <>
              <label>
                <strong>Minimum rating (optional)</strong><br />
                <input type="number" inputMode="decimal" min="1" max="7" step="0.01" value={numberInputValue(draft.skill_min_rating)} style={inputStyle} onChange={(event) => setDraft((current) => setRecordNumber(current, "skill_min_rating", event.target.value))} />
              </label>
              <label>
                <strong>Maximum rating — exclusive (optional)</strong><br />
                <input type="number" inputMode="decimal" min="1" max="7.5" step="0.01" value={numberInputValue(draft.skill_max_rating)} style={inputStyle} onChange={(event) => setDraft((current) => setRecordNumber(current, "skill_max_rating", event.target.value))} />
              </label>
            </>
          ) : null}
          {skillEligibilityMode(draft) === "COMBINED_RATING_CAP" ? (
            <label>
              <strong>Maximum combined rating</strong><br />
              <input
                type="number"
                inputMode="decimal"
                min="0.01"
                max="14"
                step="0.01"
                value={numberInputValue(draft.combined_rating_cap)}
                style={inputStyle}
                onChange={(event) => setDraft((current) => setRecordNumber(current, "combined_rating_cap", event.target.value))}
              />
            </label>
          ) : null}
          <label>
            <strong>Capacity</strong><br />
            <input
              ref={capacityRef}
              type="number"
              min="1"
              step="1"
              value={numberInputValue(draft.capacity_teams)}
              style={inputStyle}
              onChange={(event) => setDraft((current) => setRecordNumber(current, "capacity_teams", event.target.value))}
            />
          </label>
          <label>
            <strong>Entry fee (USD)</strong><br />
            <input
              ref={priceRef}
              type="number"
              inputMode="decimal"
              min="0"
              step="0.01"
              value={numberInputValue(draft.price_usd)}
              style={inputStyle}
              onChange={(event) => setDraft((current) => setRecordNumber(current, "price_usd", event.target.value))}
            />
            <small>Commerce is the consolidated place to review all fees.</small>
          </label>
        </div>
        <div style={{ marginTop: "0.85rem", padding: "0.75rem", border: "1px solid #bfdbfe", borderRadius: "10px", background: "#eff6ff" }}>
          <strong>{skillEligibilityLabel(draft)}</strong><br />
          <span>{skillEligibilitySummary(draft)}</span>
          {!["OPEN", "COMBINED_RATING_CAP"].includes(skillEligibilityMode(draft)) && cleanString(family?.participant_type).toUpperCase() !== "SINGLES" ? (
            <small style={{ display: "block", marginTop: "0.25rem" }}>For doubles, the higher-rated partner is the controlling rating.</small>
          ) : null}
        </div>

        <fieldset ref={agePolicyRef} tabIndex={-1} style={{ marginTop: "1rem", padding: "0.85rem", border: "1px solid #cbd5e1", borderRadius: "12px" }}>
          <legend style={{ fontWeight: 800 }}>Age policy</legend>
          <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            <input
              type="radio"
              name="division-age-policy-source"
              checked={agePolicySource === "INHERIT_EVENT"}
              onChange={() =>
                setDraft((current) => writeAgePolicy({ ...current, age_policy_source: "INHERIT_EVENT" }, DIVISION_AGE_POLICY_FIELDS, inheritedAgePolicy))
              }
            />
            Inherit from parent event — {agePolicySummary(inheritedAgePolicy)}
          </label>
          <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.45rem" }}>
            <input
              type="radio"
              name="division-age-policy-source"
              checked={agePolicySource === "OVERRIDE"}
              onChange={() => setDraft((current) => ({ ...current, age_policy_source: "OVERRIDE" }))}
            />
            Override for this division
          </label>
          {agePolicySource === "OVERRIDE" ? (
            <div style={{ marginTop: "0.75rem" }}>
              <TournamentAgePolicyEditor
                policy={divisionAgePolicy}
                participantType={cleanString(family?.participant_type)}
                onChange={(policy) => setDraft((current) => writeAgePolicy(current, DIVISION_AGE_POLICY_FIELDS, policy))}
                title="Division age-policy override"
              />
            </div>
          ) : null}
        </fieldset>

        <fieldset ref={scheduleRef} tabIndex={-1} style={{ marginTop: "1rem", padding: "0.85rem", border: "1px solid #cbd5e1", borderRadius: "12px" }}>
          <legend style={{ fontWeight: 800 }}>Tournament days</legend>
          <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            <input
              type="radio"
              name="new-division-schedule-mode"
              checked={scheduleMode === "INHERIT_EVENT"}
              onChange={() =>
                setDraft((current) =>
                  setEventDayReferences({ ...current, schedule_mode: "INHERIT_EVENT" }, availableDayIds)
                )
              }
            />
            Use every day selected for the parent event
          </label>
          <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", marginTop: "0.45rem" }}>
            <input
              type="radio"
              name="new-division-schedule-mode"
              checked={scheduleMode === "CUSTOM"}
              onChange={() => setDraft((current) => ({ ...current, schedule_mode: "CUSTOM" }))}
            />
            Choose specific event days
          </label>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.45rem", marginTop: "0.65rem" }}>
            {availableDayIds.map((dayId) => (
              <label key={dayId} style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                <input
                  type="checkbox"
                  checked={selectedDayIds.includes(dayId)}
                  disabled={scheduleMode !== "CUSTOM"}
                  onChange={(event) => {
                    const next = event.target.checked
                      ? [...selectedDayIds, dayId]
                      : selectedDayIds.filter((value) => value !== dayId);
                    setDraft((current) => setEventDayReferences(current, next));
                  }}
                />
                {dayById.get(dayId) || dayId}
              </label>
            ))}
          </div>
          {!availableDayIds.length ? (
            <p style={{ color: "#b91c1c" }}>
              The selected event has no tournament days. Return to Events and choose at least one day.
            </p>
          ) : null}
        </fieldset>

    </FormDialog>
  );
}
