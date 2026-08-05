"use client";

import { useEffect, useMemo, useState, type CSSProperties } from "react";
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

type Props = {
  open: boolean;
  mode?: "add" | "edit";
  initialValue: SetupRecord;
  eventFamilies: BuilderRow[];
  days: BuilderRow[];
  onCancel: () => void;
  onConfirm: (value: SetupRecord) => void;
};

type DivisionEligibilityMode = "STANDARD" | "COMBINED_RATING_CAP";

const inputStyle: CSSProperties = {
  width: "100%",
  minWidth: 0,
  boxSizing: "border-box",
  padding: "0.55rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

function eligibilityMode(value: SetupRecord): DivisionEligibilityMode {
  return cleanString(value.eligibility_mode).toUpperCase() === "COMBINED_RATING_CAP"
    ? "COMBINED_RATING_CAP"
    : "STANDARD";
}

function applyEligibilityMode(
  value: SetupRecord,
  nextMode: DivisionEligibilityMode
): SetupRecord {
  if (nextMode === "COMBINED_RATING_CAP") {
    return {
      ...value,
      eligibility_mode: "COMBINED_RATING_CAP",
      combined_rating_cap:
        Number(value.combined_rating_cap) > 0 ? Number(value.combined_rating_cap) : 8
    };
  }
  return {
    ...value,
    eligibility_mode: "STANDARD",
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
  onConfirm
}: Props) {
  const [draft, setDraft] = useState<SetupRecord>(initialValue);
  const [message, setMessage] = useState("");

  useEffect(() => {
    if (!open) return;
    const familyName = eventFamilyName(initialValue);
    const family =
      eventFamilies.find(
        (row) => eventFamilyName(row.value).toLowerCase() === familyName.toLowerCase()
      )?.value || eventFamilies[0]?.value;
    setDraft(
      dialogMode === "add" && family
        ? applyFamily({ ...initialValue }, family)
        : { ...initialValue }
    );
    setMessage("");
  }, [open, initialValue, eventFamilies, dialogMode]);

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

  function submit() {
    if (!eventFamilyName(draft)) {
      setMessage("Choose an event before adding the division.");
      return;
    }
    if (!cleanString(draft.division_name ?? draft.label)) {
      setMessage("Division name is required.");
      return;
    }
    if (!cleanString(draft.skill_label)) {
      setMessage("Skill division is required.");
      return;
    }
    if (!selectedDayIds.length) {
      setMessage("Choose at least one tournament day for this division.");
      return;
    }
    const capacity = Number(draft.capacity_teams);
    if (!Number.isInteger(capacity) || capacity < 1) {
      setMessage("Capacity must be a whole number of at least 1.");
      return;
    }
    const price = Number(draft.price_usd);
    if (!Number.isFinite(price) || price < 0) {
      setMessage("Entry fee cannot be negative.");
      return;
    }
    if (eligibilityMode(draft) === "COMBINED_RATING_CAP") {
      const cap = Number(draft.combined_rating_cap);
      if (!Number.isFinite(cap) || cap <= 0 || cap > 14) {
        setMessage("Maximum combined rating must be greater than 0 and no more than 14.");
        return;
      }
    }
    if (agePolicySource === "OVERRIDE") {
      const ageIssues = validateAgePolicy(divisionAgePolicy);
      if (ageIssues.length) {
        setMessage(ageIssues[0]);
        return;
      }
    }
    onConfirm(draft);
  }

  return (
    <div
      role="presentation"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        display: "grid",
        placeItems: "center",
        padding: "1rem",
        background: "rgba(15, 23, 42, 0.58)"
      }}
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onCancel();
      }}
    >
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="division-dialog-title"
        style={{
          width: "min(900px, 100%)",
          maxHeight: "calc(100vh - 2rem)",
          overflowY: "auto",
          padding: "1.1rem",
          borderRadius: "16px",
          background: "white",
          boxShadow: "0 24px 70px rgba(15, 23, 42, 0.35)"
        }}
      >
        <h2 id="division-dialog-title" style={{ marginTop: 0 }}>
          {dialogMode === "add"
            ? "Add division"
            : `Edit ${cleanString(draft.division_name ?? draft.label) || "division"}`}
        </h2>
        <p style={{ color: "#475569" }}>
          Divisions implement the event policy for a specific skill or age group. Saving returns a compact, read-only card; published data remains unchanged until Review.
        </p>

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
              value={eligibilityMode(draft)}
              style={inputStyle}
              onChange={(event) =>
                setDraft((current) => applyEligibilityMode(current, event.target.value as DivisionEligibilityMode))
              }
            >
              <option value="STANDARD">Standard event eligibility</option>
              <option value="COMBINED_RATING_CAP">Combined-rating doubles cap</option>
            </select>
            <small>Four-player team is configured at the parent Event level.</small>
          </label>
          {eligibilityMode(draft) === "COMBINED_RATING_CAP" ? (
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

        <fieldset style={{ marginTop: "1rem", padding: "0.85rem", border: "1px solid #cbd5e1", borderRadius: "12px" }}>
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

        <fieldset style={{ marginTop: "1rem", padding: "0.85rem", border: "1px solid #cbd5e1", borderRadius: "12px" }}>
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

        {message ? <p role="alert" style={{ color: "#b91c1c" }}>{message}</p> : null}

        <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.65rem", flexWrap: "wrap", marginTop: "1rem" }}>
          <button type="button" onClick={onCancel} style={{ padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #64748b", background: "white", color: "#0f172a", fontWeight: 800, cursor: "pointer" }}>
            Cancel
          </button>
          <button type="button" onClick={submit} style={{ padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800, cursor: "pointer" }}>
            {dialogMode === "add" ? "Add division" : "Save division"}
          </button>
        </div>
      </section>
    </div>
  );
}
