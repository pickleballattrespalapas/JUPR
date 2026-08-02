"use client";

import { useEffect, useMemo, useState, type CSSProperties } from "react";
import {
  AGE_MODES,
  SKILL_LABEL_OPTIONS,
  cleanString,
  dayLabel,
  dayReference,
  eventDayReferences,
  eventFamilyDefaults,
  eventFamilyName,
  numberInputValue,
  setEventAgeMode,
  setEventDayReferences,
  setRecordNumber,
  setRecordString,
  type BuilderRow,
  type SetupRecord
} from "../../tournament-setup/tournamentSetupBuilder";

type Props = {
  open: boolean;
  initialValue: SetupRecord;
  eventFamilies: BuilderRow[];
  days: BuilderRow[];
  onCancel: () => void;
  onConfirm: (value: SetupRecord) => void;
};

type EventMode = "STANDARD" | "COMBINED_RATING_CAP" | "FOUR_PLAYER_TEAM";

const inputStyle: CSSProperties = {
  width: "100%",
  minWidth: 0,
  boxSizing: "border-box",
  padding: "0.55rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

function optionLabel(value: string): string {
  return value
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function mode(value: SetupRecord): EventMode {
  if (cleanString(value.competition_format).toUpperCase() === "FOUR_PLAYER_TEAM") {
    return "FOUR_PLAYER_TEAM";
  }
  if (cleanString(value.eligibility_mode).toUpperCase() === "COMBINED_RATING_CAP") {
    return "COMBINED_RATING_CAP";
  }
  return "STANDARD";
}

function applyMode(value: SetupRecord, nextMode: EventMode): SetupRecord {
  if (nextMode === "COMBINED_RATING_CAP") {
    return {
      ...value,
      eligibility_mode: "COMBINED_RATING_CAP",
      combined_rating_cap:
        Number(value.combined_rating_cap) > 0 ? Number(value.combined_rating_cap) : 8,
      competition_format: "STANDARD",
      team_roster_size: 2,
      team_gender_rule: "NONE",
      team_allow_substitutes: false
    };
  }
  if (nextMode === "FOUR_PLAYER_TEAM") {
    return {
      ...value,
      eligibility_mode: "STANDARD",
      combined_rating_cap: null,
      competition_format: "FOUR_PLAYER_TEAM",
      team_roster_size: 4,
      team_gender_rule: "TWO_MEN_TWO_WOMEN",
      team_tiebreak_mode: "SINGLES",
      team_playoff_format: "NONE",
      team_allow_substitutes: false
    };
  }
  return {
    ...value,
    eligibility_mode: "STANDARD",
    combined_rating_cap: null,
    competition_format: "STANDARD",
    team_roster_size: 2,
    team_gender_rule: "NONE",
    team_allow_substitutes: false
  };
}

function applyFamily(value: SetupRecord, family: SetupRecord): SetupRecord {
  const familyName = eventFamilyName(family);
  const schedule = eventDayReferences(family);
  return setEventDayReferences(
    {
      ...value,
      event_family_label: familyName,
      event_family: familyName,
      participant_type: cleanString(family.participant_type) || value.participant_type,
      event_type: cleanString(family.participant_type) || value.event_type,
      gender_restriction:
        cleanString(family.gender_restriction) || value.gender_restriction,
      capacity_teams:
        family.default_capacity_teams ?? value.capacity_teams ?? 16,
      price_usd: family.default_price_usd ?? value.price_usd ?? 0,
      waitlist_enabled:
        family.default_waitlist ?? value.waitlist_enabled ?? true,
      partner_board_enabled:
        family.default_partner_board ?? value.partner_board_enabled ?? true,
      schedule_mode: "INHERIT_EVENT"
    },
    schedule
  );
}

export default function TournamentSetupDivisionDialog({
  open,
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
    setDraft(family ? applyFamily(initialValue, family) : initialValue);
    setMessage("");
  }, [open, initialValue, eventFamilies]);

  const family = useMemo(() => {
    const name = eventFamilyName(draft);
    return eventFamilies.find(
      (row) => eventFamilyName(row.value).toLowerCase() === name.toLowerCase()
    )?.value;
  }, [draft, eventFamilies]);
  const availableDayIds = eventDayReferences(family || {});
  const selectedDayIds = eventDayReferences(draft);
  const scheduleMode = cleanString(draft.schedule_mode) || "INHERIT_EVENT";
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
    if (!selectedDayIds.length) {
      setMessage("Choose at least one tournament day for this division.");
      return;
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
        aria-labelledby="add-division-title"
        style={{
          width: "min(760px, 100%)",
          maxHeight: "calc(100vh - 2rem)",
          overflowY: "auto",
          padding: "1.1rem",
          borderRadius: "16px",
          background: "white",
          boxShadow: "0 24px 70px rgba(15, 23, 42, 0.35)"
        }}
      >
        <h2 id="add-division-title" style={{ marginTop: 0 }}>
          Add division
        </h2>
        <p style={{ color: "#475569" }}>
          Set up the new division here. Confirming adds it to the division list,
          where every field remains editable before the setup is published.
        </p>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
            gap: "0.75rem"
          }}
        >
          <label>
            <strong>Parent event</strong>
            <br />
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
                return name ? (
                  <option key={row.key} value={name}>
                    {name}
                  </option>
                ) : null;
              })}
            </select>
          </label>
          <label>
            <strong>Division name</strong>
            <br />
            <input
              value={cleanString(draft.division_name ?? draft.label)}
              placeholder="3.5 · 50+"
              style={inputStyle}
              onChange={(event) =>
                setDraft((current) =>
                  setRecordString(
                    current,
                    ["division_name", "label"],
                    event.target.value
                  )
                )
              }
            />
          </label>
          <label>
            <strong>Skill division</strong>
            <br />
            <input
              list="new-division-skills"
              value={cleanString(draft.skill_label) || "Open"}
              style={inputStyle}
              onChange={(event) =>
                setDraft((current) =>
                  setRecordString(current, ["skill_label"], event.target.value)
                )
              }
            />
            <datalist id="new-division-skills">
              {SKILL_LABEL_OPTIONS.map((option) => (
                <option key={option} value={option} />
              ))}
            </datalist>
          </label>
          <label>
            <strong>Age format</strong>
            <br />
            <select
              value={cleanString(draft.age_mode) || "ALL_AGES"}
              style={inputStyle}
              onChange={(event) =>
                setDraft((current) => setEventAgeMode(current, event.target.value))
              }
            >
              {AGE_MODES.map((option) => (
                <option key={option} value={option}>
                  {optionLabel(option)}
                </option>
              ))}
            </select>
          </label>
          {cleanString(draft.age_mode) === "FIXED_AGE_BRACKET" ? (
            <label>
              <strong>Age bracket</strong>
              <br />
              <input
                value={cleanString(draft.age_label)}
                placeholder="50+"
                style={inputStyle}
                onChange={(event) =>
                  setDraft((current) =>
                    setRecordString(current, ["age_label"], event.target.value)
                  )
                }
              />
            </label>
          ) : null}
          <label>
            <strong>Division format</strong>
            <br />
            <select
              value={mode(draft)}
              style={inputStyle}
              onChange={(event) =>
                setDraft((current) =>
                  applyMode(current, event.target.value as EventMode)
                )
              }
            >
              <option value="STANDARD">Standard singles or doubles</option>
              <option value="COMBINED_RATING_CAP">Combined-rating doubles</option>
              <option value="FOUR_PLAYER_TEAM">Four-player team</option>
            </select>
          </label>
          {mode(draft) === "COMBINED_RATING_CAP" ? (
            <label>
              <strong>Maximum combined rating</strong>
              <br />
              <input
                type="number"
                inputMode="decimal"
                min="0.01"
                max="14"
                step="0.01"
                value={numberInputValue(draft.combined_rating_cap)}
                style={inputStyle}
                onChange={(event) =>
                  setDraft((current) =>
                    setRecordNumber(current, "combined_rating_cap", event.target.value)
                  )
                }
              />
            </label>
          ) : null}
          <label>
            <strong>Capacity</strong>
            <br />
            <input
              type="number"
              min="1"
              step="1"
              value={numberInputValue(draft.capacity_teams)}
              style={inputStyle}
              onChange={(event) =>
                setDraft((current) =>
                  setRecordNumber(current, "capacity_teams", event.target.value)
                )
              }
            />
          </label>
          <label>
            <strong>Entry fee (USD)</strong>
            <br />
            <input
              type="number"
              inputMode="decimal"
              min="0"
              step="0.01"
              value={numberInputValue(draft.price_usd)}
              style={inputStyle}
              onChange={(event) =>
                setDraft((current) =>
                  setRecordNumber(current, "price_usd", event.target.value)
                )
              }
            />
          </label>
        </div>

        <fieldset
          style={{
            marginTop: "1rem",
            padding: "0.85rem",
            border: "1px solid #cbd5e1",
            borderRadius: "12px"
          }}
        >
          <legend style={{ fontWeight: 800 }}>Tournament days</legend>
          <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            <input
              type="radio"
              name="new-division-schedule-mode"
              checked={scheduleMode === "INHERIT_EVENT"}
              onChange={() =>
                setDraft((current) =>
                  setEventDayReferences(
                    { ...current, schedule_mode: "INHERIT_EVENT" },
                    availableDayIds
                  )
                )
              }
            />
            Use every day selected for the parent event
          </label>
          <label
            style={{
              display: "flex",
              gap: "0.5rem",
              alignItems: "center",
              marginTop: "0.45rem"
            }}
          >
            <input
              type="radio"
              name="new-division-schedule-mode"
              checked={scheduleMode === "CUSTOM"}
              onChange={() =>
                setDraft((current) => ({ ...current, schedule_mode: "CUSTOM" }))
              }
            />
            Choose specific event days
          </label>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
              gap: "0.45rem",
              marginTop: "0.65rem"
            }}
          >
            {availableDayIds.map((dayId) => (
              <label
                key={dayId}
                style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}
              >
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
              The selected event has no tournament days. Return to Events and
              choose at least one day.
            </p>
          ) : null}
        </fieldset>

        {message ? <p role="alert" style={{ color: "#b91c1c" }}>{message}</p> : null}

        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            gap: "0.65rem",
            flexWrap: "wrap",
            marginTop: "1rem"
          }}
        >
          <button
            type="button"
            onClick={onCancel}
            style={{
              padding: "0.6rem 0.9rem",
              borderRadius: "999px",
              border: "1px solid #64748b",
              background: "white",
              color: "#0f172a",
              fontWeight: 800,
              cursor: "pointer"
            }}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={submit}
            style={{
              padding: "0.6rem 0.9rem",
              borderRadius: "999px",
              border: "1px solid #0f172a",
              background: "#0f172a",
              color: "white",
              fontWeight: 800,
              cursor: "pointer"
            }}
          >
            Add division
          </button>
        </div>
      </section>
    </div>
  );
}
