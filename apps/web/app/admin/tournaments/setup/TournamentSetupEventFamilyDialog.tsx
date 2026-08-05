"use client";

import { useEffect, useState, type CSSProperties } from "react";
import {
  COMPETITION_FORMATS,
  GENDER_RESTRICTIONS,
  SCORING_OPTIONS,
  cleanString,
  dayLabel,
  dayReference,
  eventDayReferences,
  eventFamilyName,
  numberInputValue,
  recordBoolean,
  setEventDayReferences,
  setRecordNumber,
  setRecordString,
  type BuilderRow,
  type SetupRecord
} from "../../tournament-setup/tournamentSetupBuilder";
import TournamentAgePolicyEditor, {
  EVENT_AGE_POLICY_FIELDS,
  readAgePolicy,
  validateAgePolicy,
  writeAgePolicy
} from "./TournamentAgePolicyEditor";

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

function optionsWithCurrent(options: readonly string[], current: string): string[] {
  return current && !options.includes(current) ? [current, ...options] : [...options];
}

type EventStructure = "SINGLES" | "GENDER_DOUBLES" | "MIXED_DOUBLES" | "FOUR_PLAYER_TEAM";

function eventStructure(value: SetupRecord): EventStructure {
  if (cleanString(value.competition_format).toUpperCase() === "FOUR_PLAYER_TEAM") {
    return "FOUR_PLAYER_TEAM";
  }
  const participantType = cleanString(value.participant_type).toUpperCase();
  if (participantType === "SINGLES") return "SINGLES";
  if (participantType === "MIXED_DOUBLES") return "MIXED_DOUBLES";
  return "GENDER_DOUBLES";
}

function applyEventStructure(value: SetupRecord, structure: EventStructure): SetupRecord {
  if (structure === "FOUR_PLAYER_TEAM") {
    return {
      ...value,
      participant_type: "MIXED_DOUBLES",
      gender_restriction: "MIXED",
      competition_format: "FOUR_PLAYER_TEAM",
      team_roster_size: 4,
      team_gender_rule: "TWO_MEN_TWO_WOMEN",
      team_tiebreak_mode: cleanString(value.team_tiebreak_mode) || "SINGLES",
      team_playoff_format: cleanString(value.team_playoff_format) || "NONE",
      team_allow_substitutes: recordBoolean(value.team_allow_substitutes, false),
      default_partner_board: true
    };
  }
  const participantType = structure;
  return {
    ...value,
    participant_type: participantType,
    gender_restriction:
      structure === "MIXED_DOUBLES"
        ? "MIXED"
        : cleanString(value.gender_restriction) === "MIXED"
          ? "ANY"
          : cleanString(value.gender_restriction) || "ANY",
    competition_format: "STANDARD",
    team_roster_size: 2,
    team_gender_rule: "NONE",
    team_tiebreak_mode: "SINGLES",
    team_playoff_format: "NONE",
    team_allow_substitutes: false,
    default_partner_board:
      structure === "SINGLES" ? false : recordBoolean(value.default_partner_board, true)
  };
}

function applyEventStructureWithAge(value: SetupRecord, structure: EventStructure): SetupRecord {
  const next = applyEventStructure(value, structure);
  const policy = readAgePolicy(next, EVENT_AGE_POLICY_FIELDS);
  if (structure === "SINGLES" && policy.mode === "SPLIT_AGE") {
    return writeAgePolicy(next, EVENT_AGE_POLICY_FIELDS, {
      ...policy,
      mode: "ALL_AGES",
      label: "All Ages",
      split_age_threshold: null
    });
  }
  return next;
}

type Props = {
  open: boolean;
  mode: "add" | "edit";
  initialValue: SetupRecord;
  days: BuilderRow[];
  onCancel: () => void;
  onConfirm: (value: SetupRecord) => void | Promise<void>;
};

export default function TournamentSetupEventFamilyDialog({
  open,
  mode,
  initialValue,
  days,
  onCancel,
  onConfirm
}: Props) {
  const [draft, setDraft] = useState<SetupRecord>(initialValue);
  const [message, setMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open) return;
    const structure = eventStructure(initialValue);
    setDraft(applyEventStructureWithAge({ ...initialValue }, structure));
    setMessage("");
    setSubmitting(false);
  }, [open, initialValue]);

  if (!open) return null;

  const name = eventFamilyName(draft);
  const currentDays = eventDayReferences(draft);
  const structure = eventStructure(draft);
  const participantType = structure === "FOUR_PLAYER_TEAM" ? "MIXED_DOUBLES" : structure;
  const gender = structure === "MIXED_DOUBLES" || structure === "FOUR_PLAYER_TEAM"
    ? "MIXED"
    : cleanString(draft.gender_restriction) || "ANY";
  const drawFormat = cleanString(draft.default_format) || "ROUND_ROBIN_PLUS_PLAYOFF";
  const scoring = cleanString(draft.default_scoring) || "GAME_TO_15";
  const agePolicy = readAgePolicy(draft, EVENT_AGE_POLICY_FIELDS);
  const dayOptions = days.map((day) => ({
    value: dayReference(day.value),
    label: dayLabel(day.value) || dayReference(day.value),
    enabled: recordBoolean(day.value.enabled, true)
  }));

  async function submit() {
    if (!name) {
      setMessage("Event name is required.");
      return;
    }
    if (!currentDays.length) {
      setMessage("Choose at least one tournament day for this event.");
      return;
    }
    const capacity = Number(draft.default_capacity_teams);
    if (!Number.isInteger(capacity) || capacity < 1) {
      setMessage("Default capacity must be a whole number of at least 1.");
      return;
    }
    const price = Number(draft.default_price_usd);
    if (!Number.isFinite(price) || price < 0) {
      setMessage("Default entry fee cannot be negative.");
      return;
    }
    const ageIssues = validateAgePolicy(agePolicy, participantType);
    if (ageIssues.length) {
      setMessage(ageIssues[0]);
      return;
    }
    setSubmitting(true);
    try {
      await onConfirm({
        ...draft,
        gender_restriction: gender,
        default_partner_board:
          participantType === "SINGLES" ? false : recordBoolean(draft.default_partner_board, true)
      });
    } finally {
      setSubmitting(false);
    }
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
        aria-labelledby="event-dialog-title"
        style={{
          width: "min(920px, 100%)",
          maxHeight: "calc(100vh - 2rem)",
          overflowY: "auto",
          padding: "1.1rem",
          borderRadius: "16px",
          background: "white",
          boxShadow: "0 24px 70px rgba(15, 23, 42, 0.35)"
        }}
      >
        <h2 id="event-dialog-title" style={{ marginTop: 0 }}>
          {mode === "add" ? "Add event" : `Edit ${name || "event"}`}
        </h2>
        <p style={{ color: "#475569" }}>
          Define the event structure and policy once. Divisions inherit these defaults unless an organizer deliberately overrides them. Saving returns a compact, read-only event card; published data stays unchanged until Review.
        </p>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
            gap: "0.75rem"
          }}
        >
          <label style={{ gridColumn: "1 / -1" }}>
            <strong>Event name</strong><br />
            <input
              value={name}
              style={inputStyle}
              placeholder="Gender Doubles"
              onChange={(event) =>
                setDraft((current) =>
                  setRecordString(current, ["event_family", "event_family_label"], event.target.value)
                )
              }
            />
          </label>

          <label>
            <strong>Event format</strong><br />
            <select
              value={structure}
              style={inputStyle}
              onChange={(event) =>
                setDraft((current) => applyEventStructureWithAge(current, event.target.value as EventStructure))
              }
            >
              <option value="SINGLES">Singles</option>
              <option value="GENDER_DOUBLES">Gender Doubles</option>
              <option value="MIXED_DOUBLES">Mixed Doubles</option>
              <option value="FOUR_PLAYER_TEAM">Four-player team</option>
            </select>
          </label>

          <label>
            <strong>Gender category</strong><br />
            <select
              value={gender}
              style={inputStyle}
              disabled={structure === "MIXED_DOUBLES" || structure === "FOUR_PLAYER_TEAM"}
              onChange={(event) =>
                setDraft((current) => setRecordString(current, ["gender_restriction"], event.target.value))
              }
            >
              {optionsWithCurrent(GENDER_RESTRICTIONS, gender).map((option) => (
                <option key={option} value={option}>{optionLabel(option)}</option>
              ))}
            </select>
            {structure === "MIXED_DOUBLES" || structure === "FOUR_PLAYER_TEAM" ? (
              <small>This event format always uses Mixed gender.</small>
            ) : null}
          </label>

          <fieldset style={{ gridColumn: "1 / -1", padding: "0.8rem", border: "1px solid #cbd5e1", borderRadius: "12px" }}>
            <legend style={{ fontWeight: 800 }}>Tournament days</legend>
            <p style={{ color: "#64748b", marginTop: 0 }}>
              Select every day on which this event may be played. Event cards sort automatically by their earliest selected day.
            </p>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "0.45rem" }}>
              {dayOptions.map((option) => (
                <label key={option.value} style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                  <input
                    type="checkbox"
                    checked={currentDays.includes(option.value)}
                    disabled={!option.enabled}
                    onChange={(event) => {
                      const next = event.target.checked
                        ? [...currentDays, option.value]
                        : currentDays.filter((value) => value !== option.value);
                      setDraft((current) => setEventDayReferences(current, next));
                    }}
                  />
                  {option.label}{option.enabled ? "" : " (disabled)"}
                </label>
              ))}
            </div>
          </fieldset>

          <label>
            <strong>Default draw format</strong><br />
            <select value={drawFormat} style={inputStyle} onChange={(event) => setDraft((current) => setRecordString(current, ["default_format"], event.target.value))}>
              {optionsWithCurrent(COMPETITION_FORMATS, drawFormat).map((option) => (
                <option key={option} value={option}>{optionLabel(option)}</option>
              ))}
            </select>
          </label>
          <label>
            <strong>Default scoring</strong><br />
            <select value={scoring} style={inputStyle} onChange={(event) => setDraft((current) => setRecordString(current, ["default_scoring"], event.target.value))}>
              {optionsWithCurrent(SCORING_OPTIONS, scoring).map((option) => (
                <option key={option} value={option}>{optionLabel(option)}</option>
              ))}
            </select>
          </label>
          <label>
            <strong>Default capacity</strong><br />
            <input
              type="number"
              min="1"
              step="1"
              value={numberInputValue(draft.default_capacity_teams)}
              style={inputStyle}
              onChange={(event) => setDraft((current) => setRecordNumber(current, "default_capacity_teams", event.target.value))}
            />
          </label>
          <label>
            <strong>Default entry fee (USD)</strong><br />
            <input
              type="number"
              min="0"
              step="0.01"
              inputMode="decimal"
              value={numberInputValue(draft.default_price_usd)}
              style={inputStyle}
              onChange={(event) => setDraft((current) => setRecordNumber(current, "default_price_usd", event.target.value))}
            />
            <small>Commerce is the consolidated place to review all event and division fees.</small>
          </label>

          {structure === "FOUR_PLAYER_TEAM" ? (
            <fieldset style={{ gridColumn: "1 / -1", padding: "0.8rem", border: "1px solid #cbd5e1", borderRadius: "12px" }}>
              <legend style={{ fontWeight: 800 }}>Four-player team rules</legend>
              <p style={{ color: "#64748b", marginTop: 0 }}>
                Teams use two men and two women. Configure the competitive tiebreak and optional playoff structure here at the event level.
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem" }}>
                <label>
                  <strong>Tiebreak</strong><br />
                  <select
                    value={cleanString(draft.team_tiebreak_mode) || "SINGLES"}
                    style={inputStyle}
                    onChange={(event) => setDraft((current) => setRecordString(current, ["team_tiebreak_mode"], event.target.value))}
                  >
                    <option value="SINGLES">Singles tiebreak</option>
                    <option value="SKINNY_RELAY">Skinny-singles relay</option>
                  </select>
                </label>
                <label>
                  <strong>Playoff format</strong><br />
                  <select
                    value={cleanString(draft.team_playoff_format) || "NONE"}
                    style={inputStyle}
                    onChange={(event) => setDraft((current) => setRecordString(current, ["team_playoff_format"], event.target.value))}
                  >
                    <option value="NONE">No playoff</option>
                    <option value="TOP_2_FINAL">Top 2 final</option>
                    <option value="TOP_4_SEMIFINALS">Top 4 semifinals</option>
                    <option value="TOP_4_SEMIFINALS_WITH_BRONZE">Top 4 with bronze match</option>
                  </select>
                </label>
                <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginTop: "1.65rem" }}>
                  <input
                    type="checkbox"
                    checked={recordBoolean(draft.team_allow_substitutes, false)}
                    onChange={(event) => setDraft((current) => ({ ...current, team_allow_substitutes: event.target.checked }))}
                  />
                  Allow substitutes
                </label>
              </div>
            </fieldset>
          ) : null}

          <TournamentAgePolicyEditor
            policy={agePolicy}
            participantType={participantType}
            onChange={(policy) => setDraft((current) => writeAgePolicy(current, EVENT_AGE_POLICY_FIELDS, policy))}
            title="Event age policy"
          />

          <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            <input
              type="checkbox"
              checked={recordBoolean(draft.default_waitlist, true)}
              onChange={(event) => setDraft((current) => ({ ...current, default_waitlist: event.target.checked }))}
            />
            Waitlist enabled by default
          </label>
          <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            <input
              type="checkbox"
              checked={participantType !== "SINGLES" && recordBoolean(draft.default_partner_board, true)}
              disabled={participantType === "SINGLES"}
              onChange={(event) => setDraft((current) => ({ ...current, default_partner_board: event.target.checked }))}
            />
            Partner Board enabled by default
          </label>
        </div>
        {message ? <p role="alert" style={{ color: "#b91c1c" }}>{message}</p> : null}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.65rem", flexWrap: "wrap", marginTop: "1rem" }}>
          <button type="button" disabled={submitting} onClick={onCancel} style={{ padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #64748b", background: "white", color: "#0f172a", fontWeight: 800, cursor: "pointer" }}>
            Cancel
          </button>
          <button type="button" disabled={submitting} onClick={() => void submit()} style={{ padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800, cursor: "pointer" }}>
            {submitting ? "Saving…" : mode === "add" ? "Add event" : "Save event"}
          </button>
        </div>
      </section>
    </div>
  );
}
