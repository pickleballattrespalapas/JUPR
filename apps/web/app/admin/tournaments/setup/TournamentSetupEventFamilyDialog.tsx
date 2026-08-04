"use client";

import { useEffect, useState, type CSSProperties } from "react";
import {
  COMPETITION_FORMATS,
  GENDER_RESTRICTIONS,
  PARTICIPANT_TYPES,
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

type Props = {
  open: boolean;
  mode: "add" | "edit";
  initialValue: SetupRecord;
  days: BuilderRow[];
  onCancel: () => void;
  onConfirm: (value: SetupRecord) => void;
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

  useEffect(() => {
    if (!open) return;
    const participantType = cleanString(initialValue.participant_type) || "GENDER_DOUBLES";
    setDraft({
      ...initialValue,
      gender_restriction:
        participantType === "MIXED_DOUBLES"
          ? "MIXED"
          : cleanString(initialValue.gender_restriction) || "ANY"
    });
    setMessage("");
  }, [open, initialValue]);

  if (!open) return null;

  const name = eventFamilyName(draft);
  const currentDays = eventDayReferences(draft);
  const participantType = cleanString(draft.participant_type) || "GENDER_DOUBLES";
  const gender = participantType === "MIXED_DOUBLES"
    ? "MIXED"
    : cleanString(draft.gender_restriction) || "ANY";
  const drawFormat = cleanString(draft.default_format) || "ROUND_ROBIN_PLUS_PLAYOFF";
  const scoring = cleanString(draft.default_scoring) || "GAME_TO_15";
  const dayOptions = days.map((day) => ({
    value: dayReference(day.value),
    label: dayLabel(day.value) || dayReference(day.value),
    enabled: recordBoolean(day.value.enabled, true)
  }));

  function updateParticipantType(nextType: string) {
    setDraft((current) => {
      const next = setRecordString(current, ["participant_type"], nextType);
      if (nextType === "MIXED_DOUBLES") {
        next.gender_restriction = "MIXED";
        next.default_partner_board = true;
      } else if (nextType === "SINGLES") {
        next.default_partner_board = false;
        if (cleanString(next.gender_restriction) === "MIXED") {
          next.gender_restriction = "ANY";
        }
      } else if (cleanString(next.gender_restriction) === "MIXED") {
        next.gender_restriction = "ANY";
      }
      return next;
    });
  }

  function submit() {
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
    onConfirm({
      ...draft,
      gender_restriction: participantType === "MIXED_DOUBLES" ? "MIXED" : gender,
      default_partner_board:
        participantType === "SINGLES" ? false : recordBoolean(draft.default_partner_board, true)
    });
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
          width: "min(820px, 100%)",
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
          Configure the parent event and its defaults. Saving closes this dialog and returns a compact, read-only event card. Published tournament data remains unchanged until final review and publication.
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
          <fieldset style={{ gridColumn: "1 / -1", padding: "0.8rem", border: "1px solid #cbd5e1", borderRadius: "12px" }}>
            <legend style={{ fontWeight: 800 }}>Tournament days</legend>
            <p style={{ color: "#64748b", marginTop: 0 }}>
              Select every day on which this event may be played. Events are displayed automatically in tournament-day order.
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
            <strong>Participant type</strong><br />
            <select value={participantType} style={inputStyle} onChange={(event) => updateParticipantType(event.target.value)}>
              {optionsWithCurrent(PARTICIPANT_TYPES, participantType).map((option) => (
                <option key={option} value={option}>{optionLabel(option)}</option>
              ))}
            </select>
          </label>
          <label>
            <strong>Gender</strong><br />
            <select
              value={gender}
              style={inputStyle}
              disabled={participantType === "MIXED_DOUBLES"}
              onChange={(event) =>
                setDraft((current) => setRecordString(current, ["gender_restriction"], event.target.value))
              }
            >
              {optionsWithCurrent(GENDER_RESTRICTIONS, gender).map((option) => (
                <option key={option} value={option}>{optionLabel(option)}</option>
              ))}
            </select>
            {participantType === "MIXED_DOUBLES" ? <small>Mixed Doubles always uses Mixed gender.</small> : null}
          </label>
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
          </label>
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
          <button type="button" onClick={onCancel} style={{ padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #64748b", background: "white", color: "#0f172a", fontWeight: 800, cursor: "pointer" }}>
            Cancel
          </button>
          <button type="button" onClick={submit} style={{ padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800, cursor: "pointer" }}>
            {mode === "add" ? "Add event" : "Save event"}
          </button>
        </div>
      </section>
    </div>
  );
}
