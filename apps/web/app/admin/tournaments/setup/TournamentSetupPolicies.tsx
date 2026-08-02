"use client";

import type { CSSProperties } from "react";
import type { SetupRecord } from "../../tournament-setup/tournamentSetupBuilder";

type PolicyTemplate = {
  key: string;
  label: string;
  text: string;
};

type Props = {
  settings: SetupRecord;
  registrationStatus: string;
  disabled: boolean;
  inputStyle: CSSProperties;
  onChange: (next: SetupRecord) => void;
};

const REGISTRATION_RULE_TEMPLATES: PolicyTemplate[] = [
  {
    key: "STANDARD",
    label: "Standard tournament rules",
    text:
      "Registration is first-come, first-served. A doubles or team entry is incomplete until all required partners or team members are confirmed. The tournament director may combine or adjust divisions for competitive balance, scheduling, or minimum participation. Players must enter the appropriate skill and age division and may be moved when their rating or playing history does not match the selected division."
  },
  {
    key: "COMPETITIVE",
    label: "Competitive and rating-verified",
    text:
      "Registration is first-come, first-served and subject to rating and eligibility review. The tournament director may move a player or team when current ratings, recent results, or roster composition do not meet the selected division rules. Partners and team rosters must be complete by the registration deadline. Draws, formats, and division groupings may be adjusted to preserve competitive integrity."
  },
  {
    key: "COMMUNITY",
    label: "Community and recreational",
    text:
      "Players should choose the division that best matches their current playing level. The tournament director may combine small divisions, move teams for fair play, or modify formats when participation requires it. Doubles entries should have a confirmed partner by the registration deadline unless the Partner Board is being used."
  }
];

const CANCELLATION_TEMPLATES: PolicyTemplate[] = [
  {
    key: "FLEXIBLE",
    label: "Flexible refund policy",
    text:
      "Cancellations received at least 14 days before the tournament receive a full refund. Cancellations received 7–13 days before the tournament receive a 50% refund. No refund is available within 7 days of the tournament unless the tournament is cancelled by the organizer. Processing fees and fulfilled merchandise are non-refundable."
  },
  {
    key: "STANDARD",
    label: "Refunds until registration closes",
    text:
      "A full refund is available until registration closes. After registration closes, fees are non-refundable unless the tournament is cancelled by the organizer or an approved replacement takes the registrant's place. Processing fees and fulfilled merchandise are non-refundable."
  },
  {
    key: "FINAL_SALE",
    label: "Registration fees are final",
    text:
      "Registration fees are non-refundable after payment. A registrant may request an approved replacement before draws are finalized. If the organizer cancels the tournament, tournament fees will be refunded or credited; processing fees and fulfilled merchandise remain non-refundable."
  }
];

const WEATHER_TEMPLATES: PolicyTemplate[] = [
  {
    key: "DELAY_RESCHEDULE",
    label: "Delay or reschedule first",
    text:
      "The tournament will use delays, shortened formats, indoor alternatives, or rescheduled play when weather or court conditions prevent safe competition. The tournament director controls all weather decisions. Once a division has begun, weather-related format changes or incomplete play do not automatically create a refund. If the organizer cancels an entire unplayed division, affected entry fees will be refunded or credited."
  },
  {
    key: "MODIFIED_FORMAT",
    label: "Play when safe with modified formats",
    text:
      "Play will continue whenever courts are safe. Weather delays may require reduced game totals, shortened scoring, revised brackets, or cancellation of consolation and playoff rounds. The tournament director's weather and safety decisions are final. Refunds are not guaranteed after play begins."
  },
  {
    key: "CREDIT_IF_CANCELLED",
    label: "Credit for fully cancelled divisions",
    text:
      "Weather may delay, shorten, move, or reschedule play. If an entire division is cancelled before any match is played and cannot be rescheduled, affected players receive a tournament credit or refund of the division fee. Partial play, shortened formats, and completed preliminary rounds are considered delivered competition."
  }
];

export function withDefaultTournamentPolicies(settings: SetupRecord): SetupRecord {
  return {
    ...settings,
    rules_markdown: text(settings.rules_markdown) || REGISTRATION_RULE_TEMPLATES[0].text,
    refund_policy_markdown:
      text(settings.refund_policy_markdown) || CANCELLATION_TEMPLATES[1].text,
    weather_policy_markdown:
      text(settings.weather_policy_markdown) || WEATHER_TEMPLATES[0].text
  };
}

function text(value: unknown): string {
  return value == null ? "" : String(value);
}

function selectedTemplate(value: unknown, templates: PolicyTemplate[]): string {
  const current = text(value).trim();
  const match = templates.find((template) => template.text === current);
  return match?.key || "CUSTOM";
}

function applyTemplate(
  settings: SetupRecord,
  field: string,
  key: string,
  templates: PolicyTemplate[]
): SetupRecord {
  if (key === "CUSTOM") return settings;
  const template = templates.find((item) => item.key === key);
  return template ? { ...settings, [field]: template.text } : settings;
}

function PolicyEditor({
  label,
  field,
  settings,
  templates,
  rows,
  required,
  disabled,
  inputStyle,
  onChange
}: {
  label: string;
  field: string;
  settings: SetupRecord;
  templates: PolicyTemplate[];
  rows: number;
  required?: boolean;
  disabled: boolean;
  inputStyle: CSSProperties;
  onChange: (next: SetupRecord) => void;
}) {
  return (
    <section style={{ display: "grid", gap: "0.5rem" }}>
      <label>
        <strong>{label}{required ? " *" : ""}</strong>
        <br />
        <select
          value={selectedTemplate(settings[field], templates)}
          disabled={disabled}
          style={inputStyle}
          onChange={(event) =>
            onChange(applyTemplate(settings, field, event.target.value, templates))
          }
        >
          {templates.map((template) => (
            <option key={template.key} value={template.key}>
              {template.label}
            </option>
          ))}
          <option value="CUSTOM">Write a custom policy</option>
        </select>
      </label>
      <textarea
        value={text(settings[field])}
        disabled={disabled}
        required={required}
        rows={rows}
        style={inputStyle}
        onChange={(event) => onChange({ ...settings, [field]: event.target.value })}
      />
    </section>
  );
}

export default function TournamentSetupPolicies({
  settings,
  registrationStatus,
  disabled,
  inputStyle,
  onChange
}: Props) {
  return (
    <section style={{ display: "grid", gap: "1rem" }}>
      <article
        style={{
          padding: "0.85rem",
          border: "1px solid #bfdbfe",
          borderRadius: "12px",
          background: "#eff6ff",
          color: "#1e3a8a"
        }}
      >
        <strong>Tournament-wide registration status</strong>
        <br />
        <small>
          Registration is controlled once for the whole tournament. Current
          status: {registrationStatus || "draft"}.
        </small>
      </article>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
          gap: "0.75rem"
        }}
      >
        <label>
          <strong>Registration link</strong>
          <br />
          <input
            value={text(settings.registration_slug)}
            disabled={disabled}
            style={inputStyle}
            onChange={(event) =>
              onChange({ ...settings, registration_slug: event.target.value })
            }
          />
        </label>
        <label>
          <strong>Registration opens</strong>
          <br />
          <input
            type="datetime-local"
            value={text(settings.registration_open_at).slice(0, 16)}
            disabled={disabled}
            style={inputStyle}
            onChange={(event) =>
              onChange({ ...settings, registration_open_at: event.target.value })
            }
          />
        </label>
        <label>
          <strong>Registration closes</strong>
          <br />
          <input
            type="datetime-local"
            value={text(settings.registration_close_at).slice(0, 16)}
            disabled={disabled}
            style={inputStyle}
            onChange={(event) =>
              onChange({ ...settings, registration_close_at: event.target.value })
            }
          />
        </label>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <input
            type="checkbox"
            checked={Boolean(settings.waitlist_enabled)}
            disabled={disabled}
            onChange={(event) =>
              onChange({ ...settings, waitlist_enabled: event.target.checked })
            }
          />
          Waitlist enabled
        </label>
        <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <input
            type="checkbox"
            checked={Boolean(settings.partner_board_enabled)}
            disabled={disabled}
            onChange={(event) =>
              onChange({ ...settings, partner_board_enabled: event.target.checked })
            }
          />
          Partner Board enabled
        </label>
      </div>

      <PolicyEditor
        label="Registration rules"
        field="rules_markdown"
        settings={settings}
        templates={REGISTRATION_RULE_TEMPLATES}
        rows={5}
        required
        disabled={disabled}
        inputStyle={inputStyle}
        onChange={onChange}
      />
      <PolicyEditor
        label="Cancellation and refund policy"
        field="refund_policy_markdown"
        settings={settings}
        templates={CANCELLATION_TEMPLATES}
        rows={4}
        required
        disabled={disabled}
        inputStyle={inputStyle}
        onChange={onChange}
      />
      <PolicyEditor
        label="Weather policy"
        field="weather_policy_markdown"
        settings={settings}
        templates={WEATHER_TEMPLATES}
        rows={5}
        required
        disabled={disabled}
        inputStyle={inputStyle}
        onChange={onChange}
      />
    </section>
  );
}
