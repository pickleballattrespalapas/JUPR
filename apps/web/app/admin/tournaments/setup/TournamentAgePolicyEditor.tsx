"use client";

import type { CSSProperties } from "react";
import { cleanString, numberInputValue, type SetupRecord } from "../../tournament-setup/tournamentSetupBuilder";

export type AgePolicyMode =
  | "ALL_AGES"
  | "FIXED_AGE_BRACKET"
  | "SPLIT_AGE"
  | "AUTO_AGE_SPLIT";

export type TeamAgeRule = "YOUNGER" | "OLDER" | "AVERAGE" | "BOTH_QUALIFY";
export type AgeMergeStrategy = "CLOSEST" | "UP" | "DOWN";

export type AgeBracket = {
  id: string;
  label: string;
  min_age: number | null;
  max_age: number | null;
};

export type AgePolicy = {
  mode: AgePolicyMode;
  label: string;
  min_age: number | null;
  max_age: number | null;
  split_age_threshold: number | null;
  min_teams_per_age_group: number;
  team_age_rule: TeamAgeRule;
  merge_strategy: AgeMergeStrategy;
  brackets: AgeBracket[];
};

export type AgePolicyFields = {
  mode: string;
  label: string;
  rules: string;
};

export const EVENT_AGE_POLICY_FIELDS: AgePolicyFields = {
  mode: "default_age_mode",
  label: "default_age_label",
  rules: "default_age_rules"
};

export const DIVISION_AGE_POLICY_FIELDS: AgePolicyFields = {
  mode: "age_mode",
  label: "age_label",
  rules: "age_rules"
};

const DEFAULT_BRACKETS: AgeBracket[] = [
  { id: "under-50", label: "Under 50", min_age: null, max_age: 49 },
  { id: "50-59", label: "50–59", min_age: 50, max_age: 59 },
  { id: "60-69", label: "60–69", min_age: 60, max_age: 69 },
  { id: "70-plus", label: "70+", min_age: 70, max_age: null }
];

const inputStyle: CSSProperties = {
  width: "100%",
  minWidth: 0,
  boxSizing: "border-box",
  padding: "0.55rem",
  border: "1px solid #cbd5e1",
  borderRadius: "8px",
  font: "inherit"
};

function newId(): string {
  return globalThis.crypto?.randomUUID?.() || `age-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function objectValue(value: unknown): SetupRecord {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return { ...(value as SetupRecord) };
  }
  if (typeof value === "string" && value.trim()) {
    try {
      const parsed = JSON.parse(value) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return { ...(parsed as SetupRecord) };
      }
    } catch {
      return {};
    }
  }
  return {};
}

function optionalNumber(value: unknown): number | null {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizedBracket(value: unknown, index: number): AgeBracket | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const row = value as SetupRecord;
  return {
    id: cleanString(row.id) || `age-${index + 1}`,
    label: cleanString(row.label) || `Age group ${index + 1}`,
    min_age: optionalNumber(row.min_age),
    max_age: optionalNumber(row.max_age)
  };
}

export function readAgePolicy(record: SetupRecord, fields: AgePolicyFields): AgePolicy {
  const rules = objectValue(record[fields.rules]);
  const rawMode = cleanString(record[fields.mode] ?? rules.mode).toUpperCase();
  const mode: AgePolicyMode = [
    "ALL_AGES",
    "FIXED_AGE_BRACKET",
    "SPLIT_AGE",
    "AUTO_AGE_SPLIT"
  ].includes(rawMode)
    ? (rawMode as AgePolicyMode)
    : "ALL_AGES";
  const rawBrackets = Array.isArray(rules.brackets) ? rules.brackets : [];
  const brackets = rawBrackets
    .map(normalizedBracket)
    .filter((row): row is AgeBracket => Boolean(row));
  const minTeams = optionalNumber(
    rules.min_teams_per_age_group ?? rules.min_teams
  );
  const splitThreshold = optionalNumber(
    rules.split_age_threshold ?? rules.threshold ?? rules.one_over
  );
  const rawTeamRule = cleanString(rules.team_age_rule).toUpperCase();
  const team_age_rule: TeamAgeRule = [
    "YOUNGER",
    "OLDER",
    "AVERAGE",
    "BOTH_QUALIFY"
  ].includes(rawTeamRule)
    ? (rawTeamRule as TeamAgeRule)
    : rules.younger_player_controls_age === false
      ? "OLDER"
      : "YOUNGER";
  const rawMerge = cleanString(rules.merge_strategy).toUpperCase();
  const merge_strategy: AgeMergeStrategy = ["CLOSEST", "UP", "DOWN"].includes(rawMerge)
    ? (rawMerge as AgeMergeStrategy)
    : "CLOSEST";

  return {
    mode,
    label: cleanString(record[fields.label] ?? rules.age_label) ||
      (mode === "ALL_AGES" ? "All Ages" : ""),
    min_age: optionalNumber(rules.min_age),
    max_age: optionalNumber(rules.max_age),
    split_age_threshold: splitThreshold,
    min_teams_per_age_group: Math.max(1, Math.trunc(minTeams || 4)),
    team_age_rule,
    merge_strategy,
    brackets: brackets.length ? brackets : DEFAULT_BRACKETS.map((row) => ({ ...row }))
  };
}

export function writeAgePolicy(
  record: SetupRecord,
  fields: AgePolicyFields,
  policy: AgePolicy
): SetupRecord {
  const rawRules = record[fields.rules];
  const rules: SetupRecord = objectValue(rawRules);
  for (const key of [
    "mode",
    "age_label",
    "min_age",
    "max_age",
    "split_age_threshold",
    "threshold",
    "one_over",
    "split_age_rule",
    "min_teams_per_age_group",
    "min_teams",
    "brackets",
    "team_age_rule",
    "younger_player_controls_age",
    "merge_strategy"
  ]) {
    delete rules[key];
  }
  rules.mode = policy.mode;
  rules.age_label = policy.label;
  rules.team_age_rule = policy.team_age_rule;
  rules.younger_player_controls_age = policy.team_age_rule === "YOUNGER";
  rules.merge_strategy = policy.merge_strategy;
  if (policy.mode === "FIXED_AGE_BRACKET") {
    rules.min_age = policy.min_age;
    rules.max_age = policy.max_age;
  }
  if (policy.mode === "SPLIT_AGE") {
    rules.split_age_threshold = policy.split_age_threshold;
    rules.split_age_rule = {
      one_player_over_or_equal: policy.split_age_threshold,
      one_player_under: policy.split_age_threshold
    };
  }
  if (policy.mode === "AUTO_AGE_SPLIT") {
    rules.min_teams_per_age_group = policy.min_teams_per_age_group;
    rules.brackets = policy.brackets.map((bracket) => ({ ...bracket }));
  }
  return {
    ...record,
    [fields.mode]: policy.mode,
    [fields.label]: policy.label || (policy.mode === "ALL_AGES" ? "All Ages" : ""),
    [fields.rules]: typeof rawRules === "string" ? JSON.stringify(rules) : rules
  };
}

export function agePolicySummary(policy: AgePolicy): string {
  if (policy.mode === "ALL_AGES") return "All ages";
  if (policy.mode === "FIXED_AGE_BRACKET") {
    return policy.label || [policy.min_age, policy.max_age].filter((value) => value != null).join("–") || "Fixed age bracket";
  }
  if (policy.mode === "SPLIT_AGE") {
    return policy.split_age_threshold
      ? `Split-age partners · one under ${policy.split_age_threshold} and one ${policy.split_age_threshold}+`
      : "Split-age partners";
  }
  return `${policy.brackets.length} candidate brackets · minimum ${policy.min_teams_per_age_group} per bracket`;
}

export function validateAgePolicy(policy: AgePolicy, participantType = ""): string[] {
  const issues: string[] = [];
  if (policy.mode === "FIXED_AGE_BRACKET" && !policy.label.trim() && policy.min_age == null && policy.max_age == null) {
    issues.push("Fixed age bracket needs a label or age range.");
  }
  if (policy.mode === "SPLIT_AGE") {
    if (cleanString(participantType).toUpperCase() === "SINGLES") {
      issues.push("Split-age partners is available only for doubles and team events.");
    }
    if (!Number.isInteger(policy.split_age_threshold) || Number(policy.split_age_threshold) < 1) {
      issues.push("Split-age partners needs a whole-number threshold of at least 1.");
    }
  }
  if (policy.mode === "AUTO_AGE_SPLIT") {
    if (!Number.isInteger(policy.min_teams_per_age_group) || policy.min_teams_per_age_group < 1) {
      issues.push("Auto age split needs a whole-number minimum of at least 1 entry per bracket.");
    }
    if (policy.brackets.length < 2) issues.push("Auto age split needs at least two candidate brackets.");
    const labels = new Set<string>();
    let previousMaximum: number | null = null;
    policy.brackets.forEach((bracket, index) => {
      const label = bracket.label.trim();
      if (!label) issues.push(`Age bracket ${index + 1} needs a label.`);
      if (label && labels.has(label.toLowerCase())) issues.push("Age bracket labels must be unique.");
      if (label) labels.add(label.toLowerCase());
      if (bracket.min_age != null && bracket.max_age != null && bracket.max_age < bracket.min_age) {
        issues.push(`${label || `Age bracket ${index + 1}`} has a maximum age below its minimum age.`);
      }
      if (previousMaximum != null && bracket.min_age != null && bracket.min_age <= previousMaximum) {
        issues.push("Auto age brackets must be ordered and may not overlap.");
      }
      if (bracket.max_age != null) previousMaximum = bracket.max_age;
    });
  }
  return [...new Set(issues)];
}

function numberField(
  value: number | null,
  onChange: (next: number | null) => void,
  props: { min?: number; max?: number; placeholder?: string; disabled?: boolean } = {}
) {
  return (
    <input
      type="number"
      min={props.min}
      max={props.max}
      step="1"
      value={numberInputValue(value)}
      placeholder={props.placeholder}
      disabled={props.disabled}
      style={inputStyle}
      onChange={(event) => {
        const next = event.target.value === "" ? null : Number(event.target.value);
        onChange(Number.isFinite(next) ? next : null);
      }}
    />
  );
}

export default function TournamentAgePolicyEditor({
  policy,
  onChange,
  participantType,
  disabled = false,
  title = "Age policy"
}: {
  policy: AgePolicy;
  onChange: (policy: AgePolicy) => void;
  participantType?: string;
  disabled?: boolean;
  title?: string;
}) {
  const isTeamEvent = cleanString(participantType).toUpperCase() !== "SINGLES";
  const issues = validateAgePolicy(policy, participantType);

  function patch(next: Partial<AgePolicy>) {
    onChange({ ...policy, ...next });
  }

  return (
    <fieldset
      style={{
        gridColumn: "1 / -1",
        padding: "0.9rem",
        border: "1px solid #cbd5e1",
        borderRadius: "12px"
      }}
    >
      <legend style={{ fontWeight: 900 }}>{title}</legend>
      <p style={{ color: "#64748b", marginTop: 0 }}>
        Define the event’s age intent first. Divisions inherit this policy unless an organizer deliberately overrides it.
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))", gap: "0.75rem" }}>
        <label>
          <strong>Age mode</strong><br />
          <select
            value={policy.mode}
            disabled={disabled}
            style={inputStyle}
            onChange={(event) => {
              const mode = event.target.value as AgePolicyMode;
              patch({
                mode,
                label: mode === "ALL_AGES" ? "All Ages" : policy.label,
                brackets: mode === "AUTO_AGE_SPLIT" && !policy.brackets.length
                  ? DEFAULT_BRACKETS.map((row) => ({ ...row }))
                  : policy.brackets
              });
            }}
          >
            <option value="ALL_AGES">All ages</option>
            <option value="FIXED_AGE_BRACKET">Fixed age bracket</option>
            {isTeamEvent ? <option value="SPLIT_AGE">Split-age partners (one under / one over)</option> : null}
            <option value="AUTO_AGE_SPLIT">Auto age split</option>
          </select>
        </label>

        {policy.mode === "FIXED_AGE_BRACKET" ? (
          <>
            <label>
              <strong>Public bracket label</strong><br />
              <input
                value={policy.label}
                disabled={disabled}
                placeholder="50+"
                style={inputStyle}
                onChange={(event) => patch({ label: event.target.value })}
              />
            </label>
            <label><strong>Minimum age</strong><br />{numberField(policy.min_age, (min_age) => patch({ min_age }), { min: 1, max: 120, disabled })}</label>
            <label><strong>Maximum age (optional)</strong><br />{numberField(policy.max_age, (max_age) => patch({ max_age }), { min: 1, max: 120, disabled })}</label>
          </>
        ) : null}

        {policy.mode === "SPLIT_AGE" ? (
          <label style={{ gridColumn: "1 / -1" }}>
            <strong>Split-age threshold</strong><br />
            {numberField(policy.split_age_threshold, (split_age_threshold) => patch({ split_age_threshold }), { min: 1, max: 120, placeholder: "50", disabled })}
            <small>Each team must include one player under the threshold and one player at or above it. Example at 50: ages 49 and 50 qualify; 49 and 49 or 50 and 50 do not. This does not create separate Under 50 and 50+ divisions.</small>
          </label>
        ) : null}

        {policy.mode === "AUTO_AGE_SPLIT" ? (
          <>
            <label>
              <strong>Minimum entries per resulting bracket</strong><br />
              <input
                type="number"
                min="1"
                step="1"
                value={policy.min_teams_per_age_group}
                disabled={disabled}
                style={inputStyle}
                onChange={(event) => patch({ min_teams_per_age_group: Math.max(1, Number(event.target.value) || 1) })}
              />
            </label>
            <label>
              <strong>Underfilled bracket fallback</strong><br />
              <select
                value={policy.merge_strategy}
                disabled={disabled}
                style={inputStyle}
                onChange={(event) => patch({ merge_strategy: event.target.value as AgeMergeStrategy })}
              >
                <option value="CLOSEST">Merge with closest bracket</option>
                <option value="UP">Merge into older bracket</option>
                <option value="DOWN">Merge into younger bracket</option>
              </select>
            </label>
          </>
        ) : null}

        {isTeamEvent && policy.mode !== "ALL_AGES" && policy.mode !== "SPLIT_AGE" ? (
          <label>
            <strong>Team age rule</strong><br />
            <select
              value={policy.team_age_rule}
              disabled={disabled}
              style={inputStyle}
              onChange={(event) => patch({ team_age_rule: event.target.value as TeamAgeRule })}
            >
              <option value="YOUNGER">Younger player determines team age</option>
              <option value="OLDER">Older player determines team age</option>
              <option value="AVERAGE">Average player age</option>
              <option value="BOTH_QUALIFY">Both players must qualify</option>
            </select>
          </label>
        ) : null}
      </div>

      {policy.mode === "AUTO_AGE_SPLIT" ? (
        <div style={{ marginTop: "0.9rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "0.65rem", alignItems: "center", flexWrap: "wrap" }}>
            <div>
              <strong>Candidate age brackets</strong>
              <p style={{ color: "#64748b", margin: "0.2rem 0 0" }}>
                These are proposed groups. The organizer reviews actual registration counts before accepting a split.
              </p>
            </div>
            <button
              type="button"
              disabled={disabled}
              style={{ ...inputStyle, width: "auto", cursor: "pointer", fontWeight: 800, background: "white" }}
              onClick={() => patch({ brackets: [...policy.brackets, { id: newId(), label: `Age group ${policy.brackets.length + 1}`, min_age: null, max_age: null }] })}
            >
              Add bracket
            </button>
          </div>
          <div style={{ display: "grid", gap: "0.65rem", marginTop: "0.65rem" }}>
            {policy.brackets.map((bracket, index) => (
              <article key={bracket.id} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem", background: "#f8fafc" }}>
                <div style={{ display: "grid", gridTemplateColumns: "minmax(170px, 2fr) minmax(110px, 1fr) minmax(110px, 1fr) auto", gap: "0.55rem", alignItems: "end" }}>
                  <label>
                    <strong>Label</strong><br />
                    <input
                      value={bracket.label}
                      disabled={disabled}
                      style={inputStyle}
                      onChange={(event) => patch({ brackets: policy.brackets.map((row) => row.id === bracket.id ? { ...row, label: event.target.value } : row) })}
                    />
                  </label>
                  <label><strong>Min age</strong><br />{numberField(bracket.min_age, (min_age) => patch({ brackets: policy.brackets.map((row) => row.id === bracket.id ? { ...row, min_age } : row) }), { min: 1, max: 120, disabled })}</label>
                  <label><strong>Max age</strong><br />{numberField(bracket.max_age, (max_age) => patch({ brackets: policy.brackets.map((row) => row.id === bracket.id ? { ...row, max_age } : row) }), { min: 1, max: 120, disabled })}</label>
                  <button
                    type="button"
                    disabled={disabled || policy.brackets.length <= 2}
                    style={{ ...inputStyle, width: "auto", color: "#991b1b", cursor: "pointer", background: "white" }}
                    onClick={() => patch({ brackets: policy.brackets.filter((row) => row.id !== bracket.id) })}
                  >
                    Remove
                  </button>
                </div>
                <small style={{ color: "#64748b" }}>Bracket {index + 1}</small>
              </article>
            ))}
          </div>
        </div>
      ) : null}

      {issues.length ? (
        <ul style={{ color: "#b91c1c", marginBottom: 0 }}>
          {issues.map((issue) => <li key={issue}>{issue}</li>)}
        </ul>
      ) : null}
    </fieldset>
  );
}
