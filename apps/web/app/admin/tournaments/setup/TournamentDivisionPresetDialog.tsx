"use client";

import { useEffect, useMemo, useState, type CSSProperties } from "react";
import {
  cleanString,
  eventDayReferences,
  eventDivisionName,
  eventFamilyName,
  newEventOptionRow,
  recordBoolean,
  setEventDayReferences,
  type BuilderRow,
  type SetupConfiguration,
  type SetupRecord
} from "../../tournament-setup/tournamentSetupBuilder";
import {
  EVENT_AGE_POLICY_FIELDS,
  readAgePolicy,
  writeAgePolicy,
  DIVISION_AGE_POLICY_FIELDS,
  type AgePolicy
} from "./TournamentAgePolicyEditor";

const SKILL_PRESETS = ["3.0", "3.5", "4.0", "4.5", "Open"] as const;

type GenderPreset = "MEN" | "WOMEN" | "MIXED" | "ANY";

type Proposal = {
  key: string;
  selected: boolean;
  duplicate: boolean;
  issue?: string;
  value: SetupRecord;
};

type Props = {
  open: boolean;
  family: BuilderRow | null;
  configuration: SetupConfiguration;
  onCancel: () => void;
  onConfirm: (values: SetupRecord[]) => void | Promise<void>;
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

function titleCase(value: string): string {
  return value
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function genderOptions(family: SetupRecord): GenderPreset[] {
  const competition = cleanString(family.competition_format).toUpperCase();
  const participant = cleanString(family.participant_type).toUpperCase();
  if (competition === "FOUR_PLAYER_TEAM" || participant === "MIXED_DOUBLES") return ["MIXED"];
  if (participant === "GENDER_DOUBLES" || participant === "SINGLES") return ["MEN", "WOMEN"];
  return ["ANY"];
}

function genderName(gender: GenderPreset): string {
  if (gender === "MEN") return "Men's";
  if (gender === "WOMEN") return "Women's";
  if (gender === "MIXED") return "Mixed";
  return "Open";
}

function fixedBracketPolicy(
  label: string,
  minAge: number | null,
  maxAge: number | null,
  inherited: AgePolicy
): AgePolicy {
  return {
    mode: "FIXED_AGE_BRACKET",
    label,
    min_age: minAge,
    max_age: maxAge,
    split_age_threshold: null,
    min_teams_per_age_group: 1,
    team_age_rule: inherited.team_age_rule,
    merge_strategy: inherited.merge_strategy,
    brackets: []
  };
}

function buildProposal(
  family: BuilderRow,
  configuration: SetupConfiguration,
  skill: string,
  gender: GenderPreset,
  age: { key: string; label: string; policy: AgePolicy | null },
  index: number
): SetupRecord {
  const familyName = eventFamilyName(family.value);
  const selectedConfiguration: SetupConfiguration = {
    ...configuration,
    eventFamilies: [family, ...configuration.eventFamilies.filter((row) => row.key !== family.key)]
  };
  let row = newEventOptionRow(selectedConfiguration);
  const participantType = cleanString(family.value.participant_type) || "GENDER_DOUBLES";
  const competitionFormat = cleanString(family.value.competition_format) || "STANDARD";
  const baseName = [genderName(gender), skill, age.label === "Event age policy" ? "" : age.label]
    .filter(Boolean)
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
  row = setEventDayReferences(
    {
      ...row,
      id: `division_${globalThis.crypto?.randomUUID?.().replaceAll("-", "").slice(0, 16) || `${Date.now().toString(36)}${index}`}`,
      event_family_label: familyName,
      event_family: familyName,
      division_name: baseName,
      label: baseName,
      participant_type: participantType,
      event_type: participantType,
      gender_restriction: gender,
      competition_format: competitionFormat,
      skill_label: skill,
      skill_mode: skill.toLowerCase() === "open" ? "OPEN" : "SKILL_BRACKET",
      capacity_teams: Number(family.value.default_capacity_teams) || 16,
      price_usd: Number(family.value.default_price_usd) || 0,
      waitlist_enabled: recordBoolean(family.value.default_waitlist, true),
      partner_board_enabled: participantType !== "SINGLES" && recordBoolean(family.value.default_partner_board, true),
      event_format_default: cleanString(family.value.default_format) || "ROUND_ROBIN_PLUS_PLAYOFF",
      scoring_default: cleanString(family.value.default_scoring) || "GAME_TO_15",
      schedule_mode: "INHERIT_EVENT",
      age_policy_source: age.policy ? "OVERRIDE" : "INHERIT_EVENT",
      sort_order: configuration.eventOptions.length + index + 1
    },
    eventDayReferences(family.value)
  );
  if (age.policy) {
    row = writeAgePolicy(row, DIVISION_AGE_POLICY_FIELDS, age.policy);
  } else {
    row = writeAgePolicy(row, DIVISION_AGE_POLICY_FIELDS, readAgePolicy(family.value, EVENT_AGE_POLICY_FIELDS));
  }
  return row;
}

export default function TournamentDivisionPresetDialog({
  open,
  family,
  configuration,
  onCancel,
  onConfirm
}: Props) {
  const [skills, setSkills] = useState<string[]>(["3.0", "3.5", "4.0", "4.5"]);
  const [genders, setGenders] = useState<GenderPreset[]>([]);
  const [ageMode, setAgeMode] = useState<"INHERIT" | "BRACKETS">("INHERIT");
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState("");

  const familyAgePolicy = useMemo(
    () => readAgePolicy(family?.value || {}, EVENT_AGE_POLICY_FIELDS),
    [family]
  );
  const supportsAgeBrackets = familyAgePolicy.mode === "AUTO_AGE_SPLIT" && familyAgePolicy.brackets.length > 0;

  useEffect(() => {
    if (!open || !family) return;
    setSkills(["3.0", "3.5", "4.0", "4.5"]);
    setGenders(genderOptions(family.value));
    setAgeMode("INHERIT");
    setProposals([]);
    setMessage("");
    setSubmitting(false);
  }, [open, family]);

  if (!open || !family) return null;
  const selectedFamily = family;

  function toggleSkill(skill: string) {
    setSkills((current) => current.includes(skill) ? current.filter((value) => value !== skill) : [...current, skill]);
    setProposals([]);
  }

  function toggleGender(gender: GenderPreset) {
    setGenders((current) => current.includes(gender) ? current.filter((value) => value !== gender) : [...current, gender]);
    setProposals([]);
  }

  function preview() {
    if (!skills.length) {
      setMessage("Choose at least one skill level.");
      return;
    }
    if (!genders.length) {
      setMessage("Choose at least one gender category.");
      return;
    }
    const ages = ageMode === "BRACKETS" && supportsAgeBrackets
      ? familyAgePolicy.brackets.map((bracket) => ({
          key: bracket.id,
          label: bracket.label,
          policy: fixedBracketPolicy(
            bracket.label,
            bracket.min_age,
            bracket.max_age,
            familyAgePolicy
          )
        }))
      : [{ key: "inherit", label: "Event age policy", policy: null }];
    const existingNames = new Set(
      configuration.eventOptions
        .filter((row) => eventFamilyName(row.value).toLowerCase() === eventFamilyName(selectedFamily.value).toLowerCase())
        .map((row) => eventDivisionName(row.value).toLowerCase())
    );
    let index = 0;
    const next: Proposal[] = [];
    for (const gender of genders) {
      for (const skill of skills) {
        for (const age of ages) {
          index += 1;
          const value = buildProposal(selectedFamily, configuration, skill, gender, age, index);
          const duplicate = existingNames.has(eventDivisionName(value).toLowerCase());
          next.push({
            key: `${gender}-${skill}-${age.key}`,
            selected: !duplicate,
            duplicate,
            issue: duplicate ? "A division with this name already exists in this Event." : undefined,
            value
          });
        }
      }
    }
    setProposals(next);
    setMessage(next.some((row) => row.duplicate) ? "Existing divisions were detected and left unselected." : "Review the proposed divisions before adding them.");
  }

  async function submit() {
    const selectedProposals = proposals.filter((row) => row.selected);
    if (!selectedProposals.length) {
      setMessage("Choose at least one new division.");
      return;
    }

    const familyName = eventFamilyName(selectedFamily.value).toLowerCase();
    const existingNames = new Set(
      configuration.eventOptions
        .filter((row) => eventFamilyName(row.value).toLowerCase() === familyName)
        .map((row) => eventDivisionName(row.value).trim().toLowerCase())
        .filter(Boolean)
    );
    const seen = new Set<string>();
    const issues = new Map<string, string>();
    for (const proposal of selectedProposals) {
      const name = eventDivisionName(proposal.value).trim();
      const normalized = name.toLowerCase();
      if (!name) {
        issues.set(proposal.key, "Division name is required.");
        continue;
      }
      if (existingNames.has(normalized)) {
        issues.set(proposal.key, "A division with this name already exists in this Event.");
        continue;
      }
      if (seen.has(normalized)) {
        issues.set(proposal.key, "Another selected proposal uses this division name.");
        continue;
      }
      seen.add(normalized);
    }

    if (issues.size) {
      setProposals((current) => current.map((row) => {
        const issue = issues.get(row.key);
        return issue ? { ...row, duplicate: true, selected: false, issue } : row;
      }));
      setMessage("Resolve the highlighted division names, then select them again.");
      return;
    }

    setSubmitting(true);
    try {
      await onConfirm(selectedProposals.map((row) => row.value));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      role="presentation"
      style={{ position: "fixed", inset: 0, zIndex: 1100, display: "grid", placeItems: "center", padding: "1rem", background: "rgba(15, 23, 42, 0.62)" }}
      onMouseDown={(event) => { if (event.target === event.currentTarget && !submitting) onCancel(); }}
    >
      <section role="dialog" aria-modal="true" aria-labelledby="division-preset-title" style={{ width: "min(1040px, 100%)", maxHeight: "calc(100vh - 2rem)", overflowY: "auto", padding: "1.1rem", borderRadius: "16px", background: "white", boxShadow: "0 24px 70px rgba(15, 23, 42, 0.35)" }}>
        <h2 id="division-preset-title" style={{ marginTop: 0 }}>Generate divisions for {eventFamilyName(selectedFamily.value)}</h2>
        <p style={{ color: "#475569" }}>
          Select common skill, gender, and age combinations once. Existing divisions are detected before anything is saved. Generated rows inherit Event defaults and can be edited later.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(250px, 1fr))", gap: "0.85rem" }}>
          <fieldset style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.75rem" }}>
            <legend style={{ fontWeight: 800 }}>Skill levels</legend>
            {SKILL_PRESETS.map((skill) => (
              <label key={skill} style={{ display: "flex", gap: "0.5rem", marginTop: "0.35rem" }}>
                <input type="checkbox" checked={skills.includes(skill)} onChange={() => toggleSkill(skill)} />
                {skill}
              </label>
            ))}
          </fieldset>
          <fieldset style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.75rem" }}>
            <legend style={{ fontWeight: 800 }}>Gender categories</legend>
            {genderOptions(selectedFamily.value).map((gender) => (
              <label key={gender} style={{ display: "flex", gap: "0.5rem", marginTop: "0.35rem" }}>
                <input type="checkbox" checked={genders.includes(gender)} onChange={() => toggleGender(gender)} />
                {titleCase(gender)}
              </label>
            ))}
          </fieldset>
          <fieldset style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.75rem" }}>
            <legend style={{ fontWeight: 800 }}>Age handling</legend>
            <label style={{ display: "flex", gap: "0.5rem" }}>
              <input type="radio" checked={ageMode === "INHERIT"} onChange={() => { setAgeMode("INHERIT"); setProposals([]); }} />
              Inherit Event age policy
            </label>
            {supportsAgeBrackets ? (
              <label style={{ display: "flex", gap: "0.5rem", marginTop: "0.4rem" }}>
                <input type="radio" checked={ageMode === "BRACKETS"} onChange={() => { setAgeMode("BRACKETS"); setProposals([]); }} />
                Create one division per candidate age bracket
              </label>
            ) : null}
          </fieldset>
        </div>

        <button type="button" style={{ marginTop: "0.85rem", padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "white", color: "#0f172a", fontWeight: 800, cursor: "pointer" }} onClick={preview}>
          Preview divisions
        </button>

        {proposals.length ? (
          <div style={{ display: "grid", gap: "0.65rem", marginTop: "0.9rem" }}>
            {proposals.map((proposal, index) => (
              <article key={proposal.key} style={{ border: `1px solid ${proposal.duplicate ? "#fde68a" : "#e2e8f0"}`, borderRadius: "12px", padding: "0.7rem", background: proposal.duplicate ? "#fffbeb" : "#f8fafc" }}>
                <div style={{ display: "grid", gridTemplateColumns: "auto minmax(220px, 2fr) minmax(120px, 1fr) minmax(120px, 1fr)", gap: "0.6rem", alignItems: "end" }}>
                  <input type="checkbox" aria-label={`Select proposal ${index + 1}`} checked={proposal.selected} disabled={submitting} onChange={(event) => setProposals((current) => current.map((row) => row.key === proposal.key ? { ...row, selected: event.target.checked } : row))} />
                  <label><strong>Division name</strong><br /><input value={eventDivisionName(proposal.value)} disabled={submitting} style={inputStyle} onChange={(event) => setProposals((current) => current.map((row) => row.key === proposal.key ? {
                    ...row,
                    duplicate: false,
                    issue: undefined,
                    value: { ...row.value, division_name: event.target.value, label: event.target.value }
                  } : row))} /></label>
                  <div><strong>Skill</strong><br />{cleanString(proposal.value.skill_label)}</div>
                  <div><strong>Gender</strong><br />{titleCase(cleanString(proposal.value.gender_restriction))}</div>
                </div>
                <small style={{ color: proposal.duplicate ? "#92400e" : "#64748b" }}>
                  {proposal.issue || (proposal.duplicate ? "Review this proposal before selecting it." : `${cleanString(proposal.value.age_label) || "Event age policy"} · $${Number(proposal.value.price_usd || 0).toFixed(2)} · capacity ${Number(proposal.value.capacity_teams || 0)}`)}
                </small>
              </article>
            ))}
          </div>
        ) : null}

        {message ? <p role="status" style={{ color: /required|choose/i.test(message) ? "#b91c1c" : "#475569" }}>{message}</p> : null}
        <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.65rem", flexWrap: "wrap", marginTop: "1rem" }}>
          <button type="button" disabled={submitting} onClick={onCancel} style={{ padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #64748b", background: "white", color: "#0f172a", fontWeight: 800, cursor: "pointer" }}>Cancel</button>
          <button type="button" disabled={submitting || !proposals.some((row) => row.selected)} onClick={() => void submit()} style={{ padding: "0.6rem 0.9rem", borderRadius: "999px", border: "1px solid #0f172a", background: "#0f172a", color: "white", fontWeight: 800, cursor: "pointer" }}>
            {submitting ? "Saving divisions…" : "Save selected divisions"}
          </button>
        </div>
      </section>
    </div>
  );
}
