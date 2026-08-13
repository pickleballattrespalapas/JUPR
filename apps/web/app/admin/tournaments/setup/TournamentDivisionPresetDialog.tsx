"use client";

import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { FormDialog, InteractionActionError, type ActionCompletion } from "@/components/interaction";
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
import {
  parsedSkillAnchor,
  skillEligibilityLabel,
  skillEligibilitySummary,
  type SkillEligibilityMode
} from "@/lib/tournamentSkillEligibility";

const SKILL_PRESETS = ["2.5", "3.0", "3.5", "4.0", "4.5", "5.0", "Open"] as const;
const COMBINED_CAP_PRESETS = [7, 7.5, 8, 8.5] as const;

type GenderPreset = "MEN" | "WOMEN" | "MIXED" | "ANY";
type EligibilityMode = "STANDARD" | "MINIMUM" | "COMBINED_RATING_CAP";

type SkillSpec = {
  key: string;
  label: string;
  skillLabel: string;
  skillMode: string;
  eligibilityMode: SkillEligibilityMode;
  skillMinRating: number | null;
  skillMaxRating: number | null;
  combinedRatingCap: number | null;
};

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
  onConfirm: (values: SetupRecord[]) => Promise<ActionCompletion>;
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

function combinedRatingAllowed(family: SetupRecord): boolean {
  const competition = cleanString(family.competition_format).toUpperCase();
  const participant = cleanString(family.participant_type).toUpperCase();
  return participant !== "SINGLES" && competition !== "FOUR_PLAYER_TEAM";
}

function genderName(gender: GenderPreset): string {
  if (gender === "MEN") return "Men's";
  if (gender === "WOMEN") return "Women's";
  if (gender === "MIXED") return "Mixed";
  return "Open";
}

function teamAgeRuleLabel(rule: string): string {
  const labels: Record<string, string> = {
    YOUNGER: "younger player determines team age",
    OLDER: "older player determines team age",
    AVERAGE: "average player age",
    BOTH_QUALIFY: "both players must qualify"
  };
  return labels[rule.toUpperCase()] || titleCase(rule);
}

function resolvedAgePolicySummary(policy: AgePolicy): string {
  if (policy.mode === "ALL_AGES") return "All ages";
  if (policy.mode === "FIXED_AGE_BRACKET") return policy.label || "Fixed age bracket";
  if (policy.mode === "SPLIT_AGE") {
    return policy.split_age_threshold
      ? `Split-age partners: one under ${policy.split_age_threshold} and one ${policy.split_age_threshold}+`
      : "Split-age partners";
  }
  const brackets = policy.brackets.map((bracket) => bracket.label).filter(Boolean).join(", ");
  return `Auto Age Split: ${brackets || "candidate brackets"}; minimum ${policy.min_teams_per_age_group} per bracket; ${teamAgeRuleLabel(policy.team_age_rule)}`;
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

function standardSkillSpec(label: string): SkillSpec {
  const open = label.toLowerCase() === "open";
  return {
    key: `${open ? "open" : "standard"}-${label.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
    label,
    skillLabel: label,
    skillMode: open ? "OPEN" : "STANDARD",
    eligibilityMode: open ? "OPEN" : "STANDARD",
    skillMinRating: null,
    skillMaxRating: null,
    combinedRatingCap: null
  };
}

function minimumSkillSpec(label: string): SkillSpec {
  const threshold = Number(label.replace(/\+$/, ""));
  const formatted = Number.isFinite(threshold)
    ? threshold.toFixed(2).replace(/0+$/, "").replace(/\.$/, "")
    : label.replace(/\+$/, "");
  return {
    key: `minimum-${formatted.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`,
    label: `${formatted}+`,
    skillLabel: `${formatted}+`,
    skillMode: "MINIMUM",
    eligibilityMode: "MINIMUM",
    skillMinRating: Number.isFinite(threshold) ? threshold : null,
    skillMaxRating: null,
    combinedRatingCap: null
  };
}

function combinedSkillSpec(cap: number): SkillSpec {
  const formatted = Number(cap).toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  return {
    key: `combined-${formatted.replace(".", "-")}`,
    label: `Combined Below ${formatted}`,
    skillLabel: `Combined < ${formatted}`,
    skillMode: "COMBINED_RATING_CAP",
    eligibilityMode: "COMBINED_RATING_CAP",
    skillMinRating: null,
    skillMaxRating: null,
    combinedRatingCap: Number(cap)
  };
}

function buildProposal(
  family: BuilderRow,
  configuration: SetupConfiguration,
  skill: SkillSpec,
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
  const baseName = [genderName(gender), skill.label, age.label === "Event age policy" ? "" : age.label]
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
      skill_label: skill.skillLabel,
      skill_mode: skill.skillMode,
      eligibility_mode: skill.eligibilityMode,
      skill_min_rating: skill.skillMinRating,
      skill_max_rating: skill.skillMaxRating,
      combined_rating_cap: skill.combinedRatingCap,
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

function generateProposals({
  family,
  configuration,
  skillSpecs,
  genders,
  ageMode,
  familyAgePolicy
}: {
  family: BuilderRow;
  configuration: SetupConfiguration;
  skillSpecs: SkillSpec[];
  genders: GenderPreset[];
  ageMode: "INHERIT" | "BRACKETS";
  familyAgePolicy: AgePolicy;
}): Proposal[] {
  if (!skillSpecs.length || !genders.length) return [];
  const ages = ageMode === "BRACKETS" && familyAgePolicy.mode === "AUTO_AGE_SPLIT" && familyAgePolicy.brackets.length
    ? familyAgePolicy.brackets.map((bracket) => ({
        key: bracket.id,
        label: bracket.label,
        policy: fixedBracketPolicy(bracket.label, bracket.min_age, bracket.max_age, familyAgePolicy)
      }))
    : [{ key: "inherit", label: "Event age policy", policy: null }];
  const existingNames = new Set(
    configuration.eventOptions
      .filter((row) => eventFamilyName(row.value).toLowerCase() === eventFamilyName(family.value).toLowerCase())
      .map((row) => eventDivisionName(row.value).trim().toLowerCase())
      .filter(Boolean)
  );
  let index = 0;
  const rows: Proposal[] = [];
  for (const gender of genders) {
    for (const skill of skillSpecs) {
      for (const age of ages) {
        index += 1;
        const value = buildProposal(family, configuration, skill, gender, age, index);
        const duplicate = existingNames.has(eventDivisionName(value).toLowerCase());
        rows.push({
          key: `${gender}-${skill.key}-${age.key}`,
          selected: !duplicate,
          duplicate,
          issue: duplicate ? "A division with this name already exists in this Event." : undefined,
          value
        });
      }
    }
  }
  return rows;
}

export default function TournamentDivisionPresetDialog({
  open,
  family,
  configuration,
  onCancel,
  onConfirm,
  onAcknowledge
}: Props) {
  const [eligibilityMode, setEligibilityMode] = useState<EligibilityMode>("STANDARD");
  const [skills, setSkills] = useState<string[]>(["3.0", "3.5", "4.0", "4.5"]);
  const [customSkillEnabled, setCustomSkillEnabled] = useState(false);
  const [customSkill, setCustomSkill] = useState("");
  const [combinedCaps, setCombinedCaps] = useState<number[]>([7, 7.5, 8]);
  const [customCapEnabled, setCustomCapEnabled] = useState(false);
  const [customCap, setCustomCap] = useState("8.25");
  const [genders, setGenders] = useState<GenderPreset[]>([]);
  const [ageMode, setAgeMode] = useState<"INHERIT" | "BRACKETS">("INHERIT");
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [message, setMessage] = useState("");

  const selectedFamily = family;
  const familyAgePolicy = useMemo(
    () => readAgePolicy(selectedFamily?.value || {}, EVENT_AGE_POLICY_FIELDS),
    [selectedFamily]
  );
  const supportsAgeBrackets = familyAgePolicy.mode === "AUTO_AGE_SPLIT" && familyAgePolicy.brackets.length > 0;
  const canUseCombined = combinedRatingAllowed(selectedFamily?.value || {});

  const skillSpecs = useMemo<SkillSpec[]>(() => {
    if (eligibilityMode === "COMBINED_RATING_CAP") {
      const caps = [...combinedCaps];
      const parsedCustom = Number(customCap);
      if (customCapEnabled && Number.isFinite(parsedCustom) && parsedCustom > 0 && parsedCustom <= 14) caps.push(parsedCustom);
      return [...new Set(caps.map((value) => Number(value.toFixed(2))))].sort((a, b) => a - b).map(combinedSkillSpec);
    }
    const numericSkills = skills.filter((value) => value.toLowerCase() !== "open");
    const specs = eligibilityMode === "MINIMUM"
      ? numericSkills.map(minimumSkillSpec)
      : skills.map(standardSkillSpec);
    const custom = customSkill.trim();
    const customAnchor = parsedSkillAnchor(custom);
    if (customSkillEnabled && customAnchor != null) {
      const canonical = customAnchor.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
      if (skills.some((value) => value.toLowerCase() === canonical.toLowerCase())) return specs;
      specs.push(
        eligibilityMode === "MINIMUM"
          ? { ...minimumSkillSpec(canonical), key: `custom-minimum-${canonical.toLowerCase().replace(/[^a-z0-9]+/g, "-")}` }
          : { ...standardSkillSpec(canonical), key: `custom-${canonical.toLowerCase().replace(/[^a-z0-9]+/g, "-")}` }
      );
    }
    return specs;
  }, [eligibilityMode, skills, customSkillEnabled, customSkill, combinedCaps, customCapEnabled, customCap]);

  const generated = useMemo(() => {
    if (!open || !selectedFamily) return [];
    return generateProposals({
      family: selectedFamily,
      configuration,
      skillSpecs,
      genders,
      ageMode,
      familyAgePolicy
    });
  }, [open, selectedFamily, configuration, skillSpecs, genders, ageMode, familyAgePolicy]);

  useEffect(() => {
    if (!open || !selectedFamily) return;
    setEligibilityMode("STANDARD");
    setSkills(["3.0", "3.5", "4.0", "4.5"]);
    setCustomSkillEnabled(false);
    setCustomSkill("");
    setCombinedCaps([7, 7.5, 8]);
    setCustomCapEnabled(false);
    setCustomCap("8.25");
    setGenders(genderOptions(selectedFamily.value));
    setAgeMode("INHERIT");
    setProposals([]);
    setMessage("");
  }, [open, selectedFamily]);

  useEffect(() => {
    if (!open || !selectedFamily) return;
    setProposals((current) => {
      const prior = new Map<string, Proposal>(current.map((row) => [row.key, row] as const));
      return generated.map((row) => {
        const existing = prior.get(row.key);
        if (!existing || row.duplicate) return row;
        return {
          ...row,
          selected: existing.selected,
          value: {
            ...row.value,
            division_name: existing.value.division_name,
            label: existing.value.label
          }
        };
      });
    });
    if (!generated.length) {
      setMessage(eligibilityMode === "COMBINED_RATING_CAP" ? "Choose at least one combined-rating cap." : "Choose at least one skill level.");
    } else if (generated.some((row) => row.duplicate)) {
      setMessage("Existing divisions were detected and left unselected.");
    } else {
      setMessage("Proposed divisions update automatically as choices change.");
    }
  }, [generated, open, selectedFamily, eligibilityMode]);

  if (!open || !selectedFamily) return null;
  const activeFamily = selectedFamily;
  const defaultGenders = genderOptions(activeFamily.value);
  const presetDirty = eligibilityMode !== "STANDARD"
    || JSON.stringify(skills) !== JSON.stringify(["3.0", "3.5", "4.0", "4.5"])
    || customSkillEnabled
    || customSkill !== ""
    || JSON.stringify(combinedCaps) !== JSON.stringify([7, 7.5, 8])
    || customCapEnabled
    || customCap !== "8.25"
    || JSON.stringify(genders) !== JSON.stringify(defaultGenders)
    || ageMode !== "INHERIT"
    || proposals.some((row) => {
      const generatedRow = generated.find((candidate) => candidate.key === row.key);
      return !generatedRow
        || row.selected !== generatedRow.selected
        || eventDivisionName(row.value) !== eventDivisionName(generatedRow.value);
    });

  function toggleSkill(skill: string) {
    setSkills((current) => current.includes(skill) ? current.filter((value) => value !== skill) : [...current, skill]);
  }

  function toggleCombinedCap(cap: number) {
    setCombinedCaps((current) => current.includes(cap) ? current.filter((value) => value !== cap) : [...current, cap]);
  }

  function toggleGender(gender: GenderPreset) {
    setGenders((current) => current.includes(gender) ? current.filter((value) => value !== gender) : [...current, gender]);
  }

  async function submit(): Promise<ActionCompletion> {
    if (customSkillEnabled && parsedSkillAnchor(customSkill) == null) {
      throw new InteractionActionError("Custom skill must be a numeric rating between 1.0 and 7.0.", { kind: "validation", fieldErrors: { "Custom skill": "Enter a numeric rating between 1.0 and 7.0." } });
    }
    const selectedProposals = proposals.filter((row) => row.selected);
    if (!selectedProposals.length) {
      throw new InteractionActionError("Choose at least one new division.", { kind: "validation", fieldErrors: { "Proposed divisions": "Select at least one new division." } });
    }

    const familyName = eventFamilyName(activeFamily.value).toLowerCase();
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
      throw new InteractionActionError("Resolve the highlighted division names, then select them again.", { kind: "validation", fieldErrors: { "Division names": "Every selected division must have a unique name." } });
    }
    return onConfirm(selectedProposals.map((row) => row.value));
  }

  return (
    <FormDialog open={open} mode="bulk" size="xwide" title={`Generate divisions for ${eventFamilyName(activeFamily.value)}`} description="Select common eligibility, gender, and age combinations once. The preview updates automatically, existing divisions are detected, and generated rows inherit Event defaults." dirty={presetDirty} submitLabel="Save selected divisions" submitDisabled={!proposals.some((row) => row.selected && !row.duplicate)} workingLabel="Saving divisions…" onSubmit={submit} onCancel={onCancel} onAcknowledge={onAcknowledge}>

        <fieldset style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.75rem", marginBottom: "0.85rem" }}>
          <legend style={{ fontWeight: 800 }}>Skill eligibility</legend>
          <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
            <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
              <input type="radio" checked={eligibilityMode === "STANDARD"} onChange={() => setEligibilityMode("STANDARD")} />
              Standard skill ceilings
            </label>
            <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
              <input type="radio" checked={eligibilityMode === "MINIMUM"} onChange={() => setEligibilityMode("MINIMUM")} />
              Minimum skill / Skill+
            </label>
            {canUseCombined ? (
              <label style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                <input type="radio" checked={eligibilityMode === "COMBINED_RATING_CAP"} onChange={() => setEligibilityMode("COMBINED_RATING_CAP")} />
                Combined team rating
              </label>
            ) : null}
          </div>
          {eligibilityMode !== "COMBINED_RATING_CAP" ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: "0.45rem", marginTop: "0.65rem" }}>
              {SKILL_PRESETS.map((skill) => (
                <label key={skill} style={{ display: "flex", gap: "0.5rem" }}>
                  <input type="checkbox" checked={skills.includes(skill)} onChange={() => toggleSkill(skill)} />
                  {skill}
                </label>
              ))}
              <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", gridColumn: "1 / -1" }}>
                <input type="checkbox" checked={customSkillEnabled} onChange={(event) => setCustomSkillEnabled(event.target.checked)} />
                Custom skill division
                <input value={customSkill} disabled={!customSkillEnabled} placeholder="Example: 3.75" style={{ ...inputStyle, maxWidth: "240px" }} onChange={(event) => setCustomSkill(event.target.value)} />
              </label>
              <small style={{ gridColumn: "1 / -1", color: "#64748b" }}>{eligibilityMode === "MINIMUM" ? "Each selected level becomes a true minimum with no upper ceiling." : "Numeric levels use the standard half-step ceiling; Open has no rating restriction."}</small>
            </div>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(170px, 1fr))", gap: "0.45rem", marginTop: "0.65rem" }}>
              {COMBINED_CAP_PRESETS.map((cap) => (
                <label key={cap} style={{ display: "flex", gap: "0.5rem" }}>
                  <input type="checkbox" checked={combinedCaps.includes(cap)} onChange={() => toggleCombinedCap(cap)} />
                  Below {cap.toFixed(1)}
                </label>
              ))}
              <label style={{ display: "flex", gap: "0.5rem", alignItems: "center", gridColumn: "1 / -1" }}>
                <input type="checkbox" checked={customCapEnabled} onChange={(event) => setCustomCapEnabled(event.target.checked)} />
                Custom combined cap
                <input type="number" min="0.01" max="14" step="0.01" value={customCap} disabled={!customCapEnabled} style={{ ...inputStyle, maxWidth: "160px" }} onChange={(event) => setCustomCap(event.target.value)} />
              </label>
              <small style={{ gridColumn: "1 / -1", color: "#64748b" }}>Teams must have a combined rating strictly below the selected cap. This option is unavailable for Singles and Four-player team Events.</small>
            </div>
          )}
        </fieldset>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "0.85rem" }}>
          <fieldset style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.75rem" }}>
            <legend style={{ fontWeight: 800 }}>Gender categories</legend>
            {genderOptions(activeFamily.value).map((gender) => (
              <label key={gender} style={{ display: "flex", gap: "0.5rem", marginTop: "0.35rem" }}>
                <input type="checkbox" checked={genders.includes(gender)} onChange={() => toggleGender(gender)} />
                {titleCase(gender)}
              </label>
            ))}
          </fieldset>
          <fieldset style={{ border: "1px solid #cbd5e1", borderRadius: "12px", padding: "0.75rem" }}>
            <legend style={{ fontWeight: 800 }}>Age handling</legend>
            <label style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start" }}>
              <input type="radio" checked={ageMode === "INHERIT"} onChange={() => setAgeMode("INHERIT")} />
              <span><strong>Inherit Event age policy</strong><br /><small>{resolvedAgePolicySummary(familyAgePolicy)}</small></span>
            </label>
            {supportsAgeBrackets ? (
              <label style={{ display: "flex", gap: "0.5rem", alignItems: "flex-start", marginTop: "0.55rem" }}>
                <input type="radio" checked={ageMode === "BRACKETS"} onChange={() => setAgeMode("BRACKETS")} />
                <span><strong>Create one division per candidate age bracket</strong><br /><small>{familyAgePolicy.brackets.map((bracket) => bracket.label).join(", ")}</small></span>
              </label>
            ) : null}
          </fieldset>
        </div>

        <div style={{ marginTop: "0.95rem" }}>
          <h3 style={{ marginBottom: "0.35rem" }}>Proposed divisions</h3>
          <p style={{ marginTop: 0, color: "#64748b" }}>This preview refreshes automatically whenever eligibility, gender, or age handling changes.</p>
          {proposals.length ? (
            <div style={{ display: "grid", gap: "0.65rem" }}>
              {proposals.map((proposal, index) => (
                <article key={proposal.key} style={{ border: `1px solid ${proposal.duplicate ? "#fde68a" : "#e2e8f0"}`, borderRadius: "12px", padding: "0.7rem", background: proposal.duplicate ? "#fffbeb" : "#f8fafc" }}>
                  <div style={{ display: "grid", gridTemplateColumns: "auto minmax(220px, 2fr) minmax(140px, 1fr) minmax(120px, 1fr)", gap: "0.6rem", alignItems: "end" }}>
                    <input type="checkbox" aria-label={`Select proposal ${index + 1}`} checked={proposal.selected} disabled={proposal.duplicate} onChange={(event) => setProposals((current) => current.map((row) => row.key === proposal.key ? { ...row, selected: event.target.checked } : row))} />
                    <label><strong>Division name</strong><br /><input value={eventDivisionName(proposal.value)} style={inputStyle} onChange={(event) => setProposals((current) => current.map((row) => row.key === proposal.key ? {
                      ...row,
                      duplicate: false,
                      issue: undefined,
                      value: { ...row.value, division_name: event.target.value, label: event.target.value }
                    } : row))} /></label>
                    <div><strong>Eligibility</strong><br />{skillEligibilityLabel(proposal.value)}<br /><small>{skillEligibilitySummary(proposal.value)}</small></div>
                    <div><strong>Gender</strong><br />{titleCase(cleanString(proposal.value.gender_restriction))}</div>
                  </div>
                  <small style={{ color: proposal.duplicate ? "#92400e" : "#64748b" }}>
                    {proposal.issue || `${cleanString(proposal.value.age_label) || "Event age policy"} · $${Number(proposal.value.price_usd || 0).toFixed(2)} · capacity ${Number(proposal.value.capacity_teams || 0)}`}
                  </small>
                </article>
              ))}
            </div>
          ) : <p role="status" style={{ color: "#92400e" }}>Choose at least one valid eligibility and gender combination.</p>}
        </div>

        {message ? <p role="status" style={{ color: /required|choose|resolve/i.test(message) ? "#b91c1c" : "#475569" }}>{message}</p> : null}
    </FormDialog>
  );
}
