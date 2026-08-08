export type TournamentSkillEligibilityMode =
  | "STANDARD"
  | "MINIMUM"
  | "OPEN"
  | "COMBINED_RATING_CAP"
  | "CUSTOM";

export type SkillEligibilityMode = TournamentSkillEligibilityMode;
export type TournamentSkillEligibilityRecord = Record<string, unknown>;

export type TournamentSkillEligibilityPolicy = {
  mode: TournamentSkillEligibilityMode;
  minimum: number | null;
  maximumExclusive: number | null;
  combinedCap: number | null;
};

function clean(value: unknown): string {
  return value == null ? "" : String(value).trim();
}

function finite(value: unknown): number | null {
  if (value == null || value === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function rounded(value: number): number {
  return Math.round(value * 100) / 100;
}

export function parsedSkillAnchor(value: unknown): number | null {
  const match = clean(value).match(/^(?:skill\s*)?([0-9](?:\.[0-9]{1,2})?)\s*(\+)?$/i);
  if (!match) return null;
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) && parsed >= 1 && parsed <= 7 ? rounded(parsed) : null;
}

export const skillAnchor = parsedSkillAnchor;

export function formatSkillNumber(value: unknown): string {
  const parsed = finite(value);
  if (parsed == null) return "";
  return parsed.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

export function resolveTournamentSkillEligibilityMode(
  value: TournamentSkillEligibilityRecord
): TournamentSkillEligibilityMode {
  const raw = clean(value.eligibility_mode).toUpperCase();
  if (["MINIMUM", "OPEN", "COMBINED_RATING_CAP", "CUSTOM"].includes(raw)) {
    return raw as TournamentSkillEligibilityMode;
  }
  const skillMode = clean(value.skill_mode).toUpperCase();
  if (["MINIMUM", "MIN", "AT_LEAST"].includes(skillMode)) {
    return "MINIMUM";
  }
  // A newly saved explicit Standard policy must win over stale display text
  // such as "Open". Legacy rows used skill_mode=OPEN/NONE instead.
  if (["STANDARD", "SKILL_BRACKET", "CEILING", "MAXIMUM"].includes(skillMode)) {
    return "STANDARD";
  }
  const label = clean(value.skill_label);
  if (label.toLowerCase() === "open") {
    return "OPEN";
  }
  // Preserve the legacy meaning of a bare trailing plus. The new lower-bound
  // behavior is activated by explicit MINIMUM metadata, not label punctuation.
  if (label.endsWith("+")) return "OPEN";
  // Legacy numeric rows sometimes retained skill_mode=OPEN. A numeric label
  // on an explicit STANDARD row remains the authoritative ceiling.
  if (raw === "STANDARD" && parsedSkillAnchor(label) != null) return "STANDARD";
  if (["OPEN", "NONE"].includes(skillMode)) return "OPEN";
  if (skillMode === "COMBINED_RATING_CAP") return "COMBINED_RATING_CAP";
  if (skillMode === "CUSTOM") return "CUSTOM";
  if (raw === "STANDARD") return "STANDARD";
  return "STANDARD";
}

export const skillEligibilityMode = resolveTournamentSkillEligibilityMode;

function explicitMaximum(value: TournamentSkillEligibilityRecord): number | null {
  return finite(
    value.skill_max_rating ??
      value.skill_max_rating_exclusive ??
      value.skill_ceiling_exclusive
  );
}

export function standardSkillCeiling(value: TournamentSkillEligibilityRecord): number | null {
  const explicit = explicitMaximum(value);
  if (explicit != null) return rounded(explicit);
  const anchor = parsedSkillAnchor(value.skill_label);
  return anchor == null ? null : rounded(anchor + 0.5);
}

export function minimumSkillRating(value: TournamentSkillEligibilityRecord): number | null {
  const explicit = finite(value.skill_min_rating);
  if (explicit != null) return rounded(explicit);
  return parsedSkillAnchor(value.skill_label);
}

export function skillEligibilityPolicy(
  value: TournamentSkillEligibilityRecord
): TournamentSkillEligibilityPolicy {
  const mode = resolveTournamentSkillEligibilityMode(value);
  return {
    mode,
    minimum:
      mode === "MINIMUM"
        ? minimumSkillRating(value)
        : mode === "CUSTOM"
          ? finite(value.skill_min_rating)
          : null,
    maximumExclusive:
      mode === "STANDARD"
        ? standardSkillCeiling(value)
        : mode === "CUSTOM"
          ? explicitMaximum(value)
          : null,
    combinedCap: mode === "COMBINED_RATING_CAP" ? finite(value.combined_rating_cap) : null
  };
}

export function normalizeTournamentSkillEligibility(
  value: TournamentSkillEligibilityRecord,
  nextMode = resolveTournamentSkillEligibilityMode(value)
): TournamentSkillEligibilityRecord {
  const next: TournamentSkillEligibilityRecord = { ...value };
  const anchor = parsedSkillAnchor(value.skill_label);

  if (nextMode === "STANDARD") {
    const resolvedAnchor = anchor ?? 3.5;
    next.eligibility_mode = "STANDARD";
    next.skill_mode = "SKILL_BRACKET";
    next.skill_label = formatSkillNumber(resolvedAnchor);
    next.skill_min_rating = null;
    next.skill_max_rating = null;
    next.combined_rating_cap = null;
    return next;
  }
  if (nextMode === "MINIMUM") {
    const minimum = finite(value.skill_min_rating) ?? anchor ?? 3.5;
    next.eligibility_mode = "MINIMUM";
    next.skill_mode = "MINIMUM";
    next.skill_min_rating = rounded(minimum);
    next.skill_max_rating = null;
    next.combined_rating_cap = null;
    next.skill_label = `${formatSkillNumber(minimum)}+`;
    return next;
  }
  if (nextMode === "OPEN") {
    next.eligibility_mode = "OPEN";
    next.skill_mode = "OPEN";
    next.skill_label = "Open";
    next.skill_min_rating = null;
    next.skill_max_rating = null;
    next.combined_rating_cap = null;
    return next;
  }
  if (nextMode === "COMBINED_RATING_CAP") {
    const cap = finite(value.combined_rating_cap) ?? 8;
    next.eligibility_mode = "COMBINED_RATING_CAP";
    next.skill_mode = "COMBINED_RATING_CAP";
    next.combined_rating_cap = rounded(cap);
    next.skill_min_rating = null;
    next.skill_max_rating = null;
    next.skill_label = `Combined < ${formatSkillNumber(cap)}`;
    return next;
  }

  const minimum = finite(value.skill_min_rating);
  const maximum = explicitMaximum(value);
  next.eligibility_mode = "CUSTOM";
  next.skill_mode = "CUSTOM";
  next.skill_min_rating = minimum == null ? null : rounded(minimum);
  next.skill_max_rating = maximum == null ? null : rounded(maximum);
  next.combined_rating_cap = null;
  next.skill_label = clean(value.skill_label) && clean(value.skill_label).toLowerCase() !== "open"
    ? clean(value.skill_label)
    : "Custom";
  return next;
}

export function validateTournamentSkillEligibility(
  value: TournamentSkillEligibilityRecord
): string | null {
  const policy = skillEligibilityPolicy(value);
  if (policy.mode === "STANDARD") {
    const anchor = parsedSkillAnchor(value.skill_label);
    if (anchor == null || policy.maximumExclusive == null || policy.maximumExclusive <= anchor || policy.maximumExclusive > 7.5) {
      return "Choose a standard skill level such as 3.5. The upper ceiling is the next half-step.";
    }
    return null;
  }
  if (policy.mode === "MINIMUM") {
    if (policy.minimum == null || policy.minimum < 1 || policy.minimum > 7) {
      return "Minimum / Skill+ requires a rating threshold between 1.0 and 7.0.";
    }
    return null;
  }
  if (policy.mode === "OPEN") return null;
  if (policy.mode === "COMBINED_RATING_CAP") {
    if (policy.combinedCap == null || policy.combinedCap <= 0 || policy.combinedCap > 14) {
      return "Combined team-rating cap must be greater than 0 and no more than 14.";
    }
    return null;
  }
  if (policy.minimum == null && policy.maximumExclusive == null) {
    return "Custom eligibility requires a minimum, a maximum, or both.";
  }
  if (policy.minimum != null && (policy.minimum < 1 || policy.minimum > 7)) {
    return "Custom minimum rating must be between 1.0 and 7.0.";
  }
  if (policy.maximumExclusive != null && (policy.maximumExclusive <= 1 || policy.maximumExclusive > 7.5)) {
    return "Custom maximum rating must be greater than 1.0 and no more than 7.5.";
  }
  if (policy.minimum != null && policy.maximumExclusive != null && policy.maximumExclusive <= policy.minimum) {
    return "Custom maximum rating must be greater than the minimum rating.";
  }
  return null;
}

export function skillEligibilityLabel(value: TournamentSkillEligibilityRecord): string {
  const policy = skillEligibilityPolicy(value);
  if (policy.mode === "OPEN") return "Open";
  if (policy.mode === "MINIMUM") return policy.minimum == null ? "Minimum / Skill+" : `${formatSkillNumber(policy.minimum)}+`;
  if (policy.mode === "COMBINED_RATING_CAP") {
    return policy.combinedCap == null ? "Combined rating" : `Combined < ${formatSkillNumber(policy.combinedCap)}`;
  }
  if (policy.mode === "CUSTOM") {
    if (policy.minimum != null && policy.maximumExclusive != null) {
      return `${formatSkillNumber(policy.minimum)}–<${formatSkillNumber(policy.maximumExclusive)}`;
    }
    if (policy.minimum != null) return `${formatSkillNumber(policy.minimum)}+`;
    if (policy.maximumExclusive != null) return `<${formatSkillNumber(policy.maximumExclusive)}`;
    return "Custom";
  }
  return clean(value.skill_label) || "Standard";
}

export function tournamentSkillEligibilitySummary(value: TournamentSkillEligibilityRecord): string {
  const policy = skillEligibilityPolicy(value);
  if (policy.mode === "OPEN") return "Open — no rating restriction";
  if (policy.mode === "MINIMUM") {
    return policy.minimum == null
      ? "Minimum / Skill+ — threshold not set"
      : `${formatSkillNumber(policy.minimum)}+ Minimum · controlling rating must be ${formatSkillNumber(policy.minimum)} or higher; no upper ceiling`;
  }
  if (policy.mode === "COMBINED_RATING_CAP") {
    return policy.combinedCap == null
      ? "Combined team rating — cap not set"
      : `Combined team rating below ${formatSkillNumber(policy.combinedCap)} (exclusive)`;
  }
  if (policy.mode === "CUSTOM") {
    if (policy.minimum != null && policy.maximumExclusive != null) {
      return `Custom · controlling rating ${formatSkillNumber(policy.minimum)} or higher and below ${formatSkillNumber(policy.maximumExclusive)}`;
    }
    if (policy.minimum != null) return `Custom · controlling rating ${formatSkillNumber(policy.minimum)} or higher`;
    if (policy.maximumExclusive != null) return `Custom · controlling rating below ${formatSkillNumber(policy.maximumExclusive)}`;
    return "Custom rating boundaries — not set";
  }
  const anchor = parsedSkillAnchor(value.skill_label);
  return anchor == null || policy.maximumExclusive == null
    ? "Standard skill ceiling — choose a skill level"
    : `${formatSkillNumber(anchor)} Standard · controlling rating must be below ${formatSkillNumber(policy.maximumExclusive)}; lower-rated players may play up`;
}

export const skillEligibilitySummary = tournamentSkillEligibilitySummary;
