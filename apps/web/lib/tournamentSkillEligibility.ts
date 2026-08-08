export type TournamentSkillEligibilityMode =
  | "STANDARD"
  | "MINIMUM"
  | "OPEN"
  | "COMBINED_RATING_CAP"
  | "CUSTOM";

export type TournamentSkillEligibilityRecord = Record<string, unknown>;

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

export function formatSkillNumber(value: unknown): string {
  const parsed = finite(value);
  if (parsed == null) return "";
  return parsed.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

export function resolveTournamentSkillEligibilityMode(
  value: TournamentSkillEligibilityRecord
): TournamentSkillEligibilityMode {
  const raw = clean(value.eligibility_mode).toUpperCase();
  if (
    ["STANDARD", "MINIMUM", "OPEN", "COMBINED_RATING_CAP", "CUSTOM"].includes(raw)
  ) {
    return raw as TournamentSkillEligibilityMode;
  }
  const skillMode = clean(value.skill_mode).toUpperCase();
  if (["MINIMUM", "MIN", "AT_LEAST"].includes(skillMode)) return "MINIMUM";
  if (["OPEN", "NONE"].includes(skillMode)) return "OPEN";
  return "STANDARD";
}

export function standardSkillCeiling(value: TournamentSkillEligibilityRecord): number | null {
  const explicit = finite(value.skill_max_rating_exclusive);
  if (explicit != null) return rounded(explicit);
  const anchor = parsedSkillAnchor(value.skill_label);
  return anchor == null ? null : rounded(anchor + 0.5);
}

export function minimumSkillRating(value: TournamentSkillEligibilityRecord): number | null {
  const explicit = finite(value.skill_min_rating);
  if (explicit != null) return rounded(explicit);
  return parsedSkillAnchor(value.skill_label);
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
    next.skill_max_rating_exclusive = rounded(resolvedAnchor + 0.5);
    next.combined_rating_cap = null;
    return next;
  }

  if (nextMode === "MINIMUM") {
    const minimum = finite(value.skill_min_rating) ?? anchor ?? 3.5;
    next.eligibility_mode = "MINIMUM";
    next.skill_mode = "MINIMUM";
    next.skill_min_rating = rounded(minimum);
    next.skill_max_rating_exclusive = null;
    next.combined_rating_cap = null;
    next.skill_label = `${formatSkillNumber(minimum)}+`;
    return next;
  }

  if (nextMode === "OPEN") {
    next.eligibility_mode = "OPEN";
    next.skill_mode = "OPEN";
    next.skill_label = "Open";
    next.skill_min_rating = null;
    next.skill_max_rating_exclusive = null;
    next.combined_rating_cap = null;
    return next;
  }

  if (nextMode === "COMBINED_RATING_CAP") {
    const cap = finite(value.combined_rating_cap) ?? 8;
    next.eligibility_mode = "COMBINED_RATING_CAP";
    next.skill_mode = "COMBINED_RATING_CAP";
    next.combined_rating_cap = rounded(cap);
    next.skill_min_rating = null;
    next.skill_max_rating_exclusive = null;
    next.skill_label = `Combined < ${formatSkillNumber(cap)}`;
    return next;
  }

  const minimum = finite(value.skill_min_rating);
  const maximum = finite(value.skill_max_rating_exclusive);
  next.eligibility_mode = "CUSTOM";
  next.skill_mode = "CUSTOM";
  next.skill_min_rating = minimum == null ? null : rounded(minimum);
  next.skill_max_rating_exclusive = maximum == null ? null : rounded(maximum);
  next.combined_rating_cap = null;
  next.skill_label = clean(value.skill_label) && clean(value.skill_label).toLowerCase() !== "open"
    ? clean(value.skill_label)
    : "Custom";
  return next;
}

export function validateTournamentSkillEligibility(
  value: TournamentSkillEligibilityRecord
): string | null {
  const mode = resolveTournamentSkillEligibilityMode(value);
  if (mode === "STANDARD") {
    const anchor = parsedSkillAnchor(value.skill_label);
    const ceiling = standardSkillCeiling(value);
    if (anchor == null || ceiling == null || ceiling <= anchor || ceiling > 7.5) {
      return "Choose a standard skill level such as 3.5. The upper ceiling is the next half-step.";
    }
    return null;
  }
  if (mode === "MINIMUM") {
    const minimum = minimumSkillRating(value);
    if (minimum == null || minimum < 1 || minimum > 7) {
      return "Minimum / Skill+ requires a rating threshold between 1.0 and 7.0.";
    }
    return null;
  }
  if (mode === "OPEN") return null;
  if (mode === "COMBINED_RATING_CAP") {
    const cap = finite(value.combined_rating_cap);
    if (cap == null || cap <= 0 || cap > 14) {
      return "Combined team-rating cap must be greater than 0 and no more than 14.";
    }
    return null;
  }
  const minimum = finite(value.skill_min_rating);
  const maximum = finite(value.skill_max_rating_exclusive);
  if (minimum == null && maximum == null) {
    return "Custom eligibility requires a minimum, a maximum, or both.";
  }
  if (minimum != null && (minimum < 1 || minimum > 7)) {
    return "Custom minimum rating must be between 1.0 and 7.0.";
  }
  if (maximum != null && (maximum <= 1 || maximum > 7.5)) {
    return "Custom maximum rating must be greater than 1.0 and no more than 7.5.";
  }
  if (minimum != null && maximum != null && maximum <= minimum) {
    return "Custom maximum rating must be greater than the minimum rating.";
  }
  return null;
}

export function tournamentSkillEligibilitySummary(
  value: TournamentSkillEligibilityRecord
): string {
  const mode = resolveTournamentSkillEligibilityMode(value);
  if (mode === "OPEN") return "Open — no rating restriction";
  if (mode === "MINIMUM") {
    const minimum = minimumSkillRating(value);
    return minimum == null
      ? "Minimum / Skill+ — threshold not set"
      : `${formatSkillNumber(minimum)}+ Minimum · controlling rating must be ${formatSkillNumber(minimum)} or higher; no upper ceiling`;
  }
  if (mode === "COMBINED_RATING_CAP") {
    const cap = finite(value.combined_rating_cap);
    return cap == null
      ? "Combined team rating — cap not set"
      : `Combined team rating below ${formatSkillNumber(cap)} (exclusive)`;
  }
  if (mode === "CUSTOM") {
    const minimum = finite(value.skill_min_rating);
    const maximum = finite(value.skill_max_rating_exclusive);
    if (minimum != null && maximum != null) {
      return `Custom · controlling rating ${formatSkillNumber(minimum)} or higher and below ${formatSkillNumber(maximum)}`;
    }
    if (minimum != null) return `Custom · controlling rating ${formatSkillNumber(minimum)} or higher`;
    if (maximum != null) return `Custom · controlling rating below ${formatSkillNumber(maximum)}`;
    return "Custom rating boundaries — not set";
  }
  const anchor = parsedSkillAnchor(value.skill_label);
  const ceiling = standardSkillCeiling(value);
  return anchor == null || ceiling == null
    ? "Standard skill ceiling — choose a skill level"
    : `${formatSkillNumber(anchor)} Standard · controlling rating must be below ${formatSkillNumber(ceiling)}; lower-rated players may play up`;
}
