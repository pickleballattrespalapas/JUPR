import type { PublicRegistrationEvent } from "@/lib/tournamentRegistrationApi";
import { skillEligibilityPolicy } from "@/lib/tournamentSkillEligibility";

const eventFormatLabels: Record<string, string> = {
  ROUND_ROBIN: "Round robin",
  SINGLE_ELIM: "Single elimination",
  DOUBLE_ELIM: "Double elimination",
  ROUND_ROBIN_PLUS_PLAYOFF: "Round robin and playoffs"
};

const scoringLabels: Record<string, string> = {
  GAME_TO_11: "Game to 11",
  ONE_GAME_TO_11: "Game to 11",
  GAME_TO_15: "Game to 15",
  ONE_GAME_TO_15: "Game to 15",
  GAME_TO_21: "Game to 21",
  ONE_GAME_TO_21: "Game to 21",
  BEST_2_OF_3: "Best two of three",
  BEST_OF_3: "Best two of three"
};

function metadataLabel(
  value: string | null | undefined,
  labels: Record<string, string>
): string | null {
  const text = String(value || "").trim();
  if (!text) return null;
  const key = text.toUpperCase().replace(/[\s-]+/g, "_");
  if (labels[key]) return labels[key];
  if (text.includes("_") || (/[A-Z]/.test(text) && text === text.toUpperCase())) {
    const words = key.toLowerCase().replace(/_+/g, " ").trim();
    return words ? `${words.charAt(0).toUpperCase()}${words.slice(1)}` : null;
  }
  return text;
}

export function publicEventFormatLabel(value?: string | null): string | null {
  return metadataLabel(value, eventFormatLabels);
}

export function publicScoringLabel(value?: string | null): string | null {
  return metadataLabel(value, scoringLabels);
}

export function publicEventCapacityLabel(
  event: PublicRegistrationEvent
): string | null {
  if (event.capacity_teams == null) return null;
  const capacity = Number(event.capacity_teams);
  const singles =
    String(event.event_type || "").toUpperCase() === "SINGLES" &&
    !event.partner_required &&
    String(event.competition_format || "").toUpperCase() !== "FOUR_PLAYER_TEAM";
  const unit = singles ? "player" : "team";
  return `Capacity: ${capacity} ${unit}${capacity === 1 ? "" : "s"}`;
}

export function formatRegistrationRating(value: number): string {
  return Number(value).toFixed(2).replace(/0$/, "");
}

export function formatPublicTournamentDate(
  value?: string | null
): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeZone: "UTC"
  }).format(date);
}

export function publicTournamentDayLabel(
  labelValue: string,
  dateValue?: string | null
): string {
  const label = String(labelValue || "Day").trim() || "Day";
  const formattedDate = formatPublicTournamentDate(dateValue);
  if (!formattedDate || !dateValue) return label;
  const date = new Date(dateValue);
  const markers = [
    String(dateValue).slice(0, 10),
    new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      timeZone: "UTC"
    }).format(date),
    new Intl.DateTimeFormat("en-US", {
      month: "long",
      day: "numeric",
      timeZone: "UTC"
    }).format(date),
    new Intl.DateTimeFormat("en-US", {
      month: "numeric",
      day: "numeric",
      timeZone: "UTC"
    }).format(date)
  ].map((value) => value.toLowerCase());
  const dayNumber = label.match(/^day\s+\d+/i)?.[0];
  if (dayNumber) {
    const suffix = label.slice(dayNumber.length).toLowerCase();
    if (!suffix.trim() || markers.some((marker) => suffix.includes(marker))) {
      return `${dayNumber} — ${formattedDate}`;
    }
  }
  return `${label} — ${formattedDate}`;
}

export function publicTournamentEventLabel(
  familyValue?: string | null,
  divisionValue?: string | null
): string {
  const family = String(familyValue || "").trim();
  const division = String(divisionValue || "").trim();
  if (!family) return division || "Event";
  if (!division) return family;
  const normalize = (value: string) =>
    value.toLowerCase().replaceAll("’", "'").replace(/\s+/g, " ").trim();
  const familyKey = normalize(family);
  const divisionKey = normalize(division);
  if (
    divisionKey === familyKey ||
    divisionKey.startsWith(`${familyKey} `) ||
    divisionKey.startsWith(`${familyKey} -`) ||
    divisionKey.startsWith(`${familyKey} —`)
  ) {
    return division;
  }
  return `${family} — ${division}`;
}

export type RegistrationEligibilityProfile = {
  gender?: string | null;
  age?: number | null;
  doublesSkill?: number | null;
  singlesSkill?: number | null;
};

function normalizedGender(value?: string | null): "MEN" | "WOMEN" | "OTHER" | "" {
  const text = String(value || "").toLowerCase().replace(/[^a-z]/g, "");
  if (["m", "male", "man", "men", "mens", "boy", "boys"].includes(text)) return "MEN";
  if (["f", "female", "woman", "women", "womens", "girl", "girls"].includes(text)) return "WOMEN";
  return text ? "OTHER" : "";
}

function finiteNumber(value?: number | null): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function ageRules(event: PublicRegistrationEvent): Record<string, unknown> {
  if (event.age_rules && typeof event.age_rules === "object") return event.age_rules;
  if (typeof event.age_rules === "string" && event.age_rules.trim()) {
    try {
      const parsed = JSON.parse(event.age_rules);
      return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed as Record<string, unknown> : {};
    } catch {
      return {};
    }
  }
  return {};
}

function hardMinimumAge(event: PublicRegistrationEvent): number | null {
  const rules = ageRules(event);
  const mode = String(event.age_mode || rules.mode || "ALL_AGES").trim().toUpperCase();
  if (["", "OPEN", "ALL", "ALL AGES", "ALL_AGES", "AUTO_AGE_SPLIT"].includes(mode)) {
    // Auto groups normally include a youngest/open group. If an organizer
    // intentionally omits one, use the lowest configured minimum.
    if (mode !== "AUTO_AGE_SPLIT") return null;
  }
  const rows = Array.isArray(rules.brackets) ? rules.brackets.filter((row): row is Record<string, unknown> => Boolean(row && typeof row === "object")) : [];
  const rawMinimum = rules.min_age ?? rows[0]?.min_age;
  const minimum = Number(rawMinimum);
  if (rawMinimum == null || rawMinimum === "" || !Number.isFinite(minimum)) return null;
  return minimum;
}

export function publicEventEligibilityReason(
  event: PublicRegistrationEvent,
  profile: RegistrationEligibilityProfile
): string | null {
  const restriction = String(event.gender_restriction || "ANY").trim().toUpperCase();
  const gender = normalizedGender(profile.gender);
  if (["MEN", "MALE"].includes(restriction) && gender !== "MEN") return "This division is limited to men's registrations.";
  if (["WOMEN", "FEMALE"].includes(restriction) && gender !== "WOMEN") return "This division is limited to women's registrations.";
  if (restriction === "MIXED" && !["MEN", "WOMEN"].includes(gender)) return "Select an eligible gender for mixed doubles.";

  const minimumAge = hardMinimumAge(event);
  const age = finiteNumber(profile.age);
  if (minimumAge != null && age != null && age < minimumAge) {
    return `Age ${age.toFixed(0)} does not meet this division's minimum age of ${minimumAge.toFixed(0)}.`;
  }

  const policy = skillEligibilityPolicy(event);
  if (policy.mode === "OPEN") return null;
  const eventType = String(event.event_type || "").toUpperCase();
  const isDoubles = Boolean(event.partner_required) || ["DOUBLES", "GENDER_DOUBLES", "MIXED_DOUBLES", "MIXED"].includes(eventType);
  const rating = isDoubles
    ? finiteNumber(profile.doublesSkill)
    : finiteNumber(profile.singlesSkill);
  if (rating == null) return null;
  if (
    policy.mode === "COMBINED_RATING_CAP" &&
    policy.combinedCap != null &&
    rating >= policy.combinedCap
  ) {
    return `Your rating is ${formatRegistrationRating(rating)}. This division requires a combined team rating below ${formatRegistrationRating(policy.combinedCap)}.`;
  }
  // Doubles lower bounds use the team's higher rating. The partner is chosen
  // after the event, so a lower-rated primary player is not conclusive here.
  if (!isDoubles && policy.minimum != null && rating < policy.minimum) {
    return `Rating ${rating.toFixed(2)} does not meet this division's minimum of ${policy.minimum.toFixed(2)}.`;
  }
  if (policy.maximumExclusive != null && rating >= policy.maximumExclusive) {
    const recommended = Math.floor(rating * 2) / 2;
    return `Rating ${rating.toFixed(2)} is above this division cap. Choose ${recommended.toFixed(1)} or higher.`;
  }
  return null;
}

export function publicEventFamilyKey(event: PublicRegistrationEvent): string {
  return `${event.registration_day_id}::${event.event_family_label.trim().toLowerCase().replace(/\s+/g, " ")}`;
}
