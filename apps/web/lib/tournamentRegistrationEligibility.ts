import { PublicRegistrationEvent } from "@/lib/tournamentRegistrationApi";
import { skillEligibilityPolicy } from "@/lib/tournamentSkillEligibility";

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
    ? finiteNumber(profile.doublesSkill) ?? finiteNumber(profile.singlesSkill)
    : finiteNumber(profile.singlesSkill) ?? finiteNumber(profile.doublesSkill);
  if (rating == null) return null;
  if (
    policy.mode === "COMBINED_RATING_CAP" &&
    policy.combinedCap != null &&
    rating >= policy.combinedCap
  ) {
    return `Rating ${rating.toFixed(2)} cannot fit a combined team-rating cap strictly below ${policy.combinedCap.toFixed(2)}.`;
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
