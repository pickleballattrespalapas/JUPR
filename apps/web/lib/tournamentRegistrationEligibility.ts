import { PublicRegistrationEvent } from "@/lib/tournamentRegistrationApi";

export type RegistrationEligibilityProfile = {
  gender?: string | null;
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

function controlledSkillFloor(label?: string | null): number | null {
  const text = String(label || "").trim();
  if (!["3.0", "3.5", "4.0", "4.5", "5.0", "5.5"].includes(text)) return null;
  const value = Number(text);
  return Number.isFinite(value) ? value : null;
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

  const floor = controlledSkillFloor(event.skill_label);
  if (floor == null) return null;
  const eventType = String(event.event_type || "").toUpperCase();
  const isDoubles = Boolean(event.partner_required) || ["DOUBLES", "GENDER_DOUBLES", "MIXED_DOUBLES", "MIXED"].includes(eventType);
  const rating = isDoubles
    ? finiteNumber(profile.doublesSkill) ?? finiteNumber(profile.singlesSkill)
    : finiteNumber(profile.singlesSkill) ?? finiteNumber(profile.doublesSkill);
  if (rating != null && rating >= floor + 0.5) {
    const recommended = Math.floor(rating * 2) / 2;
    return `Rating ${rating.toFixed(2)} is above this division cap. Choose ${recommended.toFixed(1)} or higher.`;
  }
  return null;
}

export function publicEventFamilyKey(event: PublicRegistrationEvent): string {
  return `${event.registration_day_id}::${event.event_family_label.trim().toLowerCase().replace(/\s+/g, " ")}`;
}
