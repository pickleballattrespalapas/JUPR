import type { PublicRegistrationPlayer } from "./tournamentRegistrationApi";

export function registrationNameKey(name: string): string {
  return name.normalize("NFKC").trim().replace(/\s+/g, " ").toLowerCase();
}

export function automaticRegistrationProfile(
  candidates: PublicRegistrationPlayer[],
  fullName: string,
  matchKind?: "email_exact" | "name_exact" | "none"
): PublicRegistrationPlayer | null {
  // The API's name match also handles a player's preferred display name.
  // Email alone may belong to another family member; never guess between rows.
  if (candidates.length !== 1) return null;
  const candidate = candidates[0];
  return matchKind === "name_exact" || (
    registrationNameKey(fullName) !== "" &&
    registrationNameKey(candidate.display_name) === registrationNameKey(fullName)
  ) ? candidate : null;
}
