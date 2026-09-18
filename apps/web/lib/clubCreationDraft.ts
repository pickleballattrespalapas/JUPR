// Only club details belong here. Authentication credentials stay with Auth.
export const CLUB_CREATION_DRAFT_KEY = "pcs_club_creation_draft_v1";
const MAX_AGE_MS = 24 * 60 * 60 * 1000;
export type ClubCreationStep = 1 | 2 | 3;
export type ClubCreationDraft = {
  name: string;
  slug: string;
  slugEdited: boolean;
  step: ClubCreationStep;
};

export function suggestClubSlug(name: string): string {
  return name.normalize("NFKD").replace(/[\u0300-\u036f]/g, "")
    .toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")
    .slice(0, 60).replace(/-$/, "");
}

export function clubDetailsError(name: string, slug: string): string | null {
  if (!name.trim() || name.trim().length > 120) return "Enter a club name of up to 120 characters.";
  if (slug.length < 3 || slug.length > 60 || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
    return "Choose a web address with 3–60 lowercase letters, numbers or hyphens.";
  }
  return null;
}

export function readClubCreationDraft(): ClubCreationDraft | null {
  try {
    const data = JSON.parse(window.localStorage.getItem(CLUB_CREATION_DRAFT_KEY) || "null");
    if (!data || data.version !== 1 || !Number.isFinite(data.savedAt) ||
      Date.now() - data.savedAt > MAX_AGE_MS || data.savedAt > Date.now() ||
      typeof data.name !== "string" || data.name.length > 120 ||
      typeof data.slug !== "string" || data.slug.length > 60 ||
      ![1, 2, 3].includes(data.step)) return null;
    return { name: data.name, slug: data.slug, slugEdited: data.slugEdited === true, step: data.step };
  } catch { return null; }
}

export function saveClubCreationDraft(draft: ClubCreationDraft): boolean {
  try {
    // Explicit fields prevent future callers from accidentally persisting a session.
    window.localStorage.setItem(CLUB_CREATION_DRAFT_KEY, JSON.stringify({
      version: 1, savedAt: Date.now(), name: draft.name, slug: draft.slug,
      slugEdited: draft.slugEdited, step: draft.step,
    }));
    return true;
  } catch { return false; }
}

export function clearClubCreationDraft(): void {
  try { window.localStorage.removeItem(CLUB_CREATION_DRAFT_KEY); } catch { /* Storage may be disabled. */ }
}
