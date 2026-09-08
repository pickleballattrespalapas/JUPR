// Navigation context only. Every protected API request independently checks
// the current user's assignment to the supplied club.
export const ADMIN_WORKSPACE_COOKIE = "jupr_admin_workspace_v1";
export const ADMIN_WORKSPACE_CHANGE = "jupr-admin-workspace-change";
export const ADMIN_WORKSPACE_DETAILS_CHANGE = "jupr-admin-workspace-details-change";
export type AdminWorkspace = { clubId: string; clubSlug: string };
export type AvailableWorkspace = {
  club_id: string; club_slug: string; club_name: string; roles: string[];
};

export function canChooseAdminWorkspace(workspaces: AvailableWorkspace[]): boolean {
  return workspaces.length > 1 || workspaces.some(club => club.roles.includes("super_admin"));
}

export function parseAdminWorkspace(value: string | undefined): AdminWorkspace | null {
  try {
    const data = JSON.parse(decodeURIComponent(value || ""));
    if (typeof data.clubId !== "string" || !/^[a-zA-Z0-9_-]{1,100}$/.test(data.clubId) ||
        typeof data.clubSlug !== "string" || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(data.clubSlug) ||
        data.clubSlug.length > 100) return null;
    return { clubId: data.clubId, clubSlug: data.clubSlug };
  } catch { return null; }
}

export function readBrowserWorkspace(): AdminWorkspace | null {
  const value = document.cookie.split("; ").find(item => item.startsWith(`${ADMIN_WORKSPACE_COOKIE}=`))?.slice(ADMIN_WORKSPACE_COOKIE.length + 1);
  return parseAdminWorkspace(value);
}

export function sameWorkspace(a: AdminWorkspace | null, b: AdminWorkspace | null): boolean {
  return a?.clubId === b?.clubId && a?.clubSlug === b?.clubSlug;
}

export function selectAdminWorkspace(workspace: AvailableWorkspace): void {
  const selected = { clubId: workspace.club_id, clubSlug: workspace.club_slug };
  const value = encodeURIComponent(JSON.stringify(selected));
  if (!parseAdminWorkspace(value)) throw new Error("This club address is unavailable.");
  document.cookie = `${ADMIN_WORKSPACE_COOKIE}=${value}; Path=/; SameSite=Lax; Max-Age=31536000${location.protocol === "https:" ? "; Secure" : ""}`;
  if (!sameWorkspace(readBrowserWorkspace(), selected)) throw new Error("Allow cookies to choose your club workspace.");
  // Other tabs stop accepting actions until their server and browser contexts
  // agree. Storage is a signal only; it never grants access or carries tokens.
  try { localStorage.setItem(ADMIN_WORKSPACE_CHANGE, crypto.randomUUID()); } catch { /* cookie/focus check remains active */ }
  window.dispatchEvent(new Event(ADMIN_WORKSPACE_CHANGE));
  // A full navigation clears prior club forms, route IDs, RSC cache and dialogs.
  window.location.assign("/admin");
}
