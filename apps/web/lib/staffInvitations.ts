export type StaffScope = { kind: string; program_type: string; resource_id: string };
export type StaffInvitation = {
  id: string; club_id: string; email: string; role: string; scopes: StaffScope[];
  status: "pending" | "accepted" | "cancelled" | "expired";
  expires_at: string; access_expires_at: string | null; accepted_at: string | null;
};
export function describeStaffScopes(scopes: StaffScope[]): string {
  return scopes.map(scope => scope.kind === "club" ? "All club programs" :
    `${scope.program_type.replaceAll("_", " ")}${scope.kind === "resource" ? `: ${scope.resource_id}` : ""}`).join("; ");
}
export function staffInvitationPath(id: string): string {
  return `/admin/accept-invitation?invitation=${encodeURIComponent(id)}`;
}
