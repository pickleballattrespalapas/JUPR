import "server-only";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { ADMIN_WORKSPACE_COOKIE, parseAdminWorkspace } from "./adminWorkspace";

export function getAdminWorkspace() {
  return parseAdminWorkspace(cookies().get(ADMIN_WORKSPACE_COOKIE)?.value);
}

export function requireAdminWorkspace() {
  const workspace = getAdminWorkspace();
  if (!workspace) redirect("/admin/select-club");
  return workspace;
}
