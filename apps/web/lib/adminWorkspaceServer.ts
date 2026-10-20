import "server-only";
import { cookies } from "next/headers";
import { redirect } from "next/navigation";
import { ADMIN_WORKSPACE_COOKIE, parseAdminWorkspace } from "./adminWorkspace";

export async function getAdminWorkspace() {
  return parseAdminWorkspace((await cookies()).get(ADMIN_WORKSPACE_COOKIE)?.value);
}

export async function requireAdminWorkspace() {
  const workspace = await getAdminWorkspace();
  if (!workspace) redirect("/admin/select-club");
  return workspace;
}
