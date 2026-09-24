import type { ReactNode } from "react";
import AdminShell from "@/components/AdminShell";
import { getAdminWorkspace } from "@/lib/adminWorkspaceServer";

// Rebind each route's server-rendered data and client tools to the same cookie.
export default function AdminTemplate({ children }: { children: ReactNode }) {
  return <AdminShell workspace={getAdminWorkspace()}>{children}</AdminShell>;
}
