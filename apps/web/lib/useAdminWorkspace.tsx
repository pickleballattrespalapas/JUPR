"use client";
import { createContext, useContext } from "react";
import type { AdminWorkspace } from "./adminWorkspace";

export const AdminWorkspaceContext = createContext<AdminWorkspace | null>(null);
export function useAdminWorkspace(): AdminWorkspace {
  const workspace = useContext(AdminWorkspaceContext);
  if (!workspace) throw new Error("Choose a club before opening this workspace.");
  return workspace;
}
