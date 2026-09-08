"use client";
import { useEffect, useState } from "react";
import { getAdminApiBaseUrl } from "./adminAuthClient";
import type { AvailableWorkspace } from "./adminWorkspace";

export function useAvailableWorkspaces(accessToken: string, identity = accessToken) {
  const [state, setState] = useState<{identity: string; workspaces: AvailableWorkspace[]; error: string; loaded: boolean}>({ identity: "", workspaces: [], error: "", loaded: false });
  const [revision, setRevision] = useState(0);
  useEffect(() => {
    if (!accessToken) return;
    const controller = new AbortController();
    const api = getAdminApiBaseUrl();
    setState(previous => previous.identity === identity ? previous : { identity, workspaces: [], error: "", loaded: false });
    async function load() {
      try {
        if (!api) throw new Error("Club service is not configured.");
        const response = await fetch(`${api}/admin/auth/workspaces`, { headers: { Authorization: `Bearer ${accessToken}` }, cache: "no-store", signal: controller.signal });
        if (!response.ok) throw new Error(response.status === 403 ? "Your club access has changed. Sign in again." : "Unable to load your clubs. Try again.");
        const data = await response.json();
        if (!controller.signal.aborted) setState({ identity, workspaces: data.workspaces, error: "", loaded: true });
      } catch (error) {
        if (!controller.signal.aborted) setState({ identity, workspaces: [], error: error instanceof Error ? error.message : "Unable to load clubs.", loaded: true });
      }
    }
    void load();
    return () => controller.abort();
  }, [accessToken, identity, revision]);
  return { workspaces: state.identity === identity ? state.workspaces : [], error: state.identity === identity ? state.error : "", loaded: state.identity === identity && state.loaded, retry: () => setRevision(n => n + 1) };
}
