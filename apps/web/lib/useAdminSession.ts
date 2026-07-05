"use client";

import { useEffect, useState } from "react";
import { loadAdminSession, refreshAdminSession } from "@/lib/adminAuthClient";
import type { AdminSession } from "@/lib/adminAuthClient";

type AdminSessionState = {
  session: AdminSession | null;
  accessToken: string;
  loading: boolean;
  message: string | null;
};

export function useAdminSession(): AdminSessionState {
  const [session, setSession] = useState<AdminSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setMessage(null);
      try {
        const current = loadAdminSession();
        if (!cancelled) setSession(current);
        const refreshed = await refreshAdminSession(current || undefined);
        if (!cancelled && refreshed) setSession(refreshed);
      } catch (error) {
        if (!cancelled) {
          setSession(null);
          setMessage(error instanceof Error ? error.message : "Unable to refresh admin session.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    window.addEventListener("jupr-admin-session-change", load);
    window.addEventListener("storage", load);
    return () => {
      cancelled = true;
      window.removeEventListener("jupr-admin-session-change", load);
      window.removeEventListener("storage", load);
    };
  }, []);

  return {
    session,
    accessToken: session?.access_token || "",
    loading,
    message
  };
}

export function adminSessionLabel(session: AdminSession | null): string {
  return session?.user?.email || (session?.access_token ? "signed-in admin" : "not signed in");
}
