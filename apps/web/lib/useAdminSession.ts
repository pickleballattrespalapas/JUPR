"use client";

import { useEffect, useState } from "react";
import { adminSessionStorageEventIsRelevant, restoreAuthorizedAdminSession } from "@/lib/adminAuthClient";
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
    let loadGeneration = 0;

    async function load() {
      const generation = ++loadGeneration;
      setSession(null);
      setLoading(true);
      setMessage(null);
      try {
        const authorized = await restoreAuthorizedAdminSession();
        if (!cancelled && generation === loadGeneration) setSession(authorized);
      } catch (error) {
        if (!cancelled && generation === loadGeneration) {
          setSession(null);
          setMessage(error instanceof Error ? error.message : "Unable to refresh admin session.");
        }
      } finally {
        if (!cancelled && generation === loadGeneration) setLoading(false);
      }
    }

    function handleSessionChange() {
      void load();
    }

    function handleStorage(event: StorageEvent) {
      if (adminSessionStorageEventIsRelevant(event.key)) void load();
    }

    void load();
    window.addEventListener("jupr-admin-session-change", handleSessionChange);
    window.addEventListener("storage", handleStorage);
    return () => {
      cancelled = true;
      window.removeEventListener("jupr-admin-session-change", handleSessionChange);
      window.removeEventListener("storage", handleStorage);
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
