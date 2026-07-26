"use client";

import { useEffect, useState } from "react";
import {
  adminSessionStorageEventIsRelevant,
  loadAdminSession,
  restoreAuthorizedAdminSession
} from "@/lib/adminAuthClient";
import type { AdminSession } from "@/lib/adminAuthClient";

const ADMIN_SESSION_RECHECK_MS = 60_000;
let adminSessionHookSequence = 0;

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
    let requestInFlight = false;
    let queuedBackground: boolean | null = null;
    const changeSource = `use-admin-session-${++adminSessionHookSequence}`;

    async function load(options: { background?: boolean } = {}) {
      const background = Boolean(options.background);
      if (requestInFlight && background) return;
      const generation = ++loadGeneration;
      if (!background) {
        setSession(null);
        setLoading(true);
        setMessage(null);
      }
      if (requestInFlight) {
        queuedBackground = false;
        return;
      }
      requestInFlight = true;
      try {
        const authorized = await restoreAuthorizedAdminSession(undefined, {
          changeSource
        });
        if (!cancelled && generation === loadGeneration) {
          setSession(authorized);
          setMessage(null);
        }
      } catch (error) {
        if (!cancelled && generation === loadGeneration) {
          setSession(null);
          setMessage(error instanceof Error ? error.message : "Unable to refresh admin session.");
        }
      } finally {
        requestInFlight = false;
        if (!cancelled && generation === loadGeneration && !background) {
          setLoading(false);
        }
        const nextBackground = queuedBackground;
        queuedBackground = null;
        if (!cancelled && nextBackground !== null) {
          void load({ background: nextBackground });
        }
      }
    }

    function handleSessionChange(event: Event) {
      const eventSource =
        event instanceof CustomEvent && event.detail
          ? String(event.detail.source || "")
          : "";
      if (eventSource === changeSource) return;
      if (!eventSource) {
        void load();
        return;
      }
      loadGeneration += 1;
      queuedBackground = null;
      setSession(loadAdminSession());
      setLoading(false);
      setMessage(null);
    }

    function handleStorage(event: StorageEvent) {
      if (adminSessionStorageEventIsRelevant(event.key)) void load();
    }

    function recheckVisibleSession() {
      if (document.visibilityState === "visible") {
        void load({ background: true });
      }
    }

    void load();
    window.addEventListener("jupr-admin-session-change", handleSessionChange);
    window.addEventListener("storage", handleStorage);
    window.addEventListener("focus", recheckVisibleSession);
    document.addEventListener("visibilitychange", recheckVisibleSession);
    const recheckTimer = window.setInterval(
      recheckVisibleSession,
      ADMIN_SESSION_RECHECK_MS
    );
    return () => {
      cancelled = true;
      window.clearInterval(recheckTimer);
      window.removeEventListener("jupr-admin-session-change", handleSessionChange);
      window.removeEventListener("storage", handleStorage);
      window.removeEventListener("focus", recheckVisibleSession);
      document.removeEventListener("visibilitychange", recheckVisibleSession);
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
