"use client";

import { useSyncExternalStore } from "react";
import {
  adminSessionIsFresh,
  adminSessionStorageEventIsRelevant,
  loadAdminSession,
  restoreAuthorizedAdminSession
} from "@/lib/adminAuthClient";
import type { AdminSession } from "@/lib/adminAuthClient";

const ADMIN_SESSION_RECHECK_MS = 60_000;
const SHARED_SESSION_CHANGE_SOURCE = "use-admin-session-shared";

type AdminSessionState = {
  session: AdminSession | null;
  accessToken: string;
  loading: boolean;
  message: string | null;
};

const serverSnapshot: AdminSessionState = {
  session: null,
  accessToken: "",
  loading: true,
  message: null
};

let sharedSnapshot: AdminSessionState = serverSnapshot;
let initialized = false;
let restoreRequest: Promise<void> | null = null;
const listeners = new Set<() => void>();

function emit(next: AdminSessionState): void {
  sharedSnapshot = next;
  for (const listener of listeners) listener();
}

function snapshotFromSession(
  session: AdminSession | null,
  options: { loading?: boolean; message?: string | null } = {}
): AdminSessionState {
  return {
    session,
    accessToken: session?.access_token || "",
    loading: Boolean(options.loading),
    message: options.message ?? null
  };
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getSnapshot(): AdminSessionState {
  return sharedSnapshot;
}

function getServerSnapshot(): AdminSessionState {
  return serverSnapshot;
}

async function restoreSharedSession(
  options: { background?: boolean } = {}
): Promise<void> {
  if (restoreRequest) return restoreRequest;
  const background = Boolean(options.background);
  if (!background) emit(snapshotFromSession(null, { loading: true }));

  restoreRequest = (async () => {
    try {
      const authorized = await restoreAuthorizedAdminSession(undefined, {
        changeSource: SHARED_SESSION_CHANGE_SOURCE
      });
      emit(snapshotFromSession(authorized));
    } catch (error) {
      emit(
        snapshotFromSession(null, {
          message:
            error instanceof Error
              ? error.message
              : "Unable to refresh admin session."
        })
      );
    } finally {
      restoreRequest = null;
    }
  })();

  return restoreRequest;
}

function handleSessionChange(event: Event): void {
  const eventSource =
    event instanceof CustomEvent && event.detail
      ? String(event.detail.source || "")
      : "";
  if (eventSource === SHARED_SESSION_CHANGE_SOURCE) return;

  const stored = loadAdminSession();
  const trusted = Boolean(
    stored && adminSessionIsFresh(stored) && stored.capabilities?.authorized
  );
  emit(snapshotFromSession(trusted ? stored : null, { loading: !trusted }));
  void restoreSharedSession({ background: trusted });
}

function handleStorage(event: StorageEvent): void {
  if (adminSessionStorageEventIsRelevant(event.key)) handleSessionChange(event);
}

function recheckVisibleSession(): void {
  if (document.visibilityState === "visible") {
    void restoreSharedSession({ background: Boolean(sharedSnapshot.session) });
  }
}

function ensureInitialized(): void {
  if (initialized || typeof window === "undefined") return;
  initialized = true;

  const stored = loadAdminSession();
  const trusted = Boolean(
    stored && adminSessionIsFresh(stored) && stored.capabilities?.authorized
  );
  emit(snapshotFromSession(trusted ? stored : null, { loading: !trusted }));

  window.addEventListener("jupr-admin-session-change", handleSessionChange);
  window.addEventListener("storage", handleStorage);
  window.addEventListener("focus", recheckVisibleSession);
  document.addEventListener("visibilitychange", recheckVisibleSession);
  window.setInterval(recheckVisibleSession, ADMIN_SESSION_RECHECK_MS);

  void restoreSharedSession({ background: trusted });
}

ensureInitialized();

export function useAdminSession(): AdminSessionState {
  ensureInitialized();
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

export function adminSessionLabel(session: AdminSession | null): string {
  return (
    session?.user?.email ||
    (session?.access_token ? "signed-in admin" : "not signed in")
  );
}
