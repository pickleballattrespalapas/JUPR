"use client";

import { useEffect, useRef, useState } from "react";
import { getAdminNotifications, type AdminNotifications } from "./adminNotificationsApi";
import { subscribeAdminNotifications } from "./adminNotificationsEvents";

export function useAdminNotificationBadges(accessToken: string, clubId: string, pathname: string, enabled: boolean) {
  const scope = enabled && accessToken && clubId ? `${accessToken}\u0000${clubId}` : "";
  const currentScope = useRef(scope);
  currentScope.current = scope;
  const [state, setState] = useState<{ scope: string; data: AdminNotifications | null }>({ scope: "", data: null });

  useEffect(() => {
    if (!scope) return;
    let cancelled = false;
    let revision = 0;
    let pending = false;
    const current = () => !cancelled && currentScope.current === scope;
    const unsubscribe = subscribeAdminNotifications(accessToken, clubId, data => {
      if (!current() || data.club_id !== clubId) return;
      revision += 1;
      setState({ scope, data });
    });
    const refresh = async () => {
      if (pending || !current() || document.visibilityState === "hidden") return;
      pending = true;
      const requestRevision = ++revision;
      try {
        const result = await getAdminNotifications(accessToken, clubId);
        if (!current() || requestRevision !== revision) return;
        if (result.data?.club_id === clubId) setState({ scope, data: result.data });
        else if (result.status === 401 || result.status === 403) setState({ scope, data: null });
      } finally {
        pending = false;
      }
    };
    const onVisibility = () => { if (document.visibilityState === "visible") void refresh(); };
    void refresh();
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", onVisibility);
    const timer = window.setInterval(refresh, 60_000);
    return () => {
      cancelled = true;
      unsubscribe();
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", onVisibility);
      window.clearInterval(timer);
    };
  }, [accessToken, clubId, scope, pathname]);

  // Guard during render as well as effect cleanup: never show the previous
  // account's or club's titles even for the first render after a switch.
  return scope && state.scope === scope ? state.data : null;
}
