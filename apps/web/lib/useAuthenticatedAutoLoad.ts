"use client";

import { useEffect, useRef } from "react";

/** Run an authenticated read loader once for each restored admin access token. */
export function useAuthenticatedAutoLoad(accessToken: string, load: () => Promise<unknown> | void): void {
  const loadedTokenRef = useRef("");

  useEffect(() => {
    if (!accessToken) {
      loadedTokenRef.current = "";
      return;
    }
    if (loadedTokenRef.current === accessToken) return;
    loadedTokenRef.current = accessToken;
    void load();
  }, [accessToken, load]);
}
