"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";

/**
 * Run an authenticated read loader once for each restored admin access token
 * and optional read scope. The loader itself is kept in a ref so an inline
 * callback changing identity during render cannot cause a duplicate request.
 */
export function useAuthenticatedAutoLoad(
  accessToken: string,
  load: () => Promise<unknown> | void,
  scopeKey = ""
): void {
  const loadRef = useRef(load);
  const loadedKeyRef = useRef("");
  loadRef.current = load;

  useEffect(() => {
    if (!accessToken) {
      loadedKeyRef.current = "";
      return;
    }
    const requestKey = `${accessToken}\u0000${scopeKey}`;
    if (loadedKeyRef.current === requestKey) return;
    loadedKeyRef.current = requestKey;
    void loadRef.current();
  }, [accessToken, scopeKey]);
}

/**
 * Ignore late async selection responses after the operator chooses a newer
 * record or clears the selector.
 */
export function useLatestRequestGuard(scopeKey = "", onScopeChange?: () => void) {
  const generationRef = useRef(0);
  const currentScopeRef = useRef(scopeKey);
  const requestScopeRef = useRef(scopeKey);
  const committedScopeRef = useRef(scopeKey);
  const onScopeChangeRef = useRef(onScopeChange);
  currentScopeRef.current = scopeKey;
  onScopeChangeRef.current = onScopeChange;

  useEffect(() => {
    if (committedScopeRef.current === scopeKey) return;
    committedScopeRef.current = scopeKey;
    generationRef.current += 1;
    requestScopeRef.current = currentScopeRef.current;
    onScopeChangeRef.current?.();
  }, [scopeKey]);

  const begin = useCallback(() => {
    generationRef.current += 1;
    requestScopeRef.current = currentScopeRef.current;
    return generationRef.current;
  }, []);
  const invalidate = useCallback(() => {
    generationRef.current += 1;
    requestScopeRef.current = currentScopeRef.current;
  }, []);
  const isCurrent = useCallback(
    (generation: number) => generationRef.current === generation && requestScopeRef.current === currentScopeRef.current,
    []
  );

  return useMemo(() => ({ begin, invalidate, isCurrent }), [begin, invalidate, isCurrent]);
}
