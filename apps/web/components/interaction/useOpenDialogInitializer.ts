"use client";

import { useEffect, useRef } from "react";

/**
 * Initialize local form state once for each closed-to-open dialog transition.
 *
 * Parent components may rerender while a dialog is open because of session
 * refreshes, background reads, or unrelated state changes. Those rerenders
 * must not replace unsaved form values merely because an equivalent source
 * object was allocated again.
 */
export function useOpenDialogInitializer(
  open: boolean,
  initialize: () => void
): void {
  const initializeRef = useRef(initialize);
  initializeRef.current = initialize;

  useEffect(() => {
    if (open) initializeRef.current();
  }, [open]);
}
