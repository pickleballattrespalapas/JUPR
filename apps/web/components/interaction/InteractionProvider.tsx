"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode
} from "react";

import { ActionFeedback } from "./ActionFeedback";
import { InteractionDialog } from "./InteractionDialog";
import styles from "./InteractionDialog.module.css";
import type { ActionCallback, ActionSuccess } from "./types";
import { useActionLifecycle } from "./useActionLifecycle";

export type ConfirmInteractionRequest = {
  id: string;
  title: ReactNode;
  description: ReactNode;
  preview?: ReactNode;
  confirmLabel: string;
  cancelLabel: string;
  workingLabel: string;
  confirmationText: string;
  tone: "default" | "danger";
  onConfirm: ActionCallback;
  onAcknowledge?: (success: ActionSuccess) => void;
};

type ActiveInteraction = Readonly<ConfirmInteractionRequest> & {
  origin: HTMLElement | null;
};

type InteractionContextValue = {
  activeActionId: string | null;
  openAction: (request: ConfirmInteractionRequest, origin: HTMLElement | null) => boolean;
};

const InteractionContext = createContext<InteractionContextValue | null>(null);

function isEligibleFocusTarget(element: HTMLElement | null): element is HTMLElement {
  return Boolean(
    element?.isConnected
    && !element.matches(":disabled")
    && element.getAttribute("aria-disabled") !== "true"
  );
}

function focusAfterClose(focusTargetId: string | undefined, origin: HTMLElement | null) {
  window.requestAnimationFrame(() => {
    const explicitTarget = focusTargetId ? document.getElementById(focusTargetId) : null;
    const target = isEligibleFocusTarget(explicitTarget)
      ? explicitTarget
      : isEligibleFocusTarget(origin)
        ? origin
        : document.querySelector<HTMLElement>("main");
    if (!target) return;
    const previousTabIndex = target.getAttribute("tabindex");
    if (previousTabIndex === null && target.tabIndex < 0) target.setAttribute("tabindex", "-1");
    target.focus();
    if (previousTabIndex === null && target.getAttribute("tabindex") === "-1") {
      target.addEventListener("blur", () => target.removeAttribute("tabindex"), { once: true });
    }
  });
}

/**
 * Owns guarded confirmation dialogs above page and row lifetimes. A successful
 * action therefore remains visible even when its authoritative update removes
 * or replaces the component that opened it.
 */
export function InteractionProvider({ children }: { children: ReactNode }) {
  const [active, setActive] = useState<ActiveInteraction | null>(null);
  const activeRef = useRef<ActiveInteraction | null>(null);
  const lifecycle = useActionLifecycle();
  const resetLifecycle = lifecycle.reset;

  const openAction = useCallback(
    (request: ConfirmInteractionRequest, origin: HTMLElement | null) => {
      if (activeRef.current) return false;
      const snapshot = Object.freeze({ ...request, origin });
      activeRef.current = snapshot;
      resetLifecycle();
      setActive(snapshot);
      return true;
    },
    [resetLifecycle]
  );

  const clearActive = useCallback((focusTargetId?: string) => {
    const origin = activeRef.current?.origin ?? null;
    activeRef.current = null;
    resetLifecycle();
    setActive(null);
    focusAfterClose(focusTargetId, origin);
  }, [resetLifecycle]);

  const closeDialog = useCallback(() => {
    if (lifecycle.phase === "working" || lifecycle.phase === "uncertain") return;
    clearActive();
  }, [clearActive, lifecycle.phase]);

  const acknowledge = useCallback(() => {
    const current = activeRef.current;
    const success = lifecycle.completion;
    if (!current || success?.status !== "success") return;
    try {
      current.onAcknowledge?.(success);
    } finally {
      clearActive(success.focusTargetId);
    }
  }, [clearActive, lifecycle.completion]);

  const contextValue = useMemo<InteractionContextValue>(
    () => ({ activeActionId: active?.id ?? null, openAction }),
    [active?.id, openAction]
  );

  const completion = lifecycle.completion;
  const dialogTitle = lifecycle.phase === "success" && completion?.status === "success"
    ? completion.title
    : lifecycle.phase === "uncertain" && completion?.status === "uncertain"
      ? completion.title
      : active?.title ?? "Confirm action";

  return (
    <InteractionContext.Provider value={contextValue}>
      {children}
      {active ? (
        <InteractionDialog
          open
          phase={lifecycle.phase}
          title={dialogTitle}
          description={lifecycle.phase === "success" || lifecycle.phase === "uncertain" ? undefined : active.description}
          restoreFocus={false}
          onRequestClose={closeDialog}
          actions={lifecycle.phase === "success" && completion?.status === "success" ? (
            <button
              type="button"
              className={`${styles.button} ${styles.primaryButton}`}
              onClick={acknowledge}
            >
              {completion.closeLabel ?? "OK"}
            </button>
          ) : lifecycle.phase === "uncertain" && completion?.status === "uncertain" ? (
            <button
              type="button"
              className={`${styles.button} ${styles.primaryButton}`}
              onClick={() => void lifecycle.recover(completion.onRecover)}
            >
              {completion.recoveryLabel}
            </button>
          ) : (
            <>
              <button
                type="button"
                className={`${styles.button} ${styles.secondaryButton}`}
                disabled={lifecycle.phase === "working"}
                data-autofocus
                onClick={closeDialog}
              >
                {active.cancelLabel}
              </button>
              <button
                type="button"
                className={`${styles.button} ${active.tone === "danger" ? styles.dangerButton : styles.primaryButton}`}
                disabled={lifecycle.phase === "working"}
                onClick={() => void lifecycle.run(() => active.onConfirm(active.confirmationText))}
              >
                {lifecycle.phase === "working" ? active.workingLabel : active.confirmLabel}
              </button>
            </>
          )}
        >
          {active.preview && (lifecycle.phase === "ready" || lifecycle.phase === "error") ? (
            <div>{active.preview}</div>
          ) : null}
          <ActionFeedback
            phase={lifecycle.phase}
            completion={completion}
            error={lifecycle.error}
            workingLabel={active.workingLabel}
          />
        </InteractionDialog>
      ) : null}
    </InteractionContext.Provider>
  );
}

export function useInteraction(): InteractionContextValue {
  const context = useContext(InteractionContext);
  if (!context) throw new Error("useInteraction must be used within InteractionProvider.");
  return context;
}
