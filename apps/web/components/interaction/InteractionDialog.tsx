"use client";

import {
  useEffect,
  useId,
  useRef,
  useState,
  type MouseEvent,
  type ReactNode,
  type RefObject,
  type SyntheticEvent
} from "react";
import { createPortal } from "react-dom";

import type { ActionPhase } from "./useActionLifecycle";
import styles from "./InteractionDialog.module.css";

type CloseReason = "escape" | "backdrop" | "native";

export type InteractionDialogProps = {
  open: boolean;
  phase: ActionPhase;
  size?: "standard" | "wide" | "xwide";
  title: ReactNode;
  description?: ReactNode;
  children: ReactNode;
  actions: ReactNode;
  initialFocusRef?: RefObject<HTMLElement>;
  returnFocusRef?: RefObject<HTMLElement>;
  originFocusRef?: { current: HTMLElement | null };
  restoreFocus?: boolean;
  dirty?: boolean;
  onRequestClose: () => void;
};

export function isEligibleFocusTarget(element: HTMLElement | null): element is HTMLElement {
  return Boolean(
    element?.isConnected
    && !element.matches(":disabled")
    && element.getAttribute("aria-disabled") !== "true"
  );
}

export function focusEligibleElement(element: HTMLElement | null): boolean {
  if (!isEligibleFocusTarget(element)) return false;
  const previousTabIndex = element.getAttribute("tabindex");
  if (previousTabIndex === null && element.tabIndex < 0) element.setAttribute("tabindex", "-1");
  element.focus();
  if (previousTabIndex === null && element.getAttribute("tabindex") === "-1") {
    element.addEventListener("blur", () => element.removeAttribute("tabindex"), { once: true });
  }
  return true;
}

export function InteractionDialog({
  open,
  phase,
  size = "standard",
  title,
  description,
  children,
  actions,
  initialFocusRef,
  returnFocusRef,
  originFocusRef,
  restoreFocus = true,
  dirty = false,
  onRequestClose
}: InteractionDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const rememberedFocusRef = useRef<HTMLElement | null>(null);
  const wasOpenRef = useRef(false);
  const titleId = useId();
  const descriptionId = useId();
  const [portalContainer, setPortalContainer] = useState<HTMLElement | null>(null);

  useEffect(() => {
    setPortalContainer(document.body);
  }, []);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (open) {
      if (!wasOpenRef.current) {
        rememberedFocusRef.current = document.activeElement instanceof HTMLElement
          ? document.activeElement
          : null;
        if (originFocusRef) originFocusRef.current = rememberedFocusRef.current;
        wasOpenRef.current = true;
      }
      if (!dialog.open) dialog.showModal();
      const focusFrame = window.requestAnimationFrame(() => {
        const phaseTarget = dialog.querySelector<HTMLElement>("[data-dialog-focus]");
        const initialTarget = initialFocusRef?.current ?? dialog.querySelector<HTMLElement>("[data-autofocus]");
        const firstControl = Array.from(
          dialog.querySelectorAll<HTMLElement>("button, input, select, textarea, a[href], [tabindex]")
        ).find((element) => isEligibleFocusTarget(element) && element.tabIndex >= 0) ?? null;
        focusEligibleElement(
          isEligibleFocusTarget(phaseTarget)
            ? phaseTarget
            : isEligibleFocusTarget(initialTarget)
              ? initialTarget
              : firstControl
        );
      });
      return () => window.cancelAnimationFrame(focusFrame);
    }

    if (dialog.open) dialog.close();
    if (wasOpenRef.current) {
      wasOpenRef.current = false;
      if (!restoreFocus) return;
      const restoreFrame = window.requestAnimationFrame(() => {
        const returnTarget = returnFocusRef?.current ?? null;
        focusEligibleElement(
          isEligibleFocusTarget(returnTarget)
            ? returnTarget
            : isEligibleFocusTarget(rememberedFocusRef.current)
              ? rememberedFocusRef.current
              : document.querySelector<HTMLElement>("main")
        );
      });
      return () => window.cancelAnimationFrame(restoreFrame);
    }
  }, [initialFocusRef, open, originFocusRef, phase, portalContainer, restoreFocus, returnFocusRef]);

  useEffect(() => () => {
    if (!wasOpenRef.current) return;
    wasOpenRef.current = false;
    if (!restoreFocus) return;
    window.requestAnimationFrame(() => {
      const returnTarget = returnFocusRef?.current ?? null;
      focusEligibleElement(
        isEligibleFocusTarget(returnTarget)
          ? returnTarget
          : isEligibleFocusTarget(rememberedFocusRef.current)
            ? rememberedFocusRef.current
            : document.querySelector<HTMLElement>("main")
      );
    });
  }, [restoreFocus, returnFocusRef]);

  function passiveCloseAllowed(_reason: CloseReason) {
    return (phase === "ready" || phase === "error") && !dirty;
  }

  function requestPassiveClose(reason: CloseReason) {
    if (passiveCloseAllowed(reason)) onRequestClose();
  }

  function handleCancel(event: SyntheticEvent<HTMLDialogElement>) {
    event.preventDefault();
    requestPassiveClose("escape");
  }

  function handleBackdropClick(event: MouseEvent<HTMLDialogElement>) {
    if (event.target === event.currentTarget) requestPassiveClose("backdrop");
  }

  function handleNativeClose() {
    if (open) requestPassiveClose("native");
  }

  if (!portalContainer) return null;

  return createPortal(
    <dialog
      ref={dialogRef}
      className={styles.dialog}
      data-size={size}
      aria-labelledby={titleId}
      aria-describedby={description ? descriptionId : undefined}
      aria-modal="true"
      aria-busy={phase === "working" || undefined}
      data-state={phase}
      onCancel={handleCancel}
      onClick={handleBackdropClick}
      onClose={handleNativeClose}
    >
      <div className={styles.content}>
        <header className={styles.heading}>
          <h2 id={titleId} className={styles.title}>{title}</h2>
          {description ? <div id={descriptionId} className={styles.description}>{description}</div> : null}
        </header>
        <div className={styles.body}>{children}</div>
        <footer className={styles.actions}>{actions}</footer>
      </div>
    </dialog>,
    portalContainer
  );
}
