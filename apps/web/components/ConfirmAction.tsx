"use client";

import { useEffect, useId, useRef, useState } from "react";
import type { ReactNode, SyntheticEvent } from "react";
import { createPortal } from "react-dom";

import styles from "./ConfirmAction.module.css";

export type ConfirmActionSuccess = {
  title?: string;
  description: ReactNode;
  closeLabel?: string;
};

export type ConfirmActionProps = {
  triggerLabel: string;
  title: string;
  description: ReactNode;
  confirmLabel: string;
  cancelLabel?: string;
  confirmationText: string;
  tone?: "default" | "danger";
  disabled?: boolean;
  busy?: boolean;
  onConfirm: (confirmationText: string) => void | ConfirmActionSuccess | Promise<void | ConfirmActionSuccess>;
};

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim()) return error.message;
  return "The action could not be completed. Please review the page and try again.";
}

function restoreFocusAfterDialog(trigger: HTMLButtonElement | null, rememberedFallback: HTMLElement | null) {
  if (trigger?.isConnected && !trigger.disabled) {
    trigger.focus();
    return;
  }

  const fallback = rememberedFallback?.isConnected ? rememberedFallback : document.querySelector<HTMLElement>("main");
  if (!fallback) return;
  const previousTabIndex = fallback.getAttribute("tabindex");
  if (previousTabIndex === null) fallback.setAttribute("tabindex", "-1");
  fallback.focus();
  if (previousTabIndex === null) {
    fallback.addEventListener("blur", () => {
      if (fallback.getAttribute("tabindex") === "-1") fallback.removeAttribute("tabindex");
    }, { once: true });
  }
}

export function ConfirmAction({
  triggerLabel,
  title,
  description,
  confirmLabel,
  cancelLabel = "No, go back",
  confirmationText,
  tone = "default",
  disabled = false,
  busy = false,
  onConfirm
}: ConfirmActionProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const fallbackFocusRef = useRef<HTMLElement | null>(null);
  const submittingRef = useRef(false);
  const wasOpenRef = useRef(false);
  const titleId = useId();
  const descriptionId = useId();
  const errorId = useId();
  const [open, setOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<ConfirmActionSuccess | null>(null);
  const [portalContainer, setPortalContainer] = useState<HTMLElement | null>(null);
  const actionBusy = busy || submitting;

  useEffect(() => {
    setPortalContainer(document.body);
  }, []);

  useEffect(() => () => {
    if (!wasOpenRef.current) return;
    wasOpenRef.current = false;
    window.requestAnimationFrame(() => restoreFocusAfterDialog(triggerRef.current, fallbackFocusRef.current));
  }, []);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (!open) {
      if (dialog.open) dialog.close();
      return;
    }

    if (!dialog.open) dialog.showModal();
    const focusFrame = window.requestAnimationFrame(() => cancelRef.current?.focus());
    return () => window.cancelAnimationFrame(focusFrame);
  }, [open, portalContainer, success]);

  function openDialog() {
    if (disabled || busy || submittingRef.current) return;
    fallbackFocusRef.current = triggerRef.current?.closest<HTMLElement>("article, section, main") || document.querySelector<HTMLElement>("main");
    wasOpenRef.current = true;
    setError(null);
    setSuccess(null);
    setOpen(true);
  }

  function closeDialog() {
    if (actionBusy) return;
    setOpen(false);
  }

  function handleDialogClose() {
    setOpen(false);
    wasOpenRef.current = false;
    window.requestAnimationFrame(() => restoreFocusAfterDialog(triggerRef.current, fallbackFocusRef.current));
  }

  function handleDialogCancel(event: SyntheticEvent<HTMLDialogElement>) {
    event.preventDefault();
    closeDialog();
  }

  async function handleConfirm() {
    if (disabled || busy || submittingRef.current) return;

    submittingRef.current = true;
    setSubmitting(true);
    setError(null);
    try {
      const completion = await onConfirm(confirmationText);
      if (completion) setSuccess(completion);
      else setOpen(false);
    } catch (actionError) {
      setError(errorMessage(actionError));
      window.requestAnimationFrame(() => cancelRef.current?.focus());
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  }

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className={`${styles.trigger} ${tone === "danger" ? styles.dangerTrigger : ""}`}
        disabled={disabled || busy}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-busy={busy || undefined}
        onClick={openDialog}
      >
        {triggerLabel}
      </button>

      {portalContainer ? createPortal(
        <dialog
          ref={dialogRef}
          className={styles.dialog}
          aria-labelledby={titleId}
          aria-describedby={error ? `${descriptionId} ${errorId}` : descriptionId}
          aria-modal="true"
          aria-busy={actionBusy || undefined}
          onCancel={handleDialogCancel}
          onClose={handleDialogClose}
        >
          <div className={styles.content}>
            {success ? (
              <>
                <h2 id={titleId} className={styles.title}>{success.title || "Action complete"}</h2>
                <div id={descriptionId} className={styles.description}>{success.description}</div>
                <div className={styles.actions}>
                  <button
                    ref={cancelRef}
                    type="button"
                    className={styles.confirmButton}
                    disabled={actionBusy}
                    onClick={closeDialog}
                  >
                    {success.closeLabel || "OK"}
                  </button>
                </div>
              </>
            ) : (
              <>
                <h2 id={titleId} className={styles.title}>{title}</h2>
                <div id={descriptionId} className={styles.description}>{description}</div>
                {error ? <p id={errorId} className={styles.error} role="alert">{error}</p> : null}
                <div className={styles.actions}>
                  <button
                    ref={cancelRef}
                    type="button"
                    className={styles.cancelButton}
                    disabled={actionBusy}
                    onClick={closeDialog}
                  >
                    {cancelLabel}
                  </button>
                  <button
                    type="button"
                    className={`${styles.confirmButton} ${tone === "danger" ? styles.dangerConfirm : ""}`}
                    disabled={disabled || actionBusy}
                    onClick={() => void handleConfirm()}
                  >
                    {actionBusy ? "Working…" : confirmLabel}
                  </button>
                </div>
              </>
            )}
          </div>
        </dialog>,
        portalContainer
      ) : null}
    </>
  );
}
