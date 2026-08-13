"use client";

import { useEffect, useId, useRef, useState, type ReactNode, type RefObject } from "react";

import { ActionFeedback } from "./ActionFeedback";
import { focusEligibleElement, InteractionDialog } from "./InteractionDialog";
import styles from "./InteractionDialog.module.css";
import type { ActionCompletion, ActionSuccess } from "./types";
import { useActionLifecycle } from "./useActionLifecycle";

export type FormDialogProps = {
  open: boolean;
  mode: "create" | "edit" | "bulk";
  size?: "standard" | "wide" | "xwide";
  title: ReactNode;
  description?: ReactNode;
  dirty: boolean;
  submitLabel: string;
  submitDisabled?: boolean;
  workingLabel?: string;
  cancelLabel?: string;
  children: ReactNode;
  initialFocusRef?: RefObject<HTMLElement>;
  getFirstInvalidField?: () => HTMLElement | null;
  onSubmit: () => Promise<ActionCompletion>;
  onCancel: () => void;
  onAcknowledge?: (success: ActionSuccess) => void;
};

export function FormDialog({
  open,
  mode,
  size = "standard",
  title,
  description,
  dirty,
  submitLabel,
  submitDisabled = false,
  workingLabel = "Saving…",
  cancelLabel = "Cancel",
  children,
  initialFocusRef,
  getFirstInvalidField,
  onSubmit,
  onCancel,
  onAcknowledge
}: FormDialogProps) {
  const formId = useId();
  const [discardRequested, setDiscardRequested] = useState(false);
  const rememberedOriginRef = useRef<HTMLElement | null>(null);
  const lifecycle = useActionLifecycle();
  const resetLifecycle = lifecycle.reset;

  useEffect(() => {
    if (!open) {
      setDiscardRequested(false);
      resetLifecycle();
    }
  }, [open, resetLifecycle]);

  useEffect(() => {
    if (lifecycle.phase !== "error" || lifecycle.error?.kind !== "validation") return;
    const invalidField = getFirstInvalidField?.();
    if (invalidField) window.requestAnimationFrame(() => invalidField.focus());
  }, [getFirstInvalidField, lifecycle.error, lifecycle.phase]);

  function requestCancel() {
    if (lifecycle.phase === "working" || lifecycle.phase === "uncertain") return;
    if ((lifecycle.phase === "ready" || lifecycle.phase === "error") && dirty) {
      setDiscardRequested(true);
      return;
    }
    lifecycle.reset();
    onCancel();
  }

  function discardChanges() {
    setDiscardRequested(false);
    lifecycle.reset();
    onCancel();
  }

  function acknowledge() {
    if (lifecycle.completion?.status !== "success") return;
    const success = lifecycle.completion;
    try {
      onAcknowledge?.(success);
    } finally {
      onCancel();
      window.requestAnimationFrame(() => {
        const explicitTarget = success.focusTargetId
          ? document.getElementById(success.focusTargetId)
          : null;
        if (
          !focusEligibleElement(explicitTarget)
          && !focusEligibleElement(rememberedOriginRef.current)
        ) {
          focusEligibleElement(document.querySelector<HTMLElement>("main"));
        }
      });
    }
  }

  const completion = lifecycle.completion;
  const showForm = lifecycle.phase === "ready" || lifecycle.phase === "working" || lifecycle.phase === "error";
  const dialogTitle = discardRequested
    ? "Discard unsaved changes?"
    : lifecycle.phase === "success" && completion?.status === "success"
      ? completion.title
      : lifecycle.phase === "uncertain" && completion?.status === "uncertain"
        ? completion.title
        : title;

  return (
    <InteractionDialog
      open={open}
      phase={lifecycle.phase}
      size={size}
      title={dialogTitle}
      description={discardRequested ? "Your changes have not been saved." : description}
      initialFocusRef={initialFocusRef}
      originFocusRef={rememberedOriginRef}
      restoreFocus={lifecycle.phase !== "success"}
      dirty={dirty && !discardRequested}
      onRequestClose={requestCancel}
      actions={discardRequested ? (
        <>
          <button type="button" className={`${styles.button} ${styles.secondaryButton}`} data-dialog-focus onClick={() => setDiscardRequested(false)}>Keep editing</button>
          <button type="button" className={`${styles.button} ${styles.dangerButton}`} onClick={discardChanges}>Discard changes</button>
        </>
      ) : lifecycle.phase === "success" && completion?.status === "success" ? (
        <button type="button" className={`${styles.button} ${styles.primaryButton}`} onClick={acknowledge}>{completion.closeLabel ?? "Done"}</button>
      ) : lifecycle.phase === "uncertain" && completion?.status === "uncertain" ? (
        <button type="button" className={`${styles.button} ${styles.primaryButton}`} onClick={() => void lifecycle.recover(completion.onRecover)}>{completion.recoveryLabel}</button>
      ) : (
        <>
          <button type="button" className={`${styles.button} ${styles.secondaryButton}`} disabled={lifecycle.phase === "working"} onClick={requestCancel}>{cancelLabel}</button>
          <button type="submit" form={formId} className={`${styles.button} ${styles.primaryButton}`} disabled={lifecycle.phase === "working" || submitDisabled}>{lifecycle.phase === "working" ? workingLabel : submitLabel}</button>
        </>
      )}
    >
      {discardRequested ? (
        <p>Choose <strong>Keep editing</strong> to return to the {mode} form, or discard the unsaved values.</p>
      ) : (
        <>
          {showForm ? (
            <form
              id={formId}
              onSubmit={(event) => {
                event.preventDefault();
                void lifecycle.run(onSubmit);
              }}
            >
              {children}
            </form>
          ) : null}
          <ActionFeedback
            phase={lifecycle.phase}
            completion={completion}
            error={lifecycle.error}
            workingLabel={workingLabel}
          />
        </>
      )}
    </InteractionDialog>
  );
}
