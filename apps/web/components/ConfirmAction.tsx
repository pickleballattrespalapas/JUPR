"use client";

import { useId, type ReactNode } from "react";

import { useInteraction } from "./interaction/InteractionProvider";
import type { ActionCallback, ActionSuccess } from "./interaction/types";
import styles from "./ConfirmAction.module.css";

/** @deprecated Prefer ActionSuccess from @/components/interaction for new code. */
export type ConfirmActionSuccess = ActionSuccess;

export type ConfirmActionProps = {
  triggerLabel: string;
  title: ReactNode;
  description: ReactNode;
  preview?: ReactNode;
  confirmLabel: string;
  cancelLabel?: string;
  workingLabel?: string;
  confirmationText: string;
  tone?: "default" | "danger";
  disabled?: boolean;
  disabledReason?: ReactNode;
  busy?: boolean;
  onConfirm: ActionCallback;
  onAcknowledge?: (success: ActionSuccess) => void;
};

/**
 * A lightweight trigger for the root-owned interaction lifecycle. The provider
 * snapshots the action before it runs, so its visible outcome survives removal
 * or replacement of this consumer.
 */
export function ConfirmAction({
  triggerLabel,
  title,
  description,
  preview,
  confirmLabel,
  cancelLabel = "No, go back",
  workingLabel = "Working…",
  confirmationText,
  tone = "default",
  disabled = false,
  disabledReason,
  busy = false,
  onConfirm,
  onAcknowledge
}: ConfirmActionProps) {
  const actionId = useId();
  const disabledReasonId = useId();
  const { activeActionId, openAction } = useInteraction();
  const triggerDisabled = disabled || busy;

  return (
    <span className={styles.triggerGroup}>
      <button
        type="button"
        className={`${styles.trigger} ${tone === "danger" ? styles.dangerTrigger : ""}`}
        disabled={triggerDisabled}
        aria-haspopup="dialog"
        aria-expanded={activeActionId === actionId}
        aria-busy={busy || undefined}
        aria-describedby={triggerDisabled && disabledReason ? disabledReasonId : undefined}
        onClick={(event) => {
          if (triggerDisabled) return;
          openAction(
            {
              id: actionId,
              title,
              description,
              preview,
              confirmLabel,
              cancelLabel,
              workingLabel,
              confirmationText,
              tone,
              onConfirm,
              onAcknowledge
            },
            event.currentTarget
          );
        }}
      >
        {triggerLabel}
      </button>
      {triggerDisabled && disabledReason ? (
        <span id={disabledReasonId} className={styles.disabledReason}>{disabledReason}</span>
      ) : null}
    </span>
  );
}
