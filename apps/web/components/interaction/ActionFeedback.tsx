import type { ReactNode } from "react";

import type { ActionCompletion, InteractionActionError } from "./types";
import type { ActionPhase } from "./useActionLifecycle";
import styles from "./InteractionDialog.module.css";

export type ActionFeedbackProps = {
  phase: ActionPhase;
  completion: ActionCompletion | null;
  error: InteractionActionError | null;
  workingLabel?: string;
};

const ERROR_TITLES: Record<InteractionActionError["kind"], string> = {
  validation: "Review the highlighted information",
  conflict: "The saved information changed",
  forbidden: "This action is not available to your account",
  failed: "The action did not complete"
};

export function ActionFeedback({
  phase,
  completion,
  error,
  workingLabel = "Working…"
}: ActionFeedbackProps) {
  if (phase === "ready") return null;

  if (phase === "working") {
    return (
      <section className={`${styles.feedback} ${styles.feedbackWorking}`} aria-live="polite">
        <h3 className={styles.feedbackTitle}>{workingLabel}</h3>
        <p className={styles.feedbackDescription}>Keep this window open while we finish.</p>
      </section>
    );
  }

  if (phase === "error" && error) {
    const fieldErrors = Object.entries(error.fieldErrors ?? {});
    return (
      <section
        className={`${styles.feedback} ${styles.feedbackError}`}
        role="alert"
        tabIndex={-1}
        data-dialog-focus={error.kind === "validation" ? undefined : "true"}
      >
        <h3 className={styles.feedbackTitle}>{ERROR_TITLES[error.kind]}</h3>
        <p className={styles.feedbackDescription}>{error.message}</p>
        {fieldErrors.length ? (
          <ul className={styles.fieldErrorList}>
            {fieldErrors.map(([field, message]) => <li key={field}><strong>{field}:</strong> {message}</li>)}
          </ul>
        ) : null}
        {error.recovery ? <div>{error.recovery}</div> : null}
      </section>
    );
  }

  if (phase === "success" && completion?.status === "success") {
    return (
      <section
        className={`${styles.feedback} ${styles.feedbackSuccess}`}
        aria-label="Action result"
        aria-live="polite"
        tabIndex={-1}
        data-dialog-focus
      >
        <div className={styles.feedbackDescription}>{completion.description}</div>
      </section>
    );
  }

  if (phase === "uncertain" && completion?.status === "uncertain") {
    return (
      <>
        <section
          className={`${styles.feedback} ${styles.feedbackUncertain}`}
          aria-label="Action result"
          aria-live="polite"
          tabIndex={-1}
          data-dialog-focus
        >
          <div className={styles.feedbackDescription}>{completion.description}</div>
          {completion.showOperationReference !== false ? (
            <p className={styles.operationReference}><strong>Reference number:</strong> <code>{completion.operationKey}</code></p>
          ) : null}
        </section>
        {error ? (
          <section className={`${styles.feedback} ${styles.feedbackError}`} role="alert">
            <h3 className={styles.feedbackTitle}>We couldn’t check the result</h3>
            <p className={styles.feedbackDescription}>{error.message}</p>
          </section>
        ) : null}
      </>
    );
  }

  return null;
}

export function StaticActionFeedback({
  tone,
  title,
  description
}: {
  tone: "success" | "error" | "uncertain" | "working";
  title: string;
  description: ReactNode;
}) {
  const toneClass = {
    success: styles.feedbackSuccess,
    error: styles.feedbackError,
    uncertain: styles.feedbackUncertain,
    working: styles.feedbackWorking
  }[tone];
  return (
    <section className={`${styles.feedback} ${toneClass}`} role={tone === "error" ? "alert" : "status"}>
      <h3 className={styles.feedbackTitle}>{title}</h3>
      <div className={styles.feedbackDescription}>{description}</div>
    </section>
  );
}
