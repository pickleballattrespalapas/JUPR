import type { ReactNode } from "react";

export type ActionErrorKind = "validation" | "conflict" | "forbidden" | "failed";

export type FieldErrors = Record<string, string>;

export type ActionSuccess = {
  status: "success";
  title: string;
  description: ReactNode;
  closeLabel?: string;
  focusTargetId?: string;
};

export type ActionUncertain = {
  status: "uncertain";
  title: string;
  description: ReactNode;
  operationKey: string;
  showOperationReference?: boolean;
  recoveryLabel: string;
  onRecover: () => Promise<ActionCompletion>;
};

export type ActionCompletion = ActionSuccess | ActionUncertain;

export type InteractionActionErrorOptions = {
  kind?: ActionErrorKind;
  fieldErrors?: FieldErrors;
  recovery?: ReactNode;
  cause?: unknown;
};

/**
 * A user-facing, classified mutation failure. Throw this from action callbacks
 * when the server has proved that the write did not complete.
 */
export class InteractionActionError extends Error {
  readonly kind: ActionErrorKind;
  readonly fieldErrors?: FieldErrors;
  readonly recovery?: ReactNode;

  constructor(message: string, options: InteractionActionErrorOptions = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = "InteractionActionError";
    this.kind = options.kind ?? "failed";
    this.fieldErrors = options.fieldErrors;
    this.recovery = options.recovery;
  }
}

export type ActionCallback = (confirmationText: string) => Promise<ActionCompletion>;

export function actionSuccess(
  title: string,
  description: ReactNode,
  closeLabel?: string
): ActionSuccess {
  return {
    status: "success",
    title,
    description,
    ...(closeLabel ? { closeLabel } : {})
  };
}

export function actionUncertain(
  title: string,
  description: ReactNode,
  operationKey: string,
  recoveryLabel: string,
  onRecover: () => Promise<ActionCompletion>,
  showOperationReference = true
): ActionUncertain {
  return {
    status: "uncertain",
    title,
    description,
    operationKey,
    showOperationReference,
    recoveryLabel,
    onRecover
  };
}

export function isActionCompletion(value: unknown): value is ActionCompletion {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<ActionCompletion>;
  if (candidate.status === "success") {
    return typeof candidate.title === "string" && "description" in candidate;
  }
  if (candidate.status === "uncertain") {
    return (
      typeof candidate.title === "string" &&
      "description" in candidate &&
      typeof candidate.operationKey === "string" &&
      typeof candidate.recoveryLabel === "string" &&
      typeof candidate.onRecover === "function"
    );
  }
  return false;
}

export function normalizeInteractionActionError(error: unknown): InteractionActionError {
  if (error instanceof InteractionActionError) return error;
  return new InteractionActionError(
    "We couldn’t complete that action. Your changes are still here. Check the details and try again.",
    { cause: error }
  );
}
