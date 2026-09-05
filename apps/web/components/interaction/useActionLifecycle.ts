"use client";

import { useCallback, useRef, useState } from "react";

import {
  InteractionActionError,
  isActionCompletion,
  normalizeInteractionActionError,
  type ActionCompletion
} from "./types";

export type ActionPhase = "ready" | "working" | "success" | "error" | "uncertain";

export type ActionLifecycle = {
  phase: ActionPhase;
  completion: ActionCompletion | null;
  error: InteractionActionError | null;
  run: (action: () => Promise<ActionCompletion>) => Promise<ActionCompletion | null>;
  recover: (action: () => Promise<ActionCompletion>) => Promise<ActionCompletion | null>;
  reset: () => void;
};

/** Owns the visible lifecycle and a synchronous lock for one mutation. */
export function useActionLifecycle(): ActionLifecycle {
  const inFlightRef = useRef(false);
  const [phase, setPhase] = useState<ActionPhase>("ready");
  const [completion, setCompletion] = useState<ActionCompletion | null>(null);
  const [error, setError] = useState<InteractionActionError | null>(null);

  const reset = useCallback(() => {
    if (inFlightRef.current) return;
    setPhase("ready");
    setCompletion(null);
    setError(null);
  }, []);

  const run = useCallback(async (action: () => Promise<ActionCompletion>) => {
    if (inFlightRef.current) return null;

    inFlightRef.current = true;
    setPhase("working");
    setCompletion(null);
    setError(null);

    try {
      const result = await action();
      if (!isActionCompletion(result)) {
        throw new InteractionActionError(
          "We couldn’t confirm what happened. Your changes are still here; check the page before trying again."
        );
      }
      setCompletion(result);
      setPhase(result.status);
      return result;
    } catch (actionError) {
      setError(normalizeInteractionActionError(actionError));
      setPhase("error");
      return null;
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  const recover = useCallback(async (action: () => Promise<ActionCompletion>) => {
    if (inFlightRef.current) return null;

    inFlightRef.current = true;
    setPhase("working");
    setError(null);

    try {
      const result = await action();
      if (!isActionCompletion(result)) {
        throw new InteractionActionError(
          "We still couldn’t confirm the result. Check again before repeating the action."
        );
      }
      setCompletion(result);
      setPhase(result.status);
      return result;
    } catch (recoveryError) {
      setError(normalizeInteractionActionError(recoveryError));
      // The mutation is still uncertain. Retain its exact completion record so
      // the UI cannot offer a blind repeat of the original write.
      setPhase("uncertain");
      return null;
    } finally {
      inFlightRef.current = false;
    }
  }, []);

  return { phase, completion, error, run, recover, reset };
}
