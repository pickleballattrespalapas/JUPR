"use client";
import { useEffect, useRef, useState } from "react";
import { apiError } from "@/lib/interclubRegistration";

type JsonRequest = <R>(url: string, method?: string, body?: object) => Promise<R>;
class RequestError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}
export function usePoolResource<T>(url: string, accessToken: string) {
  const [data, setData] = useState<T | null>(null), [loading, setLoading] = useState(true);
  const [error, setError] = useState(""), [busy, setBusy] = useState(false), [blocked, setBlocked] = useState(false);
  const [revision, setRevision] = useState(0);
  const token = useRef(accessToken); token.current = accessToken;
  const pending = useRef(false), mutation = useRef<AbortController | null>(null);
  const generation = useRef(0);
  useEffect(() => () => { generation.current++; mutation.current?.abort(); }, [url]);
  const request = (signal: AbortSignal): JsonRequest => async <R,>(address: string, method?: string, body?: object): Promise<R> => {
    if (signal.aborted) throw new DOMException("Aborted", "AbortError");
    const response = await fetch(address, { method, headers: { Authorization: `Bearer ${token.current}`, ...(body ? { "Content-Type": "application/json" } : {}) },
      cache: "no-store", signal, ...(body ? { body: JSON.stringify(body) } : {}) });
    const value = await response.json();
    if (signal.aborted) throw new DOMException("Aborted", "AbortError");
    if (!response.ok) throw new RequestError(apiError(value, "Unable to complete this request. Please try again."), response.status);
    return value as R;
  };
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setData(null); setError(""); setBlocked(false);
    request(controller.signal)<T>(url).then(value => { if (!controller.signal.aborted) setData(value); })
      .catch(reason => { if (!controller.signal.aborted) setError(reason instanceof RequestError ? reason.message : "Unable to load this section. Please try again."); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
    // Token refresh must preserve draft edits and selections.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, revision]);
  async function perform<R>(operation: (json: JsonRequest) => Promise<R>, onError?: (status: number) => void): Promise<R | null> {
    if (pending.current || blocked) return null;
    pending.current = true; setBusy(true); setError("");
    const controller = new AbortController(), current = generation.current; mutation.current = controller;
    try {
      const value = await operation(request(controller.signal));
      return controller.signal.aborted || current !== generation.current ? null : value;
    } catch (reason) {
      if (!controller.signal.aborted && current === generation.current) {
        onError?.(reason instanceof RequestError ? reason.status : 0);
        const uncertain = reason instanceof TypeError || reason instanceof SyntaxError || reason instanceof RequestError && [409, 503].includes(reason.status);
        setBlocked(uncertain);
        setError(reason instanceof RequestError ? reason.message : "Could not confirm the result. Reload this section before trying again.");
      }
      return null;
    } finally {
      if (current === generation.current && !controller.signal.aborted) { pending.current = false; setBusy(false); }
    }
  }
  return { data, setData, loading, error, busy, blocked, disabled: busy || blocked || loading,
    reload: () => { if (!pending.current) setRevision(value => value + 1); }, perform };
}
