/** Standings requests must settle even when headers arrive but the body stalls. */
export class StandingsRequestError extends Error {
  constructor(message: string, readonly status: number) { super(message); this.name = "StandingsRequestError"; }
}

export async function requestStandingsJson<T>(url: string, options: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  let abort: (() => void) | undefined;
  const deadline = new Promise<never>((_resolve, reject) => {
    abort = () => { controller.abort(); reject(new StandingsRequestError("The standings request was cancelled.", 499)); };
    options.signal?.addEventListener("abort", abort, { once: true });
    if (options.signal?.aborted) abort();
    timer = setTimeout(() => {
      reject(new StandingsRequestError("The request timed out. Refresh standings or retry the saved action to check its result.", 408));
      controller.abort();
    }, 30_000);
  });
  try {
    return await Promise.race([deadline, (async () => {
      const response = await fetch(url, { ...options, signal: controller.signal });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = payload?.detail;
        const message = typeof detail === "string" ? detail : detail && typeof detail.message === "string" ? detail.message : `API error (${response.status})`;
        throw new StandingsRequestError(message, response.status);
      }
      if (!payload) throw new StandingsRequestError("The server did not confirm the result. Retry the saved action to check it.", 502);
      return payload as T;
    })()]);
  } finally {
    clearTimeout(timer);
    if (abort) options.signal?.removeEventListener("abort", abort);
  }
}

export function uncertainStandingsError(error: unknown): boolean {
  return !(error instanceof StandingsRequestError) || error.status >= 500 || [408, 425, 429].includes(error.status);
}

export type StandingsMutation = { action: "advance" | "complete" | "playoff" | "submit"; body: string };

export function storedStandingsMutation(key: string): StandingsMutation | null {
  try {
    const value = JSON.parse(sessionStorage.getItem(key) || "null");
    if (!value || !["advance", "complete", "playoff", "submit"].includes(value.action) || typeof value.body !== "string") return null;
    const body = JSON.parse(value.body);
    return body && typeof body === "object" && typeof body.expected_version === "string" ? value : null;
  } catch { return null; }
}

export function rememberStandingsMutation(key: string, request: StandingsMutation | null): void {
  try { if (request) sessionStorage.setItem(key, JSON.stringify(request)); else sessionStorage.removeItem(key); } catch { /* In-memory recovery remains available. */ }
}
