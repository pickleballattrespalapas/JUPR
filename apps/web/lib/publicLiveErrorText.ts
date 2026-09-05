const SAFE_DETAIL_STATUSES = new Set([400, 403, 404, 409, 429, 503]);
const TECHNICAL_DETAIL = /\b(api|backend|database|environment|exception|fastapi|idempotenc\w*|operation[_ ]key|pgrst\w*|postgres\w*|request[_ ]fingerprint|rpc|schema|sql|supabase|traceback)\b/i;

export function publicLiveSafeDetail(status: number, detail: unknown): string | null {
  if (typeof detail !== "string" || !SAFE_DETAIL_STATUSES.has(status)) return null;
  const clean = detail.trim();
  if (!clean || clean.length > 240 || TECHNICAL_DETAIL.test(clean)) return null;
  return clean;
}

export function publicLiveErrorText(
  status: number,
  detail?: unknown,
  fallback = "We couldn’t complete that request. Please try again."
): string {
  const safeDetail = publicLiveSafeDetail(status, detail);
  if (safeDetail) return safeDetail;
  if (status === 403) return "This organizer link can’t make changes to this session.";
  if (status === 404) return "We couldn’t find this session.";
  if (status === 409) return "This session changed. Refresh the page and try again.";
  if (status === 429) return "Please wait a moment and try again.";
  if (status === 400 || status === 422) return "Check the information and try again.";
  return fallback;
}
