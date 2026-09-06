import { InteractionActionError } from "@/components/interaction/types";

/** Preserve actionable API rejections without exposing internal server errors. */
export function tournamentSetupActionError(
  response: Pick<Response, "status" | "headers">,
  payload: unknown
): InteractionActionError {
  const detail = payload && typeof payload === "object" && "detail" in payload
    ? payload.detail : null;
  const status = response.status;
  const requestId = response.headers.get("x-request-id");
  const reference = requestId && /^[a-zA-Z0-9-]{1,100}$/.test(requestId)
    ? ` Request reference: ${requestId}.` : "";
  const message = status === 401
    ? "Your sign-in expired. Sign in again before saving. Your sponsor edits are still here."
    : [400, 403, 409].includes(status) && typeof detail === "string"
      ? detail
      : status === 422
        ? "Some tournament setup values are invalid. Review the draft before saving."
        : `The server could not confirm the save (HTTP ${status}). Check the current tournament draft before retrying.${reference}`;
  return new InteractionActionError(message, {
    kind: status === 409 ? "conflict" : status === 401 || status === 403
      ? "forbidden" : status === 400 || status === 422 ? "validation" : "failed"
  });
}
