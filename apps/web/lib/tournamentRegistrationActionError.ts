import { InteractionActionError } from "@/components/interaction/types";

/** Keep registration validation visible inside the confirmation dialog. */
export function tournamentRegistrationActionError(
  response: Pick<Response, "status">,
  payload: unknown
): InteractionActionError {
  const detail = payload && typeof payload === "object" && "detail" in payload
    ? payload.detail : null;
  const status = response.status;
  const message = status === 401
    ? "Your sign-in expired. Sign in again before saving. Your edits are still here."
    : [400, 403, 409].includes(status) && typeof detail === "string"
      ? detail
      : status === 422
        ? "Some registration values are invalid. Review the registration before saving."
        : `The server could not confirm the save. Check the current registration before retrying (HTTP ${status}).`;
  return new InteractionActionError(message, {
    kind: status === 409 ? "conflict" : status === 401 || status === 403
      ? "forbidden" : status === 400 || status === 422 ? "validation" : "failed"
  });
}
