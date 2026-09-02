export type SetupDetailLoadState = "idle" | "loading" | "loaded" | "failed";

export type SetupPublicationStatus =
  | "checking"
  | "current"
  | "unpublished"
  | "unavailable";

type SetupPublicationStatusInput = {
  detailLoadState: SetupDetailLoadState;
  hasAuthoritativeDetail: boolean;
  hasUnpublishedChanges: boolean;
};

export function setupPublicationStatus({
  detailLoadState,
  hasAuthoritativeDetail,
  hasUnpublishedChanges
}: SetupPublicationStatusInput): SetupPublicationStatus {
  if (detailLoadState === "failed") return "unavailable";
  if (detailLoadState !== "loaded") return "checking";
  if (!hasAuthoritativeDetail) return "unavailable";
  return hasUnpublishedChanges ? "unpublished" : "current";
}
