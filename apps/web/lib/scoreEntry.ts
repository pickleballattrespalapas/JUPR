export function isNextAdminScoreEntryEnabled(): boolean {
  return ["1", "true", "yes", "on"].includes(
    String(process.env.NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY || "")
      .trim()
      .toLowerCase()
  );
}
