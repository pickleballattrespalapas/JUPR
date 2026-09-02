import { notFound } from "next/navigation";

import { InteractionProviderHarness } from "./InteractionProviderHarness";

// This behavioral harness must never be reachable as a product page. It is
// enabled only for an explicit local/CI interaction-foundation test run.
export const dynamic = "force-dynamic";

export default function InteractionProviderHarnessPage() {
  if (process.env.JUPR_INTERACTION_TEST_HARNESS !== "1") notFound();
  return <InteractionProviderHarness />;
}
