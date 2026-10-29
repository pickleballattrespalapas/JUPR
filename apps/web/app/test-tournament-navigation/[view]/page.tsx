import { cookies } from "next/headers";
import { notFound } from "next/navigation";

export default async function NavigationTestResult({ params, searchParams }: {
  params: Promise<{ view: string }>;
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  // Guard here too: page and layout work can be evaluated in parallel.
  if (process.env.JUPR_INTERACTION_TEST_HARNESS !== "1" || process.env.VERCEL_ENV === "production") notFound();
  const { view } = await params;
  const query = await searchParams;
  const fail = (await cookies()).get("pcs_navigation_test_fail")?.value === "1";
  if (view === "slow") await new Promise(resolve => setTimeout(resolve, 5_000));
  if (view === "failure" && fail) throw new Error("Synthetic navigation request failure");
  return <section><h2>{view} page opened</h2><p>Tournament: {query.tournament}; draw: {query.draw}; day: {query.day}</p></section>;
}
