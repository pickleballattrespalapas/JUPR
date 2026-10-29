import { notFound } from "next/navigation";
import type { ReactNode } from "react";
import Controls from "./Controls";

export const dynamic = "force-dynamic";

export default function NavigationTestLayout({ children }: { children: ReactNode }) {
  if (process.env.JUPR_INTERACTION_TEST_HARNESS !== "1" || process.env.VERCEL_ENV === "production") notFound();
  return <main><h1>Tournament navigation test</h1><Controls />{children}</main>;
}
