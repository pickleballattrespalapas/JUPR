import { Suspense, type ReactNode } from "react";
import TournamentAdminNav from "@/components/TournamentAdminNav";

export default function AdminTournamentsLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <Suspense fallback={<div aria-hidden="true" style={{ minHeight: "42px", marginBottom: "1rem" }} />}>
        <TournamentAdminNav />
      </Suspense>
      {children}
    </>
  );
}
