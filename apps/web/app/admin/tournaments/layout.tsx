import type { ReactNode } from "react";
import TournamentAdminNav from "@/components/TournamentAdminNav";

export default function AdminTournamentsLayout({ children }: { children: ReactNode }) {
  return (
    <>
      <TournamentAdminNav />
      {children}
    </>
  );
}
