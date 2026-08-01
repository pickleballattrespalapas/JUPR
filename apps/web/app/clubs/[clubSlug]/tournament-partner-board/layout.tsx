import { Suspense, type ReactNode } from "react";
import PublicTournamentRouteNav from "@/components/PublicTournamentRouteNav";

type Props = {
  children: ReactNode;
  params: { clubSlug: string };
};

export default function TournamentPartnerBoardLayout({ children, params }: Props) {
  return (
    <>
      <Suspense fallback={null}>
        <PublicTournamentRouteNav
          clubSlug={params.clubSlug}
          active="partner-board"
        />
      </Suspense>
      {children}
    </>
  );
}
