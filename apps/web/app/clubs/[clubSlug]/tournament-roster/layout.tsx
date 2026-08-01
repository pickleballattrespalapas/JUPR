import { Suspense, type ReactNode } from "react";
import PublicTournamentRouteNav from "@/components/PublicTournamentRouteNav";

type Props = {
  children: ReactNode;
  params: { clubSlug: string };
};

export default function TournamentRosterLayout({ children, params }: Props) {
  return (
    <>
      <Suspense fallback={null}>
        <PublicTournamentRouteNav clubSlug={params.clubSlug} active="roster" />
      </Suspense>
      {children}
    </>
  );
}
