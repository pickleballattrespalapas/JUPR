import { Suspense, type ReactNode } from "react";
import PublicLeagueResultsRouteNav from "@/components/PublicLeagueResultsRouteNav";

type Props = {
  children: ReactNode;
  params: { clubSlug: string };
};

export default function LeagueResultsLayout({ children, params }: Props) {
  return (
    <>
      <Suspense fallback={null}>
        <PublicLeagueResultsRouteNav clubSlug={params.clubSlug} />
      </Suspense>
      {children}
    </>
  );
}
