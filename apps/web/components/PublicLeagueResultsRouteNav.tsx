"use client";

import { useSearchParams } from "next/navigation";
import PublicLeagueNav, { type PublicLeagueModule } from "./PublicLeagueNav";

type Props = {
  clubSlug: string;
};

function activeModule(section: string): PublicLeagueModule {
  if (section === "weekly" || section === "player") return section;
  return "overall";
}

export default function PublicLeagueResultsRouteNav({ clubSlug }: Props) {
  const searchParams = useSearchParams();
  const leagueName = String(searchParams.get("league") || "").trim();
  const section = String(searchParams.get("section") || "overall").trim();
  if (!leagueName) return null;

  return (
    <PublicLeagueNav
      clubSlug={clubSlug}
      leagueName={leagueName}
      active={activeModule(section)}
    />
  );
}
