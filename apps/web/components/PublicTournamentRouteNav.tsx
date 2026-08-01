"use client";

import { useSearchParams } from "next/navigation";
import PublicTournamentNav, {
  type PublicTournamentModule
} from "./PublicTournamentNav";

type Props = {
  clubSlug: string;
  active: PublicTournamentModule;
};

export default function PublicTournamentRouteNav({ clubSlug, active }: Props) {
  const searchParams = useSearchParams();
  const registrationSlug = String(searchParams.get("tournament") || "").trim();
  const tournamentId = String(searchParams.get("tournament_id") || "").trim();

  return (
    <PublicTournamentNav
      clubSlug={clubSlug}
      tournamentId={tournamentId || null}
      registrationSlug={registrationSlug || null}
      active={active}
    />
  );
}
