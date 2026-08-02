import { redirect } from "next/navigation";

type Props = { searchParams?: Record<string, string | string[] | undefined> };

function first(value: string | string[] | undefined): string {
  return Array.isArray(value) ? String(value[0] || "") : String(value || "");
}

export default function TournamentSetupRegistrationRulesPage({ searchParams }: Props) {
  const tournament = first(searchParams?.tournament).trim();
  const name = first(searchParams?.name).trim();
  const params = new URLSearchParams();
  if (tournament) params.set("tournament", tournament);
  if (name) params.set("name", name);
  redirect(`/admin/tournaments/setup/basics?${params.toString()}`);
}
