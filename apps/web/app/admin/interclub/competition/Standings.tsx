"use client";

import type { CompetitionContext } from "@/lib/interclubCompetition";
import { CompetitionStandings } from "@/components/PublicInterclubCompetition";
import styles from "./competition.module.css";

export default function Standings({ data, clubName }: { data: CompetitionContext; clubName: (id: string) => string }) {
  const divisions = data.standings?.divisions || {};
  const ids = new Set([...(data.clubs || []).map(club => club.id), ...Object.values(divisions).flatMap(rows => rows.map(row => row.club_id)), ...(data.club_cup?.standings || []).map(row => row.club_id)]);
  return <details className={styles.section}>
    <summary>Season standings &amp; Club Cup</summary>
    <CompetitionStandings league={{ document: { name: data.season.details.name, start_date: data.season.details.start_date, end_date: data.season.details.end_date,
      timezone: data.season.details.timezone, divisions: data.season.details.divisions, clubs: [...ids].map(id => ({ id, name: clubName(id) })), meets: [], results: [], scoring_version: 1 },
      standings: Object.entries(divisions).map(([division, rows]) => ({ division, rows })),
      club_cup: { ...data.club_cup, standings: (data.club_cup?.standings || []).map(row => ({ ...row, regular_points: row.regular_points ?? 0, championship_points: row.championship_points ?? 0 })) },
      qualification: data.standings.qualification || data.qualifying,
    }} />
  </details>;
}
