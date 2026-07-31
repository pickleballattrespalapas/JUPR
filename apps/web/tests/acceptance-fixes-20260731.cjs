const fs = require("fs");

function read(path) { return fs.readFileSync(path, "utf8"); }
function expect(path, text) {
  const body = read(path);
  if (!body.includes(text)) throw new Error(`${path} is missing: ${text}`);
}

expect("app/admin/league-manager/LeagueManagerNav.tsx", 'aria-current={active ? "page" : undefined}');
expect("app/admin/league-manager/LeagueManagerNav.tsx", '"/admin/league-manager/results"');
expect("app/admin/league-manager/roster/LeagueRosterPanel.tsx", 'useState<RosterFilter>("in_league")');
expect("app/admin/league-manager/roster/LeagueRosterPanel.tsx", '"Add Player"');
expect("app/admin/league-manager/LeagueManagerPanel.tsx", 'League mode');
expect("app/admin/league-manager/LeagueManagerPanel.tsx", 'League home');
expect("app/admin/league-manager/results/page.tsx", 'League results');
expect("app/admin/players/PlayerEditorPanel.tsx", 'Player update complete');
expect("app/admin/match-log/MatchLogApplyPanel.tsx", 'score ${oldTeam1}-${oldTeam2} → ${newTeam1}-${newTeam2}');
expect("app/admin/league-manager/teams/TeamLeaguesPanel.tsx", 'America/Mazatlan');
expect("app/admin/league-manager/teams/TeamLeaguesPanel.tsx", 'league.league_type');
expect("app/admin/tournaments/TournamentAdminPanel.tsx", 'Tournament home');
expect("components/TournamentAdminNav.tsx", 'Tournament home');
console.log("July 31 acceptance source contracts passed.");

expect("app/admin/league-manager/roster/LeagueRosterPanel.tsx", 'setFilter("in_league")');
expect("app/admin/tournaments/TournamentAdminPanel.tsx", 'display: detail ? "none" : "block"');
