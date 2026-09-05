const MATCH_TYPE_LABELS: Record<string, string> = {
  popup: "Open play",
  "pop up": "Open play",
  "live match": "Club match",
  "league manager live": "League",
  "jupr live rated": "Rated event",
  "jupr live unrated": "Social event",
  challengeladder: "Challenge ladder",
  "challenge ladder": "Challenge ladder",
  "moneyball rr": "Round robin",
  "round robin": "Round robin"
};

export function publicMatchTypeLabel(value?: string | null): string {
  const raw = String(value || "").trim();
  if (!raw) return "—";
  const words = raw
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  const known = MATCH_TYPE_LABELS[words.toLowerCase()];
  if (known) return known;
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function publicSocialEventTypeLabel(value?: string | null): string {
  switch (String(value || "").trim().toLowerCase().replace(/[\s-]+/g, "_")) {
    case "round_robin":
      return "Round robin";
    case "league":
    case "league_ladder":
      return "League play";
    case "club_social":
      return "Club Social";
    default:
      return "Club Social event";
  }
}
