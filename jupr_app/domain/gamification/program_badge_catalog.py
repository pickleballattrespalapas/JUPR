"""The 22 approved program achievements; no implicit or seasonal thresholds."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProgramBadge:
    id: str
    name: str
    requirement: str
    category: str
    threshold: int
    repeatable: bool = False
    prestige: int = 20


PROGRAM_BADGES = [
    ProgramBadge(f"{kind}_completed_{n}", f"{n} {label} Completed", requirement.format(n=n), "Participation", n,
                 prestige={5: 20, 10: 35, 25: 60}[n])
    for kind, label, requirement in (
        ("leagues", "Leagues", "Finish {n} leagues at this club, meeting each league's required minimum games."),
        ("tournaments", "Tournaments", "Complete play in {n} finished tournaments at this club."),
        ("round_robins", "Round Robins", "Complete your assigned play in {n} finished round robins at this club."),
    ) for n in (5, 10, 25)
] + [
    ProgramBadge(f"{kind}_together_{n}", f"{n} {label} Together", f"{verb} {n} doubles matches with the same partner at this club.",
                 "Partnerships", n, True, {10: 20, 25: 35, 50: 60}[n])
    for kind, label, verb in (("matches", "Matches", "Play"), ("wins", "Wins", "Win")) for n in (10, 25, 50)
] + [
    ProgramBadge("five_winning_partners", "Five Winning Partners", "Win with at least 5 different partners in one completed round robin.", "Partnerships", 5, True, 40),
    ProgramBadge("triple_crown", "Triple Crown", "Earn a medal in at least 3 different events at the same completed tournament.", "Trophies", 3, True, 75),
] + [
    ProgramBadge(f"round_robin_wins_{n}", "First Round Robin Win" if n == 1 else f"{n} Round Robin Wins",
                 f"Win {n} completed round robin{'s' if n != 1 else ''} at this club. Winners are decided by wins, point differential, total points scored, then an admin decision if still tied.",
                 "Trophies", n, prestige={1: 20, 5: 35, 10: 50, 25: 75, 50: 100}[n])
    for n in (1, 5, 10, 25, 50)
]
PROGRAM_BADGE_IDS = frozenset(b.id for b in PROGRAM_BADGES)
RULE_VERSION = "program-badges-v1"
