"""Explicit community recognition; called by the admin award flow only."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from uuid import UUID

from jupr_app.domain.gamification.badge_types import BadgeCandidate


COMMUNITY_CRITERIA = {
    "good_sport": {
        "honest_calls": "Honest calls, even when they cost a point",
        "respectful_resolution": "Respectful handling of a difficult situation",
        "encouragement": "Consistently encouraging and supporting other players",
    },
    "community_builder": {
        "welcome_newcomers": "Welcoming newcomers and helping them find games",
        "volunteer": "Volunteering at club events",
        "inclusive_play": "Organizing inclusive social play",
        "introduce_participants": "Introducing new participants to the club",
    },
    "mentor": {
        "ongoing_help": "Helping a newer player improve over several visits",
        "lead_learning": "Leading beginner lessons or practice sessions",
        "rules_and_etiquette": "Helping players learn rules and court etiquette",
    },
}


def build_community_award(
    *, club_id: str, player_id: int, badge_id: str, recognition_id: str,
    criteria: Sequence[str], note: str, contribution_date: date,
) -> BadgeCandidate:
    """Build one recognition after the caller verifies club admin authorization.

    The same recognition ID must survive retries. A separate contribution gets
    a new ID, allowing repeated awards without a lifetime or season cap. The
    write service must also bind the complete request to its idempotency key.
    """
    if badge_id not in COMMUNITY_CRITERIA:
        raise ValueError("Choose a community badge.")
    if not str(club_id).strip() or isinstance(player_id, bool) or not isinstance(player_id, int) or player_id <= 0:
        raise ValueError("Choose a club player.")
    recognition_id = str(UUID(str(recognition_id)))
    if isinstance(criteria, str) or not criteria:
        raise ValueError("Choose at least one qualifying action.")
    selected = sorted(set(criteria))
    if any(key not in COMMUNITY_CRITERIA[badge_id] for key in selected):
        raise ValueError("The qualifying action does not belong to this badge.")
    note = str(note or "").strip()
    if not note or len(note) > 1000:
        raise ValueError("Add a recognition note of 1 to 1,000 characters.")
    return BadgeCandidate(
        badge_id=badge_id,
        player_id=player_id,
        club_id=club_id,
        context_type="overall",
        context_id=f"community-recognition:{recognition_id}",
        match_id=None,
        value_json={
            "recognition_id": recognition_id,
            "criteria": selected,
            "qualifying_actions": [COMMUNITY_CRITERIA[badge_id][key] for key in selected],
            "recognition_note": note,
            "contribution_date": contribution_date.isoformat(),
        },
    )
