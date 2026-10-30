"""Starting-estimate checks at rated generator admission and publication."""
from __future__ import annotations

from typing import Any

from jupr_app.services.starting_rating_review_service import require_starting_rating_review


class GeneratorProfileReviewRequired(ValueError):
    def __init__(self, names: list[str]):
        message = (
            "Before rated play, select a club profile for "
            + ", ".join(names[:6])
            + (" and the other unlinked players" if len(names) > 6 else "")
            + ". A club administrator must review starting ratings for players with no rated history."
        )
        super().__init__(message)
        self.detail = {"code": "STARTING_RATING_PROFILE_REQUIRED", "message": message}


def require_generator_starting_ratings(
    db: Any, *, club_id: str, event: dict[str, Any], before: dict[str, Any] | None = None,
) -> None:
    """Check only new identities on roster changes; never block score saving.

    Empty registration-driven groups may be created before check-in. Every
    admission is checked again when a participant is actually added.
    """
    if str(event.get("ratingMode") or "unrated") != "rated":
        return
    previous = {
        str(p.get("id")): p.get("player_id") for p in (before or {}).get("participants") or []
    }
    participants = [p for p in event.get("participants") or []
                    if before is None or str(p.get("id")) not in previous
                    or previous[str(p.get("id"))] != p.get("player_id")]
    if not participants:
        return
    missing = [str(p.get("name") or "this player") for p in participants
               if type(p.get("player_id")) is not int or p["player_id"] <= 0]
    if missing:
        raise GeneratorProfileReviewRequired(missing)
    player_ids = sorted({p["player_id"] for p in participants})
    play_format = str(event.get("playFormat") or "doubles")
    formats = ("doubles", "singles") if play_format == "doubles_singles" else (play_format,)
    for match_format in formats:
        require_starting_rating_review(db, club_id=club_id, player_ids=player_ids, match_format=match_format)


def require_published_starting_ratings(db: Any, *, club_id: str, matches: list[dict[str, Any]]) -> None:
    """Check only participants with rated games being published, by format."""
    for match_format in ("doubles", "singles"):
        ids = sorted({int(match[key]) for match in matches
                      if match.get("match_format") == match_format and match.get("rating_scope") != "unrated"
                      for key in ("t1_p1", "t1_p2", "t2_p1", "t2_p2") if match.get(key) is not None})
        if ids:
            require_starting_rating_review(db, club_id=club_id, player_ids=ids, match_format=match_format)
