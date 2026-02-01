from __future__ import annotations

from dataclasses import dataclass


ALLOWED_BADGE_STATES = ("live", "frozen", "deprecated")


@dataclass(frozen=True)
class BadgeStateTransition:
    current: str
    target: str
    allowed: bool
    reason: str | None = None


def normalize_badge_state(state: str | None) -> str:
    if not state:
        return "live"
    return str(state).strip().lower()


def can_transition_badge_state(current: str, target: str, *, force: bool = False) -> BadgeStateTransition:
    current_state = normalize_badge_state(current)
    target_state = normalize_badge_state(target)

    if current_state not in ALLOWED_BADGE_STATES or target_state not in ALLOWED_BADGE_STATES:
        return BadgeStateTransition(
            current=current_state,
            target=target_state,
            allowed=False,
            reason="Unknown badge state.",
        )

    if current_state == target_state:
        return BadgeStateTransition(
            current=current_state,
            target=target_state,
            allowed=False,
            reason="State is already set.",
        )

    if force:
        return BadgeStateTransition(current=current_state, target=target_state, allowed=True)

    allowed_paths = {
        "live": {"frozen"},
        "frozen": {"deprecated"},
        "deprecated": set(),
    }
    if target_state in allowed_paths.get(current_state, set()):
        return BadgeStateTransition(current=current_state, target=target_state, allowed=True)

    return BadgeStateTransition(
        current=current_state,
        target=target_state,
        allowed=False,
        reason="Transition not allowed without force.",
    )
