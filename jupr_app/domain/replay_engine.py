from __future__ import annotations

from typing import Any, Dict


def replay_engine(*, supabase: Any, club_id: str) -> Dict[str, Any]:
    """Deprecated: replay history must run through jupr_app.domain.replay_history.replay_history."""
    _ = (supabase, club_id)
    raise RuntimeError(
        "replay_engine is disabled to prevent divergent replay semantics. "
        "Use jupr_app.domain.replay_history.replay_history as the canonical engine."
    )
