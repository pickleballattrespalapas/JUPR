"""HTTP boundary for the shared season registration phase rules."""
from fastapi import HTTPException

from jupr_app.services.interclub_registration_phase import (
    RegistrationPhaseError, registration_season, registration_state, require_registration_phase,
)


def require_season_phase(season, *, intake=False):
    try:
        return require_registration_phase(season, intake=intake)
    except RegistrationPhaseError as exc:
        raise HTTPException(423, str(exc)) from exc
