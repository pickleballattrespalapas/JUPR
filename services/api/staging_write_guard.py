from __future__ import annotations

from fastapi import HTTPException

from jupr_app.services.staging_write_guard import (
    require_staging_league_manager_writes,
    require_staging_public_intake_writes,
    require_staging_admin_team_league_writes,
    require_staging_public_team_league_writes,
)


def require_public_intake_or_403() -> None:
    try:
        require_staging_public_intake_writes()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def require_league_manager_write_or_403() -> None:
    try:
        require_staging_league_manager_writes()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def require_admin_team_league_write_or_403() -> None:
    try:
        require_staging_admin_team_league_writes()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def require_public_team_league_write_or_403() -> None:
    try:
        require_staging_public_team_league_writes()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
