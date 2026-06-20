from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ServiceContext:
    supabase: object
    club_id: str
    actor_email: str | None = None
    actor_role: str | None = None
    source: str | None = None
    public_base_url: str | None = None
