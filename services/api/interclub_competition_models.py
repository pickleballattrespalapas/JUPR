"""Versioned Southern BCS score documents, shared by draft and official writes.

Drafts deliberately allow unfinished scores. The domain validator performs the
additional checks required to submit one complete, official meet batch.
"""
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class CompetitionGame(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=120)
    status: Literal["pending", "completed", "retired", "forfeit", "double_forfeit", "unplayed"] = "pending"
    a: int | None = Field(default=None, ge=0, strict=True)
    b: int | None = Field(default=None, ge=0, strict=True)
    winner: Literal["a", "b"] | None = None
    players_a: list[str] = Field(default_factory=list, max_length=2)
    players_b: list[str] = Field(default_factory=list, max_length=2)
    played_at: AwareDatetime | None = None
    injury_reason: str | None = Field(default=None, max_length=1000)


class CompetitionPairing(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=120)
    kind: Literal["women", "men", "mixed_a", "mixed_b"]
    court: int | None = Field(default=None, ge=1, le=100)
    eligibility_deadline: AwareDatetime | None = None
    players_a: list[str] = Field(default_factory=list, max_length=2)
    players_b: list[str] = Field(default_factory=list, max_length=2)
    games: list[CompetitionGame] = Field(min_length=1, max_length=3)


class RotatingSingles(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["pending", "completed"] = "pending"
    a: int | None = Field(default=None, ge=0, strict=True)
    b: int | None = Field(default=None, ge=0, strict=True)
    order_a: list[str] = Field(default_factory=list, max_length=4)
    order_b: list[str] = Field(default_factory=list, max_length=4)


class CompetitionEncounter(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=120)
    division: str = Field(min_length=1, max_length=30)
    club_a: str = Field(min_length=1, max_length=100)
    club_b: str = Field(min_length=1, max_length=100)
    rotation: int = Field(default=1, ge=1, le=100)
    pairings: list[CompetitionPairing] = Field(min_length=2, max_length=4)
    tiebreak: RotatingSingles | None = None


class CompetitionDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    meet_id: str = Field(min_length=1, max_length=120)
    phase: Literal["regular", "final", "qualifier"] = "regular"
    format: Literal["gender", "mixed", "mlp"] = "gender"
    schedule_mode: Literal["simultaneous", "staggered"] = "simultaneous"
    weather: Literal["normal", "delay", "rescheduled", "finalized_partial"] = "normal"
    encounters: list[CompetitionEncounter] = Field(default_factory=list, max_length=2000)
