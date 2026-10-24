"""Shared strict score input for admin, public and shared generators."""
from __future__ import annotations
from typing import TypeVar
from pydantic import BaseModel, ConfigDict, Field, model_validator


def require_complete_score_pair(score_a: int | None, score_b: int | None, match_id: str) -> None:
    if (score_a is None) != (score_b is None):
        raise ValueError(f"Enter both scores for game {match_id}, or leave both blank for not played.")


class GeneratorScoreInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    match_id: str = Field(strict=True, min_length=1, max_length=160, pattern=r"\S")
    score_a: int | None = Field(strict=True, ge=0, le=99)
    score_b: int | None = Field(strict=True, ge=0, le=99)

    @model_validator(mode="after")
    def complete_pair(self) -> "GeneratorScoreInput":
        require_complete_score_pair(self.score_a, self.score_b, self.match_id)
        return self


ScoreInput = TypeVar("ScoreInput", bound=GeneratorScoreInput)


def require_unique_match_ids(scores: list[ScoreInput]) -> list[ScoreInput]:
    seen: set[str] = set()
    for score in scores:
        if score.match_id in seen:
            raise ValueError(f"Duplicate game {score.match_id}: submit each game's scores only once.")
        seen.add(score.match_id)
    return scores
