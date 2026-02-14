from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElevationLevel:
    level: int
    light_shadow: str
    dark_shadow: str


ELEVATION_LEVELS: dict[int, ElevationLevel] = {
    0: ElevationLevel(level=0, light_shadow="none", dark_shadow="none"),
    1: ElevationLevel(
        level=1,
        light_shadow="0 1px 2px rgba(15, 23, 42, 0.08), 0 1px 1px rgba(15, 23, 42, 0.04)",
        dark_shadow="0 1px 2px rgba(0, 0, 0, 0.45), 0 1px 1px rgba(0, 0, 0, 0.28)",
    ),
    2: ElevationLevel(
        level=2,
        light_shadow="0 6px 18px rgba(15, 23, 42, 0.12), 0 2px 6px rgba(15, 23, 42, 0.08)",
        dark_shadow="0 8px 20px rgba(0, 0, 0, 0.52), 0 3px 8px rgba(0, 0, 0, 0.36)",
    ),
    3: ElevationLevel(
        level=3,
        light_shadow="0 14px 36px rgba(15, 23, 42, 0.18), 0 4px 12px rgba(15, 23, 42, 0.10)",
        dark_shadow="0 16px 40px rgba(0, 0, 0, 0.62), 0 6px 14px rgba(0, 0, 0, 0.42)",
    ),
}


def get_elevation_shadow(level: int, *, dark_mode: bool = False) -> str:
    token = ELEVATION_LEVELS.get(level, ELEVATION_LEVELS[0])
    return token.dark_shadow if dark_mode else token.light_shadow
