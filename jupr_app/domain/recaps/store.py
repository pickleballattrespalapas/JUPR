from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol


RecapVisibility = str


@dataclass(frozen=True)
class FeaturedUpcomingEvent:
    title: str
    datetime: str
    location: str
    reg_url: str
    pitch: str
    event_id: str | None = None


@dataclass(frozen=True)
class FeaturedPastEvent:
    title: str
    datetime: str
    location: str
    summary_bullets: list[str]
    link_url: str
    link_label: str
    event_id: str | None = None


@dataclass
class RecapRecord:
    recap_id: str
    club_id: str
    level: str
    status: str
    report_start: str
    report_end: str
    timezone: str = "America/Mexico_City"
    visibility: RecapVisibility = "public"
    featured_upcoming_event: FeaturedUpcomingEvent | None = None
    featured_past_event: FeaturedPastEvent | None = None
    content_snapshot: dict[str, Any] = field(default_factory=dict)
    published_at: str | None = None


class RecapStore(Protocol):
    def save(self, recap: RecapRecord) -> RecapRecord: ...

    def get(self, recap_id: str) -> RecapRecord | None: ...

    def publish(self, recap_id: str, *, visibility: RecapVisibility | None = None) -> RecapRecord: ...


class InMemoryRecapStore:
    def __init__(self) -> None:
        self._records: dict[str, RecapRecord] = {}

    def save(self, recap: RecapRecord) -> RecapRecord:
        saved = deepcopy(recap)
        self._records[recap.recap_id] = saved
        return deepcopy(saved)

    def get(self, recap_id: str) -> RecapRecord | None:
        recap = self._records.get(recap_id)
        return deepcopy(recap) if recap else None

    def publish(self, recap_id: str, *, visibility: RecapVisibility | None = None) -> RecapRecord:
        recap = self._records.get(recap_id)
        if recap is None:
            raise KeyError(f"Recap not found: {recap_id}")
        if recap.featured_upcoming_event is None:
            raise ValueError("featured_upcoming_event is required at publish time")

        recap.status = "published"
        recap.visibility = visibility or "public"
        recap.published_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._records[recap_id] = deepcopy(recap)
        return deepcopy(recap)
