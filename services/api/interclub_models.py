"""Interclub planning models: incomplete drafts are saved; invitations require a complete setup."""
from datetime import date, datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Division = Literal['2.5','3.0','3.5','4.0','4.5','5.0','Open','4.5/Open']

class DivisionRule(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    min_rating: float | None = Field(default=None, ge=1, le=7)
    max_rating: float | None = Field(default=None, ge=1, le=7)
    women_required: int | None = Field(default=None, ge=0, le=4)

    @model_validator(mode="after")
    def ordered(self):
        if self.min_rating is not None and self.max_rating is not None and self.min_rating > self.max_rating:
            raise ValueError("Minimum rating cannot exceed maximum rating.")
        return self

class PlanningMeet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host_club_id: str = Field(default="", max_length=100)
    club_ids: list[str] = Field(default_factory=list, max_length=4)
    starts_at: AwareDatetime | None = None
    duration_minutes: int | None = Field(default=180, ge=30, le=180)
    courts: int | None = Field(default=4, ge=1, le=100)

class PlanningDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(default="", max_length=120)
    start_date: date | None = None
    end_date: date | None = None
    timezone: str = "America/Mazatlan"
    divisions: list[Division] = Field(default_factory=list, max_length=8)
    club_ids: list[str] = Field(default_factory=list, max_length=32)
    meets: list[PlanningMeet] = Field(default_factory=list, max_length=100)
    registration_rules: dict[Division, DivisionRule] = Field(default_factory=dict, max_length=8)
    setup_step: int = Field(default=0, ge=0, le=4)

    @model_validator(mode="after")
    def valid_draft(self):
        try: ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError): raise ValueError("Choose a valid timezone.")
        if len(set(self.club_ids)) != len(self.club_ids) or len(set(self.divisions)) != len(self.divisions):
            raise ValueError("Remove duplicate clubs or divisions.")
        return self

class MeetDraft(BaseModel):
    host_club_id: str = Field(min_length=1, max_length=100)
    club_ids: list[str] = Field(min_length=2, max_length=4)
    starts_at: datetime
    duration_minutes: int = Field(default=180, ge=30, le=180)
    courts: int = Field(default=4, ge=1, le=100)

class SeasonDraft(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    start_date: date
    end_date: date
    timezone: str = 'America/Mazatlan'
    divisions: list[Literal['2.5','3.0','3.5','4.0','4.5','5.0','Open','4.5/Open']] = Field(min_length=1, max_length=8)
    club_ids: list[str] = Field(default_factory=list, max_length=32)
    meets: list[MeetDraft] = Field(default_factory=list, max_length=100)
    registration_rules: dict[Division, DivisionRule] = Field(default_factory=dict, max_length=8)
    setup_step: int = Field(default=0, ge=0, le=4)

    @model_validator(mode='after')
    def consistent(self):
        self.name = self.name.strip()
        if not self.name: raise ValueError('Enter a season name.')
        if self.end_date < self.start_date: raise ValueError('End date must follow start date.')
        try: zone=ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError): raise ValueError('Choose a valid timezone.')
        if len(set(self.club_ids)) != len(self.club_ids) or len(set(self.divisions)) != len(self.divisions):
            raise ValueError('Remove duplicate clubs or divisions.')
        intervals=[]
        for meet in self.meets:
            if len(set(meet.club_ids)) != len(meet.club_ids): raise ValueError('A club can appear only once in a meet.')
            if not set(meet.club_ids).issubset(self.club_ids): raise ValueError('Meet clubs must be selected for this season.')
            if meet.host_club_id not in meet.club_ids: raise ValueError('The host must play in its meet.')
            if meet.starts_at.tzinfo is None: raise ValueError('Meet time must include a timezone offset.')
            start=meet.starts_at; end=start+timedelta(minutes=meet.duration_minutes)
            if not self.start_date <= start.astimezone(zone).date() <= self.end_date or end.astimezone(zone).date()>self.end_date:
                raise ValueError('Meet time must fall within the season dates.')
            for old_start,old_end,old_clubs in intervals:
                if start<old_end and end>old_start and set(meet.club_ids)&old_clubs:
                    raise ValueError('A club cannot attend overlapping meets.')
            intervals.append((start,end,set(meet.club_ids)))
        return self
