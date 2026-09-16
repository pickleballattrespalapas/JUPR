"""Validated, portable website documents. No arbitrary HTML, CSS or scripts."""
from __future__ import annotations

import re
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def safe_link(value: str, *, image: bool = False) -> str:
    if not value:
        return value
    if any(ord(char) <= 32 for char in value):
        raise ValueError("Use a URL without spaces or control characters.")
    if image and re.fullmatch(r"data:image/(png|jpeg|webp);base64,[A-Za-z0-9+/=]+", value):
        if len(value) > 300_000:
            raise ValueError("Use an image smaller than 220 KB or an HTTPS image URL.")
        return value
    if not image and value.startswith("/") and not value.startswith("//") and "\\" not in value:
        return value
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or "\\" in value:
        raise ValueError("Use a full HTTPS address or a local page path.")
    return value


class SiteBlock(StrictModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    kind: Literal["text", "image", "button", "divider", "links"] = "text"
    heading: str = Field(default="", max_length=160)
    text: str = Field(default="", max_length=12000)
    url: str = Field(default="", max_length=300000)
    alt: str = Field(default="", max_length=240)
    span: Literal[3, 4, 6, 8, 12] = 12
    align: Literal["left", "center", "right"] = "left"
    tone: Literal["plain", "soft", "accent"] = "plain"
    padding: Literal["small", "medium", "large"] = "medium"

    @model_validator(mode="after")
    def url_safe(self):
        safe_link(self.url, image=self.kind == "image")
        if self.kind == "image" and self.url and not self.alt:
            raise ValueError("Describe each image for visitors using a screen reader.")
        return self


class SitePage(StrictModel):
    slug: str = Field(pattern=r"^(home|[a-z0-9]+(?:-[a-z0-9]+)*)$", max_length=60)
    title: str = Field(min_length=1, max_length=80)
    in_navigation: bool = True
    blocks: list[SiteBlock] = Field(default_factory=list, max_length=40)

    @model_validator(mode="after")
    def unique_blocks(self):
        if len({b.id for b in self.blocks}) != len(self.blocks):
            raise ValueError("Block IDs must be unique on each page.")
        return self


class DisplaySettings(StrictModel):
    ratings: bool = True
    records: bool = True
    match_counts: bool = True
    win_percentage: bool = True
    rating_changes: bool = True
    singles: bool = True
    last_played: bool = True
    player_status: bool = True
    qualification: bool = True
    badges: bool = True
    profile_ratings: bool = True
    profile_positions: bool = True
    profile_trophies: bool = True
    profile_social: bool = True
    profile_matches: bool = True
    result_scores: bool = True
    result_summary: bool = True
    registration_description: bool = True
    registration_schedule: bool = True


class SiteDocument(StrictModel):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=6000)
    location: str = Field(default="", max_length=300)
    visitor_info: str = Field(default="", max_length=6000)
    logo_url: str = Field(default="", max_length=300000)
    accent: str = Field(default="#1d4ed8", pattern=r"^#[0-9a-fA-F]{6}$")
    visibility: Literal["listed", "unlisted"] = "unlisted"
    display: DisplaySettings = Field(default_factory=DisplaySettings)
    pages: list[SitePage] = Field(default_factory=lambda: [SitePage(slug="home", title="Home")], min_length=1, max_length=20)

    @field_validator("logo_url")
    @classmethod
    def safe_logo(cls, value):
        return safe_link(value, image=True)

    @model_validator(mode="after")
    def unique_pages(self):
        slugs = [page.slug for page in self.pages]
        if "home" not in slugs or len(slugs) != len(set(slugs)):
            raise ValueError("Keep one home page and a unique address for each custom page.")
        if len(self.model_dump_json()) > 2_000_000:
            raise ValueError("This website is too large. Use HTTPS URLs for larger images.")
        return self


class SiteSave(StrictModel):
    revision: int = Field(ge=0)
    document: SiteDocument


class SiteAction(StrictModel):
    revision: int = Field(ge=0)


class ClubCreate(StrictModel):
    slug: str = Field(min_length=3, max_length=60, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    name: str = Field(min_length=1, max_length=120)


def default_site(club: dict) -> dict:
    return SiteDocument(name=club["name"], description=club.get("tagline") or "",
                        logo_url=club.get("logo_url") or "").model_dump()
