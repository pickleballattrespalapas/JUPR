"""Apply aliases at the public response boundary, including historical snapshots.

Staff routes and capability-protected registration editing retain canonical names.
Public GETs never gain staff names merely by supplying an Authorization header.
"""
from __future__ import annotations

import json
import csv
import io
import os
import re
import time
from functools import lru_cache

from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse, Response

from jupr_app.domain.player_aliases import load_player_aliases, public_alias_projection


def public_club_slug(path: str) -> str | None:
    match = re.match(r"^/(?:public/)?clubs/([^/]+)(?:/|$)", path)
    if not match or re.search(r"/tournament-registration/(?:edit|confirmation)(?:/|$)", path):
        return None
    return match.group(1)


def install_public_alias_middleware(app, *, get_club, get_supabase_client):
    @lru_cache(maxsize=128)
    def aliases_for(kind: str, identity: str, bucket: int):
        db = get_supabase_client()
        if kind == "club":
            club = get_club(identity)
            club_ids = {str(club.get("id") or club.get("club_id") or identity)}
        elif kind == "interclub-signups":
            rows = db.table("pcs_interclub_pool_settings").select("club_id").eq("share_id", identity).execute().data or []
            club_ids = {str(row["club_id"]) for row in rows}
        else:
            table = "pcs_managed_interclub_seasons" if kind == "managed-interclub" else "pcs_interclub_seasons"
            rows = db.table(table).select("organizer_club_id").eq("id", identity).execute().data or []
            club_ids = {str(row["organizer_club_id"]) for row in rows}
            if kind == "interclub":
                participants = db.table("pcs_interclub_participations").select("club_id").eq("season_id", identity).execute().data or []
                club_ids.update(str(row["club_id"]) for row in participants)
        if not club_ids:
            raise ValueError("Public identity scope unavailable")
        aliases = {}
        for club_id in sorted(club_ids):
            aliases.update(load_player_aliases(db, club_id))
        return aliases

    @app.middleware("http")
    async def public_aliases(request, call_next):
        response = await call_next(request)
        slug = public_club_slug(request.url.path)
        season = re.match(r"^/public/(interclub|managed-interclub|interclub-signups)/([^/]+)(?:/|$)", request.url.path)
        if (os.getenv("JUPR_ENABLE_PUBLIC_PLAYER_ALIASES", "").lower() not in {"1", "true", "yes", "on"}
                or not (slug or season)
                or response.status_code >= 400
                or not any(kind in response.headers.get("content-type", "")
                           for kind in ("application/json", "text/csv"))):
            return response
        try:
            kind, identity = ("club", slug) if slug else season.groups()
            aliases = await run_in_threadpool(aliases_for, kind, identity, int(time.monotonic() // 15))
            if not aliases:
                return response
            body = b"".join([chunk async for chunk in response.body_iterator])
            headers = {key: value for key, value in response.headers.items()
                       if key.lower() not in {"content-length", "etag", "content-encoding"}}
            if "text/csv" in response.headers.get("content-type", ""):
                output = io.StringIO()
                writer = csv.writer(output)
                for row in csv.reader(io.StringIO(body.decode("utf-8"))):
                    writer.writerow(public_alias_projection(row, aliases))
                return Response(output.getvalue(), status_code=response.status_code, headers=headers,
                                background=response.background)
            result = public_alias_projection(json.loads(body), aliases)
            return JSONResponse(result, status_code=response.status_code, headers=headers,
                                background=response.background)
        except Exception:
            # Never fall back to the canonical names if privacy lookup fails.
            return JSONResponse({"detail": "Public player names are temporarily unavailable."}, status_code=503)
