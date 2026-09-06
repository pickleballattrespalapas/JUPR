"""Additional, fail-closed scope boundary for the new Operator role.

Installed around every API route after registration so newly added route families
do not accidentally inherit club-wide operator access.
"""
from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any

from fastapi import HTTPException
from fastapi.routing import APIRoute, request_response
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from jupr_app.domain.admin.staff_policy import assignment_active, permits, operator_request_authorized


staff_request: ContextVar[dict | None] = ContextVar("staff_request", default=None)


def authorize_operator_request(db, club, assignment):
    context = staff_request.get()
    if context is None or str(context["params"].get("club_id")) != club:
        _deny()
    context["filter"] = check_operator(db, club, assignment, context["path"], context["params"], context["body"], context["method"], context["query"])
    operator_request_authorized.set(True)


def _row(db, table, club, key, value):
    rows = db.table(table).select("*").eq("club_id", club).eq(key, value).limit(1).execute().data or []
    if not rows:
        raise HTTPException(404, "Assigned program was not found.")
    return rows[0]


def _deny():
    raise HTTPException(403, "This action is outside your staff access. Ask a club administrator.")


def check_operator(db, club, assignment, path, params, body, method, query):
    if not assignment_active(assignment):
        _deny()
    scopes = assignment.get("scopes") or []
    write = method not in {"GET", "HEAD", "OPTIONS"}
    if write and (method == "DELETE" or any(part in path.split("/") for part in (
        "delete-draft", "replay", "merge", "compensate", "reconcile", "retry", "overrides"
    ))):
        _deny()
    action = str(body.get("action") or body.get("command") or "").lower()
    if write and any(word in action for word in ("delete", "reopen", "unpublish", "reset", "replay", "rebuild")):
        _deny()
    if "/players/editor" in path:
        if any(word in path for word in ("merge", "social", "operations", "league-rating")):
            _deny()
        if write and (method != "POST" or "player_id" in params):
            if any(body.get(key) is not None for key in ("rating_jupr", "starting_jupr")):
                _deny()
        return None

    resources = set()
    program = ""
    if "/league-manager/" in path:
        program = "leagues"
        league = params.get("league_name") or body.get("league_name") or query.get("league_name")
        session_id = params.get("session_id")
        if session_id:
            session = _row(db, "league_live_sessions", club, "id", session_id)
            resources.add(str(session_id))
            league = session.get("league_name")
            if write and str(session.get("status", "")).lower() in {"completed", "published", "archived"}:
                _deny()
            round_number = params.get("round_number")
            if write and round_number and path.endswith("/rounds/{round_number}"):
                rows = db.table("league_live_rounds").select("status").eq("club_id", club).eq("session_id", session_id).eq("round_number", round_number).execute().data or []
                if any(row.get("status") == "submitted" for row in rows):
                    _deny()
        if league:
            resources.add(str(league))
            # Creation refers to a new name; all other existing league accesses
            # are checked against the club-owned metadata row.
            if not (method == "POST" and path.endswith("/league-manager/leagues")):
                meta = _row(db, "leagues_metadata", club, "league_name", league)
                if write and str(meta.get("status", "")).lower() in {"completed", "published", "archived", "closed"}:
                    _deny()
        if write and path.endswith("/duplicate"):
            resources = set()  # a grant for one league cannot create a sibling
    elif "/tournaments/" in path or "/tournament-live/" in path:
        program = "tournaments"
        tid = params.get("tournament_id")
        if tid:
            tournament = _row(db, "tournaments", club, "id", tid)
            resources.add(str(tid))
            if write and str(tournament.get("status", "")).upper() in {"COMPLETED", "ARCHIVED"}:
                _deny()
        # Scoped day grants must never authorize a whole-tournament mutation.
        if params.get("day_id"):
            resources.add(str(params["day_id"]))
    elif "/play-generators/" in path:
        program = str(body.get("generator_kind") or query.get("generator_kind") or "")
        if params.get("session_key"):
            session = _row(db, "live_sessions", club, "session_key", params["session_key"])
            state = session.get("state") or {}
            program = str(state.get("generator_kind") or "")
            resources.add(str(params["session_key"]))
            if write and (state.get("official_publish") or {}).get("published_at"):
                _deny()
        if not program:
            event = body.get("event") or {}
            program = str(event.get("generatorKind") or "")
    elif "/jupr-live/" in path:
        program = "live_play"
        if params.get("session_key"):
            session = _row(db, "live_sessions", club, "session_key", params["session_key"])
            resources.add(str(params["session_key"]))
            if write and ((session.get("state") or {}).get("official_publish") or {}).get("published_at"):
                _deny()
    elif "/challenge-ladder/" in path:
        program = "challenge_ladder"
    elif "/moneyball/" in path:
        program = "moneyball"
    else:
        _deny()
    if permits(scopes, program, resources):
        return None
    # Collection reads are filtered after the endpoint builds its response.
    if not write and not resources and any(s.get("program_type") == program for s in scopes):
        return (program, scopes)
    _deny()


def install_staff_access(app, *, get_supabase_client):
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.path.startswith("/admin/clubs/{club_id}/"):
            continue
        original = route.get_route_handler()

        def wrap(handler, template):
            async def guarded(request):
                body = {}
                if request.method not in {"GET", "HEAD", "OPTIONS"}:
                    try:
                        body = await request.json()
                    except (ValueError, TypeError):
                        pass  # Endpoint validation owns invalid bodies.
                context = {"path": template, "params": request.path_params,
                           "body": body if isinstance(body, dict) else {},
                           "method": request.method, "query": request.query_params,
                           "filter": None}
                token = staff_request.set(context)
                authorization_context = operator_request_authorized.set(False)
                try:
                    response = await handler(request)
                finally:
                    staff_request.reset(token)
                    operator_request_authorized.reset(authorization_context)
                collection_filter = context["filter"]
                if collection_filter and response.status_code == 200:
                    # Unknown response structures fail closed rather than leak a
                    # club-wide collection to a resource-scoped operator.
                    data = json.loads(response.body)
                    program, scopes = collection_filter
                    found = False
                    for key in ("leagues", "sessions", "tournaments"):
                        if isinstance(data.get(key), list):
                            found = True
                            data[key] = [item for item in data[key] if permits(scopes, program, {
                                str(item.get("id", "")), str(item.get("league_name", "")), str(item.get("session_key", ""))
                            })]
                    if not found:
                        _deny()
                    # Summary counts may describe hidden rows.
                    for key in ("counts", "count", "total", "total_count", "summary"):
                        data.pop(key, None)
                    return JSONResponse(data)
                return response
            return guarded
        route.app = request_response(wrap(original, route.path))
