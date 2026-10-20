"""Loopback-only RR Hub browser fixture using real routes and generator services.

Extends the empty generator submission fixture with the transactional hub RPC
double used by service contract tests. Authentication and persistence are local
test doubles; session creation, commands and serialization remain production
code. PostgreSQL transaction/locking behavior is verified by the SQL tests.
"""
from types import SimpleNamespace

import generator_submission_api as generator
from test_admin_round_robin_hub_service import TransactionalHubDB
from jupr_app.services import admin_round_robin_hub_service as hub_service
from jupr_app.services import admin_play_generator_service as generator_service
from services.api import admin_round_robin_hub_routes as hub_routes

app = generator.app
database = generator.database
transactional = TransactionalHubDB()
database.db.setdefault(hub_service.HUB_TABLE, [])
database.db.setdefault(hub_service.OPERATION_TABLE, [])
original_rpc = database.rpc


def rpc(name, params):
    if name != "commit_round_robin_hub":
        return original_rpc(name, params)

    def execute():
        transactional.storage = database.db
        result = transactional.rpc(name, params).execute()
        database.db = transactional.storage
        return result

    return SimpleNamespace(execute=execute)


database.rpc = rpc
generator.patches.setattr(hub_routes, "_require_write_gate", lambda: None)
# The loopback fixture already authorizes its synthetic admin mutation routes.
# Expose the same capability to the real admin builder's read-only status check.
generator.patches.setattr(generator_service, "is_staging_write_gate_enabled", lambda flag: True)
generator.patches.setattr(generator_service, "is_admin_jupr_live_enabled", lambda: True)
hub_routes.install_admin_round_robin_hub_routes(app, get_supabase_client=lambda: database)


@app.get("/admin/clubs/{club_id}/notifications")
def notifications(club_id: str):
    # The surrounding admin shell reads notices; this fixture has none.
    return {"club_id": club_id, "categories": [], "items": [], "checked_at": "2026-10-20T00:00:00Z",
            "history_days": 30, "truncated": False}


@app.get("/hub-fixture-result")
def result():
    # Public serializers omit organizer secrets and submission recovery data.
    return {
        **generator.result(),
        "hubs": [hub_service.get_round_robin_hub(database, club_id="club", hub_id=row["id"])
                 for row in database.db[hub_service.HUB_TABLE]],
        "rpc_calls": len(transactional.calls),
    }
