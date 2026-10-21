"""Loopback browser rehearsal. Production services, local auth and RPC doubles."""
import os
import round_robin_hub_api as fixture
from test_round_robin_hub_registration import CheckinDB
from test_admin_round_robin_hub_service import session, create
from services.api import public_round_robin_hub_routes as public
from jupr_app.services import admin_round_robin_hub_service as hubs
from services.api.middleware import StagingWriteWaveMiddleware

app = fixture.app
intake = CheckinDB()
intake.storage = fixture.database.db
intake.storage.setdefault("round_robin_hub_checkins", [])
session(intake, key="rr-a", count=4, club="club")
session(intake, key="rr-b", count=4, club="club", offset=10)
hub = create(intake, club="club")
for key in ("rr-a", "rr-b"):
    hub = hubs.command_round_robin_hub(intake, club_id="club", hub_id=hub["hub"]["id"], action="add_group",
        payload={"session_key": key}, expected_version=hub["hub"]["version"], idempotency_key=f"attach-{key}", actor_email="reviewer@example.invalid")
fixture.database.db = intake.storage
original_rpc = fixture.database.rpc

def rpc(name, params):
    if name == "submit_round_robin_hub_checkin":
        intake.storage = fixture.database.db
        return intake.rpc(name, params)
    return original_rpc(name, params)
fixture.database.rpc = rpc
os.environ["JUPR_ENV"] = "staging"
os.environ["JUPR_STAGING_WRITE_WAVE"] = "open"
os.environ["JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES"] = "1"
app.add_middleware(StagingWriteWaveMiddleware)
public.install_public_round_robin_hub_routes(app, get_supabase_client=lambda: fixture.database)
