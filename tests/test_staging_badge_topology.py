from copy import deepcopy
import tomllib
from pathlib import Path

import pytest

from scripts.staging_badge_topology import APP, WORKER_COMMAND, verify

ROOT = Path(__file__).resolve().parents[1]


def fleet():
    env = {"JUPR_ENV": "staging", "JUPR_EMAIL_MODE": "dry_run",
           "JUPR_BADGE_WORKER_MODE": "external", "JUPR_BADGE_WORKER_CLUB_ID": "tres_palapas",
           "JUPR_BADGE_QUEUE_LEASES": "0"}
    return [{"id": str(index), "state": "started", "region": "dfw",
             "image_ref": {"registry": "registry.fly.io", "repository": APP, "digest": "a" * 64},
             "config": {"metadata": {"fly_process_group": group}, "env": deepcopy(env),
                        "guest": {"cpu_kind": "shared", "cpus": 1, "memory_mb": ram},
                        "services": [] if group == "badge_worker" else [{"internal_port": 8080}],
                        "init": {"cmd": WORKER_COMMAND.split() if group == "badge_worker" else []}}}
            for index, group, ram in [(1, "app", 1024), (2, "app", 1024), (3, "badge_worker", 2048)]]


def test_two_phase_release_preserves_api_capacity_and_one_worker():
    machines = fleet()
    assert verify(machines, phase="before")["worker_ids"] == ["3"]
    machines[2]["state"] = "stopped"
    assert verify(machines, phase="api")["image"].endswith("a" * 64)
    machines[2]["state"] = "started"
    assert verify(machines, phase="final")["api_ids"] == ["1", "2"]


@pytest.mark.parametrize("mutation", [
    lambda m: m.append(deepcopy(m[-1])),
    lambda m: m[2]["config"]["services"].append({"internal_port": 8080}),
    lambda m: m[0]["config"]["guest"].update(cpu_kind="performance"),
    lambda m: m[2]["config"]["guest"].update(memory_mb=1024),
    lambda m: m[1]["image_ref"].update(digest="b" * 64),
    lambda m: m[2]["image_ref"].update(repository="juprleagues-api"),
    lambda m: m[0]["config"]["env"].update(JUPR_EMAIL_MODE="live"),
    lambda m: m[1]["config"]["env"].update(JUPR_BADGE_QUEUE_LEASES="1"),
    lambda m: m[2]["config"]["init"].update(cmd=["python", "-m", "other"]),
    lambda m: m[2].update(region="iad"),
    lambda m: m[1].update(state="stopped"),
    lambda m: m.pop(),
])
def test_rejects_wrong_or_incomplete_runtime(mutation):
    machines = fleet()
    mutation(machines)
    with pytest.raises(ValueError):
        verify(machines, phase="final")


def test_api_phase_rejects_running_old_worker():
    with pytest.raises(ValueError, match="previous worker"):
        verify(fleet(), phase="api")


def test_staging_source_and_deploy_keep_worker_separate():
    config = tomllib.loads((ROOT / "fly.staging.toml").read_text())
    assert config["app"] == APP
    assert config["processes"]["badge_worker"] == WORKER_COMMAND
    assert config["http_service"]["processes"] == ["app"]
    assert config["kill_timeout"] == 300
    assert config["env"]["JUPR_EMAIL_MODE"] == "dry_run"
    assert config["env"]["JUPR_BADGE_QUEUE_LEASES"] == "0"
    assert config["vm"] == [
        {"processes": ["app"], "memory": "1gb", "cpu_kind": "shared", "cpus": 1},
        {"processes": ["badge_worker"], "memory": "2gb", "cpu_kind": "shared", "cpus": 1}]
    workflow = (ROOT / ".github/workflows/fly_api_staging_deploy.yml").read_text()
    assert workflow.index("--process-groups app") < workflow.index("--phase api")
    assert workflow.index("--phase api") < workflow.index("--process-groups badge_worker")
    assert "--phase final" in workflow
    assert "--ha=false" in workflow


def test_rechecks_processing_after_all_apis_are_upgraded():
    workflow = (ROOT / ".github/workflows/fly_api_staging_deploy.yml").read_text()
    phase_api = workflow.index("--phase api")
    gate = workflow.index("python scripts/check_staging_badge_prerequisites.py", phase_api)
    assert phase_api < gate < workflow.index("--process-groups badge_worker")
