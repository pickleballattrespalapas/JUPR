from copy import deepcopy
import tomllib
from pathlib import Path

import pytest

from scripts.staging_badge_topology import APP, WORKER_COMMAND, verify

ROOT = Path(__file__).resolve().parents[1]


def fleet(*, worker=True, retired=False):
    env = {"JUPR_ENV": "staging", "JUPR_EMAIL_MODE": "dry_run",
           "JUPR_BADGE_WORKER_MODE": "disabled" if retired else "external",
           "JUPR_BADGE_WORKER_CLUB_ID": "" if retired else "tres_palapas",
           "JUPR_BADGE_QUEUE_LEASES": "0"}
    spec = [(1, "app", 1024), (2, "app", 1024)]
    if worker:
        spec.append((3, "badge_worker", 2048))
    return [{"id": str(index), "state": "started", "region": "dfw",
             "image_ref": {"registry": "registry.fly.io", "repository": APP, "digest": "a" * 64},
             "config": {"metadata": {"fly_process_group": group}, "env": deepcopy(env),
                        "guest": {"cpu_kind": "shared", "cpus": 1, "memory_mb": ram},
                        "services": [] if group == "badge_worker" else [{"internal_port": 8080}],
                        "init": {"cmd": WORKER_COMMAND.split() if group == "badge_worker" else []}}}
            for index, group, ram in spec]


def test_retirement_identifies_only_worker_and_keeps_api_capacity():
    machines = fleet()
    before = verify(machines, phase="before")
    assert before["worker_ids"] == before["running_worker_ids"] == ["3"]
    assert before["api_ids"] == ["1", "2"]
    machines[2]["state"] = "stopped"
    stopped = verify(machines, phase="stopped")
    assert stopped["worker_ids"] == ["3"]
    assert stopped["running_worker_ids"] == []
    machines.pop()
    assert verify(machines, phase="retired")["worker_ids"] == []
    assert verify(fleet(worker=False, retired=True), phase="final")["api_ids"] == ["1", "2"]


def test_retirement_is_safe_to_repeat_when_worker_already_gone():
    machines = fleet(worker=False, retired=True)
    for phase in ("before", "stopped", "retired", "final"):
        assert verify(machines, phase=phase)["worker_ids"] == []


@pytest.mark.parametrize("mutation", [
    lambda m: m.append(deepcopy(m[-1])),
    lambda m: m[2]["config"]["services"].append({"internal_port": 8080}),
    lambda m: m[2]["config"].update(mounts=[{"volume": "vol_keep"}]),
    lambda m: m[0]["config"]["guest"].update(cpu_kind="performance"),
    lambda m: m[2]["config"]["guest"].update(memory_mb=1024),
    lambda m: m[2]["image_ref"].update(repository="juprleagues-api"),
    lambda m: m[0]["config"]["env"].update(JUPR_EMAIL_MODE="live"),
    lambda m: m[2]["config"]["env"].update(JUPR_ENV="production"),
    lambda m: m[2]["config"]["init"].update(cmd=["python", "-m", "other"]),
    lambda m: m[2].update(region="iad"),
    lambda m: m[1].update(state="stopped"),
    lambda m: m.pop(0),
    lambda m: m[2]["config"]["metadata"].update(fly_process_group="other_worker"),
])
def test_refuses_unsafe_machine_inventory_before_any_shutdown(mutation):
    machines = fleet()
    mutation(machines)
    with pytest.raises(ValueError):
        verify(machines, phase="before")


def test_removal_phase_rejects_running_worker():
    with pytest.raises(ValueError, match="stopped before removal"):
        verify(fleet(), phase="stopped")


@pytest.mark.parametrize("phase", ["retired", "final"])
def test_retired_fleet_must_not_contain_even_a_stopped_worker(phase):
    machines = fleet(retired=True)
    machines[2]["state"] = "stopped"
    with pytest.raises(ValueError, match="no longer exist"):
        verify(machines, phase=phase)


@pytest.mark.parametrize("mutation", [
    lambda m: m[1]["image_ref"].update(digest="b" * 64),
    lambda m: m[1]["config"]["env"].update(JUPR_BADGE_QUEUE_LEASES="1"),
    lambda m: m[1]["config"]["env"].update(JUPR_BADGE_WORKER_MODE="external"),
    lambda m: m[1]["config"]["env"].update(JUPR_BADGE_WORKER_CLUB_ID="tres_palapas"),
])
def test_final_fleet_requires_same_image_and_badge_processing_disabled(mutation):
    machines = fleet(worker=False, retired=True)
    mutation(machines)
    with pytest.raises(ValueError):
        verify(machines, phase="final")


def test_staging_source_and_deploy_retire_worker_before_api_release():
    config = tomllib.loads((ROOT / "fly.staging.toml").read_text())
    assert config["app"] == APP
    assert set(config["processes"]) == {"app"}
    assert config["http_service"]["processes"] == ["app"]
    assert config["kill_timeout"] == 300
    assert config["env"]["JUPR_EMAIL_MODE"] == "dry_run"
    assert config["env"]["JUPR_BADGE_WORKER_MODE"] == "disabled"
    assert config["env"]["JUPR_BADGE_WORKER_CLUB_ID"] == ""
    assert config["env"]["JUPR_BADGE_QUEUE_LEASES"] == "0"
    assert config["vm"] == [
        {"processes": ["app"], "memory": "1gb", "cpu_kind": "shared", "cpus": 1}]
    workflow = (ROOT / ".github/workflows/fly_api_staging_deploy.yml").read_text()
    assert workflow.index("--phase before") < workflow.index("flyctl machine stop")
    assert workflow.index("--phase stopped") < workflow.index("flyctl machine destroy")
    assert workflow.index("--phase retired") < workflow.index("--process-groups app")
    assert workflow.index("--process-groups app") < workflow.index("--phase final")
    assert "--process-groups badge_worker" not in workflow
    assert "check_badge_worker_progress.py" not in workflow
    assert "check_staging_badge_prerequisites.py" not in workflow
    assert '"badges_enabled": False' in workflow
    assert "--ha=false" in workflow
    assert '--force' not in workflow


@pytest.mark.parametrize("initial_state", ["started", "stopped", "absent", "volume"])
def test_decommission_shell_only_removes_verified_stopped_worker(tmp_path, initial_state):
    import json
    import os
    import subprocess
    import sys
    import textwrap

    machines = fleet(worker=initial_state != "absent")
    if initial_state == "stopped":
        machines[-1]["state"] = "stopped"
    elif initial_state == "volume":
        machines[-1]["config"]["mounts"] = [{"volume": "must_keep"}]
    (tmp_path / "inventory.json").write_text(json.dumps(machines))
    (tmp_path / "scripts").symlink_to(ROOT / "scripts", target_is_directory=True)
    cli = tmp_path / "flyctl"
    cli.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''
        import json
        import sys
        from pathlib import Path
        args = sys.argv[1:]
        assert "--force" not in args
        assert "-a" in args and args[args.index("-a") + 1] == "juprleagues-api-staging"
        inventory = Path("inventory.json")
        machines = json.loads(inventory.read_text())
        with Path("calls.jsonl").open("a") as output:
            output.write(json.dumps(args) + "\\n")
        if args[:2] == ["machines", "list"]:
            print(json.dumps(machines))
        elif args[:2] == ["machine", "stop"]:
            assert args[2] == "3"
            assert machines[-1]["state"] == "started"
            machines[-1]["state"] = "stopped"
        elif args[:2] == ["machine", "destroy"]:
            assert args[2] == "3"
            assert machines[-1]["state"] == "stopped"
            assert not machines[-1]["config"].get("mounts")
            machines.pop()
        else:
            raise AssertionError("Unexpected Fly command")
        inventory.write_text(json.dumps(machines))
    '''))
    cli.chmod(0o755)
    workflow = (ROOT / ".github/workflows/fly_api_staging_deploy.yml").read_text()
    block = workflow.split("      - name: Retire the isolated staging badge worker\n", 1)[1]
    script = textwrap.dedent(block.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0])
    result = subprocess.run(
        ["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, "FLY_APP_NAME": APP, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
    )
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    mutation_calls = [call for call in calls if call[0] == "machine"]
    if initial_state == "volume":
        assert result.returncode != 0
        assert mutation_calls == []
        assert json.loads((tmp_path / "inventory.json").read_text()) == machines
    else:
        assert result.returncode == 0, result.stderr
        expected_mutations = {"started": ["stop", "destroy"], "stopped": ["destroy"], "absent": []}[initial_state]
        assert [call[1] for call in mutation_calls] == expected_mutations
        assert json.loads((tmp_path / "inventory.json").read_text()) == machines[:2]
