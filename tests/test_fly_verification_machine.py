from types import SimpleNamespace
import subprocess

import pytest

from scripts import select_fly_verification_machine as selection
from scripts import initialize_registration_edit_secret as signing


def machine(identifier, state="started", group="app"):
    return {"id": identifier, "state": state, "config": {"metadata": {"fly_process_group": group}}}


def test_selection_skips_unreachable_machine_without_running_business_commands(monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[-2:] == ["--command", "/bin/true"]
        assert kwargs["timeout"] == 30
        assert kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(command, 30)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(selection.subprocess, "run", run)
    rows = [machine("22222222222222"), machine("11111111111111")]
    assert selection.select_machine(rows) == "22222222222222"
    assert len(calls) == 2


def test_selection_excludes_stopped_and_other_processes(monkeypatch):
    monkeypatch.setattr(selection.subprocess, "run", lambda *_a, **_kw: pytest.fail("no eligible machine"))
    with pytest.raises(RuntimeError, match="No running production"):
        selection.select_machine([
            machine("11111111111111", "stopped"),
            machine("22222222222222", group="worker"), machine("invalid-id"),
        ])


def test_selection_fails_closed_when_all_connections_fail(monkeypatch):
    monkeypatch.setattr(selection.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="No running production"):
        selection.select_machine([machine("11111111111111")])


@pytest.mark.parametrize("ssh", [True, False])
def test_signing_probe_uses_selected_machine_only_for_ssh(monkeypatch, ssh):
    monkeypatch.setenv("FLY_SSH_MACHINE_ID", "11111111111111")
    monkeypatch.setenv("FLY_API_TOKEN", "deploy-fixture")
    monkeypatch.setenv("FLY_SSH_TOKEN", "ssh-fixture")
    def run(command, **kwargs):
        assert ("--machine" in command) is ssh
        if ssh:
            assert command[command.index("--machine") + 1] == "11111111111111"
        assert kwargs["env"]["FLY_API_TOKEN"] == ("ssh-fixture" if ssh else "deploy-fixture")
        return SimpleNamespace(returncode=0, stdout="ok")
    monkeypatch.setattr(signing.subprocess, "run", run)
    assert signing.fly("ssh", "console", ssh=ssh) == "ok"
