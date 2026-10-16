from copy import deepcopy

import pytest

from scripts import recover_weekly_report_replica as recovery


def source():
    return {"id": recovery.SOURCE, "state": "started", "host_status": "ok", "region": "dfw",
            "config": {"image": recovery.IMAGE, "env": {"JUPR_ENV": "production"}},
            "image_ref": {"digest": recovery.DIGEST,
                          "labels": {"org.opencontainers.image.revision": recovery.SHA}},
            "checks": [{"status": "passing"}]}


class Api:
    def __init__(self):
        self.machines = [source(), {"id": recovery.UNREACHABLE, "state": "started",
                                   "host_status": "unreachable"}]
        self.volumes = []
        self.calls = []
        self.new_check = "passing"
        self.recover_old_host = False

    def __call__(self, method, path, payload=None):
        self.calls.append((method, path))
        if method == "GET" and path == "/machines":
            return deepcopy(self.machines)
        if method == "GET" and path == "/volumes":
            return self.volumes
        if method == "POST" and path == "/machines":
            assert payload["config"] == source()["config"]
            new = source()
            new["id"] = "new-healthy-replica"
            new["checks"][0]["status"] = self.new_check
            self.machines.append(new)
            if self.recover_old_host:
                self.machines[1]["host_status"] = "ok"
            return deepcopy(new)
        if method == "DELETE":
            assert path == f"/machines/{recovery.UNREACHABLE}?kill=true"
            self.machines = [m for m in self.machines if m["id"] != recovery.UNREACHABLE]
            return None
        raise AssertionError("Unexpected request")


def health():
    return {"ok": True, "git_commit_sha": recovery.SHA, "image_build_git_sha": recovery.SHA}


def test_replaces_only_exact_unreachable_replica_after_healthy_clone():
    api = Api()
    recovery.recover(api, health, pause=lambda _: None)
    assert [c for c in api.calls if c[0] != "GET"] == [
        ("POST", "/machines"), ("DELETE", f"/machines/{recovery.UNREACHABLE}?kill=true")]
    assert {m["id"] for m in api.machines} == {recovery.SOURCE, "new-healthy-replica"}


@pytest.mark.parametrize("condition", ["volume", "host_recovered", "wrong_image", "unexpected_machine"])
def test_preconditions_prevent_all_mutations(condition):
    api = Api()
    if condition == "volume":
        api.volumes = [{"id": "persistent-storage"}]
    elif condition == "host_recovered":
        api.machines[1]["host_status"] = "ok"
    elif condition == "wrong_image":
        api.machines[0]["config"]["image"] = "another-image"
    else:
        api.machines.append({"id": "unexpected"})
    with pytest.raises(AssertionError):
        recovery.recover(api, health, pause=lambda _: None)
    assert all(method == "GET" for method, _ in api.calls)


@pytest.mark.parametrize("condition", ["unhealthy_replacement", "old_host_recovers"])
def test_originals_preserved_when_health_changes(condition):
    api = Api()
    api.new_check = "critical" if condition == "unhealthy_replacement" else "passing"
    api.recover_old_host = condition == "old_host_recovers"
    with pytest.raises((AssertionError, RuntimeError)):
        recovery.recover(api, health, pause=lambda _: None)
    assert all(method != "DELETE" for method, _ in api.calls)
