from __future__ import annotations

from jupr_app.workers.player_update_email_worker import main, run_player_update_email_worker


def test_run_player_update_email_worker_passes_club_and_limit(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role")
    monkeypatch.setenv("JUPR_PUBLIC_BASE_URL", "https://jupr.example.com")

    captured = {}

    def fake_make_supabase(url, key):
        captured["url"] = url
        captured["key"] = key
        return "fake-client"

    def fake_send_pending(ctx, *, limit, public_base_url=None, smtp_config=None):
        captured["club_id"] = ctx.club_id
        captured["supabase"] = ctx.supabase
        captured["limit"] = limit
        captured["public_base_url"] = public_base_url
        return {"attempted": 8, "sent": 5, "skipped": 2, "errors": 1}

    monkeypatch.setattr("jupr_app.workers.player_update_email_worker.make_supabase", fake_make_supabase)
    monkeypatch.setattr(
        "jupr_app.workers.player_update_email_worker.send_pending_player_update_emails",
        fake_send_pending,
    )

    summary = run_player_update_email_worker("tres_palapas", limit=250)

    assert captured == {
        "url": "https://example.supabase.co",
        "key": "service-role",
        "club_id": "tres_palapas",
        "supabase": "fake-client",
        "limit": 250,
        "public_base_url": "https://jupr.example.com",
    }
    assert summary == {
        "ok": True,
        "club_id": "tres_palapas",
        "key_source": "SUPABASE_SERVICE_ROLE_KEY",
        "attempted": 8,
        "sent": 5,
        "skipped": 2,
        "errors": 1,
    }


def test_main_prints_json_summary(monkeypatch, capsys):
    monkeypatch.setattr(
        "jupr_app.workers.player_update_email_worker.run_player_update_email_worker",
        lambda club_id, limit: {
            "ok": True,
            "club_id": club_id,
            "key_source": "SUPABASE_SERVICE_ROLE_KEY",
            "attempted": 3,
            "sent": 2,
            "skipped": 1,
            "errors": 0,
        },
    )

    rc = main(["--club-id", "tres_palapas", "--limit", "250"])
    out = capsys.readouterr().out

    assert rc == 0
    assert '"club_id": "tres_palapas"' in out
    assert '"attempted": 3' in out
    assert '"sent": 2' in out
    assert '"skipped": 1' in out
    assert '"errors": 0' in out


def test_main_errors_do_not_fail_without_flag(monkeypatch):
    monkeypatch.setattr(
        "jupr_app.workers.player_update_email_worker.run_player_update_email_worker",
        lambda club_id, limit: {
            "ok": True,
            "club_id": club_id,
            "key_source": "SUPABASE_SERVICE_ROLE_KEY",
            "attempted": 2,
            "sent": 1,
            "skipped": 0,
            "errors": 1,
        },
    )

    rc = main(["--club-id", "tres_palapas", "--limit", "250"])

    assert rc == 0


def test_main_fail_on_errors_sets_nonzero_and_ok_false(monkeypatch, capsys):
    monkeypatch.setattr(
        "jupr_app.workers.player_update_email_worker.run_player_update_email_worker",
        lambda club_id, limit: {
            "ok": True,
            "club_id": club_id,
            "key_source": "SUPABASE_SERVICE_ROLE_KEY",
            "attempted": 2,
            "sent": 1,
            "skipped": 0,
            "errors": 1,
        },
    )

    rc = main(["--club-id", "tres_palapas", "--limit", "250", "--fail-on-errors"])
    out = capsys.readouterr().out

    assert rc == 1
    assert '"ok": false' in out.lower()
