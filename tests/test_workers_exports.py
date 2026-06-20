from jupr_app.workers import run_badge_queue_worker, run_player_update_email_worker


def test_workers_exports_available():
    assert callable(run_badge_queue_worker)
    assert callable(run_player_update_email_worker)
