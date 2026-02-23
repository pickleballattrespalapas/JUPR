from jupr_app.domain.tournaments.sync import cleanup_duplicate_tournament_games


class _RpcCall:
    def __init__(self):
        self.executed = False

    def execute(self):
        self.executed = True
        return {"status": "ok"}


class _Supabase:
    def __init__(self):
        self.calls = []
        self.rpc_call = _RpcCall()

    def rpc(self, name, payload):
        self.calls.append((name, payload))
        return self.rpc_call


def test_cleanup_duplicate_tournament_games_calls_expected_rpc():
    supabase = _Supabase()

    result = cleanup_duplicate_tournament_games(supabase, 123)

    assert result == {"status": "ok"}
    assert supabase.calls == [("dedupe_tournament_games", {"t_id": "123"})]
    assert supabase.rpc_call.executed is True
