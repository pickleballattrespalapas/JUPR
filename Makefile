.PHONY: db-migrate check-migration-sources check-next-parity-matrix api-test public-web-smoke

db-migrate: ## Apply pending SQL migrations (requires DATABASE_URL)
	@bash scripts/db_migrate.sh

check-migration-sources: ## Guard: block undocumented new root migrations
	@python scripts/check_migration_sources.py

check-next-parity-matrix: ## Guard: every Streamlit page is represented in the Next parity matrix
	@python scripts/check_next_parity_matrix.py


api-test: ## Run API contract tests
	@python -m pytest tests/test_api_health.py tests/test_api_contract_clubs.py tests/test_api_contract_leaderboards.py tests/test_api_contract_match_explorer.py tests/test_api_contract_league_results.py tests/test_api_contract_badge_codex.py tests/test_api_contract_challenge_ladder.py tests/test_api_contract_weekly_recaps.py tests/test_api_contract_tournament_registration.py tests/test_api_contract_admin_operations.py tests/test_api_contract_admin_match_log.py tests/test_api_contract_admin_replay.py tests/test_api_admin_score_entry_disabled.py

public-web-smoke: ## Smoke-test staging FastAPI + Next.js public routes
	@python scripts/smoke_public_web.py $(JUPR_SMOKE_ARGS)
