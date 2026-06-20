.PHONY: db-migrate check-migration-sources

db-migrate: ## Apply pending SQL migrations (requires DATABASE_URL)
	@bash scripts/db_migrate.sh

check-migration-sources: ## Guard: block undocumented new root migrations
	@python scripts/check_migration_sources.py


.PHONY: api-test
api-test: ## Run API contract tests
	@python -m pytest tests/test_api_health.py tests/test_api_contract_clubs.py tests/test_api_contract_leaderboards.py tests/test_api_admin_score_entry_disabled.py
