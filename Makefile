.PHONY: db-migrate check-migration-sources check-next-parity-matrix api-test public-web-smoke admin-pilot-smoke

db-migrate: ## Apply pending SQL migrations (requires DATABASE_URL)
	@bash scripts/db_migrate.sh

check-migration-sources: ## Guard: block undocumented new root migrations
	@python scripts/check_migration_sources.py

check-next-parity-matrix: ## Guard: every Streamlit page is represented in the Next parity matrix
	@python scripts/check_next_parity_matrix.py


api-test: ## Run API contract tests
	@python -m pytest -q tests/test_api_health.py tests/test_api_contract_*.py tests/test_api_admin_score_entry_disabled.py

public-web-smoke: ## Smoke-test staging FastAPI + Next.js public routes
	@python scripts/smoke_public_web.py $(JUPR_SMOKE_ARGS)

admin-pilot-smoke: ## Non-mutating Match Log + Replay History pilot readiness smoke
	@python scripts/smoke_admin_pilot.py $(JUPR_ADMIN_PILOT_SMOKE_ARGS)
