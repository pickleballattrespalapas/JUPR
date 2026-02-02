.PHONY: db-migrate

db-migrate: ## Apply pending SQL migrations (requires DATABASE_URL)
	@bash scripts/db_migrate.sh
