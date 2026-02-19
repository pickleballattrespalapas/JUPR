.PHONY: db-migrate schema-diff

db-migrate: ## Apply pending SQL migrations (requires DATABASE_URL)
	@bash scripts/db_migrate.sh

schema-diff: ## Compare live schema snapshot against migration-defined schema
	@python tools/schema_diff.py
