.PHONY: db-migrate schema-diff replay

db-migrate: ## Apply pending SQL migrations (requires DATABASE_URL)
	@bash scripts/db_migrate.sh

schema-diff: ## Compare live schema snapshot against migration-defined schema
	@python tools/schema_diff.py

replay: ## Replay ratings history for a club (make replay CLUB=<club_id>)
	@if [ -z "$(CLUB)" ]; then echo "Usage: make replay CLUB=<club_id>"; exit 1; fi
	@python -m jupr_app.cli.replay_ratings --club-id $(CLUB)
