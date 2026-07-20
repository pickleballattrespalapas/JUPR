.PHONY: db-migrate check-migration-sources check-next-parity-matrix check-parity-closure-program check-parity-manual-book check-parity-final-evidence check-parity-final-evidence-integrated check-parity-final-evidence-complete api-test public-web-smoke next-staging-browser-smoke admin-pilot-smoke

db-migrate: ## Apply pending SQL migrations (requires DATABASE_URL)
	@bash scripts/db_migrate.sh

check-migration-sources: ## Guard: block undocumented new root migrations
	@python scripts/check_migration_sources.py

check-next-parity-matrix: ## Guard: every Streamlit page is represented in the Next parity matrix
	@python scripts/check_next_parity_matrix.py

check-parity-closure-program: ## Guard: every Partial page has exactly one parity closure contract
	@python scripts/check_parity_closure_program.py

check-parity-manual-book: ## Guard: every Partial page has exactly one manual evidence row
	@python scripts/check_parity_manual_book.py

check-parity-final-evidence: check-next-parity-matrix check-parity-closure-program check-parity-manual-book ## Pending-safe Order-29 scaffold and manifest guards

check-parity-final-evidence-integrated: ## Final-stack guard: every staging WAVES spec and grep exists
	@python scripts/run_parity_staging_wave.py --check-integrated-manifest

check-parity-final-evidence-complete: check-parity-final-evidence ## Fail closed unless the bound manual evidence book is complete
	@test -n "$(CANDIDATE_SHA)" || (echo "CANDIDATE_SHA is required" >&2; exit 2)
	@test -n "$(VERCEL_DEPLOYMENT_ID)" || (echo "VERCEL_DEPLOYMENT_ID is required" >&2; exit 2)
	@test -n "$(VERCEL_DEPLOYMENT_ORIGIN)" || (echo "VERCEL_DEPLOYMENT_ORIGIN is required" >&2; exit 2)
	@test -n "$(FLY_IMAGE_REF)" || (echo "FLY_IMAGE_REF is required" >&2; exit 2)
	@python scripts/check_parity_manual_book.py --require-complete \
		--candidate-sha "$(CANDIDATE_SHA)" \
		--vercel-deployment-id "$(VERCEL_DEPLOYMENT_ID)" \
		--vercel-deployment-origin "$(VERCEL_DEPLOYMENT_ORIGIN)" \
		--fly-image-ref "$(FLY_IMAGE_REF)"


api-test: ## Run API contract tests
	@python -m pytest -q tests/test_api_health.py tests/test_api_contract_*.py tests/test_api_admin_score_entry_disabled.py

public-web-smoke: ## Smoke-test staging FastAPI + Next.js public routes
	@python scripts/smoke_public_web.py $(JUPR_SMOKE_ARGS)

next-staging-browser-smoke: ## Browser-test the isolated Vercel preview surface
	@cd apps/web && npm run test:e2e:staging

admin-pilot-smoke: ## Non-mutating Match Log + Replay History pilot readiness smoke
	@python scripts/smoke_admin_pilot.py $(JUPR_ADMIN_PILOT_SMOKE_ARGS)
