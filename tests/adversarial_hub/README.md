# RR Hub reliability regressions

These synthetic local tests assert the repaired behavior from the October 22
incident and adversarial audit. No production data or credentials are used.

Install root/API requirements plus `requirements.txt` in this directory, and the
locked web and SQL dependencies. Run `python -m pytest --runxfail -q
tests/adversarial_hub`; the flag prevents historical expected-failure markers
from hiding regressions. The `RR Hub Reliability` workflow also verifies the
actual React components, SQL migrations, and a loopback Next/FastAPI journey.

The browser journey supports `GENERATOR_BROWSER_PYTHON`,
`GENERATOR_BROWSER_EXECUTABLE`, `GENERATOR_BROWSER_ARGS` (JSON array),
`GENERATOR_BROWSER_ARTIFACTS`, and web/API port overrides. It blocks external
browser requests. Authentication and persistence are synthetic fixtures; real
database privileges and transaction rollback are covered separately by SQL.

Local SQL does not emulate PostgREST retry timing, and local browser tests do not
attest production load or real account authentication. Historical results and
pre-existing uncertain operations require separate authorized review.
