# Match-correction badge contract repair

New operations capture version 2 and the current frozen 41-badge scope. Existing version-1 operations retain their saved 29-badge scope and version on every generated row. Recovery validates the saved contract instead of requiring it to match today's catalog. Unsupported versions and badges outside the saved contract fail closed. New operations still reject catalog drift.

No database migration, historical rewrite, feature-flag activation, or email change is included. The existing SQL claim/apply routines already enforce each operation's stored version and scope. The match-correction gate remains disabled; the separate bounded history loader must be repaired and full workflow acceptance completed before enabling corrections for large clubs.

Validation: 77 focused Python tests passed across reconciliation, durability, catalog history, atomic recovery schema, and parity fixtures. A synthetic PGlite test executes the actual claim/apply PostgreSQL routines for both versions, rejects wrong versions and out-of-scope badges, preserves manual/unrelated awards, and verifies idempotent retries.
