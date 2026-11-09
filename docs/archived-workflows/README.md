# Archived tournament cutover workflows

These files preserve the original workflow bytes for historical review. The
`.yml.txt` extension and location outside `.github/workflows/` prevent GitHub
Actions from loading them. They are not release instructions and must not be
restored or executed without a new review.

Both were introduced for the August 3, 2026 tournament cutover:

- `tmp-record-production-tournament-cutover-status.yml.txt` was added in
  `4acee6c6347ac18a4c095e5a73fa029bd7dbefd2`. It listened for the temporary v7
  cutover, compared production with the fixed `production-tournaments-live-20260803`
  tag and hard-coded data counts, then force-pushed a status branch.
- `tmp-clean-production-tournament-cutover-helpers.yml.txt` was added in
  `44a3149d6c795d1e419c110c22b2ef4671186c7f`. It listened for that status record
  and used the production API runtime to drop three temporary database helpers.

These one-time checks are not the current release contract. Routine API releases
use `.github/workflows/fly_api_deploy.yml` and its candidate, migration, runtime,
and rollback verification.

At production baseline `ed899b6baf519f0f86418bfad9a354b8538d475a`, both archived
files already had invalid YAML because embedded Python was outside the indented
`run` block. GitHub recorded failed workflow definitions with no jobs, including
runs `42525235636` and `42525236509`. The unchanged files caused the same failures
on the November 8 repair candidate. Archiving removes those obsolete definitions
without correcting their syntax or reactivating their production operations.
