# Next Replay History

This document tracks the guarded Replay History workflow for the Next.js and FastAPI stack.

## Scope

- FastAPI status endpoint: `GET /admin/clubs/{club_id}/replay-history`.
- FastAPI run endpoint: `POST /admin/clubs/{club_id}/replay-history`.
- Next staff route: `/admin/replay-history`.
- Replay options include `ALL (Full System Reset)` plus leagues discovered from metadata or existing rating/match rows.
- Replay calls the Python `replay_history` domain function server-side.
- Replay writes are audit-attributed and flagged for review.

## Runtime flag

Replay is disabled unless the FastAPI runtime enables:

```text
JUPR_ENABLE_NEXT_ADMIN_REPLAY=1
```

The status endpoint is useful while disabled: it lets the Next page render fallback instructions without requiring database configuration.

## Authorization

When enabled, replay requires a Supabase access token with a role that has `run_replay` permission.

## Operator confirmation

Replay requires confirmation text:

```text
REPLAY
```

## Behavior

- League replay rewrites match snapshots and rebuilds `league_ratings` for the selected league.
- Full reset also updates overall player rating, wins, losses, and match count.
- Replay remains in Python; TypeScript does not implement rating or replay logic.

## Next step

Validate this during the closed-club pilot after Match Log edits or duplicate cleanup. Then decide whether to add replay job history, async job orchestration, or a stricter two-person approval flow before wider staff use.
