# Performance Notes

This repo now avoids running full badge evaluation during page load. Benchmarks below compare
inline evaluation vs. read-only summary rendering, and show worker throughput on a synthetic queue.

## Benchmark: Page render path (synthetic)

**Command:**
```bash
python - <<'PY'
import time
from types import SimpleNamespace
import pandas as pd

from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club
from jupr_app.domain.gamification.profile import build_gamification_summary

match_rows = [
    {
        "id": "m1",
        "club_id": "club",
        "league": "Open",
        "date": "2024-01-05T10:00:00Z",
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "score_t1": 11,
        "score_t2": 7,
    }
] * 200

ctx = SimpleNamespace(
    df_matches=pd.DataFrame(match_rows),
    df_players_all=pd.DataFrame(
        [
            {"id": 1, "rating": 1200},
            {"id": 2, "rating": 1200},
            {"id": 3, "rating": 1200},
            {"id": 4, "rating": 1200},
        ]
    ),
    df_leagues=pd.DataFrame(),
    club_id="club",
)

def time_call(fn, loops=20):
    start = time.perf_counter()
    for _ in range(loops):
        fn()
    return (time.perf_counter() - start) / loops

compute_time = time_call(lambda: list(compute_candidates_for_club("club", ctx=ctx)), loops=10)

badge_defs = pd.DataFrame(
    [
        {"badge_id": "participant", "name": "Participant", "prestige": 10, "is_active": True, "category": "Participation"},
        {"badge_id": "first_win", "name": "First Win", "prestige": 15, "is_active": True, "category": "Participation"},
    ]
)
player_badges = pd.DataFrame(
    [{"badge_id": "participant", "player_id": 1, "earned_at": "2024-01-01T00:00:00Z"}]
)
summary_time = time_call(lambda: build_gamification_summary(1, badge_defs, player_badges), loops=50)

print(f"compute_candidates_for_club avg: {compute_time*1000:.2f} ms")
print(f"build_gamification_summary avg: {summary_time*1000:.2f} ms")
PY
```

**Before (inline evaluation on page load):**
- `compute_candidates_for_club avg: 117.19 ms`

**After (read-only summary):**
- `build_gamification_summary avg: 7.87 ms`

## Benchmark: Worker throughput (synthetic queue)

**Command:**
```bash
python - <<'PY'
import time
from types import SimpleNamespace
import pandas as pd

from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue

class FakeTable:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
        self.filters = []
        self.sort_key = None
        self.sort_desc = False
        self.limit_count = None
        self.update_payload = None

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def order(self, column, desc=False):
        self.sort_key = column
        self.sort_desc = desc
        return self

    def limit(self, count):
        self.limit_count = count
        return self

    def update(self, payload):
        self.update_payload = payload
        return self

    def insert(self, payload):
        rows = payload if isinstance(payload, list) else [payload]
        stored = self.storage.setdefault(self.name, [])
        for row in rows:
            row = dict(row)
            row.setdefault("id", f"{self.name}_{len(stored) + 1}")
            row.setdefault("created_at", time.time())
            stored.append(row)
        return self

    def upsert(self, rows, on_conflict=None):
        existing = self.storage.setdefault(self.name, [])
        keys = [c.strip() for c in str(on_conflict or "").split(",") if c.strip()]
        existing_keys = {tuple(row.get(k) for k in keys) for row in existing} if keys else set()
        row_list = rows if isinstance(rows, list) else [rows]
        for row in row_list:
            row = dict(row)
            key = tuple(row.get(k) for k in keys) if keys else None
            if key is not None and key in existing_keys:
                continue
            row.setdefault("id", f"{self.name}_{len(existing) + 1}")
            row.setdefault("created_at", time.time())
            existing.append(row)
            if key is not None:
                existing_keys.add(key)
        return self

    def execute(self):
        data = list(self.storage.get(self.name, []))
        for op, column, value in self.filters:
            if op == "eq":
                data = [row for row in data if str(row.get(column)) == str(value)]
        if self.sort_key:
            data = sorted(data, key=lambda row: row.get(self.sort_key), reverse=self.sort_desc)
        if self.limit_count is not None:
            data = data[: int(self.limit_count)]
        if self.update_payload is not None:
            for row in data:
                row.update(self.update_payload)
        return SimpleNamespace(data=data)

class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage if storage is not None else {}

    def table(self, name):
        return FakeTable(self.storage, name)

storage = {}
supabase = FakeSupabase(storage)

club_id = "club"
for i in range(20):
    enqueue_badge_eval(
        supabase,
        club_id=club_id,
        event_type="match_recorded",
        player_ids=[1],
        match_id=f"m{i}",
    )

ctx = SimpleNamespace(
    supabase=None,
    club_id="club",
    df_matches=pd.DataFrame(
        [
            {
                "id": "m1",
                "club_id": "club",
                "league": "Open",
                "date": "2024-01-05T10:00:00Z",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
            }
        ]
    ),
    df_players_all=pd.DataFrame(),
    df_leagues=pd.DataFrame(),
    df_meta=pd.DataFrame(),
    df_badges=pd.DataFrame([
        {"badge_id": "participant", "state": "live", "eval_triggers": ["match_recorded"]}
    ]),
    df_player_badges=pd.DataFrame(),
    name_to_id={},
    id_to_name={},
    public_mode=False,
    admin_logged_in=True,
)

start = time.perf_counter()
result = process_badge_eval_queue(
    supabase,
    club_id,
    max_jobs=20,
    time_budget_seconds=5,
    ctx=ctx,
)
elapsed = time.perf_counter() - start
print(f"processed={result['processed']} elapsed={elapsed:.3f}s")
PY
```

**Before (no worker):**
- `processed=0` (no incremental processing)

**After (worker enabled, synthetic queue):**
- `processed=20 elapsed=1.445s`

Notes:
- Worker benchmark uses an in-memory fake Supabase; timing is illustrative and should be re-run
  against a dev Supabase project for real-world numbers.
