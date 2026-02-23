ARCHITECTURE_V3.md
JUPR Badge Engine — Multi-Tenant, Admin-Defined, Deterministic
1. Purpose
Badge Engine V3 provides:
Club-admin defined badges
Multi-condition AND logic
Numeric and boolean fact support
Draft → Publish lifecycle
Immutable rules after first award
Deterministic evaluation
Replayable recompute
Multi-tenant isolation
This engine must:
Never execute arbitrary SQL
Never evaluate dynamic expressions
Never change badge meaning retroactively
Always remain idempotent
2. Core Principles
Facts are the source of truth.
Rules evaluate facts only.
All conditions are AND-only.
Rules are immutable once awarded.
Publishing is required before evaluation.
Only Club Admins can publish.
All evaluation is deterministic and replayable.
3. Badge Lifecycle
Badges move through controlled states.
States
State	Editable	Evaluated	Visible	Deletable
draft	yes	no	no	yes
published	yes*	yes	yes	no
locked	no	yes	yes	no
archived	no	no	historical only	no
*Published badges remain editable until first award.
Transitions
Draft → Published
Published → Locked (automatic on first award)
Published → Archived
Locked → Archived
No backward transitions.
4. Multi-Tenant Ownership Model
System Badges
club_id = NULL
is_system_badge = true
Global
Not editable
Not deletable
Club Badges
club_id = <club_id>
is_system_badge = false
Editable until locked
Scoped to that club only
5. Role Model
Roles Per Club
admin
coordinator
score_entry
Authority Matrix
Only admin may:
Create badge
Edit draft badge
Publish badge
Archive badge
Duplicate locked badge
Run recompute
Multiple admins allowed per club.
6. Fact Registry (Controlled Surface)
Admins cannot reference raw database columns.
All rule conditions must reference registered facts.
badge_fact_registry
Fields:
fact_key
description
data_type ('numeric' | 'boolean')
allowed_scope ('overall' | 'league' | 'event')
Examples:
best_win_streak (numeric, overall)
total_matches (numeric, overall)
rating_delta (numeric, overall)
is_league_champion (boolean, league)
is_event_winner (boolean, event)
is_undefeated_event (boolean, event)
Facts must be precomputed and stored in player_badge_facts.
Badge evaluation never calculates facts dynamically.
7. Rule Model
Each badge may have 1–5 conditions.
badge_rule_conditions
Fields:
badge_id
fact_key
operator ('>=','>','=','<=','<','is')
value_numeric (nullable)
value_boolean (nullable)
Rules must obey:
AND-only logic
All facts must match badge scope
Boolean facts use operator is only
No OR logic
No nested groups
No cross-player comparisons
8. Evaluation Engine
Engine evaluates only:
WHERE status = 'published'
For each player:
for condition in badge.conditions:
    fact_value = get_fact(player, condition.fact_key, scope)
    if not compare(fact_value, operator, value):
        return False
return True
If True:
Insert into player_badges
Enforced uniqueness on (club_id, player_id, badge_id, context_id)
Increment award_count
If award_count > 0 → set is_locked = true
Evaluation must be:
Idempotent
Stateless
Deterministic
9. Immutability Contract
Once:
award_count > 0
Then:
Conditions immutable
Scope immutable
Fact keys immutable
Thresholds immutable
To modify:
Duplicate badge
Edit duplicate
Publish duplicate
Archive original (optional)
10. Publish Workflow
Badges are created in Draft state.
Publishing:
Validates scope alignment
Validates condition count (≤ 5)
Validates fact registry references
Sets published_at
Optionally triggers recompute
Recompute options:
Retroactive (all historical data)
Forward-only (from now on)
11. Archive Behavior
Archived badges:
No longer evaluated
Remain visible historically
Cannot be reactivated
Preserve award integrity
12. Performance Model
Evaluation should be triggered:
On match finalization (via queue)
On league end
On event completion
On recompute run
Never during page render.
All heavy work runs in background job.
13. Guardrails
To prevent badge explosion:
Max 25 active badges per club
Max 5 conditions per badge
No mixed-scope conditions
No deletion after publish
No deletion after award
14. Deterministic Recompute
Recompute creates:
badge_eval_run record
Queues eligible players
Re-evaluates facts
Applies rules
Recompute must not:
Duplicate awards
Modify locked badge rules
Change historical thresholds
15. Future Extension (Not Implemented Yet)
Potential future expansions:
OR logic groups
Seasonal badge collections
Prestige ranking leaderboard
Badge rarity weighting
Cross-club badge templates
Not part of V3 initial implementation.
16. Non-Goals
V3 will not support:
Arbitrary SQL rule definitions
Dynamic Python evaluation
Nested logical expressions
Retroactive rule modification
Badge deletion after award
17. Success Criteria
The engine is correct if:
Publishing never corrupts historical meaning
Recompute always produces identical results
Clubs can safely experiment
Badge meaning remains stable
Performance scales with club growth
Badge Engine V3 is a subsystem layered on top of the V2 platform architecture and must comply with all V2 constraints.
End of Architecture V3
