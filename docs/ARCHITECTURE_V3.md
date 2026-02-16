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
States
State	Editable	Evaluated	Visible	Deletable
draft	yes	no	no	yes
published	yes*	yes	yes	no
locked	no	yes	yes	no
archived	no	no	historical only	no
*Published badges remain editable until first award.
4. Multi-Tenant Ownership Model
System Badges
club_id = NULL
is_system_badge = true
Not editable. Not deletable.
Club Badges
club_id = <club_id>
is_system_badge = false
Editable until locked.
5. Role Model
Roles per club:
admin
coordinator
score_entry
Only admin may:
Create badge
Publish badge
Archive badge
Duplicate locked badge
Run recompute
Multiple admins per club allowed.
6. Fact Registry
All badge rules must reference registered facts.
Fields:
fact_key
description
data_type ('numeric' | 'boolean')
allowed_scope ('overall' | 'league' | 'event')
Facts must be precomputed and stored in player_badge_facts.
7. Rule Model
Each badge may have 1–5 conditions.
Fields per condition:
fact_key
operator ('>=','>','=','<=','<','is')
value_numeric (nullable)
value_boolean (nullable)
Rules must obey:
AND-only logic
Boolean facts use operator is
No OR logic
No nested groups
No cross-player comparisons
8. Evaluation Engine
Only badges with:
status = 'published'
are evaluated.
Evaluation must be:
Deterministic
Idempotent
Stateless
First award triggers is_locked = true.
9. Immutability Contract
Once awarded:
Conditions immutable
Scope immutable
Threshold immutable
To modify: duplicate badge.
10. Publish Workflow
Publishing validates:
Scope alignment
Condition count (≤ 5)
Fact registry compliance
Publishing may optionally trigger recompute.
11. Archive Behavior
Archived badges:
No longer evaluated
Remain visible historically
Cannot be reactivated
12. Guardrails
Max 25 active badges per club
Max 5 conditions per badge
No deletion after publish
No deletion after award
13. Deterministic Recompute
Recompute must:
Never duplicate awards
Never alter locked badge rules
Always produce consistent results
14. Relationship to V2
Badge Engine V3 is a subsystem layered on top of the V2 platform architecture and must comply with all V2 constraints.
End of Architecture V3
