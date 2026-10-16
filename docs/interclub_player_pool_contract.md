# Interclub season interest and meet availability contract

Interest belongs to a club and season. It does not reserve a team place or freeze a rating. Existing meet rosters remain the final lineup; club admins can still choose other active club players as substitutes.

## Public browser routes

- `/interclub/signup/{share_id}`: shareable season interest form.
- `/interclub/respond#token={token}`: private season management or meet RSVP page. Keep token in fragment and POST body only; never query/path/logs.

## Public API

All responses are `Cache-Control: no-store`, `X-Robots-Tag: noindex, nofollow`.

`GET /public/interclub-signups/{share_id}` -> `{club:{id,name},season:{id,name,start_date,end_date,timezone,divisions},signup:{open},meets:[{id,host_club_id,host_club_name,starts_at}]}`. Only an accepted club can open signup. Shared link is independent of the club website's directory/publication setting.

`POST /public/interclub-signups/{share_id}` body `{name,email,divisions:string[],notes:string,request_id:UUID,email_consent:true,website?:string}` -> `{status:"registered"|"already_registered",message,manage_url?:string}`. New member is active and unlinked; consent_at records agreement to receive this season’s invitations. No player ID accepted. Name 1–120, email <=254, divisions subset season divisions, notes <=1000. Same request ID and identical payload retries return original private link; duplicate normalized name+email never returns a private link and never overwrites the existing record. No player account required.

`POST /public/interclub-player-response/review` body `{token}` -> `{kind:"season"|"meet",club,season,member:{id,name,email,divisions,notes,status,revision,player_id?},meet?:{id,starts_at,host_club_id,host_club_name},availability?:{status:"invited"|"available"|"maybe"|"unavailable",revision,deadline,open},can_respond:boolean,can_withdraw?:boolean}`. Personal lookup only; never other members' data. Invalid/revoked/expired tokens: 404. Member tokens expire after season; meet tokens after meet start, but response deadline can close earlier.

`POST /public/interclub-player-response/respond` body `{token,expected_revision,action:"update_season",name,email,divisions,notes,status:"active"|"withdrawn"}` OR `{token,expected_revision,action:"respond_meet",status:"available"|"maybe"|"unavailable"}` -> same shape as review. Atomic optimistic revision checks. 409 on conflict/closed signup/RSVP, 422 malformed input. Season withdrawal remains possible while season signup is closed (`can_withdraw`); it stops future invitations; it does not edit historical rosters. Can reactivate using the same personal link while season signup is open.

## Admin API

Base `/admin/clubs/{club_id}/interclub/registrations/{season_id}`; authenticated administrator for this exact accepted club, including organizer only if it participates. No organizer access to another club's pool or contact details.

`GET {base}/pool` -> `{club,season,signup:{share_id:string|null,revision:number,open:boolean,url:string|null},members:[{id,season_id,club_id,name,email,divisions,notes,status:"active"|"withdrawn",player_id:string|null,revision,created_at,updated_at,manage_url}],email_mode:"dry_run"|"live"|"staging_redirect"}`.

`PUT {base}/pool` body `{expected_revision:number,open:boolean,rotate_link?:boolean}` -> same as GET. Revision 0 creates settings. Opening/closing doesn't delete interests. Rotation replaces old share link.

`PATCH {base}/pool/members/{member_id}` body `{expected_revision,player_id:string|null,status:"active"|"withdrawn"}` -> `{member}`. Linking requires active player from same club. One active pool entry per linked player/season. No automatic profile creation or cross-club profile linking.

`POST {base}/pool/late-requests` body `{player_id,divisions?:string[],reason:string}` -> `{member,pool}`. Available to the accepted club's administrator only after commissioner registration closes and before the season ends. Select an existing active club player; the reason is required and limited to 500 characters. Creates one pending linked member even before the season starts. Does not grant contact consent, send mail, seed a league rating, or permit roster selection before commissioner approval. Duplicate requests return 409 with reload guidance. Ordinary public signup, bulk additions and withdrawn-member restoration remain registration gated.

Pool and commissioner approval responses include private late-request metadata separately from the commissioner's decision reason. The commissioner reviews through the existing `POST {base}/pool/approvals` revision-checked endpoint. Approval records the actual decision time and does not change already closed meet eligibility. A submitted request keeps its original player identity and explanation.

`GET {base}/meets/{meet_id}/availability` -> `{meet:{id,starts_at,host_club_id,roster_deadline},settings:{revision:number,open:boolean,deadline:string|null},responses:[{id,member_id,name,email,player_id,divisions,notes,member_status,status:"invited"|"available"|"maybe"|"unavailable",revision,invited_at,responded_at,response_url}],email_mode}`. Only this club's active season members can be invited. Previously invited/withdrawn members remain visible for history. Response URLs are privileged and never exposed by public listings.

`PUT {base}/meets/{meet_id}/availability` body `{expected_revision,open:boolean,deadline:aware ISO timestamp}` -> same as GET. Deadline must be in future when opening and no later than meet start. Meet must be upcoming and include the club.

## Email extension (owned by email agent)

Use the admin base above. Season signup email goes to selected existing club players with usable recorded email; meet invitations go to selected active pool members. Preview recipient count/names and email before explicit send. Dry-run returns testable links and reports no email sent. Endpoint names and precise email bodies are owned by email agent; coordinate UI directly.

Backend helpers in `services/api/interclub_player_pool_routes.py`:
- `pool_admin_context(get_supabase_client,authorization,club_id,season_id)` -> `(db,user,club,season)`; verifies administrator and accepted participation.
- `pool_rpc(db,name,params)` maps SQL errors to HTTP errors.
- `pool_actor(user,club_id,season_id)` -> actor RPC args.
- `pool_member_url(member,season)` -> signed season management URL.
- `pool_response_url(response,season,meet)` -> signed meet response URL.
- `pool_signup_url(share_id)` -> public signup URL.
- `prepare_meet_invitations(db,user,club_id,season_id,meet_id,member_ids)` -> `{meet,settings,responses}`; atomically creates/reuses personal capabilities via `pcs_interclub_pool_action` action `invite`; repeated invitation preparation reuses existing responses, does not reset their availability or rotate links.

Token helpers use purpose-specific HMAC over stored random nonces with the existing registration-edit secret; no new environment secret. Tables/functions are service-role-only; RLS enabled; mutations use SECURITY INVOKER RPCs with explicit club/season checks. Public signup is bounded/rate-limited and does not list pool members. SQL error 42501 ->403, P0002 ->404, 40001/23505 ->409, 22023 ->422, 54000 ->429. Unknown DB failure ->503 with reload guidance.

### Email API details

Email base is deliberately separate: `/admin/clubs/{club_id}/interclub/player-pools/{season_id}/emails`. Every endpoint requires administrator access and accepted participation for this exact club; every response is no-store. Nothing in email preview sends email or creates personal invitations.

- `GET {emailBase}/audience?kind=season|meet&meet_id=UUID` returns `{kind,meet_id,candidates:[{id,name,email,available,unavailable_reason}],defaults:{subject,message},delivery_mode,send_available,send_unavailable_reason}`. Season IDs are string player IDs with current active verified club contact subscriptions; missing or ambiguous contacts cannot be selected. Meet IDs are active pool member UUIDs. Shared inboxes receive one email containing the separately scoped personal links for the selected players.
- `POST {emailBase}/preview` body `{kind,meet_id:null|UUID,recipient_ids:string[],subject,message}` returns `{recipient_count,player_count,recipients,preview:{subject,text,html},preview_fingerprint,delivery_mode,sender,send_available,send_unavailable_reason}`. At most 200 selected players. Meet preview uses a non-working placeholder for the personal response link.
- `POST {emailBase}` body is preview request plus `{operation_key:UUID,preview_fingerprint}`. Saves the reviewed batch and returns `{operation_key,kind,meet_id,subject,recipients:[{index,name,email,status,detail,links:[{name,url}]}],recipient_count,pending_count,delivery_mode}`. Reusing the operation key with different input is rejected; an identical retry resumes the saved batch.
- `POST {emailBase}/{operation_key}/recipients/{index}/send` (empty body) attempts one saved recipient, returns `{index,status,detail,links?:[{name,url}]}`. Call sequentially for pending recipients only. Claim is durable before invitation creation/provider invocation. Uncertain attempts never automatically resend. `GET {emailBase}/{operation_key}` restores saved progress after reload.
- In `dry_run`, button text is **Prepare test invitations** and results report **Test invitation prepared; no email was sent.** No SMTP setup is needed. Working links are returned only through authenticated admin results and meet availability endpoints. Refresh the availability panel after preparation. Live delivery remains subject to existing environment and communications guards.
