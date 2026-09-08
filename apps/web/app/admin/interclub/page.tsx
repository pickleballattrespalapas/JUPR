"use client";
import { useAdminWorkspace } from "@/lib/useAdminWorkspace";
import Link from "next/link";
import { useEffect, useState } from "react";
import { useAdminSession } from "@/lib/useAdminSession";
import { getAdminPlayerEditorApiBaseUrl } from "@/lib/adminPlayerEditorApi";
import OpenRegistration from "./OpenRegistration";

type Club = {id:string;name:string;slug:string};
type Meet = {host_club_id:string;club_ids:string[];starts_at:string;duration_minutes:number;courts:number};
type Draft = {name:string;start_date:string;end_date:string;timezone:string;divisions:string[];club_ids:string[];meets:Meet[]};
type Season = {id:string;revision:number;draft:Draft};
const fresh = ():Season => ({id:crypto.randomUUID(),revision:0,draft:{name:"Southern BCS",start_date:"2027-01-01",end_date:"2027-03-31",timezone:"America/Mazatlan",divisions:["3.5","4.0"],club_ids:[],meets:[]}});
const toggle = (values:string[],value:string) => values.includes(value)?values.filter(v=>v!==value):[...values,value];
function localTime(value:string) { if(!value) return ""; const parts=new Intl.DateTimeFormat("en-CA",{timeZone:"America/Mazatlan",year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hourCycle:"h23"}).formatToParts(new Date(value));const get=(type:string)=>parts.find(p=>p.type===type)?.value;return `${get("year")}-${get("month")}-${get("day")}T${get("hour")}:${get("minute")}`; }
export default function InterclubPage() {
 const {session}=useAdminSession(); const {clubId}=useAdminWorkspace();
 return <InterclubPlanner key={`${clubId}:${session?.user?.id || session?.user?.email || ""}`} />;
}
function InterclubPlanner() {
 const {session,accessToken,loading}=useAdminSession(); const api=getAdminPlayerEditorApiBaseUrl();
 const assignments=(session?.capabilities?.assignments||[]).filter(a=>["super_admin","administrator","club_owner"].includes(a.role));
 const { clubId } = useAdminWorkspace();
 const canManage = assignments.some(assignment => assignment.club_id === clubId);
 const [choices,setChoices]=useState<Club[]>([]);const [seasons,setSeasons]=useState<Season[]>([]);const [season,setSeason]=useState<Season|null>(null);
 const [busy,setBusy]=useState(false);const [message,setMessage]=useState("");const [loaded,setLoaded]=useState(false);const [reload,setReload]=useState(0);
 useEffect(()=>{if(!api||!clubId||!accessToken||!canManage)return;const controller=new AbortController();setLoaded(false);setSeason(null);setSeasons([]);setChoices([]);
 const headers={Authorization:`Bearer ${accessToken}`};
 async function load(){try{
 const r=await fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub/setup`,{headers,signal:controller.signal});const data=await r.json();if(!r.ok)throw new Error(data.detail||"Unable to load seasons.");
 const all:Club[]=[];let offset:number|null=0;while(offset!==null){const r: Response=await fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub/club-choices?offset=${offset}`,{headers,signal:controller.signal});const d: {clubs: Club[]; next_offset: number | null}=await r.json();if(!r.ok)throw new Error("Unable to load club choices.");all.push(...d.clubs);offset=d.next_offset;}
 if(!controller.signal.aborted){setSeasons(data.seasons);setChoices(all);setLoaded(true);}
 }catch(e){if(!controller.signal.aborted)setMessage(e instanceof Error?e.message:"Unable to load.");}}void load();return()=>controller.abort();
 },[api,clubId,accessToken,reload,canManage]);
 function edit(patch:Partial<Draft>){setSeason(s=>s?{...s,draft:{...s.draft,...patch}}:s);}
 function meetEdit(index:number,patch:Partial<Meet>){if(season)edit({meets:season.draft.meets.map((m,i)=>i===index?{...m,...patch}:m)});}
 async function save(){if(!season||!api||busy)return;setBusy(true);setMessage("");try{const r=await fetch(`${api}/admin/clubs/${encodeURIComponent(clubId)}/interclub/setup`,{method:"PUT",headers:{Authorization:`Bearer ${accessToken}`,"Content-Type":"application/json"},body:JSON.stringify({season_id:season.id,expected_revision:season.revision,draft:season.draft})});const d=await r.json();if(!r.ok){const detail=Array.isArray(d.detail)?d.detail.map((e:{msg:string})=>e.msg.replace(/^Value error, /,"")).join(" "):d.detail;throw new Error(detail||"Unable to save.");}setSeason(d.season);setSeasons(old=>[d.season,...old.filter(s=>s.id!==d.season.id)]);setMessage("Season draft saved.");}catch(e){setMessage(e instanceof Error?e.message:"Unable to save.");}finally{setBusy(false);}}
 if(loading)return <p>Loading…</p>;if(!canManage)return <p>Sign in as a club administrator to plan an interclub season. <Link href="/admin/login">Sign in</Link></p>;
 const draft=season?.draft;const selected=choices.filter(c=>draft?.club_ids.includes(c.id));
 return <section style={{maxWidth:1000,margin:"0 auto",padding:24}}>
 <h1>Interclub seasons</h1><p>Plan the Southern BCS season: clubs, divisions, hosts, and meet dates.</p>
 <p>Save a season plan, then set its registration rules and invite clubs below. <Link href="/admin/interclub/registrations">Club invitations and team rosters</Link></p>
 <p><button disabled={!loaded||busy} onClick={()=>{setSeason(fresh());setMessage("");}}>New season</button> <button disabled={busy} onClick={()=>setReload(n=>n+1)}>Reload saved drafts</button></p>
 <nav aria-label="Saved seasons">{seasons.map(s=><button key={s.id} disabled={busy} onClick={()=>setSeason(s)}>{s.draft.name} · {s.draft.start_date}</button>)}</nav>
 {message&&<p role="status">{message}</p>}
 {draft&&season&&<form onSubmit={e=>{e.preventDefault();void save();}} style={{display:"grid",gap:20,marginTop:20}}><fieldset disabled={busy} style={{display:"grid",gap:12}}><legend>Season details</legend>
 <label>Name <input required maxLength={120} value={draft.name} onChange={e=>edit({name:e.target.value})}/></label>
 <label>Starts <input type="date" required value={draft.start_date} onChange={e=>edit({start_date:e.target.value})}/></label><label>Ends <input type="date" required value={draft.end_date} onChange={e=>edit({end_date:e.target.value})}/></label>
 <div>Divisions {['2.5','3.0','3.5','4.0','4.5','5.0','Open',...(draft.divisions.includes('4.5/Open')?['4.5/Open']:[])].map(d=><label key={d} style={{marginLeft:16}}><input type="checkbox" checked={draft.divisions.includes(d)} onChange={()=>edit({divisions:toggle(draft.divisions,d)})}/>{d}</label>)}</div>
 <p>Four-player teams. Three games to 11, win by two, no cap. Meet times use Southern BCS time (UTC−7).</p></fieldset>
 <fieldset disabled={busy}><legend>Proposed participating clubs</legend><p>Select clubs with PCS accounts. Selection does not send an invitation or grant access.</p>{choices.map(c=><label key={c.id} style={{display:"block",padding:6}}><input type="checkbox" checked={draft.club_ids.includes(c.id)} onChange={()=>edit({club_ids:toggle(draft.club_ids,c.id)})}/>{c.name}</label>)}<p>Missing a club? A PCS Super Admin can create its account in <Link href="/admin/platform">PCS administration</Link>.</p></fieldset>
 <h2>Meets</h2>{draft.meets.map((m,index)=><fieldset key={index} disabled={busy} style={{display:"grid",gap:12}}><legend>Meet {index+1}</legend>
 <label>Host <select required value={m.host_club_id} onChange={e=>meetEdit(index,{host_club_id:e.target.value,club_ids:m.club_ids.includes(e.target.value)?m.club_ids:[...m.club_ids,e.target.value]})}><option value="">Choose host</option>{selected.map(c=><option key={c.id} value={c.id}>{c.name}</option>)}</select></label>
 <label>Date and time <input required type="datetime-local" value={localTime(m.starts_at)} onChange={e=>meetEdit(index,{starts_at:e.target.value?`${e.target.value}:00-07:00`:""})}/></label>
 <label>Minutes <input required type="number" min={30} max={180} value={m.duration_minutes} onChange={e=>meetEdit(index,{duration_minutes:Number(e.target.value)})}/></label><label>Courts <input required type="number" min={1} max={100} value={m.courts} onChange={e=>meetEdit(index,{courts:Number(e.target.value)})}/></label>
 <div>Clubs at this meet (2–4, including host){selected.map(c=><label key={c.id} style={{display:"block",padding:6}}><input type="checkbox" checked={m.club_ids.includes(c.id)} onChange={()=>meetEdit(index,{club_ids:toggle(m.club_ids,c.id)})}/>{c.name}</label>)}</div>
 <button type="button" onClick={()=>edit({meets:draft.meets.filter((_,i)=>i!==index)})}>Remove from draft</button></fieldset>)}
 <button type="button" disabled={busy||draft.meets.length>=100} onClick={()=>edit({meets:[...draft.meets,{host_club_id:"",club_ids:[],starts_at:"",duration_minutes:180,courts:4}]})}>Add meet</button>
 <button disabled={busy||!api} type="submit">{busy?"Saving…":"Save season draft"}</button></form>}
 {season && season.revision > 0 && api && !busy && (JSON.stringify(season.draft) === JSON.stringify(seasons.find(s => s.id === season.id)?.draft)
  ? <OpenRegistration key={`${season.id}:${season.revision}`} api={api} clubId={clubId} accessToken={accessToken} seasonId={season.id} revision={season.revision} divisions={season.draft.divisions} />
  : <p>Save your planning changes before opening club invitations.</p>)}
 <p><Link href="/admin">Club operations</Link> · <Link href="/admin/staff">Club staff</Link></p></section>;
}
