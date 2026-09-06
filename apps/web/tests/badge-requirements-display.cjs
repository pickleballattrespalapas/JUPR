const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');

const badge = (id, category, requirement, bucket = 'Live Now') => ({badge_id:id, name:id, category, requirements:requirement, description:'A moment on tape in the film room.', prestige:10, earners_count:2, recent_earners:[], catalog_bucket:bucket, badge_status:'live', badge_award_timing:'live'});
const badges = [badge('partner','Partnerships','Play with 20 different partners.'),badge('first','Participation','Play 1 match.'),badge('growth','Improvement','Reach a JUPR rating of 4.0.'),badge('trophy','Trophies','Finish first in a tournament.','Manual / Curated')];
const payload = {club:{name:'Test Club'},summary:{badge_count:4,earned_badge_count:4,total_unique_earners_by_badge:8,unique_earner_count:2},catalog_buckets:['Live Now','Manual / Curated'].map(name => ({name,sections:badges.filter(b=>b.catalog_bucket===name).map(b=>({name:b.category,badges:[b]}))})),filters:{categories:['Participation','Improvement','Partnerships','Trophies'],scopes:[]},trophy_room:[]};
const file = path.join(__dirname,'../app/clubs/[clubSlug]/badge-codex/page.tsx');
const output = ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,esModuleInterop:true}}).outputText;
const mod = {exports:{}};
new Function('require','module','exports',output)((name)=>{
 if(name==='next/link') return {__esModule:true,default:({children,...props})=>React.createElement('a',props,children)};
 if(name==='@/lib/badgeApi') return {getClubBadgeCodex:async()=>({data:payload,error:null})};
 return require(name);
},mod,mod.exports);
(async()=>{
 const html = renderToStaticMarkup(await mod.exports.default({params:{clubSlug:'test'},searchParams:{bucket:'all'}}));
 assert.doesNotMatch(html,/film room|moment on tape/);
 assert.match(html,/How to earn it:/);
 assert.match(html,/Badge earners<\/strong><br\/>2/);
 assert.ok(html.indexOf('data-badge-category="Participation"') < html.indexOf('data-badge-category="Improvement"'));
 assert.ok(html.indexOf('data-badge-category="Improvement"') < html.indexOf('data-badge-category="Partnerships"'));
 assert.match(html,/bucket=all&amp;category=Trophies/);
 assert.match(html,/All-time badge totals/);
 console.log('badge requirement rendering passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
