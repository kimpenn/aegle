"""Browser UI for coverage-based oocyte recall review."""

from __future__ import annotations

import html


_PAGE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="data:,">
  <title>__SAMPLE_ID__ oocyte recall review</title>
  <style>
    :root{--ink:#19251f;--paper:#eee7d9;--panel:#fffaf0;--teal:#087b78;--cyan:#00e5d8;--amber:#dd7a20;--red:#ad4436;--blue:#2e668f;--line:#cfc5b2;--muted:#67716b;--shadow:0 16px 36px rgba(31,46,38,.13)}
    *{box-sizing:border-box}html,body{margin:0;min-height:100%;color:var(--ink);background:radial-gradient(circle at 12% 0,#fff8e9 0,transparent 31%),linear-gradient(140deg,#e8dfd0,#f7f3e8 62%,#ddebe4);font-family:"Iowan Old Style","Palatino Linotype",Palatino,serif}button,input,textarea{font:inherit}button{cursor:pointer}.shell{width:min(1720px,calc(100% - 24px));margin:auto}.hero{margin:14px 0;padding:22px 28px;border:1px solid var(--line);border-radius:22px;background:linear-gradient(115deg,rgba(255,250,240,.97),rgba(225,243,235,.93));box-shadow:var(--shadow);display:flex;gap:24px;align-items:end;justify-content:space-between}.eyebrow,.mono{font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace}.eyebrow{text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;color:var(--teal);font-weight:700}.hero h1{font-size:clamp(2rem,4vw,4rem);line-height:.95;margin:.18em 0}.hero p{max-width:900px;line-height:1.48;margin:.5em 0}.hero-actions{display:flex;gap:8px;flex-wrap:wrap}.button,button,input,textarea{border:1px solid #bdb3a1;border-radius:11px;background:#fffaf0;color:var(--ink);padding:8px 11px}.button{text-decoration:none;white-space:nowrap}.button:hover,button:hover{border-color:var(--teal);background:#e3f2ed}.workspace{display:grid;grid-template-columns:minmax(420px,1.05fr) minmax(520px,1.6fr) minmax(260px,.72fr);gap:14px;align-items:start}.panel{background:rgba(255,250,240,.96);border:1px solid var(--line);border-radius:18px;padding:14px;box-shadow:0 10px 26px rgba(31,46,38,.09)}.panel h2{font-size:1.25rem;margin:0 0 8px}.subtle{color:var(--muted);font-size:.85rem;line-height:1.35}.overview-stage,.patch-stage{position:relative;overflow:hidden;border-radius:13px;background:#170d20;border:1px solid #8e877b;touch-action:none}.overview-stage img,.overview-stage canvas,.patch-stage img,.patch-stage canvas{display:block;width:100%;height:100%;position:absolute;inset:0}.overview-stage{aspect-ratio:var(--overview-aspect,1)}.overview-stage img{object-fit:fill}.overview-stage canvas{cursor:crosshair}.patch-stage{aspect-ratio:1}.patch-stage img{object-fit:fill;image-rendering:auto}.patch-stage canvas{cursor:crosshair}.toolbar{display:flex;gap:7px;align-items:center;flex-wrap:wrap;margin:9px 0}.toolbar .grow{flex:1}.toolbar input[type=number]{width:105px}.status-row{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}.status-row button.selected[data-status=complete]{background:#d9eee5;border-color:#438d73}.status-row button.selected[data-status=has_misses]{background:#f4d8cf;border-color:#a94f40}.status-row button.selected[data-status=unsure]{background:#f5e8bd;border-color:#b88728}.annotate.active{background:#f7d8b8;border-color:var(--amber);box-shadow:0 0 0 3px rgba(221,122,32,.16)}.legend{display:flex;gap:13px;flex-wrap:wrap;font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace;font-size:.7rem;margin:7px 0}.swatch{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}.progress{height:9px;border-radius:999px;background:#e0d7c8;overflow:hidden;margin:8px 0}.progress span{display:block;height:100%;width:0;background:linear-gradient(90deg,var(--teal),#59aa83)}.candidate-list,.miss-list{max-height:220px;overflow:auto;border-top:1px solid var(--line);margin-top:9px}.list-row{padding:8px 2px;border-bottom:1px solid #e4dccd;font-size:.8rem}.list-row strong{display:block}.probe{white-space:pre-wrap;font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace;font-size:.7rem;background:#f2eadc;border-radius:10px;padding:9px;min-height:70px;max-height:190px;overflow:auto}.window-note{width:100%;min-height:64px;resize:vertical}.footer{padding:20px 2px 45px;color:var(--muted);font-size:.85rem}.loading{position:absolute;inset:auto 10px 10px auto;background:rgba(18,27,24,.82);color:white;padding:5px 8px;border-radius:8px;font:12px monospace;z-index:5}.hidden{display:none!important}.error{color:#9c2e26}.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace;font-size:.72rem}.metric-grid span{background:#f0e8da;border-radius:8px;padding:6px}.review-warning{border-left:4px solid var(--amber);padding:8px 10px;background:#fff0d9;font-size:.82rem}.cross-link{color:var(--teal)}
    @media(max-width:1180px){.workspace{grid-template-columns:1fr 1fr}.side-panel{grid-column:1/-1}.hero{align-items:start;flex-direction:column}}
    @media(max-width:760px){.shell{width:min(100% - 12px,1720px)}.workspace{grid-template-columns:1fr}.hero{padding:18px}.panel{padding:10px}.toolbar input[type=number]{width:88px}}
  </style>
</head>
<body>
<main class="shell">
  <header class="hero">
    <div><div class="eyebrow">Aegle / raw UCHL1 / false-negative search</div><h1>__SAMPLE_ID__ recall review</h1><p>Move the focal window across the slide, inspect every current exact mask, and click only oocytes that lack a satisfactory mask. Overlay source: <strong id="overlay-source">loading</strong>. A window is evidence only after you explicitly mark it complete, containing misses, or unsure.</p></div>
    <div class="hero-actions"><a class="button" href="review_console.html">Sample console</a><a class="button" href="oocytes.html">Precision review</a><a class="button" href="../oocyte_review_console.html">Batch index</a><button id="export-json">Export JSON</button><button id="export-csv">Export CSV</button></div>
  </header>
  <div id="fatal" class="panel error hidden"></div>
  <section id="workspace" class="workspace hidden">
    <article class="panel">
      <h2>Whole-slide navigator</h2>
      <div class="subtle">Click or drag to select the nearest deterministic coverage window. Green, red, and amber rectangles are reviewed windows; gray rectangles remain unreviewed.</div>
      <div id="overview-stage" class="overview-stage"><img id="overview-image" alt="Whole-slide downsampled raw UCHL1"><canvas id="overview-canvas"></canvas></div>
      <div class="legend"><span><i class="swatch" style="background:#00e5d8"></i>current reviewed mask</span><span><i class="swatch" style="background:#55a77b"></i>complete</span><span><i class="swatch" style="background:#ad4436"></i>has misses</span><span><i class="swatch" style="background:#ddad3e"></i>unsure</span><span><i class="swatch" style="background:#ff6c4d"></i>manual miss</span></div>
      <div class="progress"><span id="progress-bar"></span></div>
      <div id="progress-text" class="mono subtle"></div>
      <div class="toolbar"><button id="previous-window">Previous</button><button id="next-window">Next</button><button id="next-unreviewed">Next unreviewed</button><span class="grow"></span><span id="queue-position" class="mono"></span></div>
      <div class="toolbar"><label>X <input id="center-x" type="number" min="0"></label><label>Y <input id="center-y" type="number" min="0"></label><button id="go-coordinate">Go</button></div>
      <div id="window-risk" class="metric-grid"></div>
    </article>
    <article class="panel">
      <h2>Native UCHL1 focal window</h2>
      <div class="toolbar"><button id="toggle-masks">Hide masks</button><button id="annotate" class="annotate">Add missed oocyte</button><button id="undo">Undo last</button><span class="grow"></span><label>Contrast <select id="contrast"><option value="local">Local</option><option value="global">Global</option></select></label></div>
      <div id="patch-stage" class="patch-stage"><img id="patch-image" alt="Native-resolution raw UCHL1 focal patch"><img id="mask-image" alt="Exact persisted mask overlay"><canvas id="patch-canvas"></canvas><div id="loading" class="loading hidden">loading</div></div>
      <div class="legend"><span id="coordinate-readout"></span><span id="candidate-count"></span></div>
      <div class="review-warning">A click is a manual center, not an accepted boundary. It becomes a provisional segmentation diagnostic and must be reviewed before any label delivery changes.</div>
      <h2 style="margin-top:14px">Window disposition</h2>
      <div class="status-row"><button data-status="complete">Complete</button><button data-status="has_misses">Has misses</button><button data-status="unsure">Unsure</button></div>
      <textarea id="window-note" class="window-note" placeholder="Window note"></textarea>
    </article>
    <aside class="panel side-panel">
      <h2>Window objects</h2>
      <div id="candidate-list" class="candidate-list"></div>
      <h2 style="margin-top:15px">Manual misses</h2>
      <div id="miss-list" class="miss-list"></div>
      <h2 style="margin-top:15px">Latest diagnostic</h2>
      <div id="probe" class="probe">Click “Add missed oocyte”, then click the focal image.</div>
    </aside>
  </section>
  <footer class="footer">Review state is stored only in this browser until exported. Global recall requires an explicit disposition for every full-slide survey window.</footer>
</main>
<script>
'use strict';
const SCHEMA=1;let meta,state,currentIndex=0,showMasks=true,annotating=false,dragging=false,loadToken=0;const $=id=>document.getElementById(id);
    function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v))}function key(){const grid=meta.review_identity.recall_window_geometry_sha256||'legacy';return `aegle-oocyte-recall:${meta.sample_id}:v${SCHEMA}:${grid}`}
function emptyState(){return{schema_version:SCHEMA,review_type:'oocyte_recall',sample:meta.review_identity,windows:{},missing_oocytes:[],updated_at:new Date().toISOString()}}
function save(){state.updated_at=new Date().toISOString();localStorage.setItem(key(),JSON.stringify(state));drawOverview();drawPatchAnnotations();paintDisposition();paintProgress();paintMisses()}
function currentWindow(){return meta.windows[currentIndex]}
    function loadState(){try{const stored=JSON.parse(localStorage.getItem(key())||'null'),expected=meta.review_identity,actual=stored?.sample,identityMatches=actual&&Object.entries(expected).every(([field,value])=>actual[field]===value);state=stored&&stored.schema_version===SCHEMA&&identityMatches?stored:emptyState()}catch(_){state=emptyState()}}
function canvasSize(canvas,host){const box=host.getBoundingClientRect(),ratio=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(box.width*ratio));canvas.height=Math.max(1,Math.round(box.height*ratio));canvas.style.width=box.width+'px';canvas.style.height=box.height+'px';return{w:canvas.width,h:canvas.height,ratio}}
function sourceToOverview(x,y,size){return{x:x/meta.image_width*size.w,y:y/meta.image_height*size.h}}
function overviewToSource(event){const box=$('overview-canvas').getBoundingClientRect();return{x:clamp((event.clientX-box.left)/box.width*meta.image_width,0,meta.image_width-1),y:clamp((event.clientY-box.top)/box.height*meta.image_height,0,meta.image_height-1)}}
function nearestWindow(x,y){let best=0,bestD=Infinity;meta.windows.forEach((w,i)=>{const d=(w.center_x-x)**2+(w.center_y-y)**2;if(d<bestD){best=i;bestD=d}});return best}
function drawOverview(){const canvas=$('overview-canvas'),size=canvasSize(canvas,$('overview-stage')),ctx=canvas.getContext('2d');ctx.clearRect(0,0,size.w,size.h);const reviewed=state?state.windows:{};meta.windows.forEach(w=>{const s=reviewed[w.window_id]?.status,a=sourceToOverview(w.bbox.x0,w.bbox.y0,size),b=sourceToOverview(w.bbox.x1,w.bbox.y1,size);if(s){ctx.fillStyle=s==='complete'?'rgba(69,151,111,.14)':s==='has_misses'?'rgba(173,68,54,.18)':'rgba(221,173,62,.18)';ctx.fillRect(a.x,a.y,b.x-a.x,b.y-a.y)}else{ctx.strokeStyle='rgba(220,214,201,.16)';ctx.lineWidth=Math.max(1,size.ratio*.45);ctx.strokeRect(a.x,a.y,b.x-a.x,b.y-a.y)}});meta.candidates.forEach(c=>{const p=sourceToOverview(c.center_x,c.center_y,size);ctx.fillStyle='#00e5d8';ctx.beginPath();ctx.arc(p.x,p.y,Math.max(1.5,size.ratio*1.5),0,Math.PI*2);ctx.fill()});if(state)state.missing_oocytes.forEach(m=>{const p=sourceToOverview(m.x,m.y,size);ctx.strokeStyle='#ff6c4d';ctx.lineWidth=2*size.ratio;ctx.beginPath();ctx.moveTo(p.x-4*size.ratio,p.y);ctx.lineTo(p.x+4*size.ratio,p.y);ctx.moveTo(p.x,p.y-4*size.ratio);ctx.lineTo(p.x,p.y+4*size.ratio);ctx.stroke()});const w=currentWindow();if(w){const a=sourceToOverview(w.bbox.x0,w.bbox.y0,size),b=sourceToOverview(w.bbox.x1,w.bbox.y1,size);ctx.strokeStyle='#ffffff';ctx.lineWidth=2.5*size.ratio;ctx.strokeRect(a.x,a.y,b.x-a.x,b.y-a.y);ctx.strokeStyle='#172522';ctx.lineWidth=1*size.ratio;ctx.strokeRect(a.x+3*size.ratio,a.y+3*size.ratio,b.x-a.x-6*size.ratio,b.y-a.y-6*size.ratio)}}
function patchPoint(event){const box=$('patch-canvas').getBoundingClientRect(),w=currentWindow(),span=2*meta.window_radius_px+1;return{x:clamp(w.center_x-meta.window_radius_px+(event.clientX-box.left)/box.width*span,0,meta.image_width-1),y:clamp(w.center_y-meta.window_radius_px+(event.clientY-box.top)/box.height*span,0,meta.image_height-1)}}
function drawPatchAnnotations(){const canvas=$('patch-canvas'),size=canvasSize(canvas,$('patch-stage')),ctx=canvas.getContext('2d');ctx.clearRect(0,0,size.w,size.h);if(!state||!currentWindow())return;const w=currentWindow(),span=2*meta.window_radius_px+1;state.missing_oocytes.filter(m=>m.x>=w.center_x-meta.window_radius_px&&m.x<=w.center_x+meta.window_radius_px&&m.y>=w.center_y-meta.window_radius_px&&m.y<=w.center_y+meta.window_radius_px).forEach(m=>{const x=(m.x-(w.center_x-meta.window_radius_px))/span*size.w,y=(m.y-(w.center_y-meta.window_radius_px))/span*size.h;ctx.strokeStyle='#ff6c4d';ctx.lineWidth=3*size.ratio;ctx.beginPath();ctx.arc(x,y,9*size.ratio,0,Math.PI*2);ctx.moveTo(x-13*size.ratio,y);ctx.lineTo(x+13*size.ratio,y);ctx.moveTo(x,y-13*size.ratio);ctx.lineTo(x,y+13*size.ratio);ctx.stroke()})}
async function showWindow(index){currentIndex=(index+meta.windows.length)%meta.windows.length;const w=currentWindow();$('center-x').value=Math.round(w.center_x);$('center-y').value=Math.round(w.center_y);$('queue-position').textContent=`${currentIndex+1} / ${meta.windows.length}`;$('coordinate-readout').textContent=`center (${Math.round(w.center_x)}, ${Math.round(w.center_y)}) · bbox ${w.bbox.x0},${w.bbox.y0}–${w.bbox.x1},${w.bbox.y1}`;$('window-risk').innerHTML=`<span>risk ${w.risk_score.toFixed(2)}</span><span>accepted ${w.accepted_count}</span><span>near rejects ${w.near_rejected_count}</span><span>coarse ${w.coarse_count}</span>`;const ws=state.windows[w.window_id]||{};$('window-note').value=ws.notes||'';paintDisposition();drawOverview();drawPatchAnnotations();await loadPatch(w)}
async function loadPatch(w){const token=++loadToken;$('loading').classList.remove('hidden');const q=`x=${Math.round(w.center_x)}&y=${Math.round(w.center_y)}&radius=${meta.window_radius_px}`;$('patch-image').src=`api/patch.webp?${q}&contrast=${$('contrast').value}&v=${token}`;$('mask-image').src=`api/overlay.png?${q}&v=${token}`;$('mask-image').style.display=showMasks?'block':'none';try{const response=await fetch(`api/window?${q}`);if(!response.ok)throw new Error(await response.text());const payload=await response.json();if(token!==loadToken)return;$('candidate-count').textContent=`${payload.candidates.length} intersecting masks`;$('candidate-list').innerHTML=payload.candidates.length?payload.candidates.map(c=>`<div class="list-row"><strong>${escapeHtml(c.display_id||c.detector_component_id)}</strong><span class="mono">${escapeHtml(c.detector_component_id)} · score ${Number(c.detector_score).toFixed(3)} · ${escapeHtml(c.resolution_source||c.detection_pass)}</span></div>`).join(''):'<div class="list-row subtle">No reviewed masks intersect this window.</div>'}catch(error){$('candidate-list').innerHTML=`<div class="list-row error">${escapeHtml(error.message)}</div>`}finally{if(token===loadToken)$('loading').classList.add('hidden')}}
function escapeHtml(value){return String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
function paintDisposition(){if(!state||!currentWindow())return;const status=state.windows[currentWindow().window_id]?.status||'';document.querySelectorAll('[data-status]').forEach(b=>b.classList.toggle('selected',b.dataset.status===status))}
function paintProgress(){if(!state)return;const reviewed=Object.values(state.windows).filter(w=>w.status).length,misses=state.missing_oocytes.length;$('progress-bar').style.width=(reviewed/meta.windows.length*100)+'%';$('progress-text').textContent=`${reviewed} / ${meta.windows.length} windows reviewed · ${misses} manual misses`}
function paintMisses(){if(!state)return;const rows=[...state.missing_oocytes].reverse();$('miss-list').innerHTML=rows.length?rows.map(m=>`<div class="list-row"><strong>${escapeHtml(m.annotation_id)} · (${Math.round(m.x)}, ${Math.round(m.y)})</strong><span>${escapeHtml(m.failure_class||'probe pending')}</span> <button data-remove-miss="${escapeHtml(m.annotation_id)}">Remove</button><input data-miss-note="${escapeHtml(m.annotation_id)}" value="${escapeHtml(m.notes||'')}" placeholder="Missing-oocyte note" style="width:100%;margin-top:5px"></div>`).join(''):'<div class="list-row subtle">No missing oocytes annotated.</div>';document.querySelectorAll('[data-remove-miss]').forEach(b=>b.onclick=()=>{state.missing_oocytes=state.missing_oocytes.filter(m=>m.annotation_id!==b.dataset.removeMiss);save()});document.querySelectorAll('[data-miss-note]').forEach(n=>n.onchange=()=>{const miss=state.missing_oocytes.find(m=>m.annotation_id===n.dataset.missNote);if(miss){miss.notes=n.value;save()}})}
async function addMiss(point){const w=currentWindow(),annotation={annotation_id:`miss-${Date.now()}-${state.missing_oocytes.length+1}`,x:+point.x.toFixed(2),y:+point.y.toFixed(2),window_id:w.window_id,created_at:new Date().toISOString(),notes:''};state.missing_oocytes.push(annotation);const ws=state.windows[w.window_id]||{};state.windows[w.window_id]={...ws,status:'has_misses',reviewed_at:new Date().toISOString()};save();$('probe').textContent='Running click-targeted segmentation diagnostic…';try{const response=await fetch(`api/probe?x=${annotation.x}&y=${annotation.y}`);if(!response.ok)throw new Error(await response.text());const result=await response.json();annotation.failure_class=result.failure_class;annotation.probe=result;$('probe').textContent=formatProbe(result);save()}catch(error){annotation.probe_error=error.message;$('probe').textContent=`Probe failed: ${error.message}`;save()}}
function formatProbe(p){const fmt=v=>v==null?'n/a':Number(v).toFixed(1),pct=v=>v==null?'n/a':Number(v).toFixed(0),p99=p.p99_metrics||{},conservative=p.manual_conservative_metrics||{},expanded=p.manual_expanded_metrics||{};return[`class: ${p.failure_class}`,`accepted distance: ${fmt(p.nearest_accepted_distance_px)} px`,`refined distance: ${fmt(p.nearest_refined_distance_px)} px`,`coarse distance: ${fmt(p.nearest_coarse_distance_px)} px`,`baseline P99 diameter: ${fmt(p99.equivalent_diameter_um)} um`,`conservative P${pct(p.manual_conservative_percentile)}: d ${fmt(conservative.equivalent_diameter_um)} um / circ ${fmt(conservative.circularity)}`,`expanded P${pct(p.manual_expanded_percentile)}: d ${fmt(expanded.equivalent_diameter_um)} um / circ ${fmt(expanded.circularity)}`].join('\n')}
function markStatus(status){const w=currentWindow(),old=state.windows[w.window_id]||{};state.windows[w.window_id]={...old,status,notes:$('window-note').value,reviewed_at:new Date().toISOString()};save()}
function goToCoordinate(){const x=clamp(+$('center-x').value,0,meta.image_width-1),y=clamp(+$('center-y').value,0,meta.image_height-1);showWindow(nearestWindow(x,y))}
function nextUnreviewed(){for(let step=1;step<=meta.windows.length;step++){const i=(currentIndex+step)%meta.windows.length;if(!state.windows[meta.windows[i].window_id]?.status){showWindow(i);return}}}
function exportReview(type){const windows=meta.windows.filter(w=>state.windows[w.window_id]?.status).map(w=>({...w,...state.windows[w.window_id]}));const payload={schema_version:SCHEMA,review_type:'oocyte_recall',sample:meta.review_identity,exported_at:new Date().toISOString(),settings:{window_radius_px:meta.window_radius_px,window_stride_px:meta.window_stride_px},windows,missing_oocytes:state.missing_oocytes};let blob,name;if(type==='json'){blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});name=`${meta.sample_id}_recall_review.json`}else{const rows=[];windows.forEach(w=>rows.push({record_type:'window',record_id:w.window_id,status:w.status,x:w.center_x,y:w.center_y,window_id:w.window_id,notes:w.notes||'',failure_class:''}));state.missing_oocytes.forEach(m=>rows.push({record_type:'missing_oocyte',record_id:m.annotation_id,status:'missing',x:m.x,y:m.y,window_id:m.window_id,notes:m.notes||'',failure_class:m.failure_class||''}));const keys=['record_type','record_id','status','x','y','window_id','notes','failure_class'],esc=v=>'"'+String(v??'').replaceAll('"','""')+'"';blob=new Blob([[keys.join(','),...rows.map(r=>keys.map(k=>esc(r[k])).join(','))].join('\n')],{type:'text/csv'});name=`${meta.sample_id}_recall_review.csv`}const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),0)}
async function init(){try{const response=await fetch('api/metadata');if(!response.ok)throw new Error(await response.text());meta=await response.json();const overlay=meta.mask_overlay||{};$('overlay-source').textContent=`${overlay.delivery_name||overlay.mode||'automatic detector'} / ${overlay.candidate_count??meta.candidates.length} masks`;document.documentElement.style.setProperty('--overview-aspect',`${meta.image_width}/${meta.image_height}`);$('overview-image').src='recall_review/overview.webp';$('center-x').max=meta.image_width-1;$('center-y').max=meta.image_height-1;loadState();$('workspace').classList.remove('hidden');await showWindow(0);paintProgress();paintMisses();new ResizeObserver(()=>{drawOverview();drawPatchAnnotations()}).observe($('workspace'))}catch(error){$('fatal').textContent=`Recall review failed to initialize: ${error.message}`;$('fatal').classList.remove('hidden')}}
$('overview-canvas').addEventListener('pointerdown',e=>{dragging=true;$('overview-canvas').setPointerCapture(e.pointerId);const p=overviewToSource(e);currentIndex=nearestWindow(p.x,p.y);drawOverview()});$('overview-canvas').addEventListener('pointermove',e=>{if(!dragging)return;const p=overviewToSource(e);currentIndex=nearestWindow(p.x,p.y);drawOverview()});$('overview-canvas').addEventListener('pointerup',()=>{dragging=false;showWindow(currentIndex)});$('overview-canvas').addEventListener('pointercancel',()=>dragging=false);
$('patch-canvas').addEventListener('click',e=>{if(annotating)addMiss(patchPoint(e))});$('annotate').onclick=()=>{annotating=!annotating;$('annotate').classList.toggle('active',annotating);$('annotate').textContent=annotating?'Click image to add':'Add missed oocyte'};$('undo').onclick=()=>{if(state.missing_oocytes.length){state.missing_oocytes.pop();save()}};$('toggle-masks').onclick=()=>{showMasks=!showMasks;$('mask-image').style.display=showMasks?'block':'none';$('toggle-masks').textContent=showMasks?'Hide masks':'Show masks'};$('contrast').onchange=()=>showWindow(currentIndex);$('previous-window').onclick=()=>showWindow(currentIndex-1);$('next-window').onclick=()=>showWindow(currentIndex+1);$('next-unreviewed').onclick=nextUnreviewed;$('go-coordinate').onclick=goToCoordinate;document.querySelectorAll('[data-status]').forEach(b=>b.onclick=()=>markStatus(b.dataset.status));$('window-note').onchange=()=>{const w=currentWindow(),old=state.windows[w.window_id]||{};state.windows[w.window_id]={...old,notes:$('window-note').value};save()};$('export-json').onclick=()=>exportReview('json');$('export-csv').onclick=()=>exportReview('csv');
init();
</script>
</body>
</html>'''


_CONSOLE_PAGE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <link rel="icon" href="data:,">
  <title>__SAMPLE_ID__ oocyte review console</title>
  <style>
    :root{--ink:#17231d;--muted:#657069;--paper:#f5eddf;--panel:#fffaf0;--teal:#087d79;--cyan:#00cfc5;--amber:#d47a23;--line:#cfc4b0;--shadow:0 18px 42px rgba(35,48,40,.14)}
    *{box-sizing:border-box}html,body{margin:0;min-height:100%;color:var(--ink);background:radial-gradient(circle at 10% 0,#fff8e8 0,transparent 30%),linear-gradient(145deg,#e8dfd1,#f8f4e9 58%,#dcebe4);font-family:"Iowan Old Style","Palatino Linotype",Palatino,serif}.shell{width:min(1180px,calc(100% - 28px));margin:auto;padding:24px 0 54px}.hero,.panel{border:1px solid var(--line);background:rgba(255,250,240,.96);box-shadow:var(--shadow)}.hero{border-radius:24px;padding:30px}.eyebrow,.mono{font-family:"IBM Plex Mono","Aptos Mono","Courier New",monospace}.eyebrow{font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:var(--teal);font-weight:700}.hero h1{font-size:clamp(2.6rem,7vw,5.8rem);line-height:.9;margin:.18em 0}.hero p{font-size:1.05rem;line-height:1.5;max-width:820px}.facts{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}.fact{border:1px solid var(--line);border-radius:999px;padding:7px 11px;background:#f2eadc;font-size:.78rem}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-top:14px}.panel{border-radius:18px;padding:22px;box-shadow:0 11px 28px rgba(35,48,40,.09)}.step{font:700 .72rem "IBM Plex Mono","Courier New",monospace;color:var(--teal);letter-spacing:.12em;text-transform:uppercase}.panel h2{font-size:1.65rem;margin:.35rem 0}.panel p{color:var(--muted);line-height:1.45;min-height:4.2em}.button{display:inline-block;text-decoration:none;border:1px solid #a99e8b;border-radius:11px;padding:9px 13px;color:var(--ink);background:#fffaf0}.button.primary{background:var(--teal);border-color:var(--teal);color:white}.button:hover{border-color:var(--cyan);transform:translateY(-1px)}.warning{margin-top:14px;border-left:4px solid var(--amber);background:#fff0d9;padding:13px 15px;border-radius:0 12px 12px 0;line-height:1.45}.footer{margin-top:18px;color:var(--muted);font-size:.86rem}.footer a{color:var(--teal)}
    @media(max-width:720px){.grid{grid-template-columns:1fr}.hero{padding:22px}.panel p{min-height:0}}
  </style>
</head>
<body>
<main class="shell">
  <header class="hero">
    <div class="eyebrow">Aegle / raw UCHL1 / review console</div>
    <h1>__SAMPLE_ID__</h1>
    <p>Use precision review to remove false positives and recall review to find oocytes with no satisfactory mask. These are separate biological questions and require separate exported records.</p>
    <div class="facts"><span class="fact mono">profile __PROFILE_NAME__</span><span class="fact mono">overlay __OVERLAY_NAME__</span><span class="fact mono">__CANDIDATE_COUNT__ current masks</span><span class="fact mono">__WINDOW_COUNT__ recall windows</span><span class="fact mono">source __IMAGE_WIDTH__ x __IMAGE_HEIGHT__ px</span></div>
  </header>
  <section class="grid">
    <article class="panel"><div class="step">01 / Understand</div><h2>Algorithm</h2><p>Review the detector contract, raw-UCHL1 segmentation stages, exact-mask exports, rescue behavior, and known limitations.</p><a class="button" href="../oocyte_detection_algorithm.html">Open algorithm</a></article>
    <article class="panel"><div class="step">02 / Precision</div><h2>Candidate review</h2><p>Inspect every machine-accepted mask, reject non-oocytes, flag poor boundaries, and record duplicate groups.</p><a class="button" href="oocytes.html">Open precision review</a></article>
    <article class="panel"><div class="step">03 / Recall</div><h2>Coverage review</h2><p>Move through the whole-slide queue, overlay exact masks, explicitly classify windows, and click missed oocyte centers.</p><a class="button primary" href="recall_review.html">Open recall review</a></article>
    <article class="panel"><div class="step">04 / Finalize</div><h2>Export evidence</h2><p>Export JSON from both review sessions. Manual centers require a second boundary review before final labels or profiling change.</p><a class="button" href="../oocyte_review_console.html">Return to batch index</a></article>
  </section>
  <div class="warning"><strong>Review status is not inferred from page visits.</strong> A sample remains incomplete until candidate decisions and required recall-window dispositions are exported and unresolved mask choices are finalized.</div>
  <footer class="footer">Source images and detector outputs are read-only. Browser local storage is a convenience, not the durable scientific record.</footer>
</main>
</body>
</html>'''


def recall_review_page_html(sample_id: str) -> str:
    """Return the standalone recall-review page for one sample."""

    return _PAGE.replace("__SAMPLE_ID__", html.escape(sample_id))


def review_console_page_html(
    *,
    sample_id: str,
    profile_name: str,
    candidate_count: int,
    window_count: int,
    image_shape_yx: tuple[int, int],
    overlay_name: str = "automatic detector",
) -> str:
    """Return the static landing page for one sample's review tasks."""

    image_height, image_width = image_shape_yx
    replacements = {
        "__SAMPLE_ID__": html.escape(sample_id),
        "__PROFILE_NAME__": html.escape(profile_name),
        "__OVERLAY_NAME__": html.escape(overlay_name),
        "__CANDIDATE_COUNT__": str(int(candidate_count)),
        "__WINDOW_COUNT__": str(int(window_count)),
        "__IMAGE_WIDTH__": str(int(image_width)),
        "__IMAGE_HEIGHT__": str(int(image_height)),
    }
    page = _CONSOLE_PAGE
    for placeholder, value in replacements.items():
        page = page.replace(placeholder, value)
    return page


__all__ = ["recall_review_page_html", "review_console_page_html"]
