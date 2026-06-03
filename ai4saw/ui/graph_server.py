"""Live-updating web server for the search provenance graph.

Serves a single-page vis.js graph. Polls /graph.json every 3 seconds.
vis.js is downloaded once at startup and served locally — no CDN dependency.

Usage:
    from ai4saw.ui.graph_server import start_graph_server
    port = start_graph_server(graph_path)  # returns port number
"""

from __future__ import annotations

import json
import socketserver
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


_server: Optional[_ThreadingHTTPServer] = None
_server_thread: Optional[threading.Thread] = None
_graph_path: Optional[Path] = None
_port: int = 0
_project_name: str = ""
_visjs: Optional[bytes] = None          # vis-network JS served locally
_visjs_lock = threading.Lock()


# ── vis.js local cache ────────────────────────────────────────────────────────

_VIS_URLS = [
    "https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js",
    "https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js",
    "https://unpkg.com/vis-network/standalone/umd/vis-network.min.js",
]

def _fetch_visjs() -> None:
    global _visjs
    for url in _VIS_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ai4saw/1.0"})
            with urllib.request.urlopen(req, timeout=15) as r:
                data = r.read()
            if len(data) > 10_000:          # sanity check — real file is ~600KB
                with _visjs_lock:
                    _visjs = data
                return
        except Exception:
            continue


# ── Single-page application HTML ──────────────────────────────────────────────

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>AI4SAW — Research Graph</title>
<script src="/vis-network.js" onerror="window._visFailed=true"></script>
<style>
:root {
  --seed:   #1d4ed8; --seed-l:   #dbeafe;
  --llm:    #065f46; --llm-l:    #d1fae5;
  --source: #92400e; --source-l: #fef3c7;
  --url:    #6b7280; --url-l:    #f3f4f6;
  --entity: #991b1b; --entity-l: #fee2e2;
  --border: #e5e7eb; --bg: #ffffff; --text: #111827;
  --muted:  #6b7280; --panel: #f9fafb;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:-apple-system,'Segoe UI',sans-serif;
     height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── Header ── */
#hdr{display:flex;align-items:center;gap:10px;padding:10px 18px;
     border-bottom:1px solid var(--border);background:var(--bg);flex-shrink:0;
     box-shadow:0 1px 3px rgba(0,0,0,.06)}
#logo{font-size:13px;font-weight:700;color:var(--text);letter-spacing:.5px}
#logo span{color:var(--seed)}
#stats{font-size:11px;color:var(--muted);background:var(--panel);
       padding:3px 10px;border-radius:20px;border:1px solid var(--border)}
#project-name{font-size:11px;color:var(--muted);font-style:italic}
#status{margin-left:auto;font-size:11px;display:flex;align-items:center;gap:5px}
.dot-live{width:7px;height:7px;border-radius:50%;background:#10b981;
           box-shadow:0 0 0 3px #d1fae5;display:inline-block}
.dot-off {width:7px;height:7px;border-radius:50%;background:#ef4444;display:inline-block}

/* ── Toolbar ── */
#toolbar{display:flex;align-items:center;gap:8px;padding:7px 18px;
         border-bottom:1px solid var(--border);background:var(--panel);flex-shrink:0;
         flex-wrap:wrap}
.btn{display:inline-flex;align-items:center;gap:4px;padding:4px 11px;border-radius:6px;
     border:1px solid var(--border);background:var(--bg);color:var(--text);
     font-size:11px;cursor:pointer;transition:.15s}
.btn:hover{background:var(--panel);border-color:#9ca3af}
.btn.active{background:var(--seed);color:#fff;border-color:var(--seed)}
.sep{width:1px;height:20px;background:var(--border);margin:0 2px}
.filter-chip{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
             border-radius:20px;border:1.5px solid;font-size:11px;cursor:pointer;
             user-select:none;transition:.15s;font-weight:500}
.filter-chip.off{opacity:.35;filter:grayscale(1)}

/* ── Legend chips ── */
#leg{display:flex;align-items:center;gap:8px;padding:6px 18px;
     border-bottom:1px solid var(--border);background:var(--bg);flex-shrink:0;
     font-size:11px;flex-wrap:wrap}
#leg label{font-size:11px;color:var(--muted);margin-right:2px;font-weight:500}

/* ── Graph canvas ── */
#wrap{flex:1;position:relative;overflow:hidden}
#network{width:100%;height:100%}

/* ── Empty state ── */
#empty{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
       text-align:center;pointer-events:none;display:none}
#empty svg{width:64px;height:64px;opacity:.25}
#empty h3{color:#374151;margin-top:12px;font-size:15px}
#empty p{color:var(--muted);font-size:12px;margin-top:6px;line-height:1.5}
code{background:var(--panel);padding:1px 5px;border-radius:3px;
     font-family:monospace;font-size:11px}

/* ── Tooltip ── */
#tip{position:fixed;background:#fff;border:1px solid var(--border);
     box-shadow:0 4px 12px rgba(0,0,0,.12);padding:8px 12px;border-radius:8px;
     font-size:11px;max-width:360px;display:none;pointer-events:none;z-index:200;
     word-break:break-all;line-height:1.5}
#tip strong{display:block;margin-bottom:3px;color:var(--text)}
#tip .tip-type{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;
               font-weight:600;margin-bottom:5px}

/* ── Info panel (click) ── */
#info{position:absolute;right:0;top:0;bottom:0;width:280px;background:#fff;
      border-left:1px solid var(--border);box-shadow:-4px 0 12px rgba(0,0,0,.07);
      display:none;flex-direction:column;z-index:100;overflow:hidden}
#info-hdr{padding:12px 14px;border-bottom:1px solid var(--border);
          display:flex;align-items:center;justify-content:space-between}
#info-hdr h4{font-size:13px;font-weight:600}
#info-close{cursor:pointer;color:var(--muted);font-size:16px;line-height:1;
            padding:0 4px;border-radius:4px;border:none;background:none}
#info-close:hover{background:var(--panel)}
#info-body{padding:14px;font-size:12px;overflow-y:auto;line-height:1.6;flex:1}
#info-body .row{display:flex;gap:8px;margin-bottom:6px}
#info-body .lbl{color:var(--muted);flex-shrink:0;width:64px;font-size:11px}
#info-body .val{word-break:break-all}
#info-body .nbr{margin-top:10px;padding-top:10px;border-top:1px solid var(--border)}
#info-body .nbr-item{padding:4px 0;border-bottom:1px solid #f3f4f6;font-size:11px}

/* ── Fallback ── */
#fallback{display:none;flex:1;overflow:auto;padding:20px;font-size:12px;line-height:1.7;background:#fff}
#fallback h2{font-size:13px;font-weight:600;color:#374151;margin-bottom:12px}
.frow{padding:2px 6px;border-radius:4px;margin:1px 0;font-size:11px}
.frow.seed_query{background:var(--seed-l);color:var(--seed)}
.frow.llm_query {background:var(--llm-l); color:var(--llm)}
.frow.source    {background:var(--source-l);color:var(--source)}
.frow.url       {background:var(--url-l);  color:var(--url)}
.frow.entity    {background:var(--entity-l);color:var(--entity)}
</style>
</head>
<body>

<!-- Header -->
<div id="hdr">
  <div id="logo">AI4SAW <span>Research Graph</span></div>
  <div id="project-name"></div>
  <div id="stats">Loading…</div>
  <div id="status"><span class="dot-off" id="sdot"></span><span id="stext">connecting</span></div>
</div>

<!-- Legend + type filter -->
<div id="leg">
  <label>Show:</label>
  <span class="filter-chip" id="f-seed"   data-type="seed_query" style="color:var(--seed);border-color:var(--seed);background:var(--seed-l)">◆ Seed queries</span>
  <span class="filter-chip" id="f-llm"    data-type="llm_query"  style="color:var(--llm);border-color:var(--llm);background:var(--llm-l)">● LLM queries</span>
  <span class="filter-chip" id="f-source" data-type="source"     style="color:var(--source);border-color:var(--source);background:var(--source-l)">■ Sources</span>
  <span class="filter-chip" id="f-entity" data-type="entity"     style="color:var(--entity);border-color:var(--entity);background:var(--entity-l)">★ Entities</span>
  <span class="filter-chip off" id="f-url" data-type="url"       style="color:var(--url);border-color:var(--url);background:var(--url-l)">● URLs</span>
</div>

<!-- Toolbar -->
<div id="toolbar">
  <button class="btn" id="btn-fit"  onclick="fitGraph()">⊕ Fit all</button>
  <button class="btn" id="btn-hier" onclick="toggleLayout()">⇅ Hierarchy</button>
  <div class="sep"></div>
  <button class="btn" id="btn-seeds" onclick="focusSeeds()">Show seed chains</button>
  <div class="sep"></div>
  <span id="type-counts" style="font-size:11px;color:var(--muted)"></span>
</div>

<!-- Graph -->
<div id="wrap">
  <div id="network"></div>
  <div id="empty">
    <svg viewBox="0 0 64 64" fill="none"><circle cx="32" cy="32" r="28" stroke="#9ca3af" stroke-width="2"/><circle cx="20" cy="28" r="5" fill="#9ca3af"/><circle cx="44" cy="28" r="5" fill="#9ca3af"/><circle cx="32" cy="44" r="5" fill="#9ca3af"/><line x1="25" y1="28" x2="39" y2="28" stroke="#9ca3af" stroke-width="1.5"/><line x1="22" y1="32" x2="29" y2="40" stroke="#9ca3af" stroke-width="1.5"/><line x1="42" y1="32" x2="35" y2="40" stroke="#9ca3af" stroke-width="1.5"/></svg>
    <h3>No graph data yet</h3>
    <p>Run <code>ai4saw research "your query"</code><br>with a project active — the graph<br>populates as documents are discovered.</p>
  </div>
  <div id="info">
    <div id="info-hdr"><h4 id="info-title">Node</h4><button id="info-close" onclick="closeInfo()">✕</button></div>
    <div id="info-body"></div>
  </div>
</div>
<div id="tip"></div>
<div id="fallback"><h2>Graph (text view — vis.js unavailable)</h2><div id="fnodes"></div></div>

<script>
// ── Colours & shapes ───────────────────────────────────────────────────────────
const C = {
  seed_query:{ bg:'#1d4ed8', border:'#1e40af', font:'#fff', shape:'diamond', size:22 },
  llm_query: { bg:'#065f46', border:'#064e3b', font:'#fff', shape:'ellipse', size:16 },
  source:    { bg:'#92400e', border:'#78350f', font:'#fff', shape:'box',     size:18 },
  url:       { bg:'#e5e7eb', border:'#d1d5db', font:'#374151',shape:'dot',  size:5  },
  entity:    { bg:'#991b1b', border:'#7f1d1d', font:'#fff', shape:'star',    size:20 },
};
const EDGE_COLOUR = {
  queried:     '#93c5fd', // blue
  returned:    '#d1d5db', // gray — very light, URLs are noise
  triggered_by:'#fbbf24', // amber
  triggered:   '#f87171', // red
  generated:   '#6ee7b7', // green
};

// ── State ─────────────────────────────────────────────────────────────────────
let net=null, nodeDS=null, edgeDS=null, visReady=false;
let lastData={nodes:[],edges:[]};
let hiddenTypes=new Set(['url']);   // URLs hidden by default (too noisy)
let hierarchical=false;

// ── Init vis.js ───────────────────────────────────────────────────────────────
function initVis() {
  if (window._visFailed || typeof vis==='undefined') {
    document.getElementById('fallback').style.display='block';
    document.getElementById('wrap').style.display='none';
    return false;
  }
  nodeDS = new vis.DataSet();
  edgeDS = new vis.DataSet();
  net = new vis.Network(
    document.getElementById('network'),
    { nodes:nodeDS, edges:edgeDS },
    {
      physics:{
        forceAtlas2Based:{gravitationalConstant:-80,centralGravity:.005,
                           springLength:120,springConstant:.06,damping:.5},
        solver:'forceAtlas2Based',
        stabilization:{iterations:180,fit:true},
      },
      edges:{
        arrows:{to:{enabled:true,scaleFactor:0.45}},
        smooth:{type:'dynamic'},
        color:{inherit:false},
        width:1.2,
        selectionWidth:2.5,
        font:{size:0},   // hide labels by default — use tooltip
      },
      nodes:{font:{size:11,multi:false},borderWidth:1.5,shadow:{enabled:true,size:4,x:1,y:1,color:'rgba(0,0,0,.08)'}},
      interaction:{hover:true,navigationButtons:true,keyboard:true,tooltipDelay:100},
    }
  );
  net.on('hoverNode', showTip);
  net.on('blurNode', ()=>{ document.getElementById('tip').style.display='none'; });
  net.on('click',    onClickNode);
  net.on('stabilized', ()=>{ net.fit({animation:{duration:700,easingFunction:'easeInOutQuad'}}); });
  return true;
}

// ── Tooltip ───────────────────────────────────────────────────────────────────
const BADGE = {
  seed_query:'background:#dbeafe;color:#1d4ed8',
  llm_query :'background:#d1fae5;color:#065f46',
  source    :'background:#fef3c7;color:#92400e',
  url       :'background:#f3f4f6;color:#374151',
  entity    :'background:#fee2e2;color:#991b1b',
};
function showTip(p) {
  const n=nodeDS.get(p.node); if(!n) return;
  const tip=document.getElementById('tip');
  tip.style.cssText='display:block;left:'+(p.event.clientX+14)+'px;top:'+(p.event.clientY+14)+'px';
  tip.innerHTML='<span class="tip-type" style="'+BADGE[n.ntype]+'">'+n.ntype.replace('_',' ')+'</span>'
    +'<strong>'+n.fullLabel.replace(/</g,'&lt;')+'</strong>';
}

// ── Info panel (click) ────────────────────────────────────────────────────────
function onClickNode(p) {
  if (!p.nodes.length) { closeInfo(); return; }
  const n = nodeDS.get(p.nodes[0]); if (!n) return;
  const panel = document.getElementById('info');
  panel.style.display='flex';
  document.getElementById('info-title').textContent = n.ntype.replace('_',' ');
  // Neighbours
  const connected = net.getConnectedNodes(p.nodes[0]);
  const nbrs = connected.map(id=>nodeDS.get(id)).filter(Boolean);
  const nbHTML = nbrs.slice(0,12).map(nb=>
    '<div class="nbr-item"><span style="'+BADGE[nb.ntype]+';padding:1px 6px;border-radius:8px;font-size:10px;font-weight:600">'+nb.ntype.replace('_',' ')+'</span> '+
    nb.fullLabel.replace(/</g,'&lt;').slice(0,60)+'</div>'
  ).join('');
  document.getElementById('info-body').innerHTML=
    '<div class="row"><div class="lbl">Type</div><div class="val"><span style="'+BADGE[n.ntype]+';padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600">'+n.ntype.replace('_',' ')+'</span></div></div>'
   +'<div class="row"><div class="lbl">Label</div><div class="val">'+n.fullLabel.replace(/</g,'&lt;')+'</div></div>'
   +(nbrs.length?'<div class="nbr"><b style="font-size:11px">Connected ('+nbrs.length+')</b>'+nbHTML+'</div>':'');
}
function closeInfo() { document.getElementById('info').style.display='none'; }

// ── Filter chips ──────────────────────────────────────────────────────────────
document.querySelectorAll('.filter-chip').forEach(chip=>{
  chip.addEventListener('click', ()=>{
    const t=chip.dataset.type;
    if (hiddenTypes.has(t)) { hiddenTypes.delete(t); chip.classList.remove('off'); }
    else                    { hiddenTypes.add(t);    chip.classList.add('off');    }
    rebuildGraph();
  });
});

// ── Layout toggle ─────────────────────────────────────────────────────────────
function toggleLayout() {
  hierarchical = !hierarchical;
  document.getElementById('btn-hier').classList.toggle('active', hierarchical);
  if (!net) return;
  net.setOptions({
    layout: hierarchical
      ? {hierarchical:{direction:'UD',sortMethod:'directed',levelSeparation:110,nodeSpacing:60}}
      : {hierarchical:false},
    physics: hierarchical
      ? {enabled:false}
      : {forceAtlas2Based:{gravitationalConstant:-80,centralGravity:.005,springLength:120,springConstant:.06,damping:.5},solver:'forceAtlas2Based',stabilization:{iterations:120,fit:true}},
  });
  if (!hierarchical) setTimeout(()=>net.fit({animation:true}), 800);
}

// ── Fit ───────────────────────────────────────────────────────────────────────
function fitGraph() {
  if (net) net.fit({animation:{duration:600,easingFunction:'easeInOutQuad'}});
}

// ── Focus seeds: highlight seed nodes + their direct children ─────────────────
function focusSeeds() {
  if (!net) return;
  const seeds = nodeDS.get({filter:n=>n.ntype==='seed_query'}).map(n=>n.id);
  if (!seeds.length) return;
  const kids = seeds.flatMap(id=>net.getConnectedNodes(id,'to'));
  net.selectNodes([...seeds,...kids]);
  net.fit({nodes:[...seeds,...kids],animation:{duration:700,easingFunction:'easeInOutQuad'}});
}

// ── Build vis nodes/edges from raw data ───────────────────────────────────────
function rebuildGraph() {
  if (!visReady) return;
  const ns = lastData.nodes.filter(n=>!hiddenTypes.has(n.type));
  const visibleIds = new Set(ns.map(n=>n.id));
  const es = lastData.edges.filter(e=>visibleIds.has(e.source)&&visibleIds.has(e.target));

  const inNodes = ns.map(n=>{
    const c=C[n.type]||C.url;
    const short=n.label.length>38?n.label.slice(0,36)+'…':n.label;
    return {
      id:n.id, label:short, fullLabel:n.label, ntype:n.type,
      color:{background:c.bg,border:c.border,highlight:{background:c.bg,border:'#1d4ed8'}},
      font:{color:c.font,size:n.type==='seed_query'?13:n.type==='url'?9:11},
      shape:c.shape,
      size:c.size,
      shadow:{enabled:true,size:5,x:1,y:2,color:'rgba(0,0,0,.1)'},
    };
  });
  const inEdges = es.map(e=>({
    id:e.source+'>'+e.target+'>'+e.type,
    from:e.source, to:e.target,
    color:{color:EDGE_COLOUR[e.type]||'#e5e7eb',opacity:e.type==='returned'?.4:.75},
    dashes:e.type==='queried',
    width:e.type==='triggered'||e.type==='queried'?2:1,
  }));

  const existN=new Set(nodeDS.getIds()), inNM=new Map(inNodes.map(n=>[n.id,n]));
  nodeDS.remove([...existN].filter(id=>!inNM.has(id)));
  nodeDS.add(inNodes.filter(n=>!existN.has(n.id)));

  const existE=new Set(edgeDS.getIds()), inEM=new Map(inEdges.map(e=>[e.id,e]));
  edgeDS.remove([...existE].filter(id=>!inEM.has(id)));
  edgeDS.add(inEdges.filter(e=>!existE.has(e.id)));
}

// ── Apply full graph update ────────────────────────────────────────────────────
let prevNodeCount=-1;
function applyGraph(data) {
  if (!data||!data.nodes) return;
  lastData = data;
  const total=data.nodes.length;

  // Stats bar
  const byType={};
  data.nodes.forEach(n=>{byType[n.type]=(byType[n.type]||0)+1;});
  document.getElementById('stats').textContent=total+' nodes · '+data.edges.length+' edges';
  document.getElementById('type-counts').textContent=
    Object.entries(byType).map(([t,n])=>t.replace('_',' ')+': '+n).join('  ·  ');

  document.getElementById('empty').style.display=total===0?'block':'none';

  if (!visReady||typeof vis==='undefined') {
    applyFallback(data); return;
  }

  rebuildGraph();

  // Fit on first real data load
  if (total>0 && prevNodeCount===0) {
    setTimeout(()=>net&&net.fit({animation:{duration:800,easingFunction:'easeInOutQuad'}}),400);
  }
  prevNodeCount=total;
}

function applyFallback(data) {
  const groups={};
  (data.nodes||[]).forEach(n=>{if(!groups[n.type])groups[n.type]=[];groups[n.type].push(n.label);});
  document.getElementById('fnodes').innerHTML=Object.entries(groups).map(([t,ls])=>
    '<div style="margin-bottom:10px"><b style="font-size:12px">'+t.replace('_',' ')+' ('+ls.length+')</b><br>'+
    ls.map(l=>'<div class="frow '+t+'">'+l.replace(/</g,'&lt;')+'</div>').join('')+'</div>'
  ).join('');
}

// ── Polling ────────────────────────────────────────────────────────────────────
const sdot=document.getElementById('sdot'), stext=document.getElementById('stext');
function poll() {
  fetch('/graph.json')
    .then(r=>{if(!r.ok)throw r;return r.json();})
    .then(d=>{
      applyGraph(d);
      sdot.className='dot-live'; stext.textContent='live';
    })
    .catch(()=>{sdot.className='dot-off'; stext.textContent='offline';});
}

window.addEventListener('load',()=>{
  visReady=initVis();
  // Fetch project name from URL params or page title
  const url=new URL(window.location.href);
  const proj=url.searchParams.get('project');
  if(proj) document.getElementById('project-name').textContent=proj;
  poll();
  setInterval(poll,3000);
});
</script>
</body>
</html>
"""


# ── HTTP handler ───────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path in ("/", "/index.html"):
            html = _HTML
            if _project_name:
                html = html.replace(
                    '<div id="project-name"></div>',
                    f'<div id="project-name">{_project_name}</div>',
                    1,
                )
            self._send(200, "text/html; charset=utf-8", html.encode())
        elif path == "/graph.json":
            self._send(200, "application/json", _read_graph_json())
        elif path == "/vis-network.js":
            with _visjs_lock:
                data = _visjs
            if data:
                self._send(200, "application/javascript", data)
            else:
                # vis.js not yet downloaded — redirect to CDN
                self.send_response(302)
                self.send_header("Location", _VIS_URLS[0])
                self.end_headers()
        elif path == "/favicon.ico":
            self._send(204, "text/plain", b"")
        else:
            self._send(404, "text/plain", b"Not found")


# ── Graph JSON ─────────────────────────────────────────────────────────────────

def _read_graph_json() -> bytes:
    if _graph_path and _graph_path.exists():
        try:
            data = json.loads(_graph_path.read_text(encoding="utf-8"))
            nodes = [
                {"id": n["id"], "label": n.get("label", n["id"]),
                 "type": n.get("type", "url"), "ingested": n.get("ingested", False)}
                for n in data.get("nodes", [])
            ]
            edges = [
                {"source": e["src"], "target": e["dst"], "type": e.get("type", "")}
                for e in data.get("edges", [])
            ]
            return json.dumps({
                "nodes": nodes, "edges": edges,
                "stats": {"nodes": len(nodes), "edges": len(edges)},
            }).encode()
        except Exception:
            pass
    return b'{"nodes":[],"edges":[],"stats":{"nodes":0,"edges":0}}'


# ── Public API ─────────────────────────────────────────────────────────────────

def push_update(graph_json: str) -> None:
    """No-op — clients poll /graph.json directly."""
    pass


def start_graph_server(graph_path: Path, port: int = 0, project_name: str = "") -> int:
    """Start the graph web server. Returns the port number."""
    global _server, _server_thread, _graph_path, _port, _project_name
    _graph_path = graph_path
    _project_name = project_name

    if _server is not None:
        _graph_path = graph_path
        return _port

    # Download vis.js in background so the server starts immediately
    threading.Thread(target=_fetch_visjs, daemon=True).start()

    server = _ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    _port = server.server_address[1]
    _server = server

    _server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    _server_thread.start()
    return _port


def stop_graph_server() -> None:
    global _server, _server_thread
    if _server:
        _server.shutdown()
        _server = None
        _server_thread = None
