use axum::response::Html;

pub fn dashboard_html() -> Html<&'static str> {
    Html(r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Embroidery Agent Dashboard</title>
<style>
body{font-family:system-ui;max-width:1200px;margin:0 auto;padding:20px;background:#f8f9fa}
h1{color:#2c3e50;border-bottom:2px solid #3498db;padding-bottom:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin-top:20px}
.card{background:white;border-radius:8px;padding:20px;box-shadow:0 2px 4px rgba(0,0,0,0.1)}
.card h3{margin-top:0;color:#3498db}
.stat{font-size:2em;font-weight:bold;color:#27ae60}
.btn{display:inline-block;padding:8px 16px;background:#3498db;color:white;border:none;border-radius:4px;cursor:pointer;text-decoration:none;margin:4px}
.btn:hover{background:#2980b9}
table{width:100%;border-collapse:collapse;margin-top:10px}
th,td{padding:8px;text-align:left;border-bottom:1px solid #eee}
th{background:#f0f0f0}
</style></head><body>
<h1>🧵 Embroidery Agent Dashboard</h1>
<div class="grid">
  <div class="card"><h3>Audit Chain</h3><div class="stat" id="chain-length">-</div><p>verified entries</p>
    <button class="btn" onclick="verifyChain()">Verify Chain</button>
    <div id="verify-result"></div></div>
  <div class="card"><h3>Federated Learning</h3><div class="stat" id="round-num">-</div><p>current round</p>
    <button class="btn" onclick="startRound()">Start Round</button>
    <button class="btn" onclick="aggregate()">Aggregate</button>
    <div id="fed-result"></div></div>
  <div class="card"><h3>Pattern Library</h3><div class="stat" id="pattern-count">-</div><p>stored patterns</p>
    <button class="btn" onclick="searchPatterns()">Search</button>
    <div id="search-result"></div></div>
</div>
<div class="card" style="margin-top:20px"><h3>Recent Audit Log</h3><table>
  <thead><tr><th>#</th><th>Time</th><th>Operation</th><th>Client</th><th>Hash</th></tr></thead>
  <tbody id="audit-log"></tbody></table></div>
<script>
const API = '';
async function load(){try{
  let h=await fetch(API+'/health');let d=await h.json();
  document.getElementById('chain-length').textContent=d.chain_length;
  document.getElementById('round-num').textContent=d.version;
  let c=await fetch(API+'/api/v1/audit/chain');let cd=await c.json();
  let tbody=document.getElementById('audit-log');
  tbody.innerHTML=(cd.entries||[]).map(e=>`<tr><td>${e.index}</td><td>${e.timestamp}</td><td>${e.operation}</td><td>${e.client_id}</td><td>${e.hash.slice(0,12)}...</td></tr>`).join('');
}catch(e){console.error(e)}}
async function verifyChain(){let r=await fetch(API+'/api/v1/audit/verify');let d=await r.json();
  document.getElementById('verify-result').innerHTML=d.valid?'✅ Valid':'❌ Invalid';}
async function startRound(){let r=await fetch(API+'/api/v1/fed/round/start',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  let d=await r.json();document.getElementById('fed-result').innerHTML='Round '+d.round_id;}
async function aggregate(){let r=await fetch(API+'/api/v1/fed/round/aggregate',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  let d=await r.json();document.getElementById('fed-result').innerHTML='Loss: '+d.global_loss+', Clients: '+d.clients;}
async function searchPatterns(){let r=await fetch(API+'/api/v1/fingerprint/search',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({query_vector:Array(768).fill(0).map(()=>Math.random()),top_k:5})});
  let d=await r.json();document.getElementById('search-result').innerHTML=d.length+' patterns found';}
load();
</script></body></html>"#)
}
