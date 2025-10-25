const entry=document.getElementById('entry');
const btn=document.getElementById('reflectBtn');
const statusEl=document.getElementById('status');
const archName=document.getElementById('archName');
const sentimentEl=document.getElementById('sentiment');
const mirrorText=document.getElementById('mirrorText');
const LAST_KEY='soul_mirror:last';
entry.value=localStorage.getItem(LAST_KEY)||'';
async function reflect(){
  const text=entry.value.trim();
  if(!text){statusEl.textContent="Type something first.";return;}
  statusEl.textContent="Reflecting…";
  try{
    const res=await fetch('/api/reflect',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    if(!res.ok){const e=await res.json();throw new Error(e.error||'Request failed');}
    const data=await res.json(); updateUI(data); localStorage.setItem(LAST_KEY,text);
    statusEl.textContent="✓"; setTimeout(()=>statusEl.textContent="",1200);
  }catch(e){statusEl.textContent="Error: "+e.message;}
}
function updateUI(data){
  archName.textContent=data.archetype.name;
  sentimentEl.textContent=data.sentiment.label;
  sentimentEl.classList.remove('positive','negative');
  if(data.sentiment.label==='positive') sentimentEl.classList.add('positive');
  if(data.sentiment.label==='negative') sentimentEl.classList.add('negative');
  mirrorText.textContent=data.mirror_text;
  document.documentElement.style.setProperty('--accent', data.archetype.color||'#4F46E5');
}
btn.addEventListener('click', reflect);
entry.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && (e.metaKey||e.ctrlKey)) reflect(); });
