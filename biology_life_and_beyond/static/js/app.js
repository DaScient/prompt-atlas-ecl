const tabs = document.querySelectorAll('nav button');
const sections = document.querySelectorAll('.tab');
tabs.forEach(b=>b.addEventListener('click',()=>{
  sections.forEach(s=>s.classList.remove('active'));
  document.getElementById('tab-'+b.dataset.tab).classList.add('active');
}));
document.querySelector('nav button[data-tab="essay"]').click();

// markdown loader (simple)
async function loadMD(id, url){
  const el = document.getElementById(id);
  const res = await fetch(url);
  const data = await res.json();
  el.textContent = data.text;
}
loadMD('essay','/api/essay');
loadMD('prompts','/api/prompts');

// simple plotter (canvas)
function plotLines(canvas, series, opts={}){
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = '#0b0f1e'; ctx.fillRect(0,0,W,H);
  // axes
  ctx.strokeStyle='#263048'; ctx.lineWidth=1; ctx.strokeRect(50,15,W-70,H-40);
  // compute bounds
  const xs = series[0].x;
  let ymin = Infinity, ymax = -Infinity;
  series.forEach(s => s.y.forEach(v => { ymin = Math.min(ymin,v); ymax = Math.max(ymax,v);}));
  if (ymin===ymax){ ymin-=1; ymax+=1;}
  const x0=50, y0=H-25, w=W-70, h=H-40;
  function xmap(i){ return x0 + (i/(xs.length-1))*w; }
  function ymap(v){ return y0 - ((v - ymin)/(ymax - ymin))*h; }
  // grid
  ctx.strokeStyle='#1b2338'; ctx.lineWidth=1;
  for(let k=0;k<6;k++){ let y = y0 - (k/5)*h; ctx.beginPath(); ctx.moveTo(x0,y); ctx.lineTo(x0+w,y); ctx.stroke(); }
  // lines
  const colors = opts.colors || ['#7dd3fc', '#fca5a5', '#86efac', '#fcd34d'];
  series.forEach((s,si)=>{
    ctx.strokeStyle = colors[si%colors.length];
    ctx.beginPath();
    for(let i=0;i<xs.length;i++){
      const x = xmap(i), y = ymap(s.y[i]);
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.stroke();
  });
  // labels
  ctx.fillStyle='#94a3b8'; ctx.font='12px system-ui';
  ctx.fillText(opts.title || '', x0, 12);
  // legend
  ctx.font='12px system-ui';
  series.forEach((s,si)=>{
    const lx = x0 + 10 + si*120, ly = 28;
    ctx.fillStyle = colors[si%colors.length]; ctx.fillRect(lx, ly-10, 20, 6);
    ctx.fillStyle = '#9fb0c8'; ctx.fillText(s.name || ('Series '+(si+1)), lx+28, ly);
  });
}

// Bind actions
document.getElementById('runLV').addEventListener('click', async ()=>{
  const body = {
    t_end: 50, dt: 0.05, x0: 10, y0: 5,
    params: {
      alpha: parseFloat(document.getElementById('alpha').value),
      beta: parseFloat(document.getElementById('beta').value),
      gamma: parseFloat(document.getElementById('gamma').value),
      delta: parseFloat(document.getElementById('delta').value),
      shock_t: parseFloat(document.getElementById('shock_t').value),
      shock_scale: parseFloat(document.getElementById('shock_scale').value)
    }
  };
  const res = await fetch('/api/lv',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d = await res.json();
  const series = [
    {name:'Prey', x:d.t, y:d.prey},
    {name:'Predator', x:d.t, y:d.predator}
  ];
  plotLines(document.getElementById('lvPlot'), series, {title:'Predator–Prey Dynamics'});
});

document.getElementById('runEvo').addEventListener('click', async ()=>{
  const body = {
    pop_size: parseInt(document.getElementById('pop').value,10),
    length: parseInt(document.getElementById('length').value,10),
    generations: parseInt(document.getElementById('gens').value,10),
    target: document.getElementById('target').value
  };
  const res = await fetch('/api/evo',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d = await res.json();
  const xs = Array.from({length: d.best.length}, (_,i)=>i);
  const series = [
    {name:'Best score', x: xs, y: d.best},
    {name:'Average score', x: xs, y: d.avg}
  ];
  plotLines(document.getElementById('evoPlot'), series, {title:'Evolutionary Fitness Over Generations'});
});

document.getElementById('runCoral').addEventListener('click', async ()=>{
  const body = {
    heat_t: parseFloat(document.getElementById('heat_t').value),
    heat_amp: parseFloat(document.getElementById('heat_amp').value),
    symb: parseFloat(document.getElementById('symb').value)
  };
  const res = await fetch('/api/coral',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d = await res.json();
  const series = [
    {name:'Coral', x:d.t, y:d.coral},
    {name:'Algae', x:d.t, y:d.algae}
  ];
  plotLines(document.getElementById('coralPlot'), series, {title:'Coral–Algae Symbiosis Under Heat Stress'});
});

document.getElementById('runPand').addEventListener('click', async ()=>{
  const body = {
    t_end: 120,
    R0: parseFloat(document.getElementById('R0').value),
    mut_prob: parseFloat(document.getElementById('mut_prob').value),
    variant_boost: parseFloat(document.getElementById('vboost').value)
  };
  const res = await fetch('/api/pandemic',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d = await res.json();
  const series = [
    {name:'Base lineage', x:d.t, y:d.base},
    {name:'Variant', x:d.t, y:d.variant}
  ];
  plotLines(document.getElementById('pandPlot'), series, {title:'Branching Risk (Toy)'});
});
