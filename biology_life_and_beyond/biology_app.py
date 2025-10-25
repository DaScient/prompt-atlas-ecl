from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

import json

from simulation_engine import simulate_lv, evolve_population, simulate_coral, simulate_pandemic

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="Biology, Life, and Beyond", version="0.1.0")

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((ROOT / "static" / "index.html").read_text(encoding="utf-8"))

# ---------- Schemas ----------
class LVin(BaseModel):
    t_end: float = 50.0
    dt: float = 0.05
    x0: float = 10.0
    y0: float = 5.0
    params: dict | None = None

@app.post("/api/lv")
def api_lv(payload: LVin):
    data = simulate_lv(payload.t_end, payload.dt, payload.x0, payload.y0, payload.params or {})
    return JSONResponse(data)

class EvoIn(BaseModel):
    pop_size: int = 64
    length: int = 12
    generations: int = 60
    target: str = "GATTACA"
    seed: int = 42

@app.post("/api/evo")
def api_evo(payload: EvoIn):
    data = evolve_population(payload.pop_size, payload.length, payload.generations, payload.target, payload.seed)
    return JSONResponse(data)

class CoralIn(BaseModel):
    t_end: float = 80.0
    dt: float = 0.1
    C0: float = 0.6
    A0: float = 0.6
    growth: float = 0.8
    decay: float = 0.6
    symb: float = 0.5
    heat_t: float = 30.0
    heat_amp: float = 0.7

@app.post("/api/coral")
def api_coral(payload: CoralIn):
    data = simulate_coral(payload.t_end, payload.dt, payload.C0, payload.A0,
                          payload.growth, payload.decay, payload.symb, payload.heat_t, payload.heat_amp)
    return JSONResponse(data)

class PandIn(BaseModel):
    t_end: int = 60
    R0: float = 1.2
    mut_prob: float = 0.02
    variant_boost: float = 0.3
    seed: int = 7

@app.post("/api/pandemic")
def api_pandemic(payload: PandIn):
    data = simulate_pandemic(payload.t_end, payload.R0, payload.mut_prob, payload.variant_boost, payload.seed)
    return JSONResponse(data)

@app.get("/api/essay")
def api_essay():
    text = (ROOT / "content" / "essay.md").read_text(encoding="utf-8")
    return {"text": text}

@app.get("/api/prompts")
def api_prompts():
    text = (ROOT / "content" / "prompts.md").read_text(encoding="utf-8")
    return {"text": text}

@app.get("/health")
def health():
    return {"ok": True}
