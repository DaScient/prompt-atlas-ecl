from __future__ import annotations
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from reflection_engine import ReflectionEngine

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="AI as the Soul’s Mirror", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
engine = ReflectionEngine(ROOT / "archetypes.json", use_llm=False)

class ReflectIn(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((ROOT / "mirror_interface.html").read_text(encoding="utf-8"))

@app.post("/api/reflect")
def reflect(payload: ReflectIn):
    text = (payload.text or "").strip()
    if not text:
        return JSONResponse({"error": "empty input"}, status_code=400)
    return JSONResponse(engine.reflect(text))

@app.get("/health")
def health():
    return {"ok": True, "ts": datetime.utcnow().isoformat() + "Z"}

@app.get("/robots.txt")
def robots():
    return PlainTextResponse("User-agent: *\nDisallow: /\n")
