# top-level
from server.core_bridge import Core
CORE = Core(device="mps")  # or auto-detect like your train file


# server/app.py
import os, time, uuid
from typing import Optional, Dict, Any
from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

# --- in-memory stores (swap to Postgres/Redis later) ---
RUNS: Dict[str, Dict[str, Any]] = {}

# --- auth / plan gating ---
API_KEYS = {  # stub; replace with DB
    "demo-free-key": {"user_id": "u_demo", "plan": "free", "rate_limit": 60},   # req/hour
    "demo-pro-key":  {"user_id": "u_demo_pro", "plan": "pro", "rate_limit": 10000},
}

def get_auth(x_api_key: Optional[str] = Header(default=None)):
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Missing/invalid API key")
    return API_KEYS[x_api_key]

# --- rate limiting (very simple token bucket) ---
LAST: Dict[str, list] = {}  # user_id -> [timestamps]

def rate_limit(user_id: str, limit: int):
    now = time.time()
    window = 3600
    arr = [t for t in LAST.get(user_id, []) if now - t < window]
    if len(arr) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    arr.append(now)
    LAST[user_id] = arr

# --- models ---
class Brief(BaseModel):
    goal: str
    constraints: list[str] = []
    data: dict = {}
    risks: list[str] = []

class RunCreate(BaseModel):
    brief: Brief
    prompt_pack_id: Optional[str] = None
    config: dict = {}

class StepResult(BaseModel):
    run_id: str
    t: int
    spec_json: dict
    tests_json: list[dict]
    e_star: float = Field(..., description="Entanglement certificate proxy")
    state_snapshot: list[float] = []

app = FastAPI(title="Prompt Atlas Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "version": "0.1.0"}

@app.post("/runs", response_model=dict)
def create_run(payload: RunCreate, auth=Depends(get_auth)):
    user_id = auth["user_id"]; plan = auth["plan"]
    rate_limit(user_id, auth["rate_limit"])

    run_id = str(uuid.uuid4())
    RUNS[run_id] = {
        "user_id": user_id,
        "plan": plan,
        "brief": payload.brief.model_dump(),
        "prompt_pack_id": payload.prompt_pack_id,
        "config": payload.config,
        "t": 0,
        "state": [0.0]*64,  # seed S_t; swap to Redis later
        "trace": [],
    }
    return {"run_id": run_id}

@app.post("/runs/{run_id}/step", response_model=StepResult)
def step(run_id: str, auth=Depends(get_auth)):
    user_id = auth["user_id"]; plan = auth["plan"]
    rate_limit(user_id, auth["rate_limit"])

    if run_id not in RUNS or RUNS[run_id]["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Run not found")

    r = RUNS[run_id]
    # --- call your core (stub: returns fake spec/tests + E*) ---
    t = r["t"] + 1
    spec = {"assumptions": ["stub"], "data": {}, "steps": ["stub"], "interfaces": [], "acceptance": ["stub"], "risks": ["stub"]}
    tests = [{"name": "format", "checks": ["has acceptance", "has risks"]}]
    e_star = 0.5  # replace with real metric

    # update state (stub)
    s = r["state"]
    s = s[1:] + [float((t % 100)/100.0)]
    r["state"] = s
    r["t"] = t
    r["trace"].append({"t": t, "spec": spec, "tests": tests, "e_star": e_star})

    return StepResult(run_id=run_id, t=t, spec_json=spec, tests_json=tests, e_star=e_star, state_snapshot=s)

@app.get("/runs/{run_id}/trace")
def trace(run_id: str, auth=Depends(get_auth)):
    user_id = auth["user_id"]
    rate_limit(user_id, auth["rate_limit"])
    if run_id not in RUNS or RUNS[run_id]["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run": {k:v for k,v in RUNS[run_id].items() if k != "user_id"}}

