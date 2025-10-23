import os
import time
import uuid
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

# Optional torch import for device auto-detect (works even if torch isn't installed for the API-only path)
try:
    import torch  # type: ignore
except Exception:
    torch = None

from server.core_bridge import Core

# ------------------------- device selection -------------------------
def auto_device() -> str:
    pref = os.getenv("PAE_DEVICE", "auto").strip().lower()
    if pref in {"cpu", "cuda", "mps"}:
        return pref
    if torch is not None:
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"

# ------------------------- in-memory stores (swap to DB later) -------------------------
RUNS: Dict[str, Dict[str, Any]] = {}

API_KEYS = {
    "demo-free-key": {"user_id": "u_demo", "plan": "free", "rate_limit": 120},    # req/hour
    "demo-pro-key":  {"user_id": "u_demo_pro", "plan": "pro", "rate_limit": 10000},
}

LAST_HITS: Dict[str, List[float]] = {}  # naive token bucket

def rate_limit(user_id: str, limit: int):
    now = time.time()
    window = 3600.0
    hits = [t for t in LAST_HITS.get(user_id, []) if now - t < window]
    if len(hits) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    hits.append(now)
    LAST_HITS[user_id] = hits

def get_auth(x_api_key: Optional[str] = Header(default=None)):
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Missing/invalid API key")
    return API_KEYS[x_api_key]

# ------------------------- models -------------------------
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

# ------------------------- app -------------------------
APP_VERSION = "0.2.0"
app = FastAPI(title="Prompt Atlas Engine API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CORE: Optional[Core] = None

@app.on_event("startup")
def _startup():
    global CORE
    device = auto_device()
    state_dim = int(os.getenv("PAE_STATE_DIM", "64"))
    CORE = Core(device=device, state_dim=state_dim)
    print(f"[PromptAtlas] Core initialized | device={device} | state_dim={state_dim}")

# ------------------------- routes -------------------------
@app.get("/health", response_model=dict)
def health():
    return {"ok": True, "version": APP_VERSION}

@app.get("/pricing", response_model=dict)
def pricing():
    return {
        "plans": [
            {"id": "free", "name": "Free", "limits": {"runs_per_day": 10, "multimodal": False}},
            {"id": "pro", "name": "Pro", "limits": {"runs_per_day": 1000, "multimodal": True}},
            {"id": "lab", "name": "Lab", "limits": {"private_node": True, "multimodal": True}},
        ]
    }

@app.get("/prompt-packs", response_model=dict)
def prompt_packs():
    packs = [
        {"id": "myth-1", "title": "Myth & Meaning", "domain": "myth", "tags": ["archetype", "narrative"]},
        {"id": "science-1", "title": "Science & Precision", "domain": "science", "tags": ["model", "measure"]},
        {"id": "psych-1", "title": "Psychology & Self", "domain": "psych", "tags": ["bias", "empathy"]},
        {"id": "purpose-1", "title": "Profit & Purpose", "domain": "purpose", "tags": ["ethics", "impact"]},
    ]
    return {"packs": packs}

@app.post("/runs", response_model=dict)
def create_run(payload: RunCreate, auth=Depends(get_auth)):
    user_id = auth["user_id"]
    rate_limit(user_id, auth["rate_limit"])

    if CORE is None:
        raise HTTPException(status_code=500, detail="Core not initialized")

    run_id = str(uuid.uuid4())
    try:
        state_len = CORE.bus.gru.hidden_size  # type: ignore[attr-defined]
    except Exception:
        state_len = int(os.getenv("PAE_STATE_DIM", "64"))

    RUNS[run_id] = {
        "user_id": user_id,
        "plan": auth["plan"],
        "brief": payload.brief.model_dump(),
        "prompt_pack_id": payload.prompt_pack_id,
        "config": payload.config,
        "t": 0,
        "state": [0.0] * state_len,
        "trace": [],
    }
    return {"run_id": run_id}

@app.post("/runs/{run_id}/step", response_model=StepResult)
def step(run_id: str, auth=Depends(get_auth)):
    user_id = auth["user_id"]
    rate_limit(user_id, auth["rate_limit"])

    if run_id not in RUNS or RUNS[run_id]["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Run not found")
    if CORE is None:
        raise HTTPException(status_code=500, detail="Core not initialized")

    r = RUNS[run_id]
    out = CORE.step(S_list=r["state"])  # advances state, returns spec/tests/e_star
    spec, tests, e_star, s = out["spec"], out["tests"], out["e_star"], out["state"]

    r["t"] += 1
    r["state"] = s
    r["trace"].append({"t": r["t"], "spec": spec, "tests": tests, "e_star": e_star})

    return StepResult(
        run_id=run_id,
        t=r["t"],
        spec_json=spec,
        tests_json=tests,
        e_star=e_star,
        state_snapshot=s,
    )

@app.get("/runs/{run_id}/trace", response_model=dict)
def trace(run_id: str, auth=Depends(get_auth)):
    user_id = auth["user_id"]
    rate_limit(user_id, auth["rate_limit"])

    if run_id not in RUNS or RUNS[run_id]["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Run not found")

    data = {k: v for k, v in RUNS[run_id].items() if k != "user_id"}
    return {"run": data}
