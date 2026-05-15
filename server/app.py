import asyncio
import json
import os
import time
import uuid
from typing import Optional, Dict, Any, List

from fastapi import (
    FastAPI,
    Depends,
    Header,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

# Optional torch import for device auto-detect (works even if torch isn't installed for the API-only path)
try:
    import torch  # type: ignore
except Exception:
    torch = None

from server.core_bridge import Core
from src.metrics import (
    e_star_series,
    latent_drift_series,
    summarize_run,
)
from src.plugins import get_default_registry as _get_plugin_registry
from src.registry import (
    PromptPack,
    PromptPackRegistry,
    get_default_pack_registry,
)
from src.runstore import (
    RunRecord,
    RunStore,
    StepRecord,
    get_default_runstore,
)
from src.streaming import StreamHub, get_default_hub

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

# ------------------------- stores -------------------------
# Phase 3 — runs are persisted via the RunStore protocol. The default
# backend is in-memory (identical observable behavior to the old `RUNS`
# dict). Set `PAE_DATABASE_URL` (any SQLAlchemy URL: sqlite, postgres,
# ...) to enable durable storage.
RUN_STORE: RunStore = get_default_runstore()

# Per-process pub/sub hub used by the `/runs/{id}/stream` WebSocket.
STREAM_HUB: StreamHub = get_default_hub()

# Phase 6 — Community Prompt Registry. Loaded lazily on first access;
# overlay packs can be added via PAE_PROMPT_PACKS_DIR.
PROMPT_PACK_REGISTRY: PromptPackRegistry = get_default_pack_registry()

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
APP_VERSION = "0.6.0"
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
def prompt_packs(
    domain: Optional[str] = Query(None, description="Filter by pack domain"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    q: Optional[str] = Query(None, description="Free-text search"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List packs in the Community Prompt Registry (Phase 6).

    Backwards-compatible with the Phase 1 stub: the bundled seed packs
    keep the same IDs (``myth-1``, ``science-1``, ``psych-1``,
    ``purpose-1``) so existing clients keep working.
    """
    registry = PROMPT_PACK_REGISTRY
    matching = registry.list(domain=domain, tag=tag, query=q)
    page = matching[offset : offset + limit]
    return {
        "total": len(matching),
        "offset": offset,
        "limit": limit,
        "packs": [
            {
                "id": p.id,
                "version": p.version,
                "title": p.title,
                "domain": p.domain,
                "tags": p.tags,
                "description": p.description,
                "author": p.author,
                "prompt_count": len(p.prompts),
            }
            for p in page
        ],
    }


@app.get("/prompt-packs/{pack_id}", response_model=dict)
def prompt_pack_detail(pack_id: str):
    pack = PROMPT_PACK_REGISTRY.get(pack_id)
    if pack is None:
        raise HTTPException(status_code=404, detail="Prompt pack not found")
    return pack.model_dump()


@app.get("/prompt-packs/{pack_id}/prompts/{prompt_name}", response_model=dict)
def prompt_pack_template(pack_id: str, prompt_name: str):
    pack = PROMPT_PACK_REGISTRY.get(pack_id)
    if pack is None:
        raise HTTPException(status_code=404, detail="Prompt pack not found")
    tmpl = pack.get_prompt(prompt_name)
    if tmpl is None:
        raise HTTPException(status_code=404, detail="Prompt template not found")
    return tmpl.model_dump()


@app.get("/plugins", response_model=dict)
def list_plugins():
    """List third-party extensions registered with the plugin system.

    Returns only ``namespace`` / ``name`` / ``source`` / ``meta`` —
    never the factory itself — so this endpoint stays safe to expose
    publicly. Useful for the dashboard to show which providers are
    selectable via ``PAE_LLM_PROVIDER`` / ``PAE_EMBEDDINGS_PROVIDER``.
    """
    registry = _get_plugin_registry()
    return {
        "namespaces": registry.namespaces(),
        "plugins": [
            {
                "namespace": rec.namespace,
                "name": rec.name,
                "source": rec.source,
                "meta": rec.meta,
            }
            for rec in registry.list()
        ],
    }

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

    RUN_STORE.create(
        RunRecord(
            run_id=run_id,
            user_id=user_id,
            plan=auth["plan"],
            brief=payload.brief.model_dump(),
            prompt_pack_id=payload.prompt_pack_id,
            config=payload.config,
            t=0,
            state=[0.0] * state_len,
            trace=[],
        )
    )
    return {"run_id": run_id}

@app.post("/runs/{run_id}/step", response_model=StepResult)
async def step(run_id: str, auth=Depends(get_auth)):
    user_id = auth["user_id"]
    rate_limit(user_id, auth["rate_limit"])

    record = RUN_STORE.get(run_id)
    if record is None or record.user_id != user_id:
        raise HTTPException(status_code=404, detail="Run not found")
    if CORE is None:
        raise HTTPException(status_code=500, detail="Core not initialized")

    # CORE.step is CPU-bound + sync; run it in a worker thread so we
    # don't block the event loop while still being callable from an
    # async handler (which is what we need to await STREAM_HUB.publish
    # on the same loop the WebSocket subscribers live on).
    # Phase 5: pass the brief in so the optional Orchestrator path can
    # consume it. The torch fallback inside Core.step ignores it.
    out = await asyncio.to_thread(CORE.step, record.state, brief=record.brief)
    spec, tests, e_star, s = out["spec"], out["tests"], out["e_star"], out["state"]

    new_t = record.t + 1
    RUN_STORE.update_state(run_id, t=new_t, state=s)
    RUN_STORE.append_step(
        run_id,
        # Phase 4: persist the per-step latent state so the dashboard can
        # render true per-step drift, not just the final state.
        StepRecord(t=new_t, spec=spec, tests=tests, e_star=e_star, state=list(s)),
    )
    CORE.remember_step(
        run_id=run_id, step=new_t, state=s, e_star=e_star, spec=spec, tests=tests,
    )

    # Fan the step out to any live WebSocket subscribers on this loop.
    await STREAM_HUB.publish(
        run_id,
        {
            "run_id": run_id,
            "t": new_t,
            "spec": spec,
            "tests": tests,
            "e_star": e_star,
            "state": s,
        },
    )

    return StepResult(
        run_id=run_id,
        t=new_t,
        spec_json=spec,
        tests_json=tests,
        e_star=e_star,
        state_snapshot=s,
    )

@app.get("/runs/{run_id}/trace", response_model=dict)
def trace(run_id: str, auth=Depends(get_auth)):
    user_id = auth["user_id"]
    rate_limit(user_id, auth["rate_limit"])

    record = RUN_STORE.get(run_id)
    if record is None or record.user_id != user_id:
        raise HTTPException(status_code=404, detail="Run not found")

    data = {
        "plan": record.plan,
        "brief": record.brief,
        "prompt_pack_id": record.prompt_pack_id,
        "config": record.config,
        "t": record.t,
        "state": record.state,
        "trace": [
            {"t": st.t, "spec": st.spec, "tests": st.tests, "e_star": st.e_star}
            for st in record.trace
        ],
    }
    return {"run": data}


@app.websocket("/runs/{run_id}/stream")
async def stream(websocket: WebSocket, run_id: str, x_api_key: Optional[str] = Query(default=None)):
    """Push each new ECL step for ``run_id`` to the WebSocket client.

    Auth is via the ``x_api_key`` query parameter because browsers can't
    set custom headers on the initial WebSocket handshake. The check
    mirrors the HTTP ``get_auth`` dependency, and connection attempts
    are run through the same per-user rate limiter as HTTP requests so a
    malicious client can't brute-force keys via rapid WS reconnects.

    Custom WebSocket close codes used here (per RFC 6455, codes in the
    4xxx range are application-defined):

    * ``4401`` — missing / invalid API key (mirrors HTTP 401).
    * ``4404`` — run not found or owned by another user (mirrors 404).
    * ``4429`` — caller exceeded their rate limit (mirrors 429).
    """
    if not x_api_key or x_api_key not in API_KEYS:
        await websocket.close(code=4401)  # 4xxx = application close
        return
    auth = API_KEYS[x_api_key]

    # Apply the same per-user rate limit used for HTTP routes. Going over
    # the limit closes the connection with code 4429 ("too many requests").
    try:
        rate_limit(auth["user_id"], auth["rate_limit"])
    except HTTPException:
        await websocket.close(code=4429)
        return

    record = RUN_STORE.get(run_id)
    if record is None or record.user_id != auth["user_id"]:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    try:
        async with STREAM_HUB.subscribe(run_id) as queue:
            # Send an initial snapshot so reconnecting clients catch up
            # without waiting for the next step.
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "snapshot",
                        "run_id": run_id,
                        "t": record.t,
                        "trace_len": len(record.trace),
                    }
                )
            )
            while True:
                event = await queue.get()
                await websocket.send_text(
                    json.dumps({"type": "step", **event})
                )
    except WebSocketDisconnect:
        return


# ------------------------- Phase 4: dashboard metrics -------------------------

@app.get("/runs", response_model=dict)
def list_runs(auth=Depends(get_auth)):
    """List all runs owned by the caller, with at-a-glance summaries.

    The dashboard's landing page uses this to render a run picker; the
    summary fields (latest_e_star, steps) are pre-computed server-side
    via :func:`src.metrics.summarize_run` so the browser doesn't have
    to fetch the full trace just to render a row.
    """
    user_id = auth["user_id"]
    rate_limit(user_id, auth["rate_limit"])

    records = RUN_STORE.list_for_user(user_id)
    runs = []
    for rec in records:
        summary = summarize_run(rec)
        runs.append(
            {
                "run_id": rec.run_id,
                "brief_goal": rec.brief.get("goal"),
                "prompt_pack_id": rec.prompt_pack_id,
                "t": rec.t,
                "steps": summary.steps,
                "latest_e_star": summary.latest_e_star,
                "mean_e_star": summary.mean_e_star,
            }
        )
    return {"runs": runs}


@app.get("/runs/{run_id}/metrics", response_model=dict)
def run_metrics(run_id: str, auth=Depends(get_auth)):
    """Derived metric series for the run, ready to plot.

    Returns:
        * ``e_star``      -- list of ``{t, e_star}`` points
        * ``drift``       -- list of ``{t, state_norm, state_delta}`` points
        * ``summary``     -- :class:`src.metrics.RunSummary` fields
    """
    user_id = auth["user_id"]
    rate_limit(user_id, auth["rate_limit"])

    record = RUN_STORE.get(run_id)
    if record is None or record.user_id != user_id:
        raise HTTPException(status_code=404, detail="Run not found")

    e_star = [{"t": p.t, "e_star": p.e_star} for p in e_star_series(record)]
    drift = [
        {"t": p.t, "state_norm": p.state_norm, "state_delta": p.state_delta}
        for p in latent_drift_series(record)
    ]
    summary = summarize_run(record)

    return {
        "run_id": record.run_id,
        "e_star": e_star,
        "drift": drift,
        "summary": {
            "steps": summary.steps,
            "latest_e_star": summary.latest_e_star,
            "mean_e_star": summary.mean_e_star,
            "final_state_norm": summary.final_state_norm,
            "mean_state_delta": summary.mean_state_delta,
        },
    }


# ------------------------- Phase 4: static dashboard -------------------------
# The dashboard ships as plain HTML + vanilla JS (Chart.js via CDN) so it
# adds *zero* build steps to the project. It's mounted optionally: when
# the ``web/`` directory is missing the import simply skips the mount,
# keeping the API-only install path working unchanged.

_WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.isdir(_WEB_DIR):
    from fastapi.staticfiles import StaticFiles

    app.mount("/dashboard", StaticFiles(directory=_WEB_DIR, html=True), name="dashboard")
