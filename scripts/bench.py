"""Phase 6 — reproducible benchmark harness.

Drives ``Core`` through N steps on a fixed brief and reports a single
JSON document with the E★ trajectory, mean coherence, and latent drift.
Designed to be run from CI or locally to track regressions across
commits.

Usage::

    python -m scripts.bench --steps 16 --brief "build a UI for inspecting runs"
    python -m scripts.bench --pack myth-1 --prompt user_story --steps 8 --json out.json

When ``--pack`` and ``--prompt`` are given, the prompt template is
rendered (with any ``--var KEY=VALUE`` pairs and the pack's defaults)
and used as the brief description. Otherwise the explicit ``--brief``
string is used.

The output schema lives next to this script in
``docs/research/bench_result.schema.json`` so downstream notebooks /
papers have a stable contract.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure the repo root is importable when run as ``python scripts/bench.py``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from server.core_bridge import Core  # noqa: E402
from src.registry import get_default_pack_registry  # noqa: E402


def _parse_vars(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--var must be KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v
    return out


def _resolve_brief(args: argparse.Namespace) -> Dict[str, Any]:
    if args.pack and args.prompt:
        pack = get_default_pack_registry().get(args.pack)
        if pack is None:
            raise SystemExit(f"unknown prompt pack: {args.pack!r}")
        try:
            text = pack.render(args.prompt, **_parse_vars(args.var))
        except KeyError as exc:
            raise SystemExit(f"prompt render failed: {exc}")
        return {
            "goal": f"pack:{pack.id}/{args.prompt}",
            "description": text,
        }
    return {"goal": "benchmark", "description": args.brief or "benchmark run"}


def _drift_series(states: List[List[float]]) -> List[float]:
    """L2 distance between consecutive states."""
    drift: List[float] = []
    for i in range(1, len(states)):
        a, b = states[i - 1], states[i]
        n = min(len(a), len(b))
        if n == 0:
            drift.append(0.0)
            continue
        drift.append(math.sqrt(sum((a[j] - b[j]) ** 2 for j in range(n))))
    return drift


def run_bench(
    *,
    steps: int,
    brief: Dict[str, Any],
    state_dim: int = 64,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Run ``steps`` rounds through ``Core`` and gather metrics.

    ``use_llm=True`` enables the Phase 5 orchestrator path (real or
    deterministic, depending on env). Off by default so the bench is
    reproducible bit-for-bit on identical inputs.
    """
    if use_llm:
        os.environ.setdefault("PAE_LLM", "1")
    core = Core(state_dim=state_dim)

    e_stars: List[float] = []
    states: List[List[float]] = []
    start = time.time()

    state: Optional[List[float]] = None
    for _ in range(steps):
        out = core.step(state, brief=brief if use_llm else None)
        state = out["state"]
        e_stars.append(float(out["e_star"]))
        states.append(list(state))

    elapsed = time.time() - start
    drift = _drift_series(states)

    return {
        "version": 1,
        "config": {
            "steps": steps,
            "state_dim": state_dim,
            "use_llm": use_llm,
            "brief": brief,
        },
        "metrics": {
            "e_star_series": e_stars,
            "drift_series": drift,
            "e_star_mean": statistics.fmean(e_stars) if e_stars else 0.0,
            "e_star_final": e_stars[-1] if e_stars else 0.0,
            "drift_mean": statistics.fmean(drift) if drift else 0.0,
            "wallclock_seconds": elapsed,
            "steps_per_second": (steps / elapsed) if elapsed > 0 else 0.0,
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Prompt Atlas benchmark harness")
    p.add_argument("--steps", type=int, default=16, help="number of steps to run")
    p.add_argument("--state-dim", type=int, default=64, help="latent state dim")
    p.add_argument("--brief", default="", help="freeform brief description")
    p.add_argument("--pack", default=None, help="prompt pack id (with --prompt)")
    p.add_argument("--prompt", default=None, help="prompt template name within pack")
    p.add_argument(
        "--var",
        action="append",
        default=[],
        help="KEY=VALUE substitution for the prompt template (repeatable)",
    )
    p.add_argument("--llm", action="store_true", help="enable Phase 5 orchestrator path")
    p.add_argument("--json", default=None, help="write result to this path")
    args = p.parse_args(argv)

    brief = _resolve_brief(args)
    result = run_bench(
        steps=args.steps,
        brief=brief,
        state_dim=args.state_dim,
        use_llm=args.llm,
    )
    text = json.dumps(result, indent=2)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            fh.write(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
