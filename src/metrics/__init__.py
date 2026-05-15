"""Phase 4 — derived series and summaries for dashboard visualisations.

The dashboard needs *two* shapes of data on top of the raw trace:

* **E-Star series** — what fraction of the spec's required fields are
  satisfied at each step. Already stored verbatim on every
  :class:`StepRecord`; this is essentially a projection.

* **Latent drift series** — how much the latent state changes between
  consecutive steps. We don't currently persist the full state at every
  step (we only persist *the latest* state on :class:`RunRecord`), so
  drift is computed best-effort from the information we have. When a
  future schema starts storing per-step state vectors, the same
  functions transparently produce per-step drift.

Everything here is pure-Python with no numpy / torch dependency so the
dashboard can run even on an API-only install.
"""

from src.metrics.derive import (
    LatentDriftPoint,
    EStarPoint,
    RunSummary,
    e_star_series,
    latent_drift_series,
    summarize_run,
)

__all__ = [
    "EStarPoint",
    "LatentDriftPoint",
    "RunSummary",
    "e_star_series",
    "latent_drift_series",
    "summarize_run",
]
