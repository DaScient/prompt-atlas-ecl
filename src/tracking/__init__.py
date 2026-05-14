"""Phase 2 — experiment tracking with optional MLflow backend.

Same playbook as Phase 1's NATS / Qdrant integrations:

* If ``mlflow`` is installed we use it (file store by default, remote
  tracking URI when ``MLFLOW_TRACKING_URI`` is set).
* Otherwise we use a tiny in-process no-op tracker so training and the
  API keep working in minimal environments without raising.

Callers should always go through :class:`MLflowTracker` and not import
``mlflow`` directly; that keeps the optional-dep contract centralized.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional


class MLflowTracker:
    """Thin façade over MLflow with a graceful no-op fallback.

    Parameters
    ----------
    experiment:
        Experiment name to log under. Defaults to ``"prompt-atlas-ecl"``.
    run_name:
        Optional name for the current run.
    tracking_uri:
        Optional override for the tracking URI. If omitted, MLflow's own
        ``MLFLOW_TRACKING_URI`` env var is honored.
    enabled:
        Hard switch. When ``False`` (or when ``mlflow`` cannot be
        imported) every method becomes a no-op. Defaults to honoring
        ``PAE_TRACKING=1`` from the environment.
    """

    def __init__(
        self,
        experiment: str = "prompt-atlas-ecl",
        *,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.experiment = experiment
        self.run_name = run_name
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if enabled is None:
            enabled = os.getenv("PAE_TRACKING", "0") == "1"
        self._mlflow: Any = None
        self._active_run: Any = None
        if enabled:
            try:
                import mlflow  # type: ignore

                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_experiment(self.experiment)
                self._mlflow = mlflow
            except ImportError:
                # mlflow not installed → silently degrade to no-op.
                self._mlflow = None
            except Exception:
                # Any other MLflow init failure also degrades safely so
                # training / API never crash because of telemetry.
                self._mlflow = None

    # ------------------------------------------------------------------ status

    @property
    def active(self) -> bool:
        """``True`` iff a real MLflow backend is wired up."""
        return self._mlflow is not None

    # ------------------------------------------------------------------ runs

    def start_run(self, run_name: Optional[str] = None) -> "MLflowTracker":
        if self._mlflow is None:
            return self
        self._active_run = self._mlflow.start_run(
            run_name=run_name or self.run_name
        )
        return self

    def end_run(self) -> None:
        if self._mlflow is None or self._active_run is None:
            return
        try:
            self._mlflow.end_run()
        finally:
            self._active_run = None

    def __enter__(self) -> "MLflowTracker":
        return self.start_run()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end_run()

    # ------------------------------------------------------------------ logging

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self._mlflow is None:
            return
        try:
            self._mlflow.log_params(dict(params))
        except Exception:
            pass

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        *,
        step: Optional[int] = None,
    ) -> None:
        if self._mlflow is None:
            return
        try:
            # MLflow only accepts numeric values; coerce / drop non-numerics.
            clean: Dict[str, float] = {}
            for k, v in metrics.items():
                try:
                    clean[k] = float(v)
                except (TypeError, ValueError):
                    continue
            if clean:
                self._mlflow.log_metrics(clean, step=step)
        except Exception:
            pass


__all__ = ["MLflowTracker"]
