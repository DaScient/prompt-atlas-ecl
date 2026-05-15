"""SQLAlchemy-backed :class:`RunStore`.

Activates automatically when ``PAE_DATABASE_URL`` is set (any SQLAlchemy
URL works: ``sqlite:///./runs.db``, ``postgresql+psycopg://…``, etc.).

If the optional ``sqlalchemy`` dep isn't installed, importing this module
still succeeds — :class:`SQLRunStore` simply refuses to construct, and the
factory falls back to :class:`InMemoryRunStore`. This keeps Phase 3 in
line with the optional-dep philosophy from Phases 1–2.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, List, Optional

from src.runstore.base import RunRecord, StepRecord

logger = logging.getLogger(__name__)


def _sqlalchemy_available() -> bool:
    try:
        import sqlalchemy  # noqa: F401
        return True
    except ImportError:
        return False


class SQLRunStore:
    """SQLAlchemy-backed run store.

    The schema is intentionally tiny:

    * ``runs``  — one row per run with brief/config/state serialized as JSON.
    * ``steps`` — append-only trace rows keyed by ``(run_id, t)``.

    State and trace JSON columns are simple to keep migrations almost
    free; richer normalization is a deliberate non-goal for Phase 3.
    """

    def __init__(self, database_url: str, *, create_tables: bool = True) -> None:
        if not _sqlalchemy_available():
            raise ImportError("sqlalchemy is required for SQLRunStore")

        # Local import so the module remains importable without SQLAlchemy.
        from sqlalchemy import (
            JSON,
            Column,
            Float,
            Integer,
            MetaData,
            String,
            Table,
            create_engine,
        )

        self._database_url = database_url
        # ``future=True`` enables 2.0-style API and is a no-op on 2.x.
        self._engine = create_engine(database_url, future=True)
        self._metadata = MetaData()
        self._lock = threading.Lock()

        self._runs = Table(
            "pae_runs",
            self._metadata,
            Column("run_id", String, primary_key=True),
            Column("user_id", String, nullable=False, index=True),
            Column("plan", String, nullable=False),
            Column("brief_json", JSON, nullable=False),
            Column("prompt_pack_id", String, nullable=True),
            Column("config_json", JSON, nullable=False),
            Column("t", Integer, nullable=False, default=0),
            Column("state_json", JSON, nullable=False),
        )
        self._steps = Table(
            "pae_run_steps",
            self._metadata,
            Column("run_id", String, primary_key=True, index=True),
            Column("t", Integer, primary_key=True),
            Column("spec_json", JSON, nullable=False),
            Column("tests_json", JSON, nullable=False),
            Column("e_star", Float, nullable=False),
            # Phase 4: optional per-step latent state. Nullable so legacy
            # rows from Phase 3 deserialize cleanly. SQLAlchemy's
            # ``create_all`` adds the column for fresh DBs; on a pre-Phase-4
            # database the column simply won't exist and the SELECT below
            # tolerates ``KeyError`` via ``.get(...)``.
            Column("state_json", JSON, nullable=True),
        )

        if create_tables:
            self._metadata.create_all(self._engine)

    # ----------------------------------------------------------------- CRUD

    def create(self, record: RunRecord) -> None:
        with self._lock, self._engine.begin() as conn:
            conn.execute(
                self._runs.insert().values(
                    run_id=record.run_id,
                    user_id=record.user_id,
                    plan=record.plan,
                    brief_json=record.brief,
                    prompt_pack_id=record.prompt_pack_id,
                    config_json=record.config,
                    t=record.t,
                    state_json=list(record.state),
                )
            )
            for step in record.trace:
                conn.execute(
                    self._steps.insert().values(
                        run_id=record.run_id,
                        t=step.t,
                        spec_json=step.spec,
                        tests_json=step.tests,
                        e_star=step.e_star,
                        state_json=list(step.state) if step.state is not None else None,
                    )
                )

    def get(self, run_id: str) -> Optional[RunRecord]:
        from sqlalchemy import select

        with self._engine.begin() as conn:
            row = conn.execute(
                select(self._runs).where(self._runs.c.run_id == run_id)
            ).mappings().first()
            if row is None:
                return None
            step_rows = conn.execute(
                select(self._steps)
                .where(self._steps.c.run_id == run_id)
                .order_by(self._steps.c.t.asc())
            ).mappings().all()

        # Some SQLAlchemy backends (e.g. SQLite without the JSON1
        # extension) round-trip JSON columns as strings; coerce defensively.
        def _coerce_json(value: Any, default: Any) -> Any:
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except ValueError:
                    logger.warning(
                        "SQLRunStore: failed to decode JSON column for run %s; "
                        "returning default. Raw value: %r",
                        run_id,
                        value,
                    )
                    return default
            return value if value is not None else default

        return RunRecord(
            run_id=row["run_id"],
            user_id=row["user_id"],
            plan=row["plan"],
            brief=_coerce_json(row["brief_json"], {}),
            prompt_pack_id=row["prompt_pack_id"],
            config=_coerce_json(row["config_json"], {}),
            t=row["t"],
            state=list(_coerce_json(row["state_json"], [])),
            trace=[
                StepRecord(
                    t=sr["t"],
                    spec=_coerce_json(sr["spec_json"], {}),
                    tests=_coerce_json(sr["tests_json"], []),
                    e_star=float(sr["e_star"]),
                    state=(
                        list(_coerce_json(sr["state_json"], []))
                        if sr.get("state_json") is not None
                        else None
                    ),
                )
                for sr in step_rows
            ],
        )

    def update_state(self, run_id: str, *, t: int, state: List[float]) -> None:
        with self._lock, self._engine.begin() as conn:
            conn.execute(
                self._runs.update()
                .where(self._runs.c.run_id == run_id)
                .values(t=t, state_json=list(state))
            )

    def list_for_user(self, user_id: str) -> List[RunRecord]:
        # Implemented in terms of ``get`` for simplicity: a list endpoint
        # in the dashboard fetches a few rows, not millions. If a future
        # caller needs streaming pagination we'd switch to keyset paging.
        from sqlalchemy import select

        with self._engine.begin() as conn:
            rows = conn.execute(
                select(self._runs.c.run_id)
                .where(self._runs.c.user_id == user_id)
                .order_by(self._runs.c.run_id.asc())
            ).all()
        records: List[RunRecord] = []
        for (run_id,) in rows:
            rec = self.get(run_id)
            if rec is not None:
                records.append(rec)
        return records

    def append_step(self, run_id: str, step: StepRecord) -> None:
        with self._lock, self._engine.begin() as conn:
            conn.execute(
                self._steps.insert().values(
                    run_id=run_id,
                    t=step.t,
                    spec_json=step.spec,
                    tests_json=step.tests,
                    e_star=step.e_star,
                    state_json=list(step.state) if step.state is not None else None,
                )
            )

    def close(self) -> None:
        try:
            self._engine.dispose()
        except Exception:
            logger.debug("SQLRunStore.close: engine.dispose raised", exc_info=True)


__all__ = ["SQLRunStore"]
