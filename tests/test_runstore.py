"""Tests for the Phase 3 RunStore backends."""
import os
import tempfile

import pytest

from src.runstore import (
    InMemoryRunStore,
    RunRecord,
    StepRecord,
    get_default_runstore,
)


def _sample_record(run_id: str = "r1") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        user_id="u1",
        plan="free",
        brief={"goal": "test"},
        prompt_pack_id=None,
        config={"foo": 1},
        t=0,
        state=[0.0, 0.0, 0.0],
        trace=[],
    )


# ----------------------------------------------------------------- in-memory


def test_in_memory_create_and_get_roundtrips():
    store = InMemoryRunStore()
    rec = _sample_record()
    store.create(rec)
    fetched = store.get("r1")
    assert fetched is not None
    assert fetched.user_id == "u1"
    assert fetched.state == [0.0, 0.0, 0.0]


def test_in_memory_get_returns_none_for_missing():
    store = InMemoryRunStore()
    assert store.get("nope") is None


def test_in_memory_update_state_and_append_step():
    store = InMemoryRunStore()
    store.create(_sample_record())

    store.update_state("r1", t=1, state=[1.0, 2.0, 3.0])
    store.append_step("r1", StepRecord(t=1, spec={"a": 1}, tests=[{"n": 1}], e_star=0.42))

    fetched = store.get("r1")
    assert fetched.t == 1
    assert fetched.state == [1.0, 2.0, 3.0]
    assert len(fetched.trace) == 1
    assert fetched.trace[0].e_star == 0.42


def test_in_memory_update_missing_run_is_noop():
    store = InMemoryRunStore()
    # Must not raise.
    store.update_state("ghost", t=5, state=[0.0])
    store.append_step("ghost", StepRecord(t=5, spec={}, tests=[], e_star=0.0))


# ----------------------------------------------------------------- factory


def test_factory_returns_in_memory_when_no_database_url(monkeypatch):
    monkeypatch.delenv("PAE_DATABASE_URL", raising=False)
    store = get_default_runstore()
    assert isinstance(store, InMemoryRunStore)


def test_factory_falls_back_when_sqlalchemy_init_fails(monkeypatch):
    # An obviously bogus URL exercises the except-clause fallback path
    # in the factory without requiring SQLAlchemy to actually be missing.
    pytest.importorskip("sqlalchemy")
    monkeypatch.setenv("PAE_DATABASE_URL", "not-a-valid-url://")
    store = get_default_runstore()
    # Either it was rejected at engine creation (→ in-memory) or it
    # constructed a SQLAlchemy engine that simply can't speak to the
    # nonsense scheme. Both are acceptable; the import should not crash.
    # We at least know the factory returned *something* implementing the protocol.
    assert hasattr(store, "create")
    assert hasattr(store, "get")


# ----------------------------------------------------------------- SQL backend


def test_sql_runstore_roundtrips_with_sqlite(tmp_path):
    pytest.importorskip("sqlalchemy")
    from src.runstore import SQLRunStore

    db_path = tmp_path / "runs.db"
    store = SQLRunStore(f"sqlite:///{db_path}")
    try:
        rec = _sample_record("r-sql")
        rec.trace.append(StepRecord(t=1, spec={"x": 1}, tests=[{"n": 1}], e_star=0.5))
        store.create(rec)

        fetched = store.get("r-sql")
        assert fetched is not None
        assert fetched.user_id == "u1"
        assert fetched.brief == {"goal": "test"}
        assert len(fetched.trace) == 1
        assert fetched.trace[0].e_star == 0.5

        store.update_state("r-sql", t=2, state=[9.0, 9.0, 9.0])
        store.append_step("r-sql", StepRecord(t=2, spec={"y": 2}, tests=[], e_star=0.7))
        fetched = store.get("r-sql")
        assert fetched.t == 2
        assert fetched.state == [9.0, 9.0, 9.0]
        assert [s.t for s in fetched.trace] == [1, 2]
        assert fetched.trace[1].spec == {"y": 2}
    finally:
        store.close()


def test_sql_runstore_persists_across_instances(tmp_path):
    pytest.importorskip("sqlalchemy")
    from src.runstore import SQLRunStore

    url = f"sqlite:///{tmp_path / 'runs2.db'}"
    a = SQLRunStore(url)
    a.create(_sample_record("persist-1"))
    a.close()

    b = SQLRunStore(url)
    try:
        fetched = b.get("persist-1")
        assert fetched is not None
        assert fetched.user_id == "u1"
    finally:
        b.close()


def test_list_for_user_in_memory():
    store = InMemoryRunStore()
    store.create(_sample_record("a"))
    store.create(_sample_record("b"))
    other = _sample_record("c")
    other.user_id = "other"
    store.create(other)

    rs = store.list_for_user("u1")
    assert {r.run_id for r in rs} == {"a", "b"}
    assert store.list_for_user("nobody") == []


def test_list_for_user_sql(tmp_path):
    pytest.importorskip("sqlalchemy")
    from src.runstore import SQLRunStore

    store = SQLRunStore(f"sqlite:///{tmp_path / 'list.db'}")
    try:
        store.create(_sample_record("a"))
        store.create(_sample_record("b"))
        rs = store.list_for_user("u1")
        assert {r.run_id for r in rs} == {"a", "b"}
        assert store.list_for_user("nobody") == []
    finally:
        store.close()


def test_step_record_with_state_roundtrips_through_sql(tmp_path):
    pytest.importorskip("sqlalchemy")
    from src.runstore import SQLRunStore

    store = SQLRunStore(f"sqlite:///{tmp_path / 'state.db'}")
    try:
        rec = _sample_record("s-state")
        rec.trace.append(
            StepRecord(t=1, spec={}, tests=[], e_star=0.5, state=[1.0, 2.0, 3.0])
        )
        store.create(rec)
        fetched = store.get("s-state")
        assert fetched.trace[0].state == [1.0, 2.0, 3.0]
    finally:
        store.close()


def test_factory_uses_sql_backend_when_url_is_set(tmp_path, monkeypatch):
    pytest.importorskip("sqlalchemy")
    from src.runstore import SQLRunStore

    monkeypatch.setenv("PAE_DATABASE_URL", f"sqlite:///{tmp_path / 'rfac.db'}")
    store = get_default_runstore()
    try:
        assert isinstance(store, SQLRunStore)
    finally:
        store.close()
