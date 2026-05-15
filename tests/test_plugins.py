"""Phase 6 — plugin registry tests."""
import pytest

from src.plugins import PluginRegistry, get_default_registry, register


def test_register_and_lookup():
    r = PluginRegistry()
    r.register("llm", "fake", lambda: "instance", meta={"x": 1})
    rec = r.get("llm", "fake")
    assert rec is not None
    assert rec.namespace == "llm"
    assert rec.factory() == "instance"
    assert rec.meta == {"x": 1}


def test_unknown_namespace_rejected():
    r = PluginRegistry()
    with pytest.raises(ValueError):
        r.register("lmm", "x", lambda: None)  # typo


def test_empty_name_rejected():
    r = PluginRegistry()
    with pytest.raises(ValueError):
        r.register("llm", "", lambda: None)


def test_duplicate_name_kept_unless_overwrite():
    r = PluginRegistry()
    r.register("llm", "x", lambda: "first", source="a")
    r.register("llm", "x", lambda: "second", source="b")
    assert r.get("llm", "x").factory() == "first"
    # overwrite=True wins.
    r.register("llm", "x", lambda: "third", source="c", overwrite=True)
    assert r.get("llm", "x").factory() == "third"


def test_list_filters_by_namespace():
    r = PluginRegistry()
    r.register("llm", "a", lambda: None)
    r.register("embeddings", "b", lambda: None)
    assert [p.name for p in r.list("llm")] == ["a"]
    assert [p.name for p in r.list("embeddings")] == ["b"]
    assert len(r.list()) == 2
    assert r.list("nonsense") == []


def test_get_unknown_namespace_returns_none():
    r = PluginRegistry()
    assert r.get("nonsense", "x") is None


def test_discover_entry_points_is_idempotent(monkeypatch):
    # Without any entry points installed, discover should report 0 and
    # be safe to call twice.
    r = PluginRegistry()
    first = r.discover_entry_points()
    second = r.discover_entry_points()
    assert first == 0
    assert second == 0


def test_global_registry_singleton():
    a = get_default_registry()
    b = get_default_registry()
    assert a is b


def test_register_decorator_registers(monkeypatch):
    # Decorator form uses the global registry; clear after.
    monkeypatch.delenv("PAE_PLUGINS", raising=False)

    @register("tester", "_test_tester_for_phase6", overwrite=True)
    def _factory():
        return {"hello": "world"}

    rec = get_default_registry().get("tester", "_test_tester_for_phase6")
    assert rec is not None
    assert rec.factory()["hello"] == "world"


def test_llm_factory_uses_plugin(monkeypatch):
    """The LLM factory should pick up a plugin registered by name."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("PAE_LLM_PROVIDER", "_phase6_test_llm")

    class _Fake:
        name = "_phase6_test_llm"

        def complete(self, prompt, **kw):
            from src.llm.base import LLMResponse
            return LLMResponse(text="plugin-response", provider=self.name)

    get_default_registry().register(
        "llm", "_phase6_test_llm", lambda: _Fake(), overwrite=True,
    )

    from src.llm import get_default_llm
    llm = get_default_llm()
    assert llm.complete("hi").text == "plugin-response"


def test_llm_factory_rejects_plugin_without_complete(monkeypatch):
    monkeypatch.setenv("PAE_LLM_PROVIDER", "_phase6_bad_llm")

    get_default_registry().register(
        "llm", "_phase6_bad_llm", lambda: object(), overwrite=True,
    )
    from src.llm import DeterministicProvider, get_default_llm

    # Falls back gracefully when the plugin returns the wrong shape.
    assert isinstance(get_default_llm(), DeterministicProvider)


def test_embeddings_factory_uses_plugin(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PAE_EMBEDDINGS_PROVIDER", "_phase6_test_embed")

    class _FakeEmb:
        name = "_phase6_test_embed"
        dim = 4

        def embed(self, texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    get_default_registry().register(
        "embeddings", "_phase6_test_embed", lambda: _FakeEmb(), overwrite=True,
    )

    from src.embeddings import get_default_embeddings
    emb = get_default_embeddings()
    out = emb.embed(["a", "b"])
    assert out == [[0.1, 0.2, 0.3, 0.4]] * 2
