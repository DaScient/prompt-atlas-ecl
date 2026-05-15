"""Phase 5 — LLM provider tests."""
import pytest

from src.llm import (
    DeterministicProvider,
    DualLLMProvider,
    LLMResponse,
    get_default_llm,
)


# ----------------------------------------------------------- deterministic


def test_deterministic_is_stable_for_identical_prompts():
    llm = DeterministicProvider()
    a = llm.complete("hello")
    b = llm.complete("hello")
    assert a.text == b.text
    assert a.provider == "deterministic"


def test_deterministic_differs_for_different_prompts():
    llm = DeterministicProvider()
    a = llm.complete("hello")
    b = llm.complete("goodbye")
    assert a.text != b.text


def test_deterministic_honours_max_tokens_cap():
    llm = DeterministicProvider()
    out = llm.complete("hello", max_tokens=1)
    # max_tokens=1 ⇒ at most 4 chars
    assert len(out.text) <= 4


def test_deterministic_includes_system_in_hash():
    llm = DeterministicProvider()
    a = llm.complete("p", system="A")
    b = llm.complete("p", system="B")
    assert a.text != b.text


# ----------------------------------------------------------------------- dual


class _FakeLLM:
    """Test double with explicit text & confidence."""

    def __init__(self, name, text, confidence=0.5, fail=False):
        self.name = name
        self._text = text
        self._conf = confidence
        self._fail = fail

    def complete(self, prompt, *, system=None, max_tokens=512, temperature=0.2):
        if self._fail:
            raise RuntimeError(f"{self.name} crashed")
        return LLMResponse(text=self._text, provider=self.name, confidence=self._conf)


def test_dual_picks_higher_confidence():
    dual = DualLLMProvider(
        [_FakeLLM("a", "short", 0.4), _FakeLLM("b", "much longer answer", 0.9)],
    )
    out = dual.complete("hi")
    assert out.text == "much longer answer"
    assert out.meta["winner"] == "b"


def test_dual_falls_back_when_one_provider_fails():
    dual = DualLLMProvider(
        [_FakeLLM("a", "alpha", 0.7, fail=True), _FakeLLM("b", "beta", 0.6)],
    )
    out = dual.complete("hi")
    assert out.text == "beta"


def test_dual_raises_when_all_providers_fail():
    dual = DualLLMProvider(
        [_FakeLLM("a", "", 0, fail=True), _FakeLLM("b", "", 0, fail=True)],
    )
    with pytest.raises(RuntimeError):
        dual.complete("hi")


def test_dual_computes_divergence():
    # Two different non-empty responses => divergence == 1.0
    dual = DualLLMProvider(
        [_FakeLLM("a", "alpha", 0.7), _FakeLLM("b", "beta", 0.6)],
    )
    out = dual.complete("hi")
    assert out.meta["divergence"] == pytest.approx(1.0)

    # Two identical responses => divergence == 0.0
    dual = DualLLMProvider(
        [_FakeLLM("a", "same", 0.7), _FakeLLM("b", "same", 0.6)],
    )
    out = dual.complete("hi")
    assert out.meta["divergence"] == pytest.approx(0.0)


def test_dual_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        DualLLMProvider([])


# -------------------------------------------------------------- factory


def test_default_factory_returns_deterministic_when_unset(monkeypatch):
    for var in ("PAE_LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    llm = get_default_llm()
    assert isinstance(llm, DeterministicProvider)


def test_default_factory_unknown_name_falls_back(monkeypatch):
    monkeypatch.setenv("PAE_LLM_PROVIDER", "totally-made-up")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(get_default_llm(), DeterministicProvider)


def test_default_factory_openai_without_key_falls_back(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PAE_LLM_PROVIDER", "openai")
    assert isinstance(get_default_llm(), DeterministicProvider)


def test_default_factory_dual_without_keys_falls_back(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("PAE_LLM_PROVIDER", "dual")
    assert isinstance(get_default_llm(), DeterministicProvider)
