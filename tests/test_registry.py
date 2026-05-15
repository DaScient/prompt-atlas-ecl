"""Phase 6 — prompt-pack registry tests."""
from pathlib import Path

import pytest

from src.registry import PromptPack, PromptPackRegistry, get_default_pack_registry
from src.registry.models import PromptTemplate


def test_bundled_packs_load():
    """The four legacy pack IDs must still be present (back-compat)."""
    reg = get_default_pack_registry()
    ids = set(reg.all_ids())
    for legacy in {"myth-1", "science-1", "psych-1", "purpose-1"}:
        assert legacy in ids, f"missing bundled pack {legacy}"


def test_bundled_pack_has_prompts():
    reg = get_default_pack_registry()
    pack = reg.get("myth-1")
    assert pack is not None
    assert pack.title == "Myth & Meaning"
    assert pack.domain == "myth"
    assert len(pack.prompts) >= 1
    assert all(isinstance(p, PromptTemplate) for p in pack.prompts)


def test_list_filters_by_domain():
    reg = get_default_pack_registry()
    sci = reg.list(domain="science")
    assert any(p.id == "science-1" for p in sci)
    assert all(p.domain == "science" for p in sci)


def test_list_filters_by_tag():
    reg = get_default_pack_registry()
    archetype_packs = reg.list(tag="archetype")
    assert any(p.id == "myth-1" for p in archetype_packs)


def test_list_search_query_matches_id_title_description():
    reg = get_default_pack_registry()
    out = reg.list(query="empathy")
    assert any(p.id == "psych-1" for p in out)
    # unrelated query returns nothing
    assert reg.list(query="not-a-real-token-xyz") == []


def test_get_unknown_returns_none():
    reg = get_default_pack_registry()
    assert reg.get("does-not-exist") is None


def test_pack_render_substitutes_defaults_and_args():
    pack = PromptPack(
        id="t1",
        title="t",
        domain="t",
        defaults={"audience": "users"},
        prompts=[
            PromptTemplate(
                name="hello",
                body="hi {audience}, about {topic}",
                inputs=["topic"],
            )
        ],
    )
    out = pack.render("hello", topic="cats")
    assert out == "hi users, about cats"


def test_pack_render_caller_overrides_defaults():
    pack = PromptPack(
        id="t1",
        title="t",
        domain="t",
        defaults={"audience": "users"},
        prompts=[PromptTemplate(name="h", body="hi {audience}")],
    )
    assert pack.render("h", audience="researchers") == "hi researchers"


def test_pack_render_missing_slot_raises():
    pack = PromptPack(
        id="t1",
        title="t",
        domain="t",
        prompts=[PromptTemplate(name="h", body="hi {who}")],
    )
    with pytest.raises(KeyError):
        pack.render("h")


def test_pack_render_unknown_prompt_raises():
    pack = PromptPack(id="t1", title="t", domain="t", prompts=[])
    with pytest.raises(KeyError):
        pack.render("nope")


def test_pack_id_must_be_url_safe():
    with pytest.raises(Exception):
        PromptPack(id="bad id with spaces", title="t", domain="t")


def test_registry_overlay_dir(tmp_path: Path):
    """Files in an overlay dir register on top of the bundled set."""
    overlay = tmp_path / "extra"
    overlay.mkdir()
    (overlay / "mypack.yaml").write_text(
        "id: my-pack-1\n"
        "version: 0.0.1\n"
        "title: Custom\n"
        "domain: custom\n"
        "tags: [demo]\n"
        "prompts:\n"
        "  - name: hi\n"
        "    body: hello {who}\n",
        encoding="utf-8",
    )
    reg = PromptPackRegistry(dirs=[overlay])
    assert reg.load() == 1
    pack = reg.get("my-pack-1")
    assert pack is not None
    assert pack.title == "Custom"
    assert pack.render("hi", who="world") == "hello world"


def test_registry_skips_unreadable_or_invalid_files(tmp_path: Path):
    d = tmp_path / "packs"
    d.mkdir()
    # garbled YAML
    (d / "broken.yaml").write_text("::not_yaml:::\n  - [", encoding="utf-8")
    # right ext but wrong shape (list instead of dict)
    (d / "wrong-shape.json").write_text('["just", "a", "list"]', encoding="utf-8")
    # a valid one to prove the loader keeps going past failures
    (d / "ok.yaml").write_text(
        "id: ok\ntitle: ok\ndomain: t\nprompts: []\n", encoding="utf-8",
    )
    reg = PromptPackRegistry(dirs=[d])
    reg.load()
    assert reg.get("ok") is not None
    assert reg.all_ids() == ["ok"]


def test_registry_load_is_idempotent(tmp_path: Path):
    d = tmp_path / "packs"
    d.mkdir()
    (d / "a.yaml").write_text("id: a\ntitle: A\ndomain: t\nprompts: []\n", encoding="utf-8")
    reg = PromptPackRegistry(dirs=[d])
    assert reg.load() == 1
    assert reg.load() == 1
