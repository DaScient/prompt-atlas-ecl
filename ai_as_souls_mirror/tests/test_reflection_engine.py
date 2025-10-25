from pathlib import Path
from reflection_engine import ReflectionEngine

def test_reflect_basic():
    cfg = Path(__file__).resolve().parents[1] / "archetypes.json"
    engine = ReflectionEngine(cfg)
    out = engine.reflect("I feel lost but curious and ready to keep searching.")
    assert "archetype" in out and "sentiment" in out and "mirror_text" in out
    assert isinstance(out["mirror_text"], str) and len(out["mirror_text"]) > 0
