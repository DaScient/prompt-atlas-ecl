from pathlib import Path
import json

Path("data").mkdir(exist_ok=True)
toy = {
  "goal": "Design an API for climate-aware routing.",
  "constraints": ["latency<200ms", "CO2 budget visible", "privacy-preserving"],
  "data": {"sources": ["traffic", "signals", "weather"]},
  "risks": ["data drift", "unfair routing", "sensor outages"]
}
with open("data/train.jsonl", "w") as f:
    for _ in range(100):
        f.write(json.dumps(toy) + "\n")
open("data/val.jsonl","w").write(json.dumps(toy)+"\n")
print("Wrote toy data.")
