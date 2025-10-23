import json, numpy as np

def soft_violation(spec_json: str, tests_json: str):
    s = json.loads(spec_json); t = json.loads(tests_json)
    checks = [
        0.0 if "acceptance" in s and s["acceptance"] else 1.0,
        max(0.0, 1.0 - len(t)/5),
        0.5 if "risks" in s and s["risks"] else 1.0
    ]
    return np.array(checks, dtype=np.float32)
