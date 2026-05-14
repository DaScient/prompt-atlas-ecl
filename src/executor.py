import json
import numpy as np

from src.testers import verify_spec


def soft_violation(spec_json: str, tests_json: str):
    """Numeric violation vector for a (spec, tests) pair.

    Phase 2: delegates the structural check to :func:`src.testers.verify_spec`
    (Z3 when available, pure-Python fallback otherwise) and exposes the
    result through the original ``ndarray`` shape so existing callers
    keep working unchanged.
    """
    spec = json.loads(spec_json)
    tests = json.loads(tests_json)
    report = verify_spec(spec, tests)
    return np.array(report.as_violation_vector(), dtype=np.float32)
