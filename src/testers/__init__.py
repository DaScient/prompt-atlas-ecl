"""Phase 2 — symbolic spec/tests verification.

Re-exports the public API of :mod:`src.testers.z3_tester` so callers can
do ``from src.testers import verify_spec`` regardless of backend.
"""
from src.testers.z3_tester import (
    MIN_TESTS_FOR_COVERAGE,
    REQUIRED_SPEC_FIELDS,
    VerificationReport,
    verify_spec,
)

__all__ = [
    "MIN_TESTS_FOR_COVERAGE",
    "REQUIRED_SPEC_FIELDS",
    "VerificationReport",
    "verify_spec",
]
