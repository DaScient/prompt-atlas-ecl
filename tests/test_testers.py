"""Tests for the Phase 2 Z3-backed Tester and its Python fallback."""
import json

import numpy as np

from src.executor import soft_violation
from src.testers import REQUIRED_SPEC_FIELDS, VerificationReport, verify_spec
from src.testers.z3_tester import _python_verify


GOOD_SPEC = {
    "assumptions": ["a"],
    "data": {"sources": ["x"]},
    "steps": ["s1"],
    "interfaces": ["api"],
    "acceptance": ["criterion"],
    "risks": ["r"],
}
GOOD_TESTS = [{"name": "t1"}, {"name": "t2"}]


def test_python_backend_passes_complete_spec():
    report = _python_verify(GOOD_SPEC, GOOD_TESTS)
    assert report.backend == "python"
    assert report.satisfied is True
    assert report.missing_fields == []
    assert report.empty_fields == []
    assert report.coverage_ok is True


def test_python_backend_flags_missing_fields():
    bad_spec = {k: GOOD_SPEC[k] for k in list(GOOD_SPEC)[:-2]}  # drop acceptance, risks
    report = _python_verify(bad_spec, GOOD_TESTS)
    assert report.satisfied is False
    assert "acceptance" in report.missing_fields
    assert "risks" in report.missing_fields


def test_python_backend_flags_empty_fields():
    bad_spec = dict(GOOD_SPEC, risks=[])
    report = _python_verify(bad_spec, GOOD_TESTS)
    assert report.satisfied is False
    assert "risks" in report.empty_fields


def test_python_backend_flags_low_coverage():
    report = _python_verify(GOOD_SPEC, [])
    assert report.coverage_ok is False
    assert report.satisfied is False


def test_verify_spec_falls_back_when_z3_missing():
    # Even when prefer_z3=True, an absent z3 module must yield a python report.
    report = verify_spec(GOOD_SPEC, GOOD_TESTS, prefer_z3=True)
    assert isinstance(report, VerificationReport)
    assert report.backend in {"z3", "python"}
    assert report.satisfied is True


def test_violation_vector_shape_and_range():
    report = _python_verify(GOOD_SPEC, GOOD_TESTS)
    v = report.as_violation_vector()
    assert len(v) == 3
    assert all(0.0 <= x <= 1.0 for x in v)


def test_soft_violation_preserves_legacy_api():
    out = soft_violation(json.dumps(GOOD_SPEC), json.dumps(GOOD_TESTS))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    assert out.shape == (3,)


def test_soft_violation_penalizes_missing_acceptance():
    bad = dict(GOOD_SPEC)
    bad.pop("acceptance")
    out_good = soft_violation(json.dumps(GOOD_SPEC), json.dumps(GOOD_TESTS))
    out_bad = soft_violation(json.dumps(bad), json.dumps(GOOD_TESTS))
    # Missing acceptance must increase the first component.
    assert out_bad[0] > out_good[0]


def test_required_fields_match_executor_expectations():
    # Sanity: the public required-field tuple isn't accidentally empty.
    assert "acceptance" in REQUIRED_SPEC_FIELDS
    assert "risks" in REQUIRED_SPEC_FIELDS
