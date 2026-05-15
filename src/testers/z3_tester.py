"""Phase 2 — symbolic spec/tests verification with a Z3 backend.

The Tester role in the ECL loop is responsible for checking whether a
Writer-produced ``spec`` is internally consistent and adequately covered
by ``tests``. Phase 1 used a hand-rolled numeric checker in
:func:`src.executor.soft_violation` that only counted fields.

Phase 2 promotes this to a small **structural verifier**:

* If the optional ``z3-solver`` package is installed, we build a Z3
  model and check satisfiability of the spec's required-field
  predicates plus a coverage constraint on tests.
* Otherwise we fall back to an equivalent pure-Python implementation
  so the rest of the system continues to work in minimal environments
  (e.g. CI without native Z3 wheels).

Both paths return the same :class:`VerificationReport` shape, so callers
don't care which backend ran.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Mapping, Sequence


# Fields the Writer is expected to populate. Order matters only for
# the deterministic reporting; semantics are set-based.
REQUIRED_SPEC_FIELDS: tuple = (
    "assumptions",
    "data",
    "steps",
    "interfaces",
    "acceptance",
    "risks",
)

# Minimum number of tests we consider "adequate coverage". Mirrors the
# old executor heuristic (len(tests)/5) but expressed as a hard floor.
MIN_TESTS_FOR_COVERAGE = 2


@dataclass
class VerificationReport:
    """Structured result from :func:`verify_spec`."""

    backend: str                                  # "z3" or "python"
    satisfied: bool                               # overall pass/fail
    missing_fields: List[str] = field(default_factory=list)
    empty_fields: List[str] = field(default_factory=list)
    coverage_ok: bool = True
    n_tests: int = 0
    notes: List[str] = field(default_factory=list)

    def as_violation_vector(self) -> List[float]:
        """Numeric encoding compatible with the Phase 1 ``soft_violation``.

        Returns three normalized [0, 1] scalars:

        1. acceptance / required-fields violation density
        2. test-coverage shortfall
        3. risks-field violation (binary)
        """
        n_req = len(REQUIRED_SPEC_FIELDS)
        bad = set(self.missing_fields) | set(self.empty_fields)
        acceptance_violation = 0.0 if "acceptance" not in bad else 1.0
        risks_violation = 0.0 if "risks" not in bad else 1.0
        coverage_violation = max(0.0, 1.0 - (self.n_tests / 5.0))
        field_density = len(bad) / float(n_req)
        return [
            min(1.0, 0.5 * acceptance_violation + 0.5 * field_density),
            min(1.0, coverage_violation),
            risks_violation,
        ]


def _shallow_is_nonempty(v: Any) -> bool:
    """Treat ``None``, ``""``, ``[]``, ``{}`` as empty; everything else as filled."""
    if v is None:
        return False
    if isinstance(v, (str, list, tuple, dict, set)):
        return len(v) > 0
    return True


def _python_verify(spec: Mapping[str, Any], tests: Sequence[Any]) -> VerificationReport:
    """Pure-Python equivalent of the Z3 backend, used as a fallback."""
    missing = [f for f in REQUIRED_SPEC_FIELDS if f not in spec]
    empty = [
        f for f in REQUIRED_SPEC_FIELDS
        if f in spec and not _shallow_is_nonempty(spec[f])
    ]
    n_tests = len(tests)
    coverage_ok = n_tests >= MIN_TESTS_FOR_COVERAGE
    satisfied = not missing and not empty and coverage_ok
    notes: List[str] = []
    if missing:
        notes.append(f"missing required fields: {missing}")
    if empty:
        notes.append(f"empty required fields: {empty}")
    if not coverage_ok:
        notes.append(
            f"insufficient tests: {n_tests} < {MIN_TESTS_FOR_COVERAGE}"
        )
    return VerificationReport(
        backend="python",
        satisfied=satisfied,
        missing_fields=missing,
        empty_fields=empty,
        coverage_ok=coverage_ok,
        n_tests=n_tests,
        notes=notes,
    )


def _z3_verify(spec: Mapping[str, Any], tests: Sequence[Any]) -> VerificationReport:
    """Z3-backed verifier; only called when ``z3`` is importable."""
    import z3  # local import: optional dependency

    solver = z3.Solver()
    field_vars = {f: z3.Bool(f"has_{f}") for f in REQUIRED_SPEC_FIELDS}
    missing: List[str] = []
    empty: List[str] = []
    for f, var in field_vars.items():
        if f not in spec:
            missing.append(f)
            solver.add(z3.Not(var))
        elif not _shallow_is_nonempty(spec[f]):
            empty.append(f)
            solver.add(z3.Not(var))
        else:
            solver.add(var)

    n_tests = len(tests)
    coverage_var = z3.Bool("coverage_ok")
    if n_tests >= MIN_TESTS_FOR_COVERAGE:
        solver.add(coverage_var)
    else:
        solver.add(z3.Not(coverage_var))

    goal = z3.And(*field_vars.values(), coverage_var)
    solver.push()
    solver.add(goal)
    satisfied = solver.check() == z3.sat
    solver.pop()

    notes: List[str] = []
    if missing:
        notes.append(f"missing required fields: {missing}")
    if empty:
        notes.append(f"empty required fields: {empty}")
    if n_tests < MIN_TESTS_FOR_COVERAGE:
        notes.append(
            f"insufficient tests: {n_tests} < {MIN_TESTS_FOR_COVERAGE}"
        )
    return VerificationReport(
        backend="z3",
        satisfied=satisfied,
        missing_fields=missing,
        empty_fields=empty,
        coverage_ok=n_tests >= MIN_TESTS_FOR_COVERAGE,
        n_tests=n_tests,
        notes=notes,
    )


def verify_spec(
    spec: Mapping[str, Any],
    tests: Sequence[Any],
    *,
    prefer_z3: bool = True,
) -> VerificationReport:
    """Verify a Writer spec against its Tester suite.

    Tries the Z3 backend first when ``prefer_z3`` is set; falls back to
    the pure-Python checker if ``z3-solver`` isn't installed or its
    import fails for any reason.
    """
    if prefer_z3:
        try:
            return _z3_verify(spec, tests)
        except ImportError:
            pass
        except Exception:
            # Any unexpected Z3 failure should not break the pipeline.
            pass
    return _python_verify(spec, tests)


__all__ = [
    "REQUIRED_SPEC_FIELDS",
    "MIN_TESTS_FOR_COVERAGE",
    "VerificationReport",
    "verify_spec",
]
