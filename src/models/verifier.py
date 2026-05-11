"""Static verification of model registry YAML entries (ModelForge).

Checks (no LLM, fully deterministic):
    1. schema_valid       — YAML loads and validates as ModelRegistryEntry
    2. equations_present  — ≥ 1 equation string
    3. solver_specified   — non-empty solver field
    4. outputs_present    — ≥ 1 output string
    5. parameters_documented — each parameter entry is non-empty
    6. references_present — ≥ 1 reference
    7. parameter_in_equations — each parameter name substring appears in at least
                                one equation string (WARN if missing, not FAIL)

Overall status: FAIL > WARN > PASS (worst across all checks).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from schemas import (
    ModelRegistryEntry,
    VerificationCheck,
    VerificationReport,
    VerificationStatus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PARAM_NAME_MAX_LEN: int = 40  # truncate long param strings for substring search


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of file content."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _worst(statuses: list[VerificationStatus]) -> VerificationStatus:
    """Return the most severe status in the list."""
    if VerificationStatus.FAIL in statuses:
        return VerificationStatus.FAIL
    if VerificationStatus.WARN in statuses:
        return VerificationStatus.WARN
    return VerificationStatus.PASS


def _check(
    name: str,
    condition: bool,
    fail_msg: str,
    warn: bool = False,
) -> VerificationCheck:
    """Build a VerificationCheck from a boolean condition.

    Args:
        name:      Check identifier.
        condition: True → PASS.
        fail_msg:  Message when condition is False.
        warn:      If True, failure is WARN instead of FAIL.
    """
    if condition:
        return VerificationCheck(name=name, status=VerificationStatus.PASS)
    level = VerificationStatus.WARN if warn else VerificationStatus.FAIL
    return VerificationCheck(name=name, status=level, message=fail_msg)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_equations_present(entry: ModelRegistryEntry) -> VerificationCheck:
    return _check(
        "equations_present",
        bool(entry.equations),
        "No equations defined — mathematical model must have at least one equation.",
    )


def _check_solver_specified(entry: ModelRegistryEntry) -> VerificationCheck:
    return _check(
        "solver_specified",
        bool(entry.solver and entry.solver.strip()),
        "Solver field is empty.",
    )


def _check_outputs_present(entry: ModelRegistryEntry) -> VerificationCheck:
    return _check(
        "outputs_present",
        bool(entry.outputs),
        "No outputs defined.",
    )


def _check_parameters_documented(entry: ModelRegistryEntry) -> VerificationCheck:
    if not entry.parameters:
        # No parameters declared is valid (e.g. non-parametric models)
        return VerificationCheck(name="parameters_documented", status=VerificationStatus.PASS)
    empty = [p for p in entry.parameters if not p.strip()]
    return _check(
        "parameters_documented",
        not empty,
        f"{len(empty)} parameter entry/entries are empty strings.",
    )


def _check_references_present(entry: ModelRegistryEntry) -> VerificationCheck:
    return _check(
        "references_present",
        bool(entry.references),
        "No references listed — add at least one primary reference.",
        warn=True,  # WARN, not FAIL: some models may be self-contained
    )


def _check_parameter_in_equations(entry: ModelRegistryEntry) -> VerificationCheck:
    """WARN if a declared parameter name does not appear in any equation string.

    Extracts the short name (first token before ':' or ' ') from each parameter.
    """
    if not entry.parameters or not entry.equations:
        return VerificationCheck(name="parameter_in_equations", status=VerificationStatus.PASS)

    equations_blob = " ".join(entry.equations).lower()
    missing: list[str] = []
    for param in entry.parameters:
        # Short name: text before ':' or first space
        short = param.split(":")[0].split(" ")[0].strip()[:_PARAM_NAME_MAX_LEN]
        # Strip domain annotations like ∈ ℝ, _t, {}, etc.
        short_clean = short.rstrip("{}").split("∈")[0].strip()
        if short_clean and short_clean.lower() not in equations_blob:
            missing.append(short_clean)

    return _check(
        "parameter_in_equations",
        not missing,
        f"Parameters not found in any equation: {', '.join(missing)}",
        warn=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_yaml_file(yaml_path: Path) -> VerificationReport:
    """Verify a single model registry YAML file.

    Args:
        yaml_path: Absolute or relative path to a model registry YAML.

    Returns:
        VerificationReport with all check results.

    Raises:
        FileNotFoundError: if yaml_path does not exist.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Registry YAML not found: {yaml_path}")

    raw_bytes = yaml_path.read_bytes()
    registry_hash = _sha256_bytes(raw_bytes)

    # Infer model_id from filename stem
    model_id = yaml_path.stem

    checks: list[VerificationCheck] = []

    # Check 1: schema validation
    try:
        data = yaml.safe_load(raw_bytes)
        entry = ModelRegistryEntry(**data)
        checks.append(VerificationCheck(name="schema_valid", status=VerificationStatus.PASS))
    except Exception as exc:  # noqa: BLE001
        checks.append(
            VerificationCheck(
                name="schema_valid",
                status=VerificationStatus.FAIL,
                message=f"Schema validation error: {exc}",
            )
        )
        # Cannot run further checks without a valid entry
        return VerificationReport(
            model_id=model_id,
            registry_hash=registry_hash,
            checks=checks,
            overall=VerificationStatus.FAIL,
        )

    # Checks 2–7: content checks
    checks.extend(
        [
            _check_equations_present(entry),
            _check_solver_specified(entry),
            _check_outputs_present(entry),
            _check_parameters_documented(entry),
            _check_references_present(entry),
            _check_parameter_in_equations(entry),
        ]
    )

    overall = _worst([c.status for c in checks])
    return VerificationReport(
        model_id=model_id,
        registry_hash=registry_hash,
        checks=checks,
        overall=overall,
    )


def verify_all(
    registry_dir: Path,
) -> dict[str, VerificationReport]:
    """Verify every YAML in *registry_dir*.

    Returns:
        dict mapping model_id → VerificationReport.
    """
    if not registry_dir.is_dir():
        raise FileNotFoundError(f"Registry directory not found: {registry_dir}")

    return {
        yaml_path.stem: verify_yaml_file(yaml_path)
        for yaml_path in sorted(registry_dir.glob("*.yaml"))
    }
