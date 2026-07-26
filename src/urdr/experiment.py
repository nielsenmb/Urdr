"""Checkpointed execution for scientific-scale synthetic experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .validation import (
    ReliabilityBin,
    SyntheticValidation,
    ValidationCase,
    ValidationMetric,
    benchmark_synthetic_validation,
)


@dataclass(frozen=True)
class SyntheticExperimentPlan:
    """Pre-registered configuration for a synthetic validation experiment.

    Parameters
    ----------
    name
        Human-readable experiment name.
    cases
        Target, signal-to-noise, window, and contaminant cells. Each case is
        executed and checkpointed independently.
    calibration_realizations
        Target-specific calibration simulations per case.
    evaluation_realizations
        Independent evaluation simulations per class and case.
    validation_fraction
        Fraction reserved internally by the joint detector.
    target_false_positive_rate
        Requested detection false-positive rate.
    delta_nu_tolerance
        Fractional tolerance defining successful separation recovery.
    reliability_bins
        Number of equal-width probability-reliability bins.
    seed
        Root seed. Per-case seeds are derived from this value and case names,
        so results do not depend on worker count or execution order.
    """

    name: str
    cases: tuple[ValidationCase, ...]
    calibration_realizations: int = 4096
    evaluation_realizations: int = 2048
    validation_fraction: float = 0.25
    target_false_positive_rate: float = 0.01
    delta_nu_tolerance: float = 0.05
    reliability_bins: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate and normalise the frozen experiment definition."""
        if not self.name:
            raise ValueError("experiment name cannot be empty")
        cases = tuple(self.cases)
        if not cases:
            raise ValueError("at least one validation case is required")
        names = [case.name for case in cases]
        if len(set(names)) != len(names):
            raise ValueError("validation case names must be unique")
        if self.calibration_realizations < 16:
            raise ValueError("calibration_realizations must be at least 16")
        if self.evaluation_realizations < 4:
            raise ValueError("evaluation_realizations must be at least 4")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie between zero and one")
        if not 0 < self.target_false_positive_rate < 1:
            raise ValueError(
                "target_false_positive_rate must lie between zero and one"
            )
        if not 0 < self.delta_nu_tolerance < 1:
            raise ValueError("delta_nu_tolerance must lie between zero and one")
        if self.reliability_bins < 2:
            raise ValueError("reliability_bins must be at least two")
        object.__setattr__(self, "cases", cases)

    @property
    def fingerprint(self) -> str:
        """Return a stable hash of the complete experimental design."""
        payload = json.dumps(
            self.to_manifest(include_fingerprint=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_manifest(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        """Return a JSON-compatible pre-registration manifest.

        Parameters
        ----------
        include_fingerprint
            Whether to include the SHA-256 fingerprint in the returned data.

        Returns
        -------
        dict
            Experiment settings, case summaries, and exact window hashes.
        """
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "name": self.name,
            "calibration_realizations": self.calibration_realizations,
            "evaluation_realizations": self.evaluation_realizations,
            "validation_fraction": self.validation_fraction,
            "target_false_positive_rate": self.target_false_positive_rate,
            "delta_nu_tolerance": self.delta_nu_tolerance,
            "reliability_bins": self.reliability_bins,
            "seed": self.seed,
            "cases": [_case_manifest(case) for case in self.cases],
        }
        if include_fingerprint:
            manifest["fingerprint"] = self.fingerprint
        return manifest


@dataclass(frozen=True)
class ExperimentRun:
    """Paths and counts produced by a checkpointed experiment run.

    Parameters
    ----------
    output_directory
        Root directory containing the manifest, checkpoints, and tables.
    manifest_path
        Frozen experiment manifest.
    metrics_path
        Aggregated metric CSV table.
    reliability_path
        Aggregated probability-reliability CSV table.
    completed_cases
        Number of case checkpoints included in the tables.
    reused_cases
        Number of existing checkpoints reused during this invocation.
    """

    output_directory: Path
    manifest_path: Path
    metrics_path: Path
    reliability_path: Path
    completed_cases: int
    reused_cases: int


def run_synthetic_experiment(
    plan: SyntheticExperimentPlan,
    output_directory: str | os.PathLike[str],
    *,
    workers: int = 1,
    resume: bool = True,
) -> ExperimentRun:
    """Execute a pre-registered experiment with per-case checkpoints.

    The manifest is written before the first simulation. Reusing an output
    directory with a different plan raises an error. Completed case
    checkpoints are loaded when ``resume`` is true, while missing cases are
    evaluated sequentially or with a process pool.

    Parameters
    ----------
    plan
        Frozen scientific experiment definition.
    output_directory
        Directory for the manifest, checkpoints, and aggregate CSV tables.
    workers
        Number of local worker processes. Use one for sequential execution.
    resume
        Whether to reuse valid checkpoints from an interrupted run.

    Returns
    -------
    ExperimentRun
        Output paths and checkpoint counts.
    """
    if workers < 1:
        raise ValueError("workers must be at least one")
    output = Path(output_directory)
    checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    _ensure_manifest(plan, manifest_path)

    results: dict[str, SyntheticValidation] = {}
    pending: list[ValidationCase] = []
    reused = 0
    for case in plan.cases:
        checkpoint = checkpoints / _checkpoint_name(case.name)
        if resume and checkpoint.exists():
            results[case.name] = _read_checkpoint(
                checkpoint, plan.fingerprint, case.name
            )
            reused += 1
        else:
            pending.append(case)

    failures: dict[str, str] = {}
    if workers == 1:
        for case in pending:
            try:
                result = _run_case(plan, case)
                results[case.name] = result
                _write_checkpoint(
                    checkpoints / _checkpoint_name(case.name),
                    plan,
                    case,
                    result,
                )
            except Exception as error:  # pragma: no cover - defensive reporting
                failures[case.name] = f"{type(error).__name__}: {error}"
                break
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_case, plan, case): case for case in pending
            }
            for future in as_completed(futures):
                case = futures[future]
                try:
                    result = future.result()
                    results[case.name] = result
                    _write_checkpoint(
                        checkpoints / _checkpoint_name(case.name),
                        plan,
                        case,
                        result,
                    )
                except Exception as error:  # pragma: no cover - process failure
                    failures[case.name] = f"{type(error).__name__}: {error}"

    metrics_path = output / "metrics.csv"
    reliability_path = output / "reliability.csv"
    ordered = [results[case.name] for case in plan.cases if case.name in results]
    _write_tables(ordered, metrics_path, reliability_path)
    if failures:
        details = ", ".join(f"{name} ({error})" for name, error in failures.items())
        raise RuntimeError(f"experiment cases failed: {details}")
    return ExperimentRun(
        output_directory=output,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        reliability_path=reliability_path,
        completed_cases=len(results),
        reused_cases=reused,
    )


def _run_case(
    plan: SyntheticExperimentPlan,
    case: ValidationCase,
) -> SyntheticValidation:
    return benchmark_synthetic_validation(
        [case],
        calibration_realizations=plan.calibration_realizations,
        evaluation_realizations=plan.evaluation_realizations,
        validation_fraction=plan.validation_fraction,
        target_false_positive_rate=plan.target_false_positive_rate,
        delta_nu_tolerance=plan.delta_nu_tolerance,
        reliability_bins=plan.reliability_bins,
        seed=_case_seed(plan.seed, case.name),
    )


def _case_seed(root_seed: int, case_name: str) -> int:
    payload = f"{root_seed}:{case_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def _case_manifest(case: ValidationCase) -> dict[str, Any]:
    time = np.ascontiguousarray(case.window.time, dtype="<f8")
    observed = np.ascontiguousarray(case.window.observed, dtype=np.uint8)
    duration = float(time[-1] - time[0])
    return {
        "name": case.name,
        "split": case.split,
        "simulation": asdict(case.simulation),
        "published_centres_uhz": case.published_centres_uhz.tolist(),
        "restricted_centres_uhz": case.restricted_centres_uhz.tolist(),
        "filter_width_uhz": case.filter_width_uhz,
        "delta_nu_grid_uhz": case.delta_nu_grid_uhz.tolist(),
        "segments_days": [list(segment) for segment in case.segments_days],
        "coherent_contaminants": {
            name: asdict(config)
            for name, config in case.coherent_contaminants.items()
        },
        "segment_systematics": {
            name: [asdict(config) for config in configs]
            for name, configs in case.segment_systematics.items()
        },
        "background": (
            None
            if case.background is None
            else {
                "type": type(case.background).__name__,
                "settings": asdict(case.background),
            }
        ),
        "max_lag_seconds": case.max_lag_seconds,
        "window": {
            "cadences": int(time.size),
            "duration_days": duration,
            "duty_cycle": case.window.duty_cycle,
            "time_sha256": hashlib.sha256(time.tobytes()).hexdigest(),
            "observed_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
        },
    }


def _ensure_manifest(
    plan: SyntheticExperimentPlan,
    manifest_path: Path,
) -> None:
    manifest = plan.to_manifest()
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != plan.fingerprint:
            raise ValueError(
                "output directory contains a different experiment manifest"
            )
        return
    _atomic_json_write(manifest_path, manifest)


def _write_checkpoint(
    path: Path,
    plan: SyntheticExperimentPlan,
    case: ValidationCase,
    result: SyntheticValidation,
) -> None:
    payload = {
        "schema_version": 1,
        "fingerprint": plan.fingerprint,
        "case": case.name,
        "case_seed": _case_seed(plan.seed, case.name),
        "metrics": [asdict(row) for row in result.metrics],
        "reliability": [asdict(row) for row in result.reliability],
        "settings": {
            "target_false_positive_rate": result.target_false_positive_rate,
            "delta_nu_tolerance": result.delta_nu_tolerance,
            "calibration_realizations": result.calibration_realizations,
            "evaluation_realizations": result.evaluation_realizations,
        },
    }
    _atomic_json_write(path, payload)


def _read_checkpoint(
    path: Path,
    fingerprint: str,
    case_name: str,
) -> SyntheticValidation:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("fingerprint") != fingerprint:
        raise ValueError(f"checkpoint fingerprint mismatch for {case_name}")
    if payload.get("case") != case_name:
        raise ValueError(f"checkpoint case mismatch for {case_name}")
    settings = payload["settings"]
    return SyntheticValidation(
        metrics=tuple(ValidationMetric(**row) for row in payload["metrics"]),
        reliability=tuple(
            ReliabilityBin(**row) for row in payload["reliability"]
        ),
        target_false_positive_rate=settings["target_false_positive_rate"],
        delta_nu_tolerance=settings["delta_nu_tolerance"],
        calibration_realizations=settings["calibration_realizations"],
        evaluation_realizations=settings["evaluation_realizations"],
    )


def _checkpoint_name(case_name: str) -> str:
    digest = hashlib.sha256(case_name.encode("utf-8")).hexdigest()[:12]
    return f"{digest}.json"


def _write_tables(
    results: Sequence[SyntheticValidation],
    metrics_path: Path,
    reliability_path: Path,
) -> None:
    metric_rows = [asdict(row) for result in results for row in result.metrics]
    reliability_rows = [
        asdict(row) for row in _aggregate_reliability(results)
    ]
    _atomic_csv_write(
        metrics_path,
        metric_rows,
        [field.name for field in ValidationMetric.__dataclass_fields__.values()],
    )
    _atomic_csv_write(
        reliability_path,
        reliability_rows,
        [field.name for field in ReliabilityBin.__dataclass_fields__.values()],
    )


def _aggregate_reliability(
    results: Sequence[SyntheticValidation],
) -> tuple[ReliabilityBin, ...]:
    grouped: dict[tuple[str, str, float, float], list[ReliabilityBin]] = {}
    for result in results:
        for row in result.reliability:
            key = (row.method, row.split, row.lower, row.upper)
            grouped.setdefault(key, []).append(row)
    output = []
    for key, rows in sorted(grouped.items()):
        counts = np.asarray([row.sample_count for row in rows], dtype=float)
        sample_count = int(np.sum(counts))
        output.append(
            ReliabilityBin(
                method=key[0],
                split=key[1],
                lower=key[2],
                upper=key[3],
                mean_probability=float(
                    np.average(
                        [row.mean_probability for row in rows],
                        weights=counts,
                    )
                ),
                observed_frequency=float(
                    np.average(
                        [row.observed_frequency for row in rows],
                        weights=counts,
                    )
                ),
                sample_count=sample_count,
            )
        )
    return tuple(output)


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv_write(
    path: Path,
    rows: Sequence[dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)
