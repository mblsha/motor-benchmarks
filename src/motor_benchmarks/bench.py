"""Core measurement primitives for reusable motor benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import time

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PwmCommand:
    """Generic PWM-style command for motor drivers."""

    duty: float
    freq_hz: int | None = None
    motor_volts: float | None = None


@dataclass(frozen=True)
class TachMeasurement:
    """Structured output from a tachometer measurement."""

    rpm_df: pd.DataFrame
    capture_dir: Path | None
    rpm_file: Path | None
    diag: dict[str, Any]
    motor_voltage_mean: float | None = None


@dataclass(frozen=True)
class PointResult:
    """Single datapoint result with summary stats."""

    cmd: PwmCommand
    mean_rpm: float
    std_rpm: float
    min_rpm: float
    max_rpm: float
    median_rpm: float
    cv_rpm: float
    n_rpm: int
    diag: dict[str, Any]
    motor_voltage_mean: float | None = None
    capture_dir: Path | None = None
    rpm_file: Path | None = None

    def summary_dict(self, run_dir: Path | None = None) -> dict[str, Any]:
        """Return a summary row suitable for CSV output."""

        def _rel(path: Path | None) -> str | None:
            if path is None:
                return None
            if run_dir is None:
                return str(path)
            try:
                return str(path.relative_to(run_dir))
            except ValueError:
                return str(path)

        summary = {
            "duty_cycle": self.cmd.duty,
            "mean_rpm": self.mean_rpm,
            "std_rpm": self.std_rpm,
            "min_rpm": self.min_rpm,
            "max_rpm": self.max_rpm,
            "median_rpm": self.median_rpm,
            "cv_rpm": self.cv_rpm,
            "n_rpm": self.n_rpm,
            "capture_dir": _rel(self.capture_dir),
            "rpm_file": _rel(self.rpm_file),
        }
        if self.motor_voltage_mean is not None:
            summary["motor_voltage_mean"] = self.motor_voltage_mean
        if self.cmd.freq_hz is not None:
            summary["pwm_freq_hz"] = self.cmd.freq_hz
        if self.cmd.motor_volts is not None:
            summary["motor_volts"] = self.cmd.motor_volts
        return summary


class Motor(Protocol):
    """Hardware driver interface for motors."""

    def connect(self) -> None: ...
    def apply(self, cmd: PwmCommand) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
    def healthcheck(self) -> dict[str, Any]: ...


class Tachometer(Protocol):
    """Hardware driver interface for tachometers."""

    def connect(self) -> None: ...
    def measure(
        self,
        *,
        duration_s: float,
        capture_dir: Path,
        rpm_file: Path,
    ) -> TachMeasurement: ...
    def close(self) -> None: ...
    def healthcheck(self) -> dict[str, Any]: ...


def rpm_stats(rpm_df: pd.DataFrame) -> dict[str, float | int]:
    """Compute summary RPM statistics from a dataframe."""
    if rpm_df.empty or "rpm" not in rpm_df.columns:
        return {
            "mean_rpm": 0.0,
            "std_rpm": 0.0,
            "min_rpm": 0.0,
            "max_rpm": 0.0,
            "median_rpm": 0.0,
            "cv_rpm": 0.0,
            "n_rpm": 0,
        }

    rpm = rpm_df["rpm"].dropna().to_numpy(dtype=float)
    if rpm.size == 0:
        return {
            "mean_rpm": 0.0,
            "std_rpm": 0.0,
            "min_rpm": 0.0,
            "max_rpm": 0.0,
            "median_rpm": 0.0,
            "cv_rpm": 0.0,
            "n_rpm": 0,
        }

    mean = float(rpm.mean())
    std = float(rpm.std(ddof=1)) if rpm.size > 1 else 0.0
    median = float(np.median(rpm))
    cv = float(std / mean * 100.0) if mean > 0 else 0.0
    return {
        "mean_rpm": mean,
        "std_rpm": std,
        "min_rpm": float(rpm.min()),
        "max_rpm": float(rpm.max()),
        "median_rpm": median,
        "cv_rpm": cv,
        "n_rpm": int(rpm.size),
    }


def measure_point(
    motor: Motor,
    tach: Tachometer,
    cmd: PwmCommand,
    *,
    settle_s: float,
    capture_s: float,
    capture_dir: Path,
    rpm_file: Path,
) -> PointResult:
    """Measure a single datapoint (the core primitive)."""
    if not 0.0 <= cmd.duty <= 1.0:
        raise ValueError(f"duty must be between 0.0 and 1.0, got {cmd.duty}")

    motor.apply(cmd)
    time.sleep(settle_s)

    measurement = tach.measure(
        duration_s=capture_s,
        capture_dir=capture_dir,
        rpm_file=rpm_file,
    )

    stats = rpm_stats(measurement.rpm_df)
    return PointResult(
        cmd=cmd,
        diag=measurement.diag,
        motor_voltage_mean=measurement.motor_voltage_mean,
        capture_dir=measurement.capture_dir,
        rpm_file=measurement.rpm_file,
        **stats,
    )
