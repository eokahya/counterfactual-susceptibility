"""Native-MPS memory, host-memory, swap, and thermal safety sampler."""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cfsus.reproduction.artifacts import write_json_atomic


def _run_text(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env={
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GIT_")
        },
    )
    return result.stdout.strip()


def swap_used_bytes() -> int:
    output = _run_text(["sysctl", "vm.swapusage"])
    match = re.search(r"used = ([0-9.]+)M", output)
    if match is None:
        raise RuntimeError("swap telemetry is unavailable")
    return round(float(match.group(1)) * 1024**2)


def thermal_state() -> str:
    output = _run_text(["pmset", "-g", "therm"]).casefold()
    if "serious" in output or "critical" in output:
        return "serious_or_critical"
    if "no thermal warning level has been recorded" in output:
        return "nominal"
    if "fair" in output:
        return "fair"
    return "unknown"


@dataclass(frozen=True, slots=True)
class MPSSample:
    unix_time: float
    stage: str
    mps_current_bytes: int
    mps_driver_bytes: int
    process_rss_bytes: int
    available_memory_bytes: int
    swap_used_bytes: int
    thermal_state: str


class MPSTelemetrySampler:
    """Sample in a background thread and terminate on a frozen safety breach."""

    def __init__(
        self,
        torch: Any,
        limits: Mapping[str, Any],
        emergency_path: Path,
    ) -> None:
        self.torch = torch
        self.limits = limits
        self.emergency_path = emergency_path
        self.interval = float(limits["sample_interval_seconds"])
        self.started_at = time.time()
        self.swap_start = swap_used_bytes()
        self._stage = "worker_start"
        self._samples: list[MPSSample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._emergency_sent = False
        self._telemetry_failures = 0
        self.sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        with self._lock:
            self._stage = name
        self.sample()
        try:
            yield
        finally:
            self.sample()

    def _violations(self, sample: MPSSample) -> list[str]:
        violations: list[str] = []
        if sample.mps_driver_bytes > int(self.limits["maximum_mps_driver_bytes"]):
            violations.append("maximum_mps_driver_bytes")
        if sample.process_rss_bytes > int(self.limits["maximum_process_rss_bytes"]):
            violations.append("maximum_process_rss_bytes")
        if sample.swap_used_bytes - self.swap_start > int(
            self.limits["maximum_swap_growth_bytes"]
        ):
            violations.append("maximum_swap_growth_bytes")
        if sample.available_memory_bytes < int(
            self.limits["minimum_available_memory_bytes"]
        ):
            violations.append("minimum_available_memory_bytes")
        if sample.thermal_state not in set(self.limits["accepted_thermal_states"]):
            violations.append("accepted_thermal_states")
        return violations

    def _emergency(self, violations: list[str], detail: str | None = None) -> None:
        if self._emergency_sent:
            return
        self._emergency_sent = True
        write_json_atomic(
            self.emergency_path,
            {
                "schema_version": 1,
                "artifact_type": "stage1b_measurement_worker_emergency",
                "violations": violations,
                "detail_class": detail,
                "sample_count": len(self._samples),
                "last_sample": (asdict(self._samples[-1]) if self._samples else None),
            },
        )
        os.kill(os.getpid(), signal.SIGTERM)

    def sample(self) -> None:
        import psutil  # type: ignore[import-untyped]

        with self._lock:
            stage = self._stage
        sample = MPSSample(
            unix_time=time.time(),
            stage=stage,
            mps_current_bytes=int(self.torch.mps.current_allocated_memory()),
            mps_driver_bytes=int(self.torch.mps.driver_allocated_memory()),
            process_rss_bytes=int(psutil.Process().memory_info().rss),
            available_memory_bytes=int(psutil.virtual_memory().available),
            swap_used_bytes=swap_used_bytes(),
            thermal_state=thermal_state(),
        )
        with self._lock:
            self._samples.append(sample)
        violations = self._violations(sample)
        if violations:
            self._emergency(violations)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample()
                self._telemetry_failures = 0
            except BaseException as error:
                self._telemetry_failures += 1
                if self._telemetry_failures >= int(
                    self.limits["telemetry_failure_limit"]
                ):
                    self._emergency(["telemetry_failure_limit"], type(error).__name__)

    def finish(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=5)
        self.sample()
        with self._lock:
            samples = list(self._samples)
        if not samples:
            raise RuntimeError("telemetry sampler produced no samples")
        peak_metrics = (
            "mps_current_bytes",
            "mps_driver_bytes",
            "process_rss_bytes",
            "swap_used_bytes",
        )
        stage_peaks: dict[str, dict[str, int]] = {}
        for stage in sorted({item.stage for item in samples}):
            selected = [item for item in samples if item.stage == stage]
            stage_peaks[stage] = {
                metric: max(int(getattr(item, metric)) for item in selected)
                for metric in peak_metrics
            }
            stage_peaks[stage]["swap_growth_bytes"] = max(
                0, stage_peaks[stage]["swap_used_bytes"] - self.swap_start
            )
            stage_peaks[stage]["minimum_available_memory_bytes"] = min(
                item.available_memory_bytes for item in selected
            )
        attempt_peaks = {
            metric: max(int(getattr(item, metric)) for item in samples)
            for metric in peak_metrics
        }
        attempt_peaks["swap_growth_bytes"] = max(
            0, attempt_peaks["swap_used_bytes"] - self.swap_start
        )
        attempt_peaks["minimum_available_memory_bytes"] = min(
            item.available_memory_bytes for item in samples
        )
        violations = sorted(
            {violation for sample in samples for violation in self._violations(sample)}
        )
        return {
            "started_at_unix": self.started_at,
            "finished_at_unix": time.time(),
            "sample_count": len(samples),
            "sampling_interval_seconds": self.interval,
            "attempt_peaks": attempt_peaks,
            "stage_peaks": stage_peaks,
            "thermal_states": sorted({item.thermal_state for item in samples}),
            "violations": violations,
            "telemetry_failures": self._telemetry_failures,
        }


__all__ = [
    "MPSSample",
    "MPSTelemetrySampler",
    "swap_used_bytes",
    "thermal_state",
]
