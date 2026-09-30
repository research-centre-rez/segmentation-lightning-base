from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path
from statistics import mean
from typing import Any

import psutil
import pynvml
import torch


class ResourceMonitor:
    """
    Monitor CPU, RAM, GPU utilization, and PyTorch CUDA memory.

    CPU and RAM are measured for the current Python process. If
    `include_children=True`, child processes such as PyTorch DataLoader
    workers are included as well.

    GPU utilization and device memory are obtained through NVML and therefore
    refer to the entire GPU.

    PyTorch allocated/reserved memory refers specifically to this process.
    """

    def __init__(
        self,
        interval: float = 0.5,
        gpu_id: int | None = None,
        include_children: bool = True,
    ) -> None:
        self.interval = interval
        self.include_children = include_children

        self.process = psutil.Process(os.getpid())

        if gpu_id is None and torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()

        self.gpu_id = gpu_id
        self.gpu_handle = None

        if self.gpu_id is not None:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)

        self.samples: list[dict[str, Any]] = []

        self._running = False
        self._thread: threading.Thread | None = None
        self._start_time: float | None = None
        self._end_time: float | None = None

    def _processes(self) -> list[psutil.Process]:
        processes = [self.process]

        if self.include_children:
            try:
                processes.extend(self.process.children(recursive=True))
            except psutil.Error:
                pass

        return processes

    def _cpu_percent(self) -> float:
        total = 0.0

        for process in self._processes():
            try:
                total += process.cpu_percent(interval=None)
            except psutil.Error:
                pass

        return total

    def _ram_bytes(self) -> int:
        total = 0

        for process in self._processes():
            try:
                total += process.memory_info().rss
            except psutil.Error:
                pass

        return total

    def _sample(self) -> None:
        # cpu_percent() needs one initial call to establish a baseline.
        for process in self._processes():
            try:
                process.cpu_percent(interval=None)
            except psutil.Error:
                pass

        while self._running:
            now = time.perf_counter()

            sample = {
                "elapsed_s": now - self._start_time,
                "cpu_percent": self._cpu_percent(),
                "ram_mb": self._ram_bytes() / 1024**2,
            }

            if self.gpu_handle is not None:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(
                    self.gpu_handle
                )
                memory = pynvml.nvmlDeviceGetMemoryInfo(
                    self.gpu_handle
                )

                sample.update(
                    {
                        "gpu_percent": utilization.gpu,
                        "gpu_memory_used_mb": memory.used / 1024**2,
                        "gpu_memory_total_mb": memory.total / 1024**2,
                    }
                )

            if (
                self.gpu_id is not None
                and torch.cuda.is_available()
            ):
                sample.update(
                    {
                        "torch_allocated_mb":
                            torch.cuda.memory_allocated(self.gpu_id)
                            / 1024**2,
                        "torch_reserved_mb":
                            torch.cuda.memory_reserved(self.gpu_id)
                            / 1024**2,
                    }
                )

            self.samples.append(sample)

            time.sleep(self.interval)

    def start(self) -> ResourceMonitor:
        self.samples.clear()

        if (
            self.gpu_id is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.reset_peak_memory_stats(self.gpu_id)

        self._start_time = time.perf_counter()
        self._end_time = None
        self._running = True

        self._thread = threading.Thread(
            target=self._sample,
            daemon=True,
        )
        self._thread.start()

        return self

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False

        if self._thread is not None:
            self._thread.join()

        # Synchronize before recording runtime so asynchronous CUDA work
        # has actually completed.
        if (
            self.gpu_id is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.gpu_id)

        self._end_time = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0

        end = (
            self._end_time
            if self._end_time is not None
            else time.perf_counter()
        )

        return end - self._start_time

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {
                "elapsed_s": self.elapsed_seconds,
            }

        result = {
            "elapsed_s": self.elapsed_seconds,
            "mean_cpu_percent": mean(
                x["cpu_percent"] for x in self.samples
            ),
            "max_cpu_percent": max(
                x["cpu_percent"] for x in self.samples
            ),
            "mean_ram_mb": mean(
                x["ram_mb"] for x in self.samples
            ),
            "max_ram_mb": max(
                x["ram_mb"] for x in self.samples
            ),
        }

        if "gpu_percent" in self.samples[0]:
            result.update(
                {
                    "mean_gpu_percent": mean(
                        x["gpu_percent"] for x in self.samples
                    ),
                    "max_gpu_percent": max(
                        x["gpu_percent"] for x in self.samples
                    ),
                    "max_gpu_memory_used_mb": max(
                        x["gpu_memory_used_mb"]
                        for x in self.samples
                    ),
                }
            )

        if (
            self.gpu_id is not None
            and torch.cuda.is_available()
        ):
            result.update(
                {
                    "torch_peak_allocated_mb":
                        torch.cuda.max_memory_allocated(
                            self.gpu_id
                        )
                        / 1024**2,
                    "torch_peak_reserved_mb":
                        torch.cuda.max_memory_reserved(
                            self.gpu_id
                        )
                        / 1024**2,
                }
            )

        return result

    def save_csv(self, path: str | Path) -> None:
        if not self.samples:
            raise RuntimeError("No samples have been recorded.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self.samples[0].keys(),
            )
            writer.writeheader()
            writer.writerows(self.samples)

    def __enter__(self) -> ResourceMonitor:
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def __del__(self) -> None:
        if self._running:
            self.stop()
