"""High-resolution inference instrumentation for the vllama engine.

Metrics: TTFT (prefill latency), ITL (decode latency), MBU (model bandwidth
utilization), KV-cache memory efficiency, and throughput (tokens/sec/GPU).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class InferenceMetrics:
    ttft_ms: float = 0.0
    avg_itl_ms: float = 0.0
    tokens_generated: int = 0
    throughput_tokens_per_sec: float = 0.0
    mbu_pct: float = 0.0
    kv_cache_efficiency_pct: float = 0.0
    peak_memory_mb: float = 0.0
    device_name: str = ""
    peak_bandwidth_gbps: float = 0.0
    model_params: int = 0


# Peak theoretical memory bandwidth (GB/s) for common accelerators. Used as the
# MBU denominator; unknown devices fall back to a measured microbenchmark.
_PEAK_BANDWIDTH_GBPS = {
    "h100": 3350.0,
    "a100": 1555.0,
    "rtx 4090": 1008.0,
    "rtx 3090": 936.2,
    "v100": 900.0,
    "rtx a6000": 768.0,
    "rtx 3080": 760.3,
    "rtx 4080": 716.8,
    "rtx 4070": 504.2,
}


class ServingPerformanceProfiler:
    """Times the compute-bound prefill (TTFT) vs memory-bound decode (ITL)
    using hardware stream synchronization and a high-resolution monotonic clock."""

    def __init__(self, device: torch.device, model: Optional[torch.nn.Module] = None):
        self.device = device
        self.model = model
        self._is_cuda = device.type == "cuda"
        self.peak_bandwidth_gbps = self._resolve_peak_bandwidth()

    def _sync(self) -> None:
        if self._is_cuda:
            torch.cuda.synchronize(self.device)

    def _resolve_peak_bandwidth(self) -> float:
        if self._is_cuda:
            name = torch.cuda.get_device_name(self.device).lower()
            for key, bw in _PEAK_BANDWIDTH_GBPS.items():
                if key in name:
                    return bw
        return self._measure_peak_bandwidth()

    def _measure_peak_bandwidth(self, n_bytes: int = 256 * 1024 * 1024) -> float:
        try:
            n_elems = max(1, n_bytes // 4)
            buf = torch.empty(n_elems, dtype=torch.float32, device=self.device)
            self._sync()
            t0 = time.perf_counter_ns()
            for _ in range(3):
                buf.sum()
            self._sync()
            dt_sec = (time.perf_counter_ns() - t0) / 1e9
            bytes_read = buf.numel() * buf.element_size() * 3
            return bytes_read / dt_sec / 1e9 if dt_sec > 0 else 0.0
        except Exception:
            return 0.0

    def model_weight_bytes(self) -> float:
        if self.model is None:
            return 0.0
        return float(sum(p.numel() * p.element_size() for p in self.model.parameters()))

    def profile_generation_run(self, stream: Iterator[int]) -> tuple[list[int], InferenceMetrics]:
        if self._is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

        metrics = InferenceMetrics()
        if self.model is not None:
            metrics.model_params = int(sum(p.numel() for p in self.model.parameters()))
        metrics.device_name = torch.cuda.get_device_name(self.device) if self._is_cuda else "cpu"
        metrics.peak_bandwidth_gbps = self.peak_bandwidth_gbps

        tokens: list[int] = []
        decode_latencies: list[float] = []

        # 1) Compute-bound prefill + first token (TTFT).
        self._sync()
        t0 = time.perf_counter_ns()
        first = next(stream, None)
        self._sync()
        metrics.ttft_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        if first is not None:
            tokens.append(first)
            metrics.tokens_generated += 1

        # 2) Memory-bound decode steps (ITL).
        while True:
            self._sync()
            t_step = time.perf_counter_ns()
            token = next(stream, None)
            self._sync()
            step_ms = (time.perf_counter_ns() - t_step) / 1_000_000.0
            if token is None:
                break
            tokens.append(token)
            metrics.tokens_generated += 1
            decode_latencies.append(step_ms)

        # 3) Compile diagnostics.
        if decode_latencies:
            metrics.avg_itl_ms = sum(decode_latencies) / len(decode_latencies)
            total_decode_sec = sum(decode_latencies) / 1000.0
            if total_decode_sec > 0:
                metrics.throughput_tokens_per_sec = len(decode_latencies) / total_decode_sec
            weight_bytes = self.model_weight_bytes()
            if weight_bytes > 0 and metrics.avg_itl_ms > 0 and self.peak_bandwidth_gbps > 0:
                achieved_gbps = (weight_bytes / 1e9) / (metrics.avg_itl_ms / 1000.0)
                metrics.mbu_pct = achieved_gbps / self.peak_bandwidth_gbps * 100.0

        if self._is_cuda:
            metrics.peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)

        # Overlay telemetry captured by ``generate_stream`` onto the report
        # (KV-cache efficiency + peak memory stored on the model).
        if self.model is not None and hasattr(self.model, "profiler_metrics"):
            pm = self.model.profiler_metrics
            metrics.kv_cache_efficiency_pct = float(pm.get("kv_cache_efficiency", 0.0))
            metrics.peak_memory_mb = float(pm.get("peak_memory_mb", metrics.peak_memory_mb))

        return tokens, metrics


def format_metrics(m: InferenceMetrics) -> str:
    lines = [
        "=" * 60,
        " INFERENCE METRICS",
        "=" * 60,
        f" Device              {m.device_name}",
        f" Model params        {m.model_params:,}",
        f" Peak bandwidth      {m.peak_bandwidth_gbps:,.0f} GB/s",
        "-" * 60,
        f" TTFT (prefill)      {m.ttft_ms:10.2f} ms",
        f" Avg ITL (decode)    {m.avg_itl_ms:10.2f} ms/token",
        f" Tokens generated    {m.tokens_generated:10d}",
        f" Throughput          {m.throughput_tokens_per_sec:10.2f} tok/s",
        f" MBU                 {m.mbu_pct:10.2f} %",
        f" KV-cache efficiency {m.kv_cache_efficiency_pct:10.2f} %",
        f" Peak memory         {m.peak_memory_mb:10.2f} MB",
        "=" * 60,
    ]
    return "\n".join(lines)
