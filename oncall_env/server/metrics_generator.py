"""
Metrics generation for the incident triage environment.

Generates time-series metric data points for each service.
Healthy services get stable values; affected services show
anomalous patterns based on the scenario configuration.

All functions are pure — randomness injected via ``seed``.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from oncall_env.models import MetricDataPoint

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Healthy metric ranges per metric type
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HEALTHY_RANGES: Dict[str, Dict[str, float]] = {
    "cpu": {"min": 15.0, "max": 45.0},
    "memory": {"min": 40.0, "max": 65.0},
    "latency_p99": {"min": 50.0, "max": 200.0},
    "error_rate": {"min": 0.0, "max": 2.0},
    "connections": {"min": 5.0, "max": 25.0},
    "request_rate": {"min": 100.0, "max": 500.0},
}

ALL_METRIC_NAMES = list(HEALTHY_RANGES.keys())

# Time range → data point interval mapping
TIME_RANGE_CONFIG = {
    "last_5m": {"minutes": 5, "interval_s": 30},
    "last_15m": {"minutes": 15, "interval_s": 60},
    "last_1h": {"minutes": 60, "interval_s": 300},
}


def generate_metric_data(
    service: str,
    scenario_config: Dict[str, Any],
    metric_name: Optional[str] = None,
    time_range: str = "last_15m",
    seed: int = 0,
) -> Dict[str, List[MetricDataPoint]]:
    """Generate time-series metric data points for a service.

    Args:
        service: Service name to generate metrics for.
        scenario_config: Scenario dict with ``metric_config`` mapping
            service names to metric overrides, e.g.
            ``{"order-svc": {"cpu": {"min": 90, "max": 100}}}``.
        metric_name: Specific metric to return, or ``None`` for all.
        time_range: One of ``last_5m``, ``last_15m``, ``last_1h``.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping metric names to lists of ``MetricDataPoint``.
    """
    rng = random.Random(seed)
    cfg = TIME_RANGE_CONFIG.get(time_range, TIME_RANGE_CONFIG["last_15m"])
    window_minutes = cfg["minutes"]
    interval_s = cfg["interval_s"]

    base_time = datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
    start_time = base_time - timedelta(minutes=window_minutes)
    num_points = (window_minutes * 60) // interval_s

    # Get service-specific metric overrides from the scenario
    service_overrides = scenario_config.get("metric_config", {}).get(service, {})

    # Which metrics to generate
    metrics_to_generate = [metric_name] if metric_name else ALL_METRIC_NAMES

    result: Dict[str, List[MetricDataPoint]] = {}

    for m_name in metrics_to_generate:
        if m_name not in HEALTHY_RANGES:
            continue

        # Determine value range: scenario override or healthy default
        if m_name in service_overrides:
            val_range = service_overrides[m_name]
        else:
            val_range = HEALTHY_RANGES[m_name]

        vmin = val_range["min"]
        vmax = val_range["max"]

        # Generate data points with smooth progression
        points: List[MetricDataPoint] = []
        current_val = rng.uniform(vmin, vmax)

        for i in range(num_points):
            ts = start_time + timedelta(seconds=i * interval_s)

            # Smooth random walk within bounds
            drift = rng.gauss(0, (vmax - vmin) * 0.05)
            current_val = max(vmin, min(vmax, current_val + drift))

            # For anomalous metrics, ramp up toward the end of the window
            if m_name in service_overrides and i > num_points * 0.6:
                # Pull values toward the upper end of the anomalous range
                pull = (vmax - current_val) * 0.15
                current_val += pull

            points.append(MetricDataPoint(
                timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                value=round(current_val, 2),
            ))

        result[m_name] = points

    return result
