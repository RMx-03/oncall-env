"""
Scenario engine for the incident triage environment.

Manages scenario selection and loading.  Scenarios can come from:
  1. Built-in defaults (always available, one per difficulty)
  2. JSON files in the ``scenarios/`` directory (loaded lazily)

Selection is deterministic given a ``(task_id, seed)`` pair.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AVAILABLE_SERVICES: List[str] = [
    "api-gw",
    "order-svc",
    "order-db",
    "payment-svc",
    "auth-svc",
    "notification-svc",
    "cache",
    "search-svc",
]

SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

# Task → difficulty mapping
TASK_DIFFICULTY_MAP: Dict[str, str] = {
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Built-in scenarios (one per difficulty for testing)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_BUILTIN_SCENARIOS: Dict[str, List[Dict[str, Any]]] = {
    "easy": [
        {
            "scenario_id": "oom_crash",
            "difficulty": "easy",
            "title": "Order Service OOM Crash",
            "incident_summary": (
                "Multiple 5xx errors reported on /api/orders. "
                "The order-svc appears unresponsive and health checks "
                "are failing. Customer-facing impact confirmed."
            ),
            "ground_truth": {
                "severity": "P2",
                "root_cause": (
                    "Out of memory error in order-svc due to a memory leak "
                    "in the request handler causing the JVM to exceed its "
                    "heap limit and crash repeatedly."
                ),
                "root_cause_category": "resource",
                "remediation_action": "restart_service",
                "remediation_target": "order-svc",
            },
            "affected_services": ["order-svc"],
            "root_service": "order-svc",
            "service_states": {
                "api-gw": "degraded",
                "order-svc": "down",
                "order-db": "healthy",
                "payment-svc": "healthy",
                "auth-svc": "healthy",
                "notification-svc": "healthy",
                "cache": "healthy",
                "search-svc": "healthy",
            },
            "log_config": {
                "order-svc": {
                    "error_patterns": [
                        "java.lang.OutOfMemoryError: Java heap space",
                        "Service terminated — exit code 137 (OOM killed)",
                        "GC overhead limit exceeded — pausing all threads",
                        "Failed to allocate 256MB for new buffer pool",
                    ],
                    "warn_patterns": [
                        "Heap usage at 92% — approaching configured limit",
                        "GC pause time: 4500ms exceeds 1000ms threshold",
                        "Memory pool 'Old Gen' usage above 85%",
                    ],
                },
                "api-gw": {
                    "error_patterns": [
                        "Upstream service order-svc returned 502 Bad Gateway",
                        "Circuit breaker OPEN for upstream order-svc",
                    ],
                    "warn_patterns": [
                        "Upstream order-svc response time 12000ms exceeds 5000ms threshold",
                    ],
                },
            },
            "metric_config": {
                "order-svc": {
                    "cpu": {"min": 0.0, "max": 5.0},
                    "memory": {"min": 98.0, "max": 100.0},
                    "error_rate": {"min": 80.0, "max": 100.0},
                    "latency_p99": {"min": 5000.0, "max": 30000.0},
                    "request_rate": {"min": 0.0, "max": 10.0},
                },
                "api-gw": {
                    "error_rate": {"min": 15.0, "max": 35.0},
                    "latency_p99": {"min": 3000.0, "max": 12000.0},
                },
            },
            "alert_config": [
                {
                    "alert_id": "ALT-001",
                    "severity": "critical",
                    "service": "order-svc",
                    "title": "Service order-svc Not Responding",
                    "description": (
                        "Health check failed for order-svc. Last 5 probes "
                        "timed out. Service appears down."
                    ),
                    "fired_at": "2026-01-15T13:55:00Z",
                },
                {
                    "alert_id": "ALT-002",
                    "severity": "warning",
                    "service": "api-gw",
                    "title": "High Error Rate on API Gateway",
                    "description": (
                        "Error rate for /api/orders endpoint exceeded 15% "
                        "threshold. Currently at 28%."
                    ),
                    "fired_at": "2026-01-15T13:56:00Z",
                },
            ],
        },
    ],
    "medium": [
        {
            "scenario_id": "db_connection_pool_exhaustion",
            "difficulty": "medium",
            "title": "Database Connection Pool Exhaustion",
            "incident_summary": (
                "Intermittent 503 errors on order-related endpoints. "
                "Latency spikes observed across multiple services. "
                "Database team has not reported any issues."
            ),
            "ground_truth": {
                "severity": "P1",
                "root_cause": (
                    "PostgreSQL connection pool exhaustion on order-db "
                    "caused by a connection leak in order-svc after a "
                    "recent deployment. Connections are opened but never "
                    "returned to the pool, causing cascading failures."
                ),
                "root_cause_category": "database",
                "remediation_action": "failover_db",
                "remediation_target": "order-db",
            },
            "affected_services": ["order-db", "order-svc", "api-gw"],
            "root_service": "order-db",
            "service_states": {
                "api-gw": "degraded",
                "order-svc": "degraded",
                "order-db": "degraded",
                "payment-svc": "healthy",
                "auth-svc": "healthy",
                "notification-svc": "healthy",
                "cache": "healthy",
                "search-svc": "healthy",
            },
            "log_config": {
                "order-db": {
                    "error_patterns": [
                        "FATAL: remaining connection slots are reserved for superuser",
                        "Connection pool exhausted — max_connections=100 all in use",
                        "Unable to acquire connection within 30s timeout",
                        "pg_stat_activity shows 98 active connections (limit: 100)",
                    ],
                    "warn_patterns": [
                        "Connection pool utilization at 95% — approaching limit",
                        "Slow query detected: 8200ms for SELECT on orders table",
                        "Lock contention detected on table: order_items",
                    ],
                },
                "order-svc": {
                    "error_patterns": [
                        "Database connection timeout after 30000ms",
                        "Failed to execute query — no available connections",
                        "Transaction rolled back — connection pool timeout",
                    ],
                    "warn_patterns": [
                        "Retrying database connection (attempt 3/5)",
                        "Query latency 4500ms exceeds 2000ms SLA threshold",
                    ],
                },
                "api-gw": {
                    "error_patterns": [
                        "Upstream order-svc returned 503 Service Unavailable",
                    ],
                    "warn_patterns": [
                        "Retry budget exhausted for order-svc — returning 503",
                        "Circuit breaker HALF-OPEN for order-svc",
                    ],
                },
            },
            "metric_config": {
                "order-db": {
                    "connections": {"min": 90.0, "max": 100.0},
                    "cpu": {"min": 70.0, "max": 95.0},
                    "latency_p99": {"min": 3000.0, "max": 12000.0},
                },
                "order-svc": {
                    "error_rate": {"min": 20.0, "max": 50.0},
                    "latency_p99": {"min": 4000.0, "max": 15000.0},
                },
                "api-gw": {
                    "error_rate": {"min": 10.0, "max": 25.0},
                },
            },
            "alert_config": [
                {
                    "alert_id": "ALT-001",
                    "severity": "critical",
                    "service": "order-db",
                    "title": "Database Connection Pool Near Capacity",
                    "description": (
                        "PostgreSQL connections at 98/100. "
                        "New connections are being rejected."
                    ),
                    "fired_at": "2026-01-15T13:50:00Z",
                },
                {
                    "alert_id": "ALT-002",
                    "severity": "critical",
                    "service": "order-svc",
                    "title": "High Error Rate on Order Service",
                    "description": (
                        "Error rate exceeded 20% for order-svc. "
                        "Multiple database connection timeouts observed."
                    ),
                    "fired_at": "2026-01-15T13:52:00Z",
                },
                {
                    "alert_id": "ALT-003",
                    "severity": "warning",
                    "service": "api-gw",
                    "title": "Elevated 503 Responses",
                    "description": (
                        "503 response rate on /api/orders at 12%. "
                        "Upstream order-svc returning errors."
                    ),
                    "fired_at": "2026-01-15T13:53:00Z",
                },
            ],
        },
    ],
    "hard": [
        {
            "scenario_id": "dns_intermittent",
            "difficulty": "hard",
            "title": "Intermittent DNS Resolution Failures",
            "incident_summary": (
                "Sporadic timeout errors reported across multiple services. "
                "No single service appears fully down. Error rates are "
                "fluctuating unpredictably. Some requests succeed normally."
            ),
            "ground_truth": {
                "severity": "P2",
                "root_cause": (
                    "Intermittent DNS resolution failures caused by a "
                    "misconfigured ndots setting in the cluster DNS config. "
                    "Service-to-service calls randomly fail when the DNS "
                    "resolver times out, causing cascading partial failures."
                ),
                "root_cause_category": "network",
                "remediation_action": "fix_config",
                "remediation_target": "api-gw",
            },
            "affected_services": [
                "api-gw", "order-svc", "payment-svc",
                "notification-svc", "search-svc",
            ],
            "root_service": "api-gw",
            "service_states": {
                "api-gw": "degraded",
                "order-svc": "degraded",
                "order-db": "healthy",
                "payment-svc": "degraded",
                "auth-svc": "healthy",
                "notification-svc": "degraded",
                "cache": "healthy",
                "search-svc": "degraded",
            },
            "log_config": {
                "api-gw": {
                    "error_patterns": [
                        "DNS resolution failed for order-svc.default.svc.cluster.local",
                        "Temporary failure in name resolution — retrying",
                        "Connection timed out: failed to resolve upstream hostname",
                        "getaddrinfo ETIMEOUT order-svc.default.svc.cluster.local",
                    ],
                    "warn_patterns": [
                        "DNS lookup took 4800ms (threshold: 1000ms)",
                        "Upstream health check flapping for order-svc",
                        "ndots:5 search path expansion causing excess DNS queries",
                    ],
                },
                "order-svc": {
                    "error_patterns": [
                        "Failed to connect to payment-svc: Name resolution timeout",
                        "DNS SERVFAIL for payment-svc.default.svc.cluster.local",
                    ],
                    "warn_patterns": [
                        "Retrying RPC to payment-svc after DNS failure (attempt 2/3)",
                    ],
                },
                "payment-svc": {
                    "error_patterns": [
                        "DNS resolution timeout for notification-svc",
                        "Connection to notification-svc failed: ETIMEOUT",
                    ],
                    "warn_patterns": [],
                },
                "notification-svc": {
                    "error_patterns": [
                        "Failed to resolve smtp.example.com — DNS timeout",
                    ],
                    "warn_patterns": [
                        "Email delivery delayed due to DNS resolution issues",
                    ],
                },
                "search-svc": {
                    "error_patterns": [
                        "Elasticsearch cluster node discovery failed — DNS timeout",
                    ],
                    "warn_patterns": [
                        "Index sync delayed: unable to resolve elasticsearch.default.svc",
                    ],
                },
                # RED HERRING: auth-svc has a few warn logs unrelated to DNS
                "auth-svc": {
                    "error_patterns": [],
                    "warn_patterns": [
                        "Token refresh rate spike detected — 3x normal volume",
                        "OAuth2 nonce cache size approaching limit",
                    ],
                },
            },
            "metric_config": {
                "api-gw": {
                    "error_rate": {"min": 5.0, "max": 18.0},
                    "latency_p99": {"min": 2000.0, "max": 8000.0},
                },
                "order-svc": {
                    "error_rate": {"min": 3.0, "max": 12.0},
                    "latency_p99": {"min": 1500.0, "max": 6000.0},
                },
                "payment-svc": {
                    "error_rate": {"min": 2.0, "max": 10.0},
                },
                "notification-svc": {
                    "error_rate": {"min": 4.0, "max": 15.0},
                },
                "search-svc": {
                    "error_rate": {"min": 3.0, "max": 8.0},
                },
            },
            "alert_config": [
                {
                    "alert_id": "ALT-001",
                    "severity": "warning",
                    "service": "api-gw",
                    "title": "Intermittent Gateway Errors",
                    "description": (
                        "Sporadic 502/504 errors on multiple endpoints. "
                        "Error rate fluctuating between 5-18%."
                    ),
                    "fired_at": "2026-01-15T13:40:00Z",
                },
                {
                    "alert_id": "ALT-002",
                    "severity": "warning",
                    "service": "order-svc",
                    "title": "Order Service Intermittent Failures",
                    "description": (
                        "Order processing failing intermittently. "
                        "Some requests succeed, others time out."
                    ),
                    "fired_at": "2026-01-15T13:42:00Z",
                },
                {
                    "alert_id": "ALT-003",
                    "severity": "info",
                    "service": "auth-svc",
                    "title": "Elevated Token Refresh Rate",
                    "description": (
                        "Token refresh requests 3x above baseline. "
                        "May indicate downstream retry storms."
                    ),
                    "fired_at": "2026-01-15T13:45:00Z",
                },
                {
                    "alert_id": "ALT-004",
                    "severity": "warning",
                    "service": "payment-svc",
                    "title": "Payment Processing Delays",
                    "description": (
                        "Payment confirmation latency increased. "
                        "Sporadic connection timeouts reported."
                    ),
                    "fired_at": "2026-01-15T13:48:00Z",
                },
            ],
        },
    ],
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scenario Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ScenarioEngine:
    """Loads and selects incident scenarios for the environment.

    Scenarios are loaded from JSON files when available; otherwise
    built-in defaults are used.  Selection is deterministic given
    the same ``(task_id, seed)`` pair.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def _load_scenarios(self, difficulty: str) -> List[Dict[str, Any]]:
        """Load scenarios for a difficulty level.

        Checks JSON files first, then falls back to built-in defaults.

        Args:
            difficulty: One of ``easy``, ``medium``, ``hard``.

        Returns:
            List of scenario configuration dicts.

        Raises:
            ValueError: If no scenarios found for the difficulty.
        """
        if difficulty in self._cache:
            return self._cache[difficulty]

        scenarios: List[Dict[str, Any]] = []

        # Try loading from JSON file
        json_path = SCENARIOS_DIR / f"{difficulty}_scenarios.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    scenarios.extend(loaded)
                elif isinstance(loaded, dict) and "scenarios" in loaded:
                    scenarios.extend(loaded["scenarios"])

        # Fall back to built-in scenarios
        if not scenarios:
            scenarios = _BUILTIN_SCENARIOS.get(difficulty, [])

        if not scenarios:
            raise ValueError(
                f"No scenarios found for difficulty '{difficulty}'. "
                f"Checked: {json_path} and built-in defaults."
            )

        self._cache[difficulty] = scenarios
        return scenarios

    def select_scenario(
        self,
        task_id: str,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Select a scenario for the given task and seed.

        Args:
            task_id: Task identifier (``easy``, ``medium``, ``hard``).
            seed: Random seed for deterministic selection.
                  ``None`` means random choice.

        Returns:
            A complete scenario configuration dict.

        Raises:
            ValueError: If task_id is not recognized.
        """
        difficulty = TASK_DIFFICULTY_MAP.get(task_id)
        if difficulty is None:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid values: {list(TASK_DIFFICULTY_MAP.keys())}"
            )

        scenarios = self._load_scenarios(difficulty)

        if seed is not None:
            idx = seed % len(scenarios)
        else:
            idx = random.randint(0, len(scenarios) - 1)

        return scenarios[idx]

    def list_task_ids(self) -> List[str]:
        """Return all valid task IDs."""
        return list(TASK_DIFFICULTY_MAP.keys())
