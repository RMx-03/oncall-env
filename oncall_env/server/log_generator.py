"""
Log line generation for the incident triage environment.

Generates realistic, chronologically-ordered log entries for each
service based on the active scenario configuration.

All functions are pure — randomness injected via ``seed``.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from oncall_env.models import LogEntry

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Healthy (noise) log templates per service
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HEALTHY_TEMPLATES: Dict[str, List[str]] = {
    "api-gw": [
        "Request GET /api/orders completed in {latency}ms — 200 OK",
        "Request POST /api/orders completed in {latency}ms — 201 Created",
        "Rate limit check passed for client client-{cid}",
        "TLS handshake completed for api.example.com",
        "Health check probe returned 200 for upstream {upstream}",
        "Routing request to upstream {upstream}",
    ],
    "order-svc": [
        "Order ORD-{oid} created successfully",
        "Processing order ORD-{oid} — validating inventory",
        "Payment authorized for order ORD-{oid}",
        "Order ORD-{oid} dispatched to fulfillment queue",
        "Inventory check passed for SKU SKU-{sku}",
        "Database query completed in {latency}ms",
    ],
    "order-db": [
        "Connection pool stats: active={active}, idle={idle}, total=20",
        "Query executed in {latency}ms: SELECT * FROM orders WHERE id = ?",
        "Checkpoint completed — WAL flushed to disk",
        "Autovacuum running on table: orders",
        "Replication lag: {lag}ms",
    ],
    "payment-svc": [
        "Payment PAY-{pid} processed via Stripe — success",
        "Idempotency key validated for request req-{rid}",
        "Webhook delivered to merchant callback URL",
        "Currency conversion EUR→USD applied at rate 1.08",
        "Fraud check passed for transaction PAY-{pid}",
    ],
    "auth-svc": [
        "Token issued for user usr-{uid} — expires in 3600s",
        "JWT validated successfully for request to /api/orders",
        "OAuth2 refresh token rotated for client app-{cid}",
        "Rate limit check passed for login endpoint",
        "Session cache hit for token tok-{tid}",
    ],
    "notification-svc": [
        "Email queued for recipient user-{uid}@example.com",
        "Push notification sent to device dev-{did}",
        "SMS delivered to +1-555-{phone}",
        "Notification template rendered in {latency}ms",
        "Batch processing: {count} notifications dispatched",
    ],
    "cache": [
        "Cache HIT for key user:{uid}:profile (TTL {ttl}s remaining)",
        "Cache MISS for key order:{oid}:status — fetching from origin",
        "Eviction: LRU removed {count} keys — memory at {mem}%",
        "Connected clients: {clients}, used memory: {mem}MB",
        "Key space: {keys} keys across 16 databases",
    ],
    "search-svc": [
        "Search query completed in {latency}ms — {hits} hits",
        "Index refresh completed for index: products",
        "Bulk indexing: {count} documents processed",
        "Cluster health: green — {shards} active shards",
        "Query cache hit rate: {rate}%",
    ],
}

# Fallback for unknown services
_DEFAULT_HEALTHY = [
    "Request processed successfully in {latency}ms",
    "Health check passed",
    "Background task completed",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Time range → log count mapping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIME_RANGE_CONFIG = {
    "last_5m": {"minutes": 5, "log_count": 12},
    "last_15m": {"minutes": 15, "log_count": 25},
    "last_1h": {"minutes": 60, "log_count": 50},
}


def _fill_template(template: str, rng: random.Random) -> str:
    """Fill a log template with random but realistic values."""
    replacements = {
        "{latency}": str(rng.randint(5, 250)),
        "{cid}": str(rng.randint(1000, 9999)),
        "{oid}": str(rng.randint(10000, 99999)),
        "{sku}": str(rng.randint(100, 999)),
        "{pid}": str(rng.randint(10000, 99999)),
        "{rid}": str(rng.randint(1000, 9999)),
        "{uid}": str(rng.randint(1000, 9999)),
        "{tid}": str(rng.randint(10000, 99999)),
        "{did}": str(rng.randint(100, 999)),
        "{phone}": f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}",
        "{count}": str(rng.randint(10, 500)),
        "{ttl}": str(rng.randint(60, 3600)),
        "{mem}": str(rng.randint(40, 75)),
        "{clients}": str(rng.randint(20, 150)),
        "{keys}": str(rng.randint(5000, 50000)),
        "{hits}": str(rng.randint(0, 200)),
        "{shards}": str(rng.randint(5, 20)),
        "{rate}": str(rng.randint(70, 99)),
        "{active}": str(rng.randint(3, 8)),
        "{idle}": str(rng.randint(10, 17)),
        "{lag}": str(rng.randint(1, 50)),
        "{upstream}": rng.choice([
            "order-svc", "payment-svc", "auth-svc", "search-svc",
        ]),
    }
    result = template
    for placeholder, value in replacements.items():
        result = result.replace(placeholder, value)
    return result


def generate_log_entries(
    service: str,
    scenario_config: Dict[str, Any],
    time_range: str = "last_15m",
    seed: int = 0,
) -> List[LogEntry]:
    """Generate realistic log entries for a service.

    Args:
        service: Service name to generate logs for.
        scenario_config: Scenario dict with ``log_config`` mapping
            service names to ``{"error_patterns": [...], "warn_patterns": [...]}``.
        time_range: One of ``last_5m``, ``last_15m``, ``last_1h``.
        seed: Random seed for reproducibility.

    Returns:
        Chronologically sorted list of ``LogEntry`` objects.
    """
    rng = random.Random(seed)
    cfg = TIME_RANGE_CONFIG.get(time_range, TIME_RANGE_CONFIG["last_15m"])
    total_logs = cfg["log_count"]
    window_minutes = cfg["minutes"]

    # Base timestamp: use a fixed reproducible time
    base_time = datetime(2026, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
    start_time = base_time - timedelta(minutes=window_minutes)

    # Get service-specific log config from the scenario
    log_cfg = scenario_config.get("log_config", {}).get(service, {})
    error_patterns: List[str] = log_cfg.get("error_patterns", [])
    warn_patterns: List[str] = log_cfg.get("warn_patterns", [])

    # Determine the split: affected services get ~40% error logs
    is_affected = bool(error_patterns or warn_patterns)
    if is_affected:
        error_count = max(1, int(total_logs * 0.35))
        warn_count = max(1, int(total_logs * 0.15)) if warn_patterns else 0
        healthy_count = total_logs - error_count - warn_count
    else:
        error_count = 0
        warn_count = 0
        healthy_count = total_logs

    entries: List[LogEntry] = []
    templates = HEALTHY_TEMPLATES.get(service, _DEFAULT_HEALTHY)

    # Generate healthy (INFO) logs
    for _ in range(healthy_count):
        offset = rng.uniform(0, window_minutes * 60)
        ts = start_time + timedelta(seconds=offset)
        entries.append(LogEntry(
            timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            service=service,
            level="INFO",
            message=_fill_template(rng.choice(templates), rng),
        ))

    # Generate ERROR logs
    for _ in range(error_count):
        # Errors cluster toward the end of the window (incident onset)
        offset = rng.uniform(window_minutes * 30, window_minutes * 60)
        ts = start_time + timedelta(seconds=offset)
        entries.append(LogEntry(
            timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            service=service,
            level=rng.choice(["ERROR", "FATAL"]) if rng.random() < 0.2 else "ERROR",
            message=rng.choice(error_patterns),
        ))

    # Generate WARN logs
    for _ in range(warn_count):
        # Warnings appear slightly before errors
        offset = rng.uniform(window_minutes * 20, window_minutes * 55)
        ts = start_time + timedelta(seconds=offset)
        entries.append(LogEntry(
            timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            service=service,
            level="WARN",
            message=rng.choice(warn_patterns),
        ))

    # Sort chronologically
    entries.sort(key=lambda e: e.timestamp)
    return entries
