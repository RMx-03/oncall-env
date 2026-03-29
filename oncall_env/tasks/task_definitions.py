"""
Task definitions for the Incident Triage Environment.

Defines 3 difficulty levels (easy, medium, hard) with metadata
describing characteristics and expected agent performance.

These are static configuration — NO logic, NO imports from server/.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, FrozenSet, List, Mapping

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task configurations (frozen dicts — immutable at runtime)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TASKS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    # ─────────────────── EASY ───────────────────
    "easy": MappingProxyType({
        "task_id": "easy",
        "name": "Single Service Failure",
        "description": (
            "A single service is clearly down. Logs contain obvious error "
            "messages. One or two alerts firing. The agent must identify "
            "which service failed, classify the severity, and restart it."
        ),
        "difficulty": "easy",
        "expected_score_range": "0.6 — 1.0 for capable models",
        "scenarios": ("oom_crash",),
        "characteristics": MappingProxyType({
            "num_affected_services": 1,
            "red_herrings": False,
            "cascading_failure": False,
            "ambiguous_logs": False,
        }),
    }),

    # ─────────────────── MEDIUM ───────────────────
    "medium": MappingProxyType({
        "task_id": "medium",
        "name": "Multi-Service Cascading Issue",
        "description": (
            "Multiple services are showing errors. The root cause is in "
            "one service causing cascading failures to downstream dependants. "
            "Logs contain some noise. The agent must correlate across "
            "services to identify the true root service."
        ),
        "difficulty": "medium",
        "expected_score_range": "0.3 — 0.7 for capable models",
        "scenarios": ("db_connection_pool_exhaustion",),
        "characteristics": MappingProxyType({
            "num_affected_services": 3,
            "red_herrings": True,
            "cascading_failure": True,
            "ambiguous_logs": False,
        }),
    }),

    # ─────────────────── HARD ───────────────────
    "hard": MappingProxyType({
        "task_id": "hard",
        "name": "Subtle Degradation with Red Herrings",
        "description": (
            "Intermittent failures across the system. Multiple alerts from "
            "different services. The root cause is subtle (e.g., DNS resolution "
            "issue). Logs contain red herrings and ambiguous messages. "
            "The agent must reason carefully to distinguish symptoms from causes."
        ),
        "difficulty": "hard",
        "expected_score_range": "0.1 — 0.4 for frontier models",
        "scenarios": ("dns_intermittent",),
        "characteristics": MappingProxyType({
            "num_affected_services": 5,
            "red_herrings": True,
            "cascading_failure": True,
            "ambiguous_logs": True,
        }),
    }),
})


def get_task(task_id: str) -> Mapping[str, Any]:
    """Retrieve a task configuration by ID.

    Args:
        task_id: One of ``easy``, ``medium``, ``hard``.

    Returns:
        Frozen task configuration dict.

    Raises:
        ValueError: If task_id is not recognized.
    """
    if task_id not in TASKS:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Valid values: {list(TASKS.keys())}"
        )
    return TASKS[task_id]


def list_task_ids() -> List[str]:
    """Return all valid task IDs."""
    return list(TASKS.keys())
