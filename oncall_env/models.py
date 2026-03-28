"""
Pydantic models for the Incident Triage OpenEnv Environment.

Defines the complete Action/Observation/State contract between
the AI agent and the simulated production incident environment.

Models:
    IncidentAction      — What the agent can do (8 action types)
    IncidentObservation  — What the agent sees after each step
    IncidentState        — Internal episode metadata (hidden from agent)
    LogEntry             — A single structured log line
    MetricDataPoint      — A single time-series metric value
    Alert                — A monitoring alert
    ServiceStatus        — Health status of a service
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# OpenEnv base classes — these are Pydantic BaseModel subclasses.
# Action has: metadata (optional dict)
# Observation has: done (bool), reward (Optional[float]), metadata (optional dict)
# State has: episode_id (str), step_count (int)
from openenv.core.env_server.types import Action, Observation, State


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Models (used inside Observation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class LogEntry(BaseModel):
    """A single structured log line from a service."""

    timestamp: str = Field(
        ..., description="ISO 8601 timestamp, e.g. '2026-03-28T10:05:00Z'"
    )
    service: str = Field(..., description="Service that emitted this log")
    level: Literal["DEBUG", "INFO", "WARN", "ERROR", "FATAL"] = Field(
        ..., description="Log severity level"
    )
    message: str = Field(..., description="Human-readable log message")


class MetricDataPoint(BaseModel):
    """A single time-series metric value."""

    timestamp: str = Field(..., description="ISO 8601 timestamp")
    value: float = Field(..., description="Metric value at this timestamp")


class Alert(BaseModel):
    """A monitoring alert from the observability stack."""

    alert_id: str = Field(..., description="Unique alert identifier, e.g. 'ALT-001'")
    severity: Literal["critical", "warning", "info"] = Field(
        ..., description="Alert severity level"
    )
    service: str = Field(..., description="Service this alert is associated with")
    title: str = Field(..., description="Short alert title")
    description: str = Field(..., description="Detailed alert description")
    fired_at: str = Field(..., description="ISO 8601 timestamp when alert fired")


class ServiceStatus(BaseModel):
    """Health status of a single service."""

    service: str = Field(..., description="Service name")
    status: Literal["healthy", "degraded", "down"] = Field(
        ..., description="Current health status"
    )
    uptime: str = Field(..., description="Time since last restart, e.g. '14d 3h'")
    last_deploy: str = Field(
        ..., description="ISO 8601 timestamp of last deployment"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ACTION: What the agent can do at each step
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Constrained literal types for strict validation
ActionType = Literal[
    "query_logs",
    "query_metrics",
    "check_alerts",
    "check_service_status",
    "set_severity",
    "identify_root_cause",
    "execute_remediation",
    "escalate",
]

SeverityLevel = Literal["P1", "P2", "P3", "P4"]

RootCauseCategory = Literal[
    "database", "network", "config", "resource", "dependency", "security"
]

RemediationAction = Literal[
    "restart_service",
    "scale_up",
    "rollback_deploy",
    "flush_cache",
    "fix_config",
    "failover_db",
]

EscalationTeam = Literal["database", "network", "security", "platform"]

TimeRange = Literal["last_5m", "last_15m", "last_1h"]

MetricName = Literal[
    "cpu", "memory", "latency_p99", "error_rate", "connections", "request_rate"
]


class IncidentAction(Action):
    """Agent's action at each step of the incident response.

    Inherits from OpenEnv Action (has: metadata).

    The action_type field determines which optional parameters are relevant.
    Invalid combinations (e.g., service_name with set_severity) are
    silently ignored, not rejected — this simplifies the agent's task.

    Action Types:
        query_logs           — Fetch log entries from a specific service
        query_metrics        — Fetch time-series metrics from a service
        check_alerts         — View active monitoring alerts
        check_service_status — Check service health (up/down/degraded)
        set_severity         — Classify the incident severity (P1–P4)
        identify_root_cause  — Declare the root cause category + description
        execute_remediation  — Execute a fix action (TERMINAL)
        escalate             — Escalate to another team (TERMINAL)
    """

    action_type: ActionType = Field(
        ..., description="The type of action to perform"
    )

    # ── Investigation parameters ──
    service_name: Optional[str] = Field(
        default=None,
        description="Target service, e.g. 'api-gw', 'order-svc', 'order-db'",
    )
    time_range: Optional[TimeRange] = Field(
        default="last_15m",
        description="Time window for log/metric queries",
    )
    metric_name: Optional[MetricName] = Field(
        default=None,
        description="Specific metric to query; omit for all metrics",
    )

    # ── Triage parameters ──
    severity: Optional[SeverityLevel] = Field(
        default=None,
        description="Incident severity classification (P1=critical, P4=low)",
    )

    # ── Diagnosis parameters ──
    root_cause: Optional[str] = Field(
        default=None,
        description="Free-text description of the identified root cause",
    )
    root_cause_category: Optional[RootCauseCategory] = Field(
        default=None,
        description="Categorical classification of the root cause",
    )

    # ── Remediation parameters ──
    remediation_action: Optional[RemediationAction] = Field(
        default=None,
        description="The fix action to execute",
    )
    remediation_target: Optional[str] = Field(
        default=None,
        description="Which service to apply the remediation to",
    )

    # ── Escalation parameters ──
    escalation_team: Optional[EscalationTeam] = Field(
        default=None,
        description="Team to escalate the incident to",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OBSERVATION: What the agent sees after each step
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class IncidentObservation(Observation):
    """What the agent receives after taking an action.

    Inherits from OpenEnv Observation (has: done, reward, metadata).

    Contains the result of the last action, any returned data
    (logs, metrics, alerts, statuses), and progress indicators.
    """

    # done: bool — inherited from Observation
    # reward: Optional[float] — inherited from Observation
    # metadata: Optional[dict] — inherited from Observation

    # ── Context ──
    goal: str = Field(
        default="", description="What the agent needs to accomplish"
    )
    incident_summary: str = Field(
        default="", description="Brief description of the ongoing incident"
    )

    # ── Action result ──
    action_result: str = Field(
        default="", description="Human-readable result of the last action"
    )
    action_success: bool = Field(
        default=True, description="Whether the last action executed successfully"
    )
    last_action_error: Optional[str] = Field(
        default=None, description="Error message if last action failed"
    )

    # ── Data returned by investigative actions ──
    log_entries: Optional[List[LogEntry]] = Field(
        default=None, description="Log entries returned by query_logs"
    )
    metric_data: Optional[Dict[str, List[MetricDataPoint]]] = Field(
        default=None,
        description="Metric key → data points, returned by query_metrics",
    )
    alerts: Optional[List[Alert]] = Field(
        default=None, description="Active alerts returned by check_alerts"
    )
    service_statuses: Optional[List[ServiceStatus]] = Field(
        default=None, description="Service statuses from check_service_status"
    )

    # ── Progress tracking ──
    investigation_actions_taken: int = Field(
        default=0, description="Number of actions taken so far"
    )
    max_actions: int = Field(
        default=30, description="Maximum actions allowed per episode"
    )
    services_investigated: List[str] = Field(
        default_factory=list,
        description="Services the agent has queried so far",
    )
    severity_set: bool = Field(
        default=False, description="Whether the agent has classified severity"
    )
    root_cause_identified: bool = Field(
        default=False, description="Whether the agent has identified root cause"
    )
    remediation_executed: bool = Field(
        default=False, description="Whether the agent has executed remediation"
    )

    # ── Hints ──
    available_services: List[str] = Field(
        default_factory=list,
        description="List of services the agent can query",
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATE: Internal episode metadata (hidden from agent)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class IncidentState(State):
    """Internal state tracking for the current episode.

    Inherits from OpenEnv State (has: episode_id, step_count).

    Contains both the ground truth (set on reset) and the agent's
    answers (accumulated during the episode). Used by the grader
    to compute the final deterministic score.

    Fields:
        episode_id        — (inherited) Unique episode identifier
        step_count        — (inherited) Number of steps taken
        task_id           — Which task is being evaluated
        ground_truth_*    — Correct answers (set on reset, never shown to agent)
        agent_*           — Agent's answers (filled during episode)
        done              — Whether the episode has ended
        total_reward      — Cumulative reward across all steps
    """

    # episode_id: str — inherited from State
    # step_count: int — inherited from State

    # ── Episode metadata ──
    task_id: str = Field(default="", description="Task being evaluated")
    task_difficulty: str = Field(
        default="", description="Difficulty level: easy, medium, hard"
    )

    # ── Ground truth (hidden from agent, set on reset) ──
    ground_truth_severity: str = Field(
        default="", description="Correct severity (P1–P4)"
    )
    ground_truth_root_cause: str = Field(
        default="", description="Correct root cause description"
    )
    ground_truth_category: str = Field(
        default="", description="Correct root cause category"
    )
    ground_truth_remediation: str = Field(
        default="", description="Correct remediation action"
    )
    ground_truth_target: str = Field(
        default="", description="Correct remediation target service"
    )

    # ── Agent's answers (filled during episode) ──
    agent_severity: Optional[str] = Field(
        default=None, description="Agent's severity classification"
    )
    agent_root_cause: Optional[str] = Field(
        default=None, description="Agent's root cause description"
    )
    agent_category: Optional[str] = Field(
        default=None, description="Agent's root cause category"
    )
    agent_remediation: Optional[str] = Field(
        default=None, description="Agent's remediation action"
    )
    agent_remediation_target: Optional[str] = Field(
        default=None, description="Agent's remediation target service"
    )

    # ── Episode status ──
    done: bool = Field(default=False, description="Whether episode has ended")
    total_reward: float = Field(
        default=0.0, description="Cumulative reward across all steps"
    )
    services_investigated: List[str] = Field(
        default_factory=list,
        description="Unique services the agent has queried",
    )
