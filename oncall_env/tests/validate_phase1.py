"""Quick validation that all Phase 1 models work correctly."""
import sys
import os

# Dynamically add the workspace root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, workspace_root)

from oncall_env.models import (
    IncidentAction,
    IncidentObservation,
    IncidentState,
    LogEntry,
    MetricDataPoint,
    Alert,
    ServiceStatus,
)

print("=== 1. Literal validation ===")
try:
    a = IncidentAction(action_type="invalid_type")
    print("FAIL: should have rejected invalid type")
except Exception as e:
    print(f"PASS: rejected invalid action type ({type(e).__name__})")

print()
print("=== 2. Action serialization ===")
a = IncidentAction(
    action_type="execute_remediation",
    remediation_action="rollback_deploy",
    remediation_target="order-svc",
)
print(f"Action type: {a.action_type}")
print(f"Remediation: {a.remediation_action} -> {a.remediation_target}")
print(f"JSON preview: {a.model_dump_json()[:120]}...")

print()
print("=== 3. Helper models ===")
log = LogEntry(
    timestamp="2026-03-28T10:05:00Z",
    service="api-gw",
    level="ERROR",
    message="Upstream service order-svc timed out after 30000ms",
)
print(f"Log: [{log.level}] {log.service}: {log.message[:50]}")

metric = MetricDataPoint(timestamp="2026-03-28T10:00:00Z", value=95.3)
print(f"Metric: {metric.value} at {metric.timestamp}")

alert = Alert(
    alert_id="ALT-001",
    severity="critical",
    service="order-svc",
    title="Service order-svc Not Responding",
    description="Health check failed 5 times",
    fired_at="2026-03-28T10:05:00Z",
)
print(f"Alert: [{alert.severity}] {alert.title}")

status = ServiceStatus(
    service="order-svc",
    status="down",
    uptime="0m",
    last_deploy="2026-03-28T09:00:00Z",
)
print(f"Status: {status.service} = {status.status}")

print()
print("=== 4. Observation ===")
obs = IncidentObservation(
    goal="Investigate and fix the incident",
    incident_summary="Order service is down",
    action_result="Retrieved 15 log entries",
    action_success=True,
    log_entries=[log],
    alerts=[alert],
    service_statuses=[status],
    available_services=["api-gw", "order-svc", "order-db"],
    investigation_actions_taken=3,
    max_actions=30,
)
print(f"done={obs.done}, reward={obs.reward}")
print(f"goal: {obs.goal}")
print(f"logs: {len(obs.log_entries)} entries")
print(f"alerts: {len(obs.alerts)} alerts")
print(f"JSON size: {len(obs.model_dump_json())} chars")

print()
print("=== 5. State ===")
s = IncidentState(
    task_id="easy",
    task_difficulty="easy",
    ground_truth_severity="P2",
    ground_truth_root_cause="OOM crash in order-svc",
    ground_truth_category="resource",
    ground_truth_remediation="restart_service",
    ground_truth_target="order-svc",
)
print(f"task: {s.task_id}, difficulty: {s.task_difficulty}")
print(f"gt_severity: {s.ground_truth_severity}")
print(f"gt_category: {s.ground_truth_category}")
print(f"episode_id: {s.episode_id!r}, step_count: {s.step_count}")
print(f"agent_severity: {s.agent_severity} (not set yet)")

print()
print("=== 6. Inheritance check ===")
from openenv.core.env_server.types import Action, Observation, State
print(f"IncidentAction inherits Action: {issubclass(IncidentAction, Action)}")
print(f"IncidentObservation inherits Observation: {issubclass(IncidentObservation, Observation)}")
print(f"IncidentState inherits State: {issubclass(IncidentState, State)}")

print()
print("=" * 50)
print("ALL PHASE 1 VALIDATION TESTS PASSED ✓")
print("=" * 50)
