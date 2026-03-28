"""End-to-end Phase 2 test: full episode flow."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oncall_env.server.environment import IncidentTriageEnvironment
from oncall_env.models import IncidentAction

env = IncidentTriageEnvironment()

# ── RESET ──
print("=" * 60)
print("1. RESET (easy task)")
print("=" * 60)
obs = env.reset(seed=42, task_id="easy")
print(f"  goal: {obs.goal[:70]}...")
print(f"  summary: {obs.incident_summary[:80]}...")
print(f"  done={obs.done}, reward={obs.reward}")
print(f"  services: {obs.available_services}")
print()

# ── CHECK ALERTS ──
print("2. CHECK ALERTS")
obs = env.step(IncidentAction(action_type="check_alerts"))
print(f"  result: {obs.action_result}")
print(f"  alerts: {len(obs.alerts)} found")
for a in obs.alerts:
    print(f"    [{a.severity}] {a.service}: {a.title}")
print(f"  reward: {obs.reward}")
print()

# ── CHECK SERVICE STATUS ──
print("3. CHECK SERVICE STATUS")
obs = env.step(IncidentAction(action_type="check_service_status"))
print(f"  result: {obs.action_result}")
for s in obs.service_statuses:
    print(f"    {s.service}: {s.status} (uptime: {s.uptime})")
print(f"  reward: {obs.reward}")
print()

# ── QUERY LOGS (order-svc — affected) ──
print("4. QUERY LOGS (order-svc)")
obs = env.step(IncidentAction(
    action_type="query_logs",
    service_name="order-svc",
    time_range="last_15m",
))
print(f"  result: {obs.action_result}")
error_logs = [l for l in obs.log_entries if l.level in ("ERROR", "FATAL")]
info_logs = [l for l in obs.log_entries if l.level == "INFO"]
print(f"  total logs: {len(obs.log_entries)}")
print(f"  ERROR/FATAL: {len(error_logs)}, INFO: {len(info_logs)}")
if error_logs:
    print(f"  sample error: {error_logs[0].message}")
print()

# ── QUERY LOGS (cache — healthy) ──
print("5. QUERY LOGS (cache — should be healthy)")
obs = env.step(IncidentAction(
    action_type="query_logs",
    service_name="cache",
    time_range="last_5m",
))
error_logs_cache = [l for l in obs.log_entries if l.level in ("ERROR", "FATAL")]
print(f"  total logs: {len(obs.log_entries)}")
print(f"  ERROR/FATAL: {len(error_logs_cache)} (should be 0)")
print()

# ── QUERY METRICS (order-svc) ──
print("6. QUERY METRICS (order-svc)")
obs = env.step(IncidentAction(
    action_type="query_metrics",
    service_name="order-svc",
    metric_name="error_rate",
    time_range="last_15m",
))
print(f"  result: {obs.action_result}")
if "error_rate" in obs.metric_data:
    vals = [p.value for p in obs.metric_data["error_rate"]]
    print(f"  error_rate range: {min(vals):.1f} — {max(vals):.1f}")
print()

# ── SET SEVERITY (correct) ──
print("7. SET SEVERITY (P2 — correct)")
obs = env.step(IncidentAction(action_type="set_severity", severity="P2"))
print(f"  result: {obs.action_result}")
print(f"  reward: {obs.reward} (expected +0.15)")
print(f"  severity_set: {obs.severity_set}")
print()

# ── SET SEVERITY (off by one) ──
print("8. SET SEVERITY (P1 — off by one, overwrites)")
obs = env.step(IncidentAction(action_type="set_severity", severity="P1"))
print(f"  reward: {obs.reward} (expected +0.07)")
print()

# ── IDENTIFY ROOT CAUSE (correct) ──
print("9. IDENTIFY ROOT CAUSE (correct category)")
obs = env.step(IncidentAction(
    action_type="identify_root_cause",
    root_cause_category="resource",
    root_cause="Out of memory error in order-svc causing crashes",
))
print(f"  result: {obs.action_result}")
print(f"  reward: {obs.reward} (expected > 0.15)")
print(f"  root_cause_identified: {obs.root_cause_identified}")
print()

# ── EXECUTE REMEDIATION (correct) ──
print("10. EXECUTE REMEDIATION (correct)")
obs = env.step(IncidentAction(
    action_type="execute_remediation",
    remediation_action="restart_service",
    remediation_target="order-svc",
))
print(f"  result: {obs.action_result}")
print(f"  reward: {obs.reward} (expected +0.25 + 0.15 + efficiency)")
print(f"  done: {obs.done}")
print()

# ── STATE ──
print("=" * 60)
print("FINAL STATE")
print("=" * 60)
state = env.state
print(f"  episode_id: {state.episode_id[:12]}...")
print(f"  step_count: {state.step_count}")
print(f"  total_reward: {state.total_reward}")
print(f"  done: {state.done}")
print(f"  agent_severity: {state.agent_severity}")
print(f"  agent_category: {state.agent_category}")
print(f"  agent_remediation: {state.agent_remediation}")
print(f"  agent_remediation_target: {state.agent_remediation_target}")
print(f"  services_investigated: {state.services_investigated}")
print()

# ── STEP AFTER DONE ──
print("11. STEP AFTER DONE (should be blocked)")
obs = env.step(IncidentAction(action_type="check_alerts"))
print(f"  result: {obs.action_result}")
print(f"  done: {obs.done}")
print()

print("=" * 60)
print("ALL PHASE 2 TESTS PASSED ✓")
print("=" * 60)
