"""Phase 4 validation: Scenario data integrity and completeness."""
import json
import sys
import os
from pathlib import Path

# Add workspace root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, workspace_root)

from oncall_env.server.scenario_engine import ScenarioEngine, AVAILABLE_SERVICES

SCENARIOS_DIR = Path(workspace_root) / "oncall_env" / "scenarios"

# Required keys in every scenario dict
REQUIRED_SCENARIO_KEYS = {
    "scenario_id", "difficulty", "title", "incident_summary",
    "ground_truth", "affected_services", "root_service",
    "service_states", "log_config", "metric_config", "alert_config",
}

REQUIRED_GROUND_TRUTH_KEYS = {
    "severity", "root_cause", "root_cause_category",
    "remediation_action", "remediation_target",
}

VALID_SEVERITIES = {"P1", "P2", "P3", "P4"}
VALID_CATEGORIES = {"database", "network", "config", "resource", "dependency", "security"}
VALID_REMEDIATIONS = {"restart_service", "scale_up", "rollback_deploy", "flush_cache", "fix_config", "failover_db"}
VALID_STATUSES = {"healthy", "degraded", "down"}

errors = []
warnings = []

def error(msg: str) -> None:
    errors.append(msg)

def warn(msg: str) -> None:
    warnings.append(msg)


print("=" * 60)
print("PHASE 4 VALIDATION — Scenario Data Integrity")
print("=" * 60)
print()

# ── 1. Load all JSON files ──
print("1. LOADING SCENARIO FILES")
print("-" * 40)

all_scenarios = {}
for difficulty in ["easy", "medium", "hard"]:
    path = SCENARIOS_DIR / f"{difficulty}_scenarios.json"
    if not path.exists():
        error(f"Missing file: {path}")
        continue
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "scenarios" in data:
        scenarios = data["scenarios"]
    elif isinstance(data, list):
        scenarios = data
    else:
        error(f"{path.name}: Invalid format — expected list or {{scenarios: [...]}}")
        continue
    
    all_scenarios[difficulty] = scenarios
    print(f"  {difficulty}: {len(scenarios)} scenarios loaded")

total = sum(len(s) for s in all_scenarios.values())
print(f"  Total: {total} scenarios")
assert total >= 9, f"Expected at least 9 scenarios, got {total}"
print()

# ── 2. Validate each scenario ──
print("2. VALIDATING SCENARIO STRUCTURE")
print("-" * 40)

for difficulty, scenarios in all_scenarios.items():
    for sc in scenarios:
        sid = sc.get("scenario_id", "UNKNOWN")
        prefix = f"  [{difficulty}/{sid}]"
        
        # 2a. Required keys
        missing = REQUIRED_SCENARIO_KEYS - set(sc.keys())
        if missing:
            error(f"{prefix} Missing keys: {missing}")
        
        # 2b. Difficulty matches
        if sc.get("difficulty") != difficulty:
            error(f"{prefix} difficulty={sc.get('difficulty')} doesn't match file ({difficulty})")
        
        # 2c. Ground truth validation
        gt = sc.get("ground_truth", {})
        gt_missing = REQUIRED_GROUND_TRUTH_KEYS - set(gt.keys())
        if gt_missing:
            error(f"{prefix} ground_truth missing: {gt_missing}")
        
        if gt.get("severity") not in VALID_SEVERITIES:
            error(f"{prefix} Invalid severity: {gt.get('severity')}")
        
        if gt.get("root_cause_category") not in VALID_CATEGORIES:
            error(f"{prefix} Invalid category: {gt.get('root_cause_category')}")
        
        if gt.get("remediation_action") not in VALID_REMEDIATIONS:
            error(f"{prefix} Invalid remediation: {gt.get('remediation_action')}")
        
        if not gt.get("root_cause"):
            error(f"{prefix} Empty root_cause description")
        
        if not gt.get("remediation_target"):
            error(f"{prefix} Empty remediation_target")
        
        # 2d. Service states — all 8 services present
        states = sc.get("service_states", {})
        for svc in AVAILABLE_SERVICES:
            if svc not in states:
                error(f"{prefix} service_states missing: {svc}")
            elif states[svc] not in VALID_STATUSES:
                error(f"{prefix} service_states[{svc}] invalid: {states[svc]}")
        
        # 2e. At least one alert
        alerts = sc.get("alert_config", [])
        if len(alerts) == 0:
            error(f"{prefix} No alerts defined")
        
        for alert in alerts:
            for key in ["alert_id", "severity", "service", "title", "description", "fired_at"]:
                if key not in alert:
                    error(f"{prefix} Alert missing key: {key}")
        
        # 2f. Log config — at least the affected services have error patterns
        affected = sc.get("affected_services", [])
        log_config = sc.get("log_config", {})
        for aff_svc in affected:
            if aff_svc not in log_config:
                warn(f"{prefix} No log_config for affected service: {aff_svc}")
            else:
                ep = log_config[aff_svc].get("error_patterns", [])
                if not ep:
                    warn(f"{prefix} No error_patterns for affected service: {aff_svc}")
        
        # 2g. Metric config — at least the root service has anomalous metrics
        metric_config = sc.get("metric_config", {})
        root_svc = sc.get("root_service")
        if root_svc and root_svc not in metric_config:
            warn(f"{prefix} No metric_config for root service: {root_svc}")
        
        print(f"{prefix} OK — {len(alerts)} alerts, {len(log_config)} services logged, {len(metric_config)} services metriced")

print()

# ── 3. ScenarioEngine integration ──
print("3. SCENARIO ENGINE INTEGRATION")
print("-" * 40)

engine = ScenarioEngine()

for tid in ["easy", "medium", "hard"]:
    # Test seed=0
    sc0 = engine.select_scenario(tid, seed=0)
    print(f"  {tid} (seed=0): {sc0['scenario_id']} — {sc0['title']}")
    
    # Test seed=1
    sc1 = engine.select_scenario(tid, seed=1)
    print(f"  {tid} (seed=1): {sc1['scenario_id']} — {sc1['title']}")
    
    # Test seed=2
    sc2 = engine.select_scenario(tid, seed=2)
    print(f"  {tid} (seed=2): {sc2['scenario_id']} — {sc2['title']}")
    
    # Verify different seeds give different scenarios
    ids = {sc0["scenario_id"], sc1["scenario_id"], sc2["scenario_id"]}
    if len(ids) == 3:
        print(f"  ✓ All 3 seeds produced different scenarios")
    else:
        print(f"  ⚠ Only {len(ids)} unique scenarios from 3 seeds")
    print()

# ── 4. Full episode test with JSON-loaded scenario ──
print("4. FULL EPISODE WITH JSON SCENARIO")
print("-" * 40)

from oncall_env.server.environment import IncidentTriageEnvironment
from oncall_env.models import IncidentAction
from oncall_env.tasks.graders import grade_episode, grade_episode_breakdown

env = IncidentTriageEnvironment()

# Test with a non-default seed to get a different scenario
obs = env.reset(seed=1, task_id="easy")
print(f"  Task: easy, Scenario: {env.state.task_id}")
print(f"  Summary: {obs.incident_summary[:60]}...")

# Quick play-through
env.step(IncidentAction(action_type="check_alerts"))
env.step(IncidentAction(action_type="check_service_status"))
env.step(IncidentAction(action_type="query_logs", service_name="order-db"))

# Set correct answers for disk_full scenario
env.step(IncidentAction(action_type="set_severity", severity="P1"))
env.step(IncidentAction(
    action_type="identify_root_cause",
    root_cause_category="resource",
    root_cause="Database disk full due to unarchived transaction logs",
))
obs = env.step(IncidentAction(
    action_type="execute_remediation",
    remediation_action="scale_up",
    remediation_target="order-db",
))

print(f"  Done: {obs.done}")
score = grade_episode(env.state)
breakdown = grade_episode_breakdown(env.state)
print(f"  Grade: {score}")
print(f"  Breakdown: {breakdown}")
assert 0.0 <= score <= 1.0
print()

# ── 5. Test hard scenario ──
print("5. HARD SCENARIO TEST (TLS cert)")
print("-" * 40)

obs = env.reset(seed=1, task_id="hard")
print(f"  Summary: {obs.incident_summary[:60]}...")

env.step(IncidentAction(action_type="check_alerts"))
env.step(IncidentAction(action_type="query_logs", service_name="api-gw"))
env.step(IncidentAction(action_type="query_logs", service_name="payment-svc"))

env.step(IncidentAction(action_type="set_severity", severity="P1"))
env.step(IncidentAction(
    action_type="identify_root_cause",
    root_cause_category="security",
    root_cause="TLS certificate expired for internal service communication",
))
obs = env.step(IncidentAction(
    action_type="execute_remediation",
    remediation_action="fix_config",
    remediation_target="api-gw",
))

score = grade_episode(env.state)
breakdown = grade_episode_breakdown(env.state)
print(f"  Done: {obs.done}, Grade: {score}")
print(f"  Breakdown: {breakdown}")
assert 0.0 <= score <= 1.0
print()

# ── Results ──
print("=" * 60)
if errors:
    print(f"ERRORS ({len(errors)}):")
    for e in errors:
        print(f"  ❌ {e}")
    print()

if warnings:
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  ⚠ {w}")
    print()

if not errors:
    print("ALL PHASE 4 VALIDATION TESTS PASSED ✓")
else:
    print("PHASE 4 VALIDATION FAILED — fix errors above")
    sys.exit(1)

print("=" * 60)
