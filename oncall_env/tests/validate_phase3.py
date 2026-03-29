"""Phase 3 validation: Task definitions and grader correctness."""
import sys
import os

# Add workspace root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, workspace_root)

from oncall_env.models import IncidentState
from oncall_env.tasks import (
    TASKS,
    get_task,
    list_task_ids,
    grade_episode,
    grade_episode_breakdown,
)

print("=" * 60)
print("PHASE 3 VALIDATION — Tasks & Graders")
print("=" * 60)
print()

# ── 1. Task definitions ──
print("1. TASK DEFINITIONS")
print("-" * 40)
task_ids = list_task_ids()
print(f"  Task IDs: {task_ids}")
assert task_ids == ["easy", "medium", "hard"], f"Expected 3 tasks, got {task_ids}"

for tid in task_ids:
    task = get_task(tid)
    print(f"\n  [{tid}] {task['name']}")
    print(f"    Difficulty: {task['difficulty']}")
    print(f"    Expected scores: {task['expected_score_range']}")
    print(f"    Scenarios: {task['scenarios']}")
    chars = task["characteristics"]
    print(f"    Affected services: {chars['num_affected_services']}")
    print(f"    Red herrings: {chars['red_herrings']}")
    print(f"    Cascading: {chars['cascading_failure']}")

# Verify tasks are immutable
try:
    TASKS["easy"]["name"] = "Hacked"
    print("\n  FAIL: TASKS should be immutable!")
except TypeError:
    print("\n  PASS: TASKS are immutable (MappingProxyType)")

# Verify invalid task_id raises
try:
    get_task("nonexistent")
    print("  FAIL: should raise ValueError")
except ValueError:
    print("  PASS: Invalid task_id raises ValueError")

print()

# ── 2. Perfect score (max possible = 1.0) ──
print("2. GRADER: PERFECT SCORE")
print("-" * 40)
perfect_state = IncidentState(
    episode_id="test-perfect",
    step_count=8,  # Fast = 0.05 efficiency
    task_id="easy",
    task_difficulty="easy",
    ground_truth_severity="P2",
    ground_truth_root_cause=(
        "Out of memory error in order-svc due to a memory leak "
        "in the request handler causing the JVM to exceed its "
        "heap limit and crash repeatedly."
    ),
    ground_truth_category="resource",
    ground_truth_remediation="restart_service",
    ground_truth_target="order-svc",
    # Agent gets everything exactly right
    agent_severity="P2",
    agent_root_cause=(
        "Out of memory error in order-svc due to a memory leak "
        "in the request handler causing the JVM to exceed its "
        "heap limit and crash repeatedly."
    ),
    agent_category="resource",
    agent_remediation="restart_service",
    agent_remediation_target="order-svc",
    done=True,
)

perfect_score = grade_episode(perfect_state)
breakdown = grade_episode_breakdown(perfect_state)
print(f"  Total score: {perfect_score}")
for k, v in breakdown.items():
    print(f"    {k}: {v}")

assert perfect_score == 1.0, f"Expected 1.0, got {perfect_score}"
print("  PASS: Perfect score = 1.0 ✓")
print()

# ── 3. Zero score (agent did nothing) ──
print("3. GRADER: ZERO SCORE")
print("-" * 40)
empty_state = IncidentState(
    episode_id="test-empty",
    step_count=30,  # Hit max actions
    task_id="easy",
    task_difficulty="easy",
    ground_truth_severity="P2",
    ground_truth_root_cause="Out of memory error",
    ground_truth_category="resource",
    ground_truth_remediation="restart_service",
    ground_truth_target="order-svc",
    # Agent did nothing — all None
    done=True,
)

zero_score = grade_episode(empty_state)
breakdown_zero = grade_episode_breakdown(empty_state)
print(f"  Total score: {zero_score}")
for k, v in breakdown_zero.items():
    print(f"    {k}: {v}")

assert zero_score == 0.0, f"Expected 0.0, got {zero_score}"
print("  PASS: Zero score = 0.0 ✓")
print()

# ── 4. Partial credit: severity off by one ──
print("4. GRADER: SEVERITY OFF-BY-ONE")
print("-" * 40)
partial_state = IncidentState(
    episode_id="test-partial-sev",
    step_count=12,
    task_id="easy",
    task_difficulty="easy",
    ground_truth_severity="P2",
    ground_truth_root_cause="Out of memory error in order-svc",
    ground_truth_category="resource",
    ground_truth_remediation="restart_service",
    ground_truth_target="order-svc",
    agent_severity="P3",  # Off by one → 0.10 instead of 0.20
    agent_category="resource",
    agent_root_cause="memory leak in order-svc",
    agent_remediation="restart_service",
    agent_remediation_target="order-svc",
    done=True,
)

partial_score = grade_episode(partial_state)
breakdown_partial = grade_episode_breakdown(partial_state)
print(f"  Total score: {partial_score}")
print(f"  Severity: {breakdown_partial['severity']} (expected 0.10)")
assert breakdown_partial["severity"] == 0.10
print(f"  Root cause category: {breakdown_partial['root_cause_category']} (expected 0.25)")
assert breakdown_partial["root_cause_category"] == 0.25
print(f"  Remediation action: {breakdown_partial['remediation_action']} (expected 0.25)")
assert breakdown_partial["remediation_action"] == 0.25
print(f"  Efficiency: {breakdown_partial['efficiency']} (expected 0.02, 12 steps)")
assert breakdown_partial["efficiency"] == 0.02
print("  PASS: Partial credit calculated correctly ✓")
print()

# ── 5. Wrong everything ──
print("5. GRADER: ALL WRONG ANSWERS")
print("-" * 40)
wrong_state = IncidentState(
    episode_id="test-wrong",
    step_count=25,  # Slow, no efficiency bonus
    task_id="easy",
    task_difficulty="easy",
    ground_truth_severity="P2",
    ground_truth_root_cause="Out of memory error in order-svc",
    ground_truth_category="resource",
    ground_truth_remediation="restart_service",
    ground_truth_target="order-svc",
    agent_severity="P4",  # Off by 2 → 0.00
    agent_category="network",  # Wrong → 0.00
    agent_root_cause="network partition between services",  # No overlap → ~0.00
    agent_remediation="flush_cache",  # Wrong → 0.00
    agent_remediation_target="cache",  # Wrong → 0.00
    done=True,
)

wrong_score = grade_episode(wrong_state)
breakdown_wrong = grade_episode_breakdown(wrong_state)
print(f"  Total score: {wrong_score}")
for k, v in breakdown_wrong.items():
    print(f"    {k}: {v}")

assert wrong_score == 0.0, f"Expected 0.0, got {wrong_score}"
print("  PASS: All wrong = 0.0 ✓")
print()

# ── 6. Boundary: score never exceeds 1.0 ──
print("6. BOUNDARY: Score clamped to [0.0, 1.0]")
print("-" * 40)
assert 0.0 <= perfect_score <= 1.0
assert 0.0 <= zero_score <= 1.0
assert 0.0 <= partial_score <= 1.0
assert 0.0 <= wrong_score <= 1.0
print("  PASS: All scores in [0.0, 1.0] ✓")
print()

# ── 7. Weight sum verification ──
print("7. WEIGHT SUM VERIFICATION")
print("-" * 40)
from oncall_env.tasks.graders import (
    WEIGHT_SEVERITY,
    WEIGHT_ROOT_CAUSE_CATEGORY,
    WEIGHT_ROOT_CAUSE_DESCRIPTION,
    WEIGHT_REMEDIATION_ACTION,
    WEIGHT_REMEDIATION_TARGET,
    WEIGHT_EFFICIENCY_FAST,
)
total_weight = (
    WEIGHT_SEVERITY
    + WEIGHT_ROOT_CAUSE_CATEGORY
    + WEIGHT_ROOT_CAUSE_DESCRIPTION
    + WEIGHT_REMEDIATION_ACTION
    + WEIGHT_REMEDIATION_TARGET
    + WEIGHT_EFFICIENCY_FAST
)
print(f"  Sum of max weights: {total_weight}")
assert total_weight == 1.0, f"Weights must sum to 1.0, got {total_weight}"
print("  PASS: Weights sum to exactly 1.0 ✓")
print()

# ── 8. Keyword overlap scoring ──
print("8. KEYWORD OVERLAP DETAIL")
print("-" * 40)
overlap_state = IncidentState(
    episode_id="test-overlap",
    step_count=5,
    ground_truth_severity="P2",
    ground_truth_root_cause="Out of memory error in order-svc causing crashes",
    ground_truth_category="resource",
    ground_truth_remediation="restart_service",
    ground_truth_target="order-svc",
    agent_severity="P2",
    agent_root_cause="memory error in order-svc",  # Partial overlap
    agent_category="resource",
    agent_remediation="restart_service",
    agent_remediation_target="order-svc",
    done=True,
)
overlap_breakdown = grade_episode_breakdown(overlap_state)
print(f"  Agent text: 'memory error in order-svc'")
print(f"  GT text: 'Out of memory error in order-svc causing crashes'")
print(f"  Description score: {overlap_breakdown['root_cause_description']}")
assert 0.0 < overlap_breakdown["root_cause_description"] < WEIGHT_ROOT_CAUSE_DESCRIPTION
print("  PASS: Partial keyword overlap gives partial credit ✓")
print()

print("=" * 60)
print("ALL PHASE 3 VALIDATION TESTS PASSED ✓")
print("=" * 60)
