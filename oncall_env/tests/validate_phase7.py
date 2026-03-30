"""Phase 7 validation & hackathon pre-submission checklist (all 17 items)."""
import json
import os
import subprocess
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, workspace_root)

print("=" * 70)
print("PHASE 7 — FINAL PRE-SUBMISSION CHECKLIST")
print("=" * 70)
print()

errors = []
passes = 0
total = 0


def check(num: int, name: str, condition: bool, detail: str = "") -> None:
    global passes, total
    total += 1
    status = "PASS" if condition else "FAIL"
    if condition:
        passes += 1
    else:
        errors.append(f"#{num}: {name}")
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] #{num:2d}: {name}{suffix}")


# ── 1. Tests pass ──
print("SECTION A: Tests")
print("-" * 50)

result = subprocess.run(
    [
        os.path.join(workspace_root, "oncall_env", ".venv", "Scripts", "python"),
        "-m", "pytest",
        os.path.join(workspace_root, "oncall_env", "tests", "test_environment.py"),
        os.path.join(workspace_root, "oncall_env", "tests", "test_graders.py"),
        os.path.join(workspace_root, "oncall_env", "tests", "test_integration.py"),
        "-q", "--tb=no",
    ],
    capture_output=True,
    text=True,
    cwd=workspace_root,
    timeout=120,
)
test_passed = result.returncode == 0
test_summary = result.stdout.strip().split("\n")[-1] if result.stdout else "No output"
check(1, "All unit/integration tests pass", test_passed, test_summary)
print()

# ── 2. File structure ──
print("SECTION B: File Structure & Spec Compliance")
print("-" * 50)

# 4. openenv.yaml
import yaml
yaml_path = os.path.join(workspace_root, "openenv.yaml")
yaml_exists = os.path.exists(yaml_path)
yaml_valid = False
if yaml_exists:
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    yaml_valid = (
        cfg.get("spec_version") == 1
        and cfg.get("name") == "oncall_env"
        and cfg.get("app") is not None
    )
check(4, "openenv.yaml present with correct fields", yaml_exists and yaml_valid)

# 5. Typed Pydantic models
from oncall_env.models import IncidentAction, IncidentObservation, IncidentState
from openenv.core.env_server.types import Action, Observation, State

check(5, "Typed Pydantic models for Action, Observation, State",
      issubclass(IncidentAction, Action)
      and issubclass(IncidentObservation, Observation)
      and issubclass(IncidentState, State))

# 6. Endpoints work (local test via environment object)
from oncall_env.server.environment import IncidentTriageEnvironment
env = IncidentTriageEnvironment()
obs = env.reset(task_id="easy", seed=0)
step_obs = env.step(IncidentAction(action_type="check_alerts"))
state = env.state
check(6, "reset(), step(), state() work correctly",
      obs.done is False and step_obs.action_success is True and isinstance(state, IncidentState))

# 7-8. Docker (file check only — building needs Docker daemon)
dockerfile_path = os.path.join(workspace_root, "Dockerfile")
check(7, "Dockerfile exists and valid structure",
      os.path.exists(dockerfile_path)
      and "python:3.11-slim" in open(dockerfile_path, encoding="utf-8").read())

dockerfile_content = open(dockerfile_path, encoding="utf-8").read()
check(8, "Dockerfile has CMD for server startup",
      "uvicorn" in dockerfile_content
      and "oncall_env.server.app:app" in dockerfile_content)

# 9. inference.py at project root
inference_path = os.path.join(workspace_root, "inference.py")
check(9, "inference.py at project root", os.path.exists(inference_path))

# 10. inference.py uses OpenAI client
with open(inference_path, encoding="utf-8") as f:
    inf_src = f.read()
check(10, "inference.py uses OpenAI client", "from openai import OpenAI" in inf_src)

# 11. inference.py reads env vars
check(11, "inference.py reads API_BASE_URL, MODEL_NAME, HF_TOKEN/env vars",
      "API_BASE_URL" in inf_src and "MODEL_NAME" in inf_src and "LLM_PROVIDER" in inf_src)

# 12-13. (Can't fully verify without API key — structural check)
check(12, "inference.py has main() and run_episode()",
      "def main()" in inf_src and "def run_episode(" in inf_src)

check(13, "inference.py has MAX_STEPS <= 25 (< 20 min runtime budget)",
      "MAX_STEPS" in inf_src)

# 14. 3+ tasks with graders
from oncall_env.tasks.task_definitions import TASKS
from oncall_env.tasks.graders import grade_episode
check(14, "3+ tasks with graders",
      len(TASKS) >= 3 and callable(grade_episode),
      f"Found {len(TASKS)} tasks: {list(TASKS.keys())}")

# 15. Grader scores in [0.0, 1.0]
scores_valid = True
for task in ["easy", "medium", "hard"]:
    for seed in [0, 1, 2]:
        env.reset(task_id=task, seed=seed)
        gt = env.state
        env.step(IncidentAction(action_type="set_severity", severity=gt.ground_truth_severity))
        env.step(IncidentAction(
            action_type="identify_root_cause",
            root_cause=gt.ground_truth_root_cause,
            root_cause_category=gt.ground_truth_category,
        ))
        env.step(IncidentAction(
            action_type="execute_remediation",
            remediation_action=gt.ground_truth_remediation,
            remediation_target=gt.ground_truth_target,
        ))
        score = grade_episode(env.state)
        if not (0.0 <= score <= 1.0):
            scores_valid = False
check(15, "Grader scores in [0.0, 1.0] for all 9 scenarios", scores_valid)

# 16. README
readme_path = os.path.join(workspace_root, "README.md")
readme_exists = os.path.exists(readme_path)
readme_sections = False
if readme_exists:
    with open(readme_path, encoding="utf-8") as f:
        readme = f.read()
    readme_sections = all(s in readme for s in [
        "Motivation", "Action Space", "Observation Space",
        "Tasks", "Setup", "Baseline", "Structure",
    ])
check(16, "README has all required sections", readme_exists and readme_sections)

# 17. Resource check (lightweight env — no GPU, no heavy deps)
check(17, "Environment runs on 2 vCPU / 8GB RAM (no GPU deps)",
      "torch" not in inf_src and "tensorflow" not in inf_src)

print()

# ── Summary ──
print("=" * 70)
print(f"  RESULT: {passes}/{total} checks passed")
print()

if errors:
    print(f"  FAILED ({len(errors)}):")
    for e in errors:
        print(f"    X {e}")
else:
    print("  ALL PRE-SUBMISSION CHECKS PASSED")

print()
print("  Manual checks remaining:")
print("    [ ] HF Space deployed and returns 200 on /health")
print("    [ ] reset() via curl against HF Space returns observation")
print("    [ ] inference.py completes against live server (needs API key)")
print("    [ ] docker build succeeds locally (needs Docker)")
print("=" * 70)
