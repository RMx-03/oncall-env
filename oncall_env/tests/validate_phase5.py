"""Phase 5 validation: Client and inference script structure."""
import sys
import os
import json

# Add workspace root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, workspace_root)

print("=" * 60)
print("PHASE 5 VALIDATION - Client & Inference Script")
print("=" * 60)
print()

# ── 1. Client import and typing ──
print("1. CLIENT IMPORT & TYPING")
print("-" * 40)

from oncall_env.client import IncidentTriageEnv
from openenv.core.env_client import EnvClient
from oncall_env.models import IncidentAction, IncidentObservation, IncidentState

assert issubclass(IncidentTriageEnv, EnvClient), "Must subclass EnvClient"
print("  PASS: IncidentTriageEnv is EnvClient subclass")

# Check abstract methods are implemented
env = IncidentTriageEnv.__new__(IncidentTriageEnv)
assert hasattr(env, "_step_payload"), "Missing _step_payload"
assert hasattr(env, "_parse_result"), "Missing _parse_result"
assert hasattr(env, "_parse_state"), "Missing _parse_state"
print("  PASS: All 3 abstract methods implemented")

# ── 2. _step_payload serialization ──
print()
print("2. _STEP_PAYLOAD SERIALIZATION")
print("-" * 40)

action_logs = IncidentAction(action_type="query_logs", service_name="order-svc")
payload_logs = env._step_payload(action_logs)
print(f"  query_logs payload: {payload_logs}")
assert payload_logs["action_type"] == "query_logs"
assert payload_logs["service_name"] == "order-svc"
assert "severity" not in payload_logs, "None fields should be excluded"
print("  PASS: exclude_none works correctly")

action_sev = IncidentAction(action_type="set_severity", severity="P1")
payload_sev = env._step_payload(action_sev)
print(f"  set_severity payload: {payload_sev}")
assert payload_sev["severity"] == "P1"
assert "service_name" not in payload_sev
print("  PASS: Severity action serialized correctly")

action_rem = IncidentAction(
    action_type="execute_remediation",
    remediation_action="restart_service",
    remediation_target="order-svc",
)
payload_rem = env._step_payload(action_rem)
print(f"  execute_remediation payload: {json.dumps(payload_rem, indent=2)}")
assert payload_rem["remediation_action"] == "restart_service"
assert payload_rem["remediation_target"] == "order-svc"
print("  PASS: Remediation action serialized correctly")

# ── 3. _parse_result deserialization ──
print()
print("3. _PARSE_RESULT DESERIALIZATION")
print("-" * 40)

mock_payload = {
    "observation": {
        "done": False,
        "goal": "Investigate the incident",
        "incident_summary": "Test incident",
        "action_result": "Logs fetched",
        "action_success": True,
        "log_entries": [
            {"timestamp": "2026-01-01T00:00:00Z", "service": "order-svc", "level": "ERROR", "message": "OOM"}
        ],
        "available_services": ["api-gw", "order-svc"],
        "investigation_actions_taken": 1,
        "max_actions": 30,
    },
    "reward": 0.15,
    "done": False,
}

step_result = env._parse_result(mock_payload)
print(f"  observation type: {type(step_result.observation).__name__}")
assert isinstance(step_result.observation, IncidentObservation)
assert step_result.observation.incident_summary == "Test incident"
assert step_result.observation.log_entries is not None
assert len(step_result.observation.log_entries) == 1
assert step_result.reward == 0.15
assert step_result.done is False
print("  PASS: StepResult[IncidentObservation] parsed correctly")

# ── 4. _parse_state deserialization ──
print()
print("4. _PARSE_STATE DESERIALIZATION")
print("-" * 40)

mock_state = {
    "episode_id": "test-ep",
    "step_count": 5,
    "task_id": "easy",
    "task_difficulty": "easy",
    "ground_truth_severity": "P2",
    "ground_truth_root_cause": "OOM in order-svc",
    "ground_truth_category": "resource",
    "ground_truth_remediation": "restart_service",
    "ground_truth_target": "order-svc",
    "agent_severity": "P2",
    "done": True,
    "total_reward": 0.5,
}

state = env._parse_state(mock_state)
print(f"  state type: {type(state).__name__}")
assert isinstance(state, IncidentState)
assert state.episode_id == "test-ep"
assert state.ground_truth_severity == "P2"
assert state.agent_severity == "P2"
print("  PASS: IncidentState parsed correctly")

# ── 5. Package re-exports ──
print()
print("5. PACKAGE RE-EXPORTS")
print("-" * 40)

from oncall_env import IncidentTriageEnv as EnvFromPackage
assert EnvFromPackage is IncidentTriageEnv
print("  PASS: IncidentTriageEnv re-exported from oncall_env.__init__")

# ── 6. Inference script structure ──
print()
print("6. INFERENCE SCRIPT STRUCTURE")
print("-" * 40)

inference_path = os.path.join(workspace_root, "inference.py")
assert os.path.exists(inference_path), f"inference.py not found at {inference_path}"
print(f"  Found: {inference_path}")

with open(inference_path, "r", encoding="utf-8") as f:
    src = f.read()

# Check required components
checks = {
    "imports openai": "from openai import OpenAI" in src,
    "SYSTEM_PROMPT defined": "SYSTEM_PROMPT" in src,
    "build_user_prompt function": "def build_user_prompt" in src,
    "parse_model_action function": "def parse_model_action" in src,
    "run_episode function": "def run_episode" in src,
    "main function": "def main" in src,
    "PROVIDER_CONFIGS dict": "PROVIDER_CONFIGS" in src,
    "openai provider": '"openai"' in src,
    "openrouter provider": '"openrouter"' in src,
    "groq provider": '"groq"' in src,
    "gemini provider": '"gemini"' in src,
    "custom provider": '"custom"' in src,
    "API_BASE_URL override": "API_BASE_URL" in src,
    "LLM_PROVIDER env var": "LLM_PROVIDER" in src,
    "FALLBACK_ACTION defined": "FALLBACK_ACTION" in src,
    "ENV_URL env var": "ENV_URL" in src,
    "easy/medium/hard tasks": '"easy"' in src and '"medium"' in src and '"hard"' in src,
}

all_pass = True
for name, passed in checks.items():
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  {status}: {name}")

# ── 7. parse_model_action tests ──
print()
print("7. PARSE_MODEL_ACTION TESTS")
print("-" * 40)

# Import from inference.py
sys.path.insert(0, workspace_root)

# Can't import directly due to API key check, test via exec
import importlib.util
spec = importlib.util.spec_from_file_location("inference", inference_path)
# We need to mock env vars to avoid sys.exit
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-validation")
os.environ.setdefault("MODEL_NAME", "test-model")

# Load the module
loader = spec.loader
mod = importlib.util.module_from_spec(spec)

# Patch sys.exit to prevent it
original_exit = sys.exit
sys.exit = lambda *a, **kw: None
try:
    loader.exec_module(mod)
finally:
    sys.exit = original_exit

parse_fn = mod.parse_model_action

# Test cases
test_cases = [
    ('{"action_type": "check_alerts"}', "check_alerts"),
    ('```json\n{"action_type": "query_logs", "service_name": "order-svc"}\n```', "query_logs"),
    ('I think we should check alerts.\n{"action_type": "check_alerts"}', "check_alerts"),
    ("", "check_alerts"),  # Fallback
    ("not json at all", "check_alerts"),  # Fallback
]

for raw, expected_type in test_cases:
    result = parse_fn(raw)
    actual_type = result.get("action_type", "?")
    status = "PASS" if actual_type == expected_type else "FAIL"
    print(f"  {status}: '{raw[:50]}...' -> {actual_type}")
    if actual_type != expected_type:
        all_pass = False

print()
print("=" * 60)
if all_pass:
    print("ALL PHASE 5 VALIDATION TESTS PASSED")
else:
    print("SOME TESTS FAILED - review above")
print("=" * 60)
