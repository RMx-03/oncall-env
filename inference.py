"""
Baseline inference script for the OnCall Incident Triage Environment.

Runs an LLM agent against all 3 task difficulties (easy, medium, hard)
and reports the grader scores.

Required environment variables:
    OPENAI_API_KEY   - API key (or provider-specific key)
    MODEL_NAME       - Model identifier (default per provider)

Optional environment variables:
    LLM_PROVIDER     - Provider: openai, openrouter, groq, gemini, custom
    API_BASE_URL     - Overrides provider's default base URL
    ENV_URL          - Environment server URL (default: http://localhost:8000)

Usage:
    # Start server first:
    uvicorn oncall_env.server.app:app --host 0.0.0.0 --port 8000

    # Run inference:
    set OPENAI_API_KEY=sk-...
    python inference.py

    # With a different provider:
    set LLM_PROVIDER=groq
    set GROQ_API_KEY=gsk_...
    python inference.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Provider configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROVIDER_CONFIGS: Dict[str, Dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "default_model": "meta-llama/llama-3-70b",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.0-flash",
    },
    "custom": {
        "base_url": "",  # Must set API_BASE_URL
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "",  # Must set MODEL_NAME
    },
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Resolve configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").lower()
if LLM_PROVIDER not in PROVIDER_CONFIGS:
    print(f"WARNING: Unknown LLM_PROVIDER '{LLM_PROVIDER}', falling back to 'openai'")
    LLM_PROVIDER = "openai"

_provider = PROVIDER_CONFIGS[LLM_PROVIDER]

# API_BASE_URL always overrides the provider default (hackathon compliance)
API_BASE_URL = os.environ.get("API_BASE_URL", _provider["base_url"])

# Hackathon spec: HF_TOKEN > OPENAI_API_KEY > provider-specific key
API_KEY = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get(_provider["api_key_env"], "")
)
MODEL_NAME = os.environ.get("MODEL_NAME", _provider["default_model"])

if not API_KEY:
    print(
        f"ERROR: No API key found. Set HF_TOKEN, OPENAI_API_KEY, or "
        f"{_provider['api_key_env']} environment variable."
    )
    sys.exit(1)

if not MODEL_NAME:
    print("ERROR: No model specified. Set MODEL_NAME environment variable.")
    sys.exit(1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_STEPS: int = 25
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 1024
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:8000")

FALLBACK_ACTION: Dict[str, str] = {"action_type": "check_alerts"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenAI client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# System prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = """You are an expert SRE/DevOps engineer responding to a production incident.
You have a LIMITED budget of actions. You MUST complete the full incident workflow before running out.

You interact with an incident response environment by outputting a SINGLE JSON action object.

Available actions:
1. INVESTIGATE (do 2-4 of these, no more):
   {"action_type": "check_alerts"}
   {"action_type": "check_service_status"}
   {"action_type": "query_logs", "service_name": "<service>"}
   {"action_type": "query_metrics", "service_name": "<service>"}

2. DIAGNOSE (do these after investigating):
   {"action_type": "set_severity", "severity": "<P1|P2|P3|P4>"}
   {"action_type": "identify_root_cause", "root_cause": "<description>", "root_cause_category": "<database|network|config|resource|dependency|security>"}

3. REMEDIATE (do this LAST — it ends the episode):
   {"action_type": "execute_remediation", "remediation_action": "<restart_service|scale_up|rollback_deploy|flush_cache|fix_config|failover_db>", "remediation_target": "<service>"}

Available services: api-gw, order-svc, order-db, payment-svc, auth-svc, notification-svc, cache, search-svc

You MUST follow this exact workflow:
  Step 1: check_alerts
  Step 2: check_service_status
  Step 3: query_logs for the most suspicious service
  Step 4: query_metrics for the most suspicious service
  Step 5: set_severity
  Step 6: identify_root_cause
  Step 7: execute_remediation

Do NOT repeat the same action. Do NOT keep investigating forever.
After 4-5 investigation actions, you MUST move to set_severity, then identify_root_cause, then execute_remediation.

CRITICAL RULES:
- Output ONLY a single JSON object. No text before or after.
- NEVER repeat an action you already took.
- By step 5 you MUST start diagnosing (set_severity).
- By step 7 you MUST execute remediation."""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt building
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_user_prompt(
    step: int,
    observation: Dict[str, Any],
    history: List[str],
) -> str:
    """Build the user prompt from the current observation and action history.

    Args:
        step: Current step number.
        observation: Dict of the current observation.
        history: List of previous action summaries.

    Returns:
        Formatted prompt string for the LLM.
    """
    remaining = MAX_STEPS - step + 1
    severity_set = observation.get('severity_set', False)
    root_cause_identified = observation.get('root_cause_identified', False)
    remediation_executed = observation.get('remediation_executed', False)

    parts: List[str] = [
        f"## Step {step}/{MAX_STEPS} ({remaining} steps remaining)",
        f"**Incident**: {observation.get('incident_summary', 'No summary available')}",
    ]

    # Show last action result
    action_result = observation.get('action_result', 'Episode started')
    parts.append(f"\n**Last action result**: {action_result}")

    # Log entries — show ERROR/FATAL/WARN first, then INFO
    log_entries = observation.get("log_entries")
    if log_entries:
        # Sort: errors first, then warnings, then info
        severity_order = {"FATAL": 0, "ERROR": 1, "WARN": 2, "WARNING": 2, "INFO": 3, "DEBUG": 4}
        sorted_logs = sorted(
            log_entries,
            key=lambda e: severity_order.get(e.get('level', 'INFO'), 3)
        )
        logs_text = "\n".join(
            f"  [{entry['level']}] [{entry['service']}] {entry['message']}"
            for entry in sorted_logs[:20]
        )
        parts.append(f"\n**Log entries** (errors shown first):\n{logs_text}")

    # Metric data
    metric_data = observation.get("metric_data")
    if metric_data:
        metrics_text = json.dumps(metric_data, indent=2, default=str)[:1500]
        parts.append(f"\n**Metrics**:\n{metrics_text}")

    # Alerts
    alerts = observation.get("alerts")
    if alerts:
        alerts_text = "\n".join(
            f"  [{a['severity'].upper()}] {a['title']} - {a['description']}"
            for a in alerts
        )
        parts.append(f"\n**Active alerts**:\n{alerts_text}")

    # Service statuses
    statuses = observation.get("service_statuses")
    if statuses:
        status_text = "\n".join(
            f"  {s['service']}: {s['status']}"
            for s in statuses
        )
        parts.append(f"\n**Service statuses**:\n{status_text}")

    # Progress flags
    parts.append(
        f"\n**Progress**: severity_set={severity_set}, "
        f"root_cause_identified={root_cause_identified}, "
        f"remediation_executed={remediation_executed}"
    )

    # Phase-aware urgency — this is the key to making the agent progress
    if step >= 5 and not severity_set:
        parts.append(
            "\n⚠️ URGENT: You have investigated enough. You MUST now set_severity. "
            "Output: {\"action_type\": \"set_severity\", \"severity\": \"P1|P2|P3|P4\"}"
        )
    elif severity_set and not root_cause_identified:
        parts.append(
            "\n⚠️ URGENT: Severity is set. You MUST now identify_root_cause. "
            "Output: {\"action_type\": \"identify_root_cause\", \"root_cause\": \"<description>\", "
            "\"root_cause_category\": \"database|network|config|resource|dependency|security\"}"
        )
    elif severity_set and root_cause_identified and not remediation_executed:
        parts.append(
            "\n⚠️ URGENT: Root cause identified. You MUST now execute_remediation. "
            "Output: {\"action_type\": \"execute_remediation\", \"remediation_action\": "
            "\"restart_service|scale_up|rollback_deploy|flush_cache|fix_config|failover_db\", "
            "\"remediation_target\": \"<service>\"}"
        )
    elif remaining <= 3 and not severity_set:
        parts.append(
            "\n🚨 CRITICAL: Almost out of steps! You MUST set_severity NOW."
        )

    parts.append("\nOutput your next action as a single JSON object:")

    return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _guess_target_service(observation: Dict[str, Any]) -> str:
    """Guess the most likely broken service from the observation.

    Looks at service_statuses for 'down' or 'degraded' services.
    Falls back to 'order-svc' as a reasonable default.
    """
    statuses = observation.get("service_statuses") or []
    # Prefer 'down' services
    for s in statuses:
        if isinstance(s, dict) and s.get("status") == "down":
            return s["service"]
    # Then 'degraded'
    for s in statuses:
        if isinstance(s, dict) and s.get("status") == "degraded":
            return s["service"]
    return "order-svc"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Action parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _extract_json_object(text: str) -> Optional[str]:
    """Extract the first top-level JSON object from text using bracket matching.

    Unlike regex, this properly handles nested braces, escaped quotes,
    and long string values.
    """
    start = text.find('{')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == '\\':
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def parse_model_action(response_text: str) -> Dict[str, Any]:
    """Extract a JSON action dict from the model's response text.

    Uses bracket-matching to properly handle long strings and nested objects.
    Falls back to ``check_alerts`` on parse failure.
    """
    if not response_text:
        return dict(FALLBACK_ACTION)

    # Strip markdown code fences
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try parsing the entire cleaned text first
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "action_type" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    # Use bracket-matching to extract the first JSON object
    json_str = _extract_json_object(response_text)
    if json_str:
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, dict) and "action_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    print(f"    WARNING: Could not parse action, using fallback. Raw: {response_text[:200]}")
    return dict(FALLBACK_ACTION)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Episode runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def run_episode(env_url: str, task_id: str) -> Dict[str, Any]:
    """Run a single episode against the environment and return results.

    Args:
        env_url: Base URL of the environment server.
        task_id: Task difficulty (easy, medium, hard).

    Returns:
        Dict with ``score``, ``steps``, ``total_reward``, ``duration_s``.
    """
    import requests

    start_time = time.time()

    # Reset environment
    resp = requests.post(
        f"{env_url}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    observation = result.get("observation", result)

    history: List[str] = []
    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    total_reward: float = 0.0
    steps_taken: int = 0

    for step in range(1, MAX_STEPS + 1):
        # Check if episode is done
        if result.get("done", False) or observation.get("done", False):
            break

        # Build prompt and call LLM with full conversation history
        user_prompt = build_user_prompt(step, observation, history)
        conversation.append({"role": "user", "content": user_prompt})

        # Keep conversation manageable — system + last 14 messages
        if len(conversation) > 15:
            messages = [conversation[0]] + conversation[-14:]
        else:
            messages = list(conversation)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as e:
            print(f"    LLM error at step {step}: {e}")
            response_text = json.dumps(FALLBACK_ACTION)

        # Add assistant response to conversation history
        conversation.append({"role": "assistant", "content": response_text})

        # Parse action
        action = parse_model_action(response_text)
        action_type = action.get("action_type", "?")

        # ── Force progression: prevent infinite loops ──
        severity_set = observation.get("severity_set", False)
        root_cause_identified = observation.get("root_cause_identified", False)

        if severity_set and root_cause_identified and action_type not in ("execute_remediation", "escalate"):
            # Agent already diagnosed but won't remediate — force it
            print(f"    (forced progression: {action_type} → execute_remediation)")
            # Use whatever the agent last said for remediation context
            if not action.get("remediation_action"):
                action = {
                    "action_type": "execute_remediation",
                    "remediation_action": "restart_service",
                    "remediation_target": _guess_target_service(observation),
                }
            else:
                action["action_type"] = "execute_remediation"
            action_type = "execute_remediation"

        print(f"    Step {step:2d}: {action_type:<25s}", end="")

        # Execute action (OpenEnv HTTP API expects {"action": {...}})
        try:
            resp = requests.post(
                f"{env_url}/step",
                json={"action": action},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            print(f" ERROR: {e}")
            result = {"done": True, "observation": {}, "reward": 0}

        observation = result.get("observation", result)
        reward = result.get("reward") or observation.get("reward", 0)
        total_reward += reward
        steps_taken = step
        done = result.get("done", False) or observation.get("done", False)

        print(f" reward={reward:+.3f}  done={done}")

        history.append(
            f"Step {step}: {action_type} -> reward {reward:+.3f}"
        )

        if done:
            break

    # Get final state for grading
    try:
        state_resp = requests.get(f"{env_url}/state", timeout=10)
        state_resp.raise_for_status()
        final_state = state_resp.json()
    except Exception:
        final_state = {}

    duration = time.time() - start_time

    # Compute grader score using the tasks module
    grader_score = _compute_grader_score(final_state)

    return {
        "score": grader_score,
        "steps": steps_taken,
        "total_reward": round(total_reward, 4),
        "duration_s": round(duration, 1),
    }


def _compute_grader_score(state_dict: Dict[str, Any]) -> float:
    """Compute the grader score from the final state dict.

    Tries to use the tasks.graders module; falls back to
    total_reward if the import fails.

    Args:
        state_dict: Final state dictionary from the server.

    Returns:
        Float score in [0.0, 1.0].
    """
    try:
        # Add project root to path for import
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from oncall_env.models import IncidentState
        from oncall_env.tasks.graders import grade_episode

        state = IncidentState.model_validate(state_dict)
        return grade_episode(state)
    except Exception as e:
        print(f"    WARNING: Could not compute grader score: {e}")
        return max(0.0, min(1.0, state_dict.get("total_reward", 0.0)))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    """Run baseline inference on all 3 tasks and print results."""
    print("=" * 60)
    print("OnCall Incident Triage | Baseline Inference")
    print("=" * 60)
    print(f"  Provider:    {LLM_PROVIDER}")
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Base URL:    {API_BASE_URL}")
    print(f"  Environment: {ENV_URL}")
    print(f"  Max steps:   {MAX_STEPS}")
    print("=" * 60)

    results: Dict[str, Dict[str, Any]] = {}

    for task_id in ["easy", "medium", "hard"]:
        print(f"\n{'─' * 50}")
        print(f"  TASK: {task_id.upper()}")
        print(f"{'─' * 50}")

        try:
            episode_result = run_episode(ENV_URL, task_id)
            results[task_id] = episode_result
            print(
                f"\n  Score: {episode_result['score']:.4f} | "
                f"Steps: {episode_result['steps']} | "
                f"Reward: {episode_result['total_reward']:+.4f} | "
                f"Time: {episode_result['duration_s']}s"
            )
        except Exception as e:
            print(f"\n  FAILED: {e}")
            results[task_id] = {"score": 0.0, "steps": 0, "total_reward": 0.0, "duration_s": 0.0}

    # Summary
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Task':<10s} {'Score':>8s} {'Steps':>7s} {'Reward':>9s} {'Time':>7s}")
    print(f"  {'─' * 44}")

    scores = []
    for task_id, r in results.items():
        score = r["score"]
        scores.append(score)
        print(
            f"  {task_id:<10s} {score:>8.4f} {r['steps']:>7d} "
            f"{r['total_reward']:>+9.4f} {r['duration_s']:>6.1f}s"
        )

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"  {'─' * 44}")
    print(f"  {'AVERAGE':<10s} {avg_score:>8.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
