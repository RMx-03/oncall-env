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
API_KEY = os.environ.get(
    "OPENAI_API_KEY",
    os.environ.get(_provider["api_key_env"], ""),
)
MODEL_NAME = os.environ.get("MODEL_NAME", _provider["default_model"])

if not API_KEY:
    print(
        f"ERROR: No API key found. Set OPENAI_API_KEY or "
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

You interact with an incident response environment by outputting a JSON action at each step.

Available actions:
- {"action_type": "query_logs", "service_name": "<service>", "time_range": "last_15m"}
- {"action_type": "query_metrics", "service_name": "<service>", "metric_name": "<cpu|memory|latency_p99|error_rate|connections>"}
- {"action_type": "check_alerts"}
- {"action_type": "check_alerts", "service_name": "<service>"}
- {"action_type": "check_service_status"} (all services) or {"action_type": "check_service_status", "service_name": "<service>"}
- {"action_type": "set_severity", "severity": "<P1|P2|P3|P4>"}
- {"action_type": "identify_root_cause", "root_cause": "<description>", "root_cause_category": "<database|network|config|resource|dependency|security>"}
- {"action_type": "execute_remediation", "remediation_action": "<restart_service|scale_up|rollback_deploy|flush_cache|fix_config|failover_db>", "remediation_target": "<service>"}
- {"action_type": "escalate", "escalation_team": "<database|network|security|platform>"}

Available services: api-gw, order-svc, order-db, payment-svc, auth-svc, notification-svc, cache, search-svc

Strategy:
1. First check all alerts and service statuses to get an overview
2. Query logs from affected services to find error patterns
3. Query metrics to confirm hypotheses
4. Set severity based on impact
5. Identify root cause category and description
6. Execute the appropriate remediation

IMPORTANT: Output ONLY a single JSON action object. No explanation before or after."""


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
    parts: List[str] = [
        f"## Step {step}/{MAX_STEPS}",
        f"**Incident**: {observation.get('incident_summary', 'No summary available')}",
        f"**Goal**: {observation.get('goal', 'Investigate and resolve the incident')}",
        f"\n**Last action result**: {observation.get('action_result', 'Episode started')}",
    ]

    # Log entries
    log_entries = observation.get("log_entries")
    if log_entries:
        logs_text = "\n".join(
            f"  [{entry['timestamp']}] {entry['level']} [{entry['service']}] {entry['message']}"
            for entry in log_entries[:20]  # Cap to avoid token overflow
        )
        parts.append(f"\n**Log entries**:\n{logs_text}")

    # Metric data
    metric_data = observation.get("metric_data")
    if metric_data:
        metrics_text = json.dumps(metric_data, indent=2, default=str)[:2000]
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
            f"  {s['service']}: {s['status']} (uptime: {s['uptime']}, last deploy: {s['last_deploy']})"
            for s in statuses
        )
        parts.append(f"\n**Service statuses**:\n{status_text}")

    # Progress
    parts.append(
        f"\n**Progress**: severity_set={observation.get('severity_set', False)}, "
        f"root_cause_identified={observation.get('root_cause_identified', False)}, "
        f"remediation_executed={observation.get('remediation_executed', False)}"
    )
    parts.append(
        f"**Available services**: {observation.get('available_services', [])}"
    )

    # Action history (last 10)
    if history:
        parts.append("\n**Action history**:\n" + "\n".join(history[-10:]))

    parts.append("\n**Output your next action as a JSON object:**")

    return "\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Action parsing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def parse_model_action(response_text: str) -> Dict[str, Any]:
    """Extract a JSON action dict from the model's response text.

    Handles responses wrapped in markdown code blocks or with
    surrounding explanation text.

    Args:
        response_text: Raw text from the LLM.

    Returns:
        Parsed action dict. Falls back to ``check_alerts`` on failure.
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

    # Try to find JSON object in the text
    json_match = re.search(r"\{[^{}]*\}", response_text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict) and "action_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Try nested JSON (for cases with nested objects)
    json_match = re.search(r"\{[^}]*\{[^}]*\}[^}]*\}", response_text)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, dict) and "action_type" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    print(f"    WARNING: Could not parse action, using fallback. Raw: {response_text[:100]}")
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
    total_reward: float = 0.0
    steps_taken: int = 0

    for step in range(1, MAX_STEPS + 1):
        # Check if episode is done
        if result.get("done", False) or observation.get("done", False):
            break

        # Build prompt and call LLM
        user_prompt = build_user_prompt(step, observation, history)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

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

        # Parse action
        action = parse_model_action(response_text)
        action_type = action.get("action_type", "?")
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
