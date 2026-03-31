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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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

MAX_STEPS: int = 10
MAX_POLICY_STEPS: int = 7
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 1024
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:8000")
DETERMINISTIC_BASELINE: bool = os.environ.get(
    "DETERMINISTIC_BASELINE", "1"
).lower() not in {
    "0",
    "false",
    "no",
}

FALLBACK_ACTION: Dict[str, str] = {"action_type": "check_alerts"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OpenAI client
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# System prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = """You are an expert SRE agent for incident triage.
Return exactly one valid JSON object and nothing else.
Keep root_cause under 20 words.
Use only these enums:
- severity: P1, P2, P3, P4
- root_cause_category: database, network, config, resource, dependency, security
- remediation_action: restart_service, scale_up, rollback_deploy, flush_cache, fix_config, failover_db
"""

VALID_ACTION_TYPES = {
    "check_alerts",
    "check_service_status",
    "query_logs",
    "query_metrics",
    "set_severity",
    "identify_root_cause",
    "execute_remediation",
    "escalate",
}

INVESTIGATION_ACTIONS = {
    "check_alerts",
    "check_service_status",
    "query_logs",
    "query_metrics",
}

VALID_SEVERITIES = {"P1", "P2", "P3", "P4"}
VALID_CATEGORIES = {
    "database",
    "network",
    "config",
    "resource",
    "dependency",
    "security",
}
VALID_REMEDIATIONS = {
    "restart_service",
    "scale_up",
    "rollback_deploy",
    "flush_cache",
    "fix_config",
    "failover_db",
}
VALID_SERVICES = {
    "api-gw",
    "order-svc",
    "order-db",
    "payment-svc",
    "auth-svc",
    "notification-svc",
    "cache",
    "search-svc",
}

PHASES = ("INVESTIGATE", "SEVERITY", "ROOT_CAUSE", "REMEDIATION", "DONE")
INVESTIGATION_PLAN = (
    "check_alerts",
    "check_service_status",
    "query_logs",
    "query_metrics",
)


@dataclass
class PolicyState:
    phase: str = "INVESTIGATE"
    step_in_phase: int = 0
    target_service: str = "order-svc"
    severity: str = "P2"
    category: str = "resource"
    remediation_action: str = "restart_service"
    root_cause_text: str = "Resource exhaustion causing instability."
    evidence: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    last_action_type: str = ""


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return {}
    return {}


def _normalize_observation(obj: Any) -> Dict[str, Any]:
    obs = _as_dict(obj)
    out: Dict[str, Any] = dict(obs)
    out.setdefault("alerts", [])
    out.setdefault("service_statuses", [])
    out.setdefault("log_entries", [])
    out.setdefault("metric_data", {})
    out["alerts"] = [_as_dict(a) for a in out.get("alerts") or []]
    out["service_statuses"] = [_as_dict(s) for s in out.get("service_statuses") or []]
    out["log_entries"] = [_as_dict(e) for e in out.get("log_entries") or []]
    return out


def _determine_phase(policy: PolicyState) -> str:
    return policy.phase


def build_user_prompt(
    step: int,
    observation: Dict[str, Any],
    phase: str,
    target_service: str,
) -> str:
    remaining = MAX_POLICY_STEPS - step + 1
    parts: List[str] = [
        f"step={step}/{MAX_POLICY_STEPS} remaining={remaining}",
        f"phase={phase}",
        f"incident={observation.get('incident_summary', '')}",
        f"last_result={observation.get('action_result', 'episode started')}",
        f"target_service_hint={target_service}",
    ]

    alerts = observation.get("alerts") or []
    if alerts:
        alert_lines = [
            f"[{a.get('severity', 'info')}] {a.get('service', '?')}: {a.get('title', '')}"
            for a in alerts[:8]
        ]
        parts.append("alerts:\n" + "\n".join(alert_lines))

    statuses = observation.get("service_statuses") or []
    if statuses:
        status_lines = [
            f"{s.get('service', '?')}={s.get('status', 'unknown')}" for s in statuses
        ]
        parts.append("statuses:\n" + "\n".join(status_lines))

    logs = observation.get("log_entries") or []
    if logs:
        log_lines = [
            f"[{e.get('level', 'INFO')}] {e.get('service', '?')}: {e.get('message', '')}"
            for e in logs[:12]
        ]
        parts.append("logs:\n" + "\n".join(log_lines))

    metrics = observation.get("metric_data")
    if metrics:
        parts.append("metrics:\n" + json.dumps(metrics, default=str)[:800])

    parts.append("Return one JSON action only.")
    return "\n".join(parts)


def _extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
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
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_model_action(response_text: str) -> Optional[Dict[str, Any]]:
    if not response_text:
        return None
    cleaned = response_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    json_str = _extract_json_object(cleaned)
    if not json_str:
        return None
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def _service_from_statuses(observation: Dict[str, Any]) -> str:
    statuses = observation.get("service_statuses") or []
    for entry in statuses:
        if isinstance(entry, dict) and entry.get("status") == "down":
            svc = entry.get("service")
            if svc in VALID_SERVICES:
                return svc
    for entry in statuses:
        if isinstance(entry, dict) and entry.get("status") == "degraded":
            svc = entry.get("service")
            if svc in VALID_SERVICES:
                return svc
    return "order-svc"


def _update_evidence(evidence: Dict[str, Any], observation: Dict[str, Any]) -> None:
    for service in VALID_SERVICES:
        evidence.setdefault("service_scores", {}).setdefault(service, 0)
    evidence.setdefault(
        "signals",
        {"database": 0, "network": 0, "config": 0, "resource": 0, "security": 0},
    )

    alerts = observation.get("alerts") or []
    critical_alerts = 0
    for a in alerts:
        if not isinstance(a, dict):
            continue
        sev = str(a.get("severity", "")).lower()
        svc = a.get("service")
        text = f"{a.get('title', '')} {a.get('description', '')}".lower()
        if sev == "critical":
            critical_alerts += 1
            if svc in VALID_SERVICES:
                evidence["service_scores"][svc] += 3
        if "database" in text or "connection" in text:
            evidence["signals"]["database"] += 1
        if "dns" in text or "tls" in text or "certificate" in text or "timeout" in text:
            evidence["signals"]["network"] += 1
        if "deploy" in text or "version" in text or "config" in text:
            evidence["signals"]["config"] += 1
        if (
            "oom" in text
            or "memory" in text
            or "cpu" in text
            or "disk" in text
            or "cache" in text
        ):
            evidence["signals"]["resource"] += 1
        if "auth" in text or "ssl" in text or "certificate" in text:
            evidence["signals"]["security"] += 1
    evidence["critical_alerts"] = max(
        evidence.get("critical_alerts", 0), critical_alerts
    )

    statuses = observation.get("service_statuses") or []
    down_count = 0
    degraded_count = 0
    for s in statuses:
        if not isinstance(s, dict):
            continue
        svc = s.get("service")
        status = s.get("status")
        if status == "down":
            down_count += 1
            if svc in VALID_SERVICES:
                evidence["service_scores"][svc] += 4
        elif status == "degraded":
            degraded_count += 1
            if svc in VALID_SERVICES:
                evidence["service_scores"][svc] += 2
    evidence["down_count"] = max(evidence.get("down_count", 0), down_count)
    evidence["degraded_count"] = max(evidence.get("degraded_count", 0), degraded_count)

    logs = observation.get("log_entries") or []
    for e in logs:
        if not isinstance(e, dict):
            continue
        svc = e.get("service")
        lvl = str(e.get("level", "INFO")).upper()
        msg = str(e.get("message", "")).lower()
        if svc in VALID_SERVICES and lvl in {"ERROR", "FATAL", "WARN"}:
            evidence["service_scores"][svc] += 2 if lvl in {"ERROR", "FATAL"} else 1
        if any(
            k in msg
            for k in (
                "connection pool",
                "too many connections",
                "remaining connection slots",
                "postgres",
                "database",
            )
        ):
            evidence["signals"]["database"] += 3
        if any(
            k in msg
            for k in (
                "dns",
                "tls",
                "certificate",
                "name resolution",
                "etimeout",
                "handshake",
                "timeout",
            )
        ):
            evidence["signals"]["network"] += 3
        if any(
            k in msg
            for k in ("deploy", "rollback", "version", "config", "route", "mismatch")
        ):
            evidence["signals"]["config"] += 3
        if any(
            k in msg
            for k in (
                "oom",
                "outofmemory",
                "heap",
                "exit code 137",
                "disk",
                "no space left",
                "cache miss",
                "eviction",
                "memory",
            )
        ):
            evidence["signals"]["resource"] += 3
        if any(
            k in msg
            for k in (
                "ssl",
                "certificate",
                "mtls",
                "handshake",
                "token validation",
            )
        ):
            evidence["signals"]["security"] += 2


def _infer_severity(evidence: Dict[str, Any]) -> str:
    down = evidence.get("down_count", 0)
    degraded = evidence.get("degraded_count", 0)
    critical = evidence.get("critical_alerts", 0)
    if down >= 2:
        return "P1"
    if down == 1:
        return "P2"
    if degraded >= 3 and critical >= 2:
        return "P1"
    if degraded >= 1:
        return "P2"
    if critical >= 1:
        return "P3"
    return "P4"


def _infer_category_and_remediation(evidence: Dict[str, Any]) -> Tuple[str, str, str]:
    signals = evidence.get("signals", {})
    category = max(
        ["database", "network", "config", "resource", "security"],
        key=lambda k: signals.get(k, 0),
    )
    action_map = {
        "database": "failover_db",
        "network": "fix_config",
        "config": "rollback_deploy",
        "resource": "restart_service",
        "security": "fix_config",
    }
    cause_map = {
        "database": "Database connection pool exhaustion causing request failures.",
        "network": "DNS/TLS resolution timeouts causing intermittent service failures.",
        "config": "Configuration mismatch after rollout causing endpoint failures.",
        "resource": "Resource exhaustion (memory/disk/cache) causing service instability.",
        "security": "Certificate/TLS validation failures breaking secure service communication.",
    }
    return category, action_map[category], cause_map[category]


def _advance_phase(policy: PolicyState, action_type: str, done: bool) -> None:
    if done:
        policy.phase = "DONE"
        policy.step_in_phase = 0
        return

    if policy.phase == "INVESTIGATE":
        if action_type in INVESTIGATION_ACTIONS:
            policy.step_in_phase += 1
        if policy.step_in_phase >= len(INVESTIGATION_PLAN):
            policy.phase = "SEVERITY"
            policy.step_in_phase = 0
        return

    if policy.phase == "SEVERITY" and action_type == "set_severity":
        policy.phase = "ROOT_CAUSE"
        policy.step_in_phase = 0
        return

    if policy.phase == "ROOT_CAUSE" and action_type == "identify_root_cause":
        policy.phase = "REMEDIATION"
        policy.step_in_phase = 0
        return

    if policy.phase == "REMEDIATION" and action_type in {
        "execute_remediation",
        "escalate",
    }:
        policy.phase = "DONE"
        policy.step_in_phase = 0


def _category_to_remediation(category: str) -> str:
    return {
        "database": "failover_db",
        "network": "fix_config",
        "config": "rollback_deploy",
        "resource": "restart_service",
        "security": "fix_config",
        "dependency": "scale_up",
    }.get(category, "restart_service")


def _build_root_cause_text(category: str, target: str) -> str:
    by_category = {
        "database": f"Database connection saturation affecting {target}.",
        "network": f"DNS/TLS networking instability impacting {target}.",
        "config": f"Configuration mismatch causing failures in {target}.",
        "resource": f"Resource exhaustion causing instability in {target}.",
        "security": f"Certificate or TLS validation failures affecting {target}.",
        "dependency": f"Downstream dependency degradation impacting {target}.",
    }
    return by_category.get(category, f"Operational issue affecting {target}.")


def _planned_action(
    policy: PolicyState, observation: Dict[str, Any], global_step: int
) -> Dict[str, Any]:
    category, remediation_action, _ = _infer_category_and_remediation(policy.evidence)
    target = _pick_target_service(observation, policy.evidence, category)
    policy.target_service = target
    policy.category = category
    policy.remediation_action = remediation_action
    policy.severity = _infer_severity(policy.evidence)
    policy.root_cause_text = _build_root_cause_text(category, target)

    if policy.phase == "INVESTIGATE":
        idx = min(policy.step_in_phase, len(INVESTIGATION_PLAN) - 1)
        planned = INVESTIGATION_PLAN[idx]
        if planned in {"query_logs", "query_metrics"}:
            return {"action_type": planned, "service_name": target}
        return {"action_type": planned}
    if policy.phase == "SEVERITY":
        return {"action_type": "set_severity", "severity": policy.severity}
    if policy.phase == "ROOT_CAUSE":
        return {
            "action_type": "identify_root_cause",
            "root_cause_category": policy.category,
            "root_cause": policy.root_cause_text,
        }
    if policy.phase == "REMEDIATION":
        return {
            "action_type": "execute_remediation",
            "remediation_action": policy.remediation_action,
            "remediation_target": target,
        }
    return {
        "action_type": "execute_remediation",
        "remediation_action": policy.remediation_action,
        "remediation_target": target,
    }


def _pick_target_service(
    observation: Dict[str, Any], evidence: Dict[str, Any], category: str
) -> str:
    statuses = observation.get("service_statuses") or []
    broken = [
        s.get("service")
        for s in statuses
        if isinstance(s, dict)
        and s.get("status") in ("down", "degraded")
        and s.get("service") in VALID_SERVICES
    ]
    service_scores = evidence.get("service_scores", {})

    preferred = {
        "database": "order-db",
        "network": "api-gw",
        "config": "order-svc",
        "resource": "order-svc",
        "security": "api-gw",
    }.get(category, "order-svc")

    if preferred in broken:
        return preferred

    if broken:
        return max(broken, key=lambda s: service_scores.get(s, 0))

    fallback = _service_from_statuses(observation)
    if fallback in VALID_SERVICES:
        return fallback
    return "order-svc"


def _planned_investigation_action(step: int, target_service: str) -> Dict[str, Any]:
    if step == 1:
        return {"action_type": "check_alerts"}
    if step == 2:
        return {"action_type": "check_service_status"}
    if step == 3:
        return {"action_type": "query_logs", "service_name": target_service}
    return {"action_type": "query_metrics", "service_name": target_service}


def _phase_fallback_action(
    policy: PolicyState,
    step: int,
    observation: Dict[str, Any],
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    policy.evidence = evidence
    return _planned_action(policy, observation, step)


def _validate_and_repair_action(
    action: Optional[Dict[str, Any]],
    policy: PolicyState,
    step: int,
    observation: Dict[str, Any],
    evidence: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool]:
    phase = policy.phase
    planned = _phase_fallback_action(policy, step, observation, evidence)

    if not isinstance(action, dict):
        return planned, True

    action_type = action.get("action_type")
    if action_type not in VALID_ACTION_TYPES:
        return planned, True

    expected_type = planned.get("action_type")
    if action_type != expected_type:
        return planned, True

    repaired = dict(action)

    if action_type in {"query_logs", "query_metrics"}:
        svc = repaired.get("service_name")
        if svc not in VALID_SERVICES:
            repaired["service_name"] = _pick_target_service(
                observation,
                evidence,
                _infer_category_and_remediation(evidence)[0],
            )
            return repaired, True
        return repaired, False

    if action_type == "set_severity":
        sev = repaired.get("severity")
        if sev not in VALID_SEVERITIES:
            repaired["severity"] = planned.get("severity", _infer_severity(evidence))
            return repaired, True
        return repaired, False

    if action_type == "identify_root_cause":
        category = planned.get(
            "root_cause_category", _infer_category_and_remediation(evidence)[0]
        )
        root_cause = planned.get(
            "root_cause", _build_root_cause_text(category, policy.target_service)
        )
        changed = False
        if repaired.get("root_cause_category") not in VALID_CATEGORIES:
            repaired["root_cause_category"] = category
            changed = True
        if not repaired.get("root_cause"):
            repaired["root_cause"] = root_cause
            changed = True
        return repaired, changed

    if action_type == "execute_remediation":
        category = planned.get(
            "root_cause_category", _infer_category_and_remediation(evidence)[0]
        )
        remediation_action = planned.get(
            "remediation_action", _category_to_remediation(category)
        )
        changed = False
        if repaired.get("remediation_action") not in VALID_REMEDIATIONS:
            repaired["remediation_action"] = remediation_action
            changed = True
        if repaired.get("remediation_target") not in VALID_SERVICES:
            repaired["remediation_target"] = _pick_target_service(
                observation, evidence, category
            )
            changed = True
        return repaired, changed

    return repaired, False


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

    # Reset environment (fixed seed for reproducibility — hackathon requirement)
    resp = requests.post(
        f"{env_url}/reset",
        json={"task_id": task_id, "seed": 42},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    observation = _normalize_observation(result.get("observation", result))

    history: List[str] = []
    conversation: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    total_reward: float = 0.0
    steps_taken: int = 0
    policy = PolicyState()
    evidence: Dict[str, Any] = policy.evidence
    diagnostics: Dict[str, Any] = {
        "parse_failures": 0,
        "validation_repairs": 0,
        "phase_transitions": [],
        "terminal_reached": False,
        "executed_remediation": False,
        "action_validity_rate": 0.0,
        "valid_actions": 0,
        "total_actions": 0,
    }
    policy.diagnostics = diagnostics
    last_phase = ""

    for step in range(1, MAX_POLICY_STEPS + 1):
        # Check if episode is done
        if result.get("done", False) or observation.get("done", False):
            diagnostics["terminal_reached"] = True
            break

        _update_evidence(evidence, observation)
        phase = _determine_phase(policy)
        if phase != last_phase:
            diagnostics["phase_transitions"].append(f"{step}:{phase}")
            last_phase = phase

        category_hint, _, _ = _infer_category_and_remediation(evidence)
        target_hint = _pick_target_service(observation, evidence, category_hint)
        policy.target_service = target_hint
        planned_action = _planned_action(policy, observation, step)

        action: Dict[str, Any]
        repaired = False
        if DETERMINISTIC_BASELINE:
            action = planned_action
        else:
            user_prompt = build_user_prompt(step, observation, phase, target_hint)
            conversation.append({"role": "user", "content": user_prompt})

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
                response_text = ""

            conversation.append({"role": "assistant", "content": response_text})
            raw_action = parse_model_action(response_text)
            if raw_action is None:
                diagnostics["parse_failures"] += 1

            action, repaired = _validate_and_repair_action(
                raw_action,
                policy,
                step,
                observation,
                evidence,
            )
            if repaired:
                diagnostics["validation_repairs"] += 1

        action_type = action.get("action_type", "?")

        # Debug: print what the LLM is actually sending for diagnosis actions
        if action_type == "set_severity":
            print(f"    [DEBUG] LLM severity='{action.get('severity')}'")
        elif action_type == "identify_root_cause":
            print(
                f"    [DEBUG] LLM category='{action.get('root_cause_category')}' cause='{str(action.get('root_cause', ''))[:60]}'"
            )
        elif action_type == "execute_remediation":
            print(
                f"    [DEBUG] LLM action='{action.get('remediation_action')}' target='{action.get('remediation_target')}'"
            )

        diagnostics["total_actions"] += 1
        if not repaired:
            diagnostics["valid_actions"] += 1
        if action_type == "execute_remediation":
            diagnostics["executed_remediation"] = True

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

        observation = _normalize_observation(result.get("observation", result))
        reward = result.get("reward") or observation.get("reward", 0)
        total_reward += reward
        steps_taken = step
        done = result.get("done", False) or observation.get("done", False)
        _advance_phase(policy, action_type, done)

        print(f" reward={reward:+.3f}  done={done}")

        history.append(f"Step {step}: {action_type} -> reward {reward:+.3f}")

        if done:
            diagnostics["terminal_reached"] = True
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
    grader_score, grader_breakdown = _compute_grader_score_and_breakdown(final_state)
    diagnostics["action_validity_rate"] = round(
        diagnostics["valid_actions"] / max(1, diagnostics["total_actions"]),
        3,
    )

    return {
        "score": grader_score,
        "steps": steps_taken,
        "total_reward": round(total_reward, 4),
        "duration_s": round(duration, 1),
        "diagnostics": diagnostics,
        "grader_breakdown": grader_breakdown,
    }


def _compute_grader_score_and_breakdown(
    state_dict: Dict[str, Any],
) -> Tuple[float, Dict[str, float]]:
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
        from oncall_env.tasks.graders import grade_episode, grade_episode_breakdown

        state = IncidentState.model_validate(state_dict)
        return grade_episode(state), grade_episode_breakdown(state)
    except Exception as e:
        print(f"    WARNING: Could not compute grader score: {e}")
        fallback = max(0.0, min(1.0, state_dict.get("total_reward", 0.0)))
        return fallback, {}


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
    print(f"  Max steps:   {MAX_POLICY_STEPS}")
    print(f"  Deterministic baseline: {DETERMINISTIC_BASELINE}")
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
            d = episode_result.get("diagnostics", {})
            print(
                "  Diagnostics: "
                f"parse_failures={d.get('parse_failures', 0)}, "
                f"repairs={d.get('validation_repairs', 0)}, "
                f"terminal_reached={d.get('terminal_reached', False)}, "
                f"executed_remediation={d.get('executed_remediation', False)}, "
                f"action_validity_rate={d.get('action_validity_rate', 0.0):.3f}"
            )
            transitions = d.get("phase_transitions", [])
            if transitions:
                print(f"  Phase transitions: {' -> '.join(transitions)}")
            br = episode_result.get("grader_breakdown", {})
            if br:
                print(
                    "  Grader breakdown: "
                    f"severity={br.get('severity', 0.0):.2f}, "
                    f"category={br.get('root_cause_category', 0.0):.2f}, "
                    f"description={br.get('root_cause_description', 0.0):.2f}, "
                    f"action={br.get('remediation_action', 0.0):.2f}, "
                    f"target={br.get('remediation_target', 0.0):.2f}, "
                    f"efficiency={br.get('efficiency', 0.0):.2f}"
                )
        except Exception as e:
            print(f"\n  FAILED: {e}")
            results[task_id] = {
                "score": 0.0,
                "steps": 0,
                "total_reward": 0.0,
                "duration_s": 0.0,
                "diagnostics": {},
                "grader_breakdown": {},
            }

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
