"""
Alert generation for the incident triage environment.

Converts scenario alert configuration into typed ``Alert`` objects.
This is intentionally simple — alerts are fully defined in the
scenario data; the generator just instantiates them.

All functions are pure — no side effects.
"""

from __future__ import annotations

from typing import Any, Dict, List

from oncall_env.models import Alert


def generate_alerts(scenario_config: Dict[str, Any]) -> List[Alert]:
    """Convert scenario alert configuration into Alert objects.

    Args:
        scenario_config: Scenario dict containing ``alert_config``,
            a list of dicts with keys matching the ``Alert`` model.

    Returns:
        List of ``Alert`` objects, one per alert in the config.
    """
    alert_configs: List[Dict[str, Any]] = scenario_config.get("alert_config", [])
    return [Alert(**cfg) for cfg in alert_configs]
