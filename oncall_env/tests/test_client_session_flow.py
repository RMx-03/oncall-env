"""Integration guard for client session persistence over reset/step/state."""

from __future__ import annotations

import os

import pytest
import requests


pytest.importorskip("openenv.core.env_client")


def _health_ok(base_url: str) -> bool:
    try:
        resp = requests.get(f"{base_url}/health", timeout=3)
        resp.raise_for_status()
        return True
    except Exception:
        return False


def _has_incident_signals(alert_obs, status_obs) -> bool:
    alerts = alert_obs.alerts or []
    statuses = status_obs.service_statuses or []
    has_alerts = len(alerts) > 0
    has_broken_status = any(s.status in {"down", "degraded"} for s in statuses)
    return has_alerts or has_broken_status


def test_client_reset_step_state_persistence() -> None:
    """Verify reset -> step -> state persistence via typed client session."""
    base_url = os.getenv("ONCALL_TEST_ENV_URL", "http://localhost:8000")
    if not _health_ok(base_url):
        pytest.skip(f"Environment unavailable at {base_url}")

    from oncall_env import IncidentAction, IncidentTriageEnv

    with IncidentTriageEnv(base_url=base_url).sync() as env:
        reset_result = env.reset(task_id="easy", seed=42)
        assert reset_result.observation.incident_summary != ""

        alerts_result = env.step(IncidentAction(action_type="check_alerts"))
        status_result = env.step(IncidentAction(action_type="check_service_status"))

        # EnvClient implementations may expose state as method or property.
        state_attr = getattr(env, "state")
        state = state_attr() if callable(state_attr) else state_attr

        assert _has_incident_signals(
            alerts_result.observation,
            status_result.observation,
        ), "Expected alerts or degraded/down statuses after reset"
        assert status_result.observation.investigation_actions_taken >= 2
        assert state.step_count >= 2
        assert (
            state.ground_truth_category is not None
            and state.ground_truth_category != ""
        )
