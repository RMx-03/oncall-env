"""Unit tests for IncidentTriageEnvironment — reset, step, state, all action handlers."""
import sys
import os

import pytest

# Ensure workspace root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oncall_env.server.environment import (
    IncidentTriageEnvironment,
    MAX_ACTIONS,
    AVAILABLE_SERVICES,
)
from oncall_env.models import IncidentAction, IncidentObservation, IncidentState


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def env():
    """Return a fresh environment instance."""
    return IncidentTriageEnvironment()


@pytest.fixture
def env_easy(env):
    """Return an environment reset with the easy task."""
    env.reset(task_id="easy", seed=0)
    return env


@pytest.fixture
def env_medium(env):
    """Return an environment reset with the medium task."""
    env.reset(task_id="medium", seed=0)
    return env


@pytest.fixture
def env_hard(env):
    """Return an environment reset with the hard task."""
    env.reset(task_id="hard", seed=0)
    return env


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7.1 — Reset / Step / State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestReset:
    """Tests for environment reset."""

    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="easy", seed=0)
        assert isinstance(obs, IncidentObservation)

    def test_reset_not_done(self, env):
        obs = env.reset(task_id="easy")
        assert obs.done is False

    def test_reset_has_incident_summary(self, env):
        obs = env.reset(task_id="easy")
        assert obs.incident_summary != ""
        assert len(obs.incident_summary) > 10

    def test_reset_has_goal(self, env):
        obs = env.reset(task_id="easy")
        assert obs.goal != ""
        assert "SRE" in obs.goal or "incident" in obs.goal.lower()

    def test_reset_has_available_services(self, env):
        obs = env.reset(task_id="easy")
        assert obs.available_services == list(AVAILABLE_SERVICES)
        assert len(obs.available_services) == 8

    def test_reset_max_actions(self, env):
        obs = env.reset(task_id="easy")
        assert obs.max_actions == MAX_ACTIONS

    def test_reset_zero_actions_taken(self, env):
        obs = env.reset(task_id="easy")
        assert obs.investigation_actions_taken == 0

    def test_reset_no_data_fields(self, env):
        obs = env.reset(task_id="easy")
        assert obs.log_entries is None
        assert obs.metric_data is None
        assert obs.alerts is None
        assert obs.service_statuses is None

    def test_reset_progress_flags_false(self, env):
        obs = env.reset(task_id="easy")
        assert obs.severity_set is False
        assert obs.root_cause_identified is False
        assert obs.remediation_executed is False

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    def test_reset_all_difficulties(self, env, task_id):
        obs = env.reset(task_id=task_id)
        assert obs.done is False
        assert obs.incident_summary != ""

    def test_reset_deterministic_seed(self, env):
        obs1 = env.reset(task_id="easy", seed=42)
        summary1 = obs1.incident_summary
        obs2 = env.reset(task_id="easy", seed=42)
        assert obs2.incident_summary == summary1

    def test_reset_clears_previous_episode(self, env):
        env.reset(task_id="easy")
        env.step(IncidentAction(action_type="check_alerts"))
        obs = env.reset(task_id="medium")
        assert obs.investigation_actions_taken == 0
        assert obs.done is False


class TestState:
    """Tests for the state property."""

    def test_state_type(self, env_easy):
        assert isinstance(env_easy.state, IncidentState)

    def test_state_has_ground_truth(self, env_easy):
        state = env_easy.state
        assert state.ground_truth_severity != ""
        assert state.ground_truth_root_cause != ""
        assert state.ground_truth_category != ""
        assert state.ground_truth_remediation != ""
        assert state.ground_truth_target != ""

    def test_state_step_count_increments(self, env_easy):
        assert env_easy.state.step_count == 0
        env_easy.step(IncidentAction(action_type="check_alerts"))
        assert env_easy.state.step_count == 1
        env_easy.step(IncidentAction(action_type="check_service_status"))
        assert env_easy.state.step_count == 2

    def test_state_task_id(self, env_easy):
        assert env_easy.state.task_id == "easy"

    def test_state_done_false_initially(self, env_easy):
        assert env_easy.state.done is False

    def test_state_agent_answers_none_initially(self, env_easy):
        state = env_easy.state
        assert state.agent_severity is None
        assert state.agent_root_cause is None
        assert state.agent_category is None
        assert state.agent_remediation is None
        assert state.agent_remediation_target is None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7.2 — All 8 Action Handlers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestQueryLogs:
    """Tests for query_logs action."""

    def test_query_logs_returns_entries(self, env_easy):
        obs = env_easy.step(IncidentAction(action_type="query_logs", service_name="order-svc"))
        assert obs.action_success is True
        assert obs.log_entries is not None
        assert len(obs.log_entries) > 0

    def test_query_logs_invalid_service(self, env_easy):
        obs = env_easy.step(IncidentAction(action_type="query_logs", service_name="nonexistent"))
        assert obs.action_success is False

    def test_query_logs_missing_service(self, env_easy):
        obs = env_easy.step(IncidentAction(action_type="query_logs"))
        assert obs.action_success is False

    def test_query_logs_tracks_investigation(self, env_easy):
        env_easy.step(IncidentAction(action_type="query_logs", service_name="order-svc"))
        assert "order-svc" in env_easy.state.services_investigated


class TestQueryMetrics:
    """Tests for query_metrics action."""

    def test_query_metrics_returns_data(self, env_easy):
        obs = env_easy.step(IncidentAction(action_type="query_metrics", service_name="order-svc"))
        assert obs.action_success is True
        assert obs.metric_data is not None

    def test_query_metrics_invalid_service(self, env_easy):
        obs = env_easy.step(IncidentAction(action_type="query_metrics", service_name="bad-svc"))
        assert obs.action_success is False


class TestCheckAlerts:
    """Tests for check_alerts action."""

    def test_check_alerts_all(self, env_easy):
        obs = env_easy.step(IncidentAction(action_type="check_alerts"))
        assert obs.action_success is True
        assert obs.alerts is not None

    def test_check_alerts_for_service(self, env_easy):
        obs = env_easy.step(IncidentAction(action_type="check_alerts", service_name="order-svc"))
        assert obs.action_success is True
        assert obs.alerts is not None


class TestCheckServiceStatus:
    """Tests for check_service_status action."""

    def test_check_all_statuses(self, env_easy):
        obs = env_easy.step(IncidentAction(action_type="check_service_status"))
        assert obs.action_success is True
        assert obs.service_statuses is not None
        assert len(obs.service_statuses) == 8

    def test_check_single_service_status(self, env_easy):
        obs = env_easy.step(IncidentAction(
            action_type="check_service_status",
            service_name="api-gw",
        ))
        assert obs.action_success is True
        assert obs.service_statuses is not None


class TestSetSeverity:
    """Tests for set_severity action."""

    def test_set_severity(self, env_easy):
        obs = env_easy.step(IncidentAction(action_type="set_severity", severity="P1"))
        assert obs.action_success is True
        assert obs.severity_set is True
        assert env_easy.state.agent_severity == "P1"

    def test_set_severity_updates_state(self, env_easy):
        env_easy.step(IncidentAction(action_type="set_severity", severity="P3"))
        assert env_easy.state.agent_severity == "P3"
        # Can override
        env_easy.step(IncidentAction(action_type="set_severity", severity="P1"))
        assert env_easy.state.agent_severity == "P1"


class TestIdentifyRootCause:
    """Tests for identify_root_cause action."""

    def test_identify_root_cause(self, env_easy):
        obs = env_easy.step(IncidentAction(
            action_type="identify_root_cause",
            root_cause="OOM crash in order-svc",
            root_cause_category="resource",
        ))
        assert obs.action_success is True
        assert obs.root_cause_identified is True
        assert env_easy.state.agent_root_cause == "OOM crash in order-svc"
        assert env_easy.state.agent_category == "resource"


class TestExecuteRemediation:
    """Tests for execute_remediation action (terminal)."""

    def test_remediation_terminates_episode(self, env_easy):
        obs = env_easy.step(IncidentAction(
            action_type="execute_remediation",
            remediation_action="restart_service",
            remediation_target="order-svc",
        ))
        assert obs.done is True
        assert env_easy.state.done is True
        assert env_easy.state.agent_remediation == "restart_service"
        assert env_easy.state.agent_remediation_target == "order-svc"

    def test_no_actions_after_remediation(self, env_easy):
        env_easy.step(IncidentAction(
            action_type="execute_remediation",
            remediation_action="restart_service",
            remediation_target="order-svc",
        ))
        obs = env_easy.step(IncidentAction(action_type="check_alerts"))
        assert obs.action_success is False
        assert obs.done is True


class TestEscalate:
    """Tests for escalate action (terminal)."""

    def test_escalate_terminates_episode(self, env_easy):
        obs = env_easy.step(IncidentAction(
            action_type="escalate",
            escalation_team="database",
        ))
        assert obs.done is True
        assert env_easy.state.done is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7.3 — Episode Termination
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEpisodeTermination:
    """Tests for episode termination conditions."""

    def test_max_actions_terminates(self, env_easy):
        obs = None
        for i in range(MAX_ACTIONS + 5):
            obs = env_easy.step(IncidentAction(action_type="check_alerts"))
            if obs.done:
                break
        assert obs is not None
        assert obs.done is True

    def test_max_actions_count(self, env_easy):
        for _ in range(MAX_ACTIONS):
            obs = env_easy.step(IncidentAction(action_type="check_alerts"))
            if obs.done:
                break
        assert env_easy.state.step_count <= MAX_ACTIONS

    def test_remediation_is_terminal(self, env_easy):
        obs = env_easy.step(IncidentAction(
            action_type="execute_remediation",
            remediation_action="scale_up",
            remediation_target="order-db",
        ))
        assert obs.done is True

    def test_escalation_is_terminal(self, env_easy):
        obs = env_easy.step(IncidentAction(
            action_type="escalate",
            escalation_team="network",
        ))
        assert obs.done is True

    def test_investigation_not_terminal(self, env_easy):
        obs = env_easy.step(IncidentAction(action_type="check_alerts"))
        assert obs.done is False

        obs = env_easy.step(IncidentAction(
            action_type="query_logs",
            service_name="order-svc",
        ))
        assert obs.done is False

        obs = env_easy.step(IncidentAction(action_type="check_service_status"))
        assert obs.done is False

    def test_set_severity_not_terminal(self, env_easy):
        obs = env_easy.step(IncidentAction(action_type="set_severity", severity="P2"))
        assert obs.done is False

    def test_identify_root_cause_not_terminal(self, env_easy):
        obs = env_easy.step(IncidentAction(
            action_type="identify_root_cause",
            root_cause="test",
            root_cause_category="database",
        ))
        assert obs.done is False
