"""Integration test: full episode flows through the environment + grading."""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oncall_env.server.environment import IncidentTriageEnvironment
from oncall_env.models import IncidentAction
from oncall_env.tasks.graders import grade_episode, grade_episode_breakdown


@pytest.fixture
def env():
    return IncidentTriageEnvironment()


class TestFullEpisode:
    """End-to-end episode: investigate → diagnose → remediate → grade."""

    def test_easy_perfect_episode(self, env):
        """Play through an easy scenario with correct answers."""
        obs = env.reset(task_id="easy", seed=0)
        assert obs.done is False

        # Get ground truth for this scenario
        gt = env.state
        gt_severity = gt.ground_truth_severity
        gt_category = gt.ground_truth_category
        gt_root_cause = gt.ground_truth_root_cause
        gt_remediation = gt.ground_truth_remediation
        gt_target = gt.ground_truth_target

        # Step 1: Check alerts
        obs = env.step(IncidentAction(action_type="check_alerts"))
        assert obs.action_success is True
        assert obs.done is False

        # Step 2: Check service status
        obs = env.step(IncidentAction(action_type="check_service_status"))
        assert obs.service_statuses is not None

        # Step 3: Query logs from a service
        obs = env.step(IncidentAction(
            action_type="query_logs",
            service_name=gt_target,
        ))
        assert obs.log_entries is not None

        # Step 4: Query metrics
        obs = env.step(IncidentAction(
            action_type="query_metrics",
            service_name=gt_target,
        ))
        assert obs.metric_data is not None

        # Step 5: Set severity
        obs = env.step(IncidentAction(
            action_type="set_severity",
            severity=gt_severity,
        ))
        assert obs.severity_set is True

        # Step 6: Identify root cause
        obs = env.step(IncidentAction(
            action_type="identify_root_cause",
            root_cause=gt_root_cause,
            root_cause_category=gt_category,
        ))
        assert obs.root_cause_identified is True

        # Step 7: Execute remediation (terminal)
        obs = env.step(IncidentAction(
            action_type="execute_remediation",
            remediation_action=gt_remediation,
            remediation_target=gt_target,
        ))
        assert obs.done is True

        # Grade
        score = grade_episode(env.state)
        assert 0.9 <= score <= 1.0

        breakdown = grade_episode_breakdown(env.state)
        assert breakdown["total"] >= 0.9

    @pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_perfect_across_difficulties(self, env, task_id, seed):
        """Perfect play on every scenario should yield >= 0.9."""
        obs = env.reset(task_id=task_id, seed=seed)
        gt = env.state

        env.step(IncidentAction(action_type="check_alerts"))
        env.step(IncidentAction(action_type="check_service_status"))
        env.step(IncidentAction(
            action_type="query_logs",
            service_name=gt.ground_truth_target,
        ))
        env.step(IncidentAction(
            action_type="set_severity",
            severity=gt.ground_truth_severity,
        ))
        env.step(IncidentAction(
            action_type="identify_root_cause",
            root_cause=gt.ground_truth_root_cause,
            root_cause_category=gt.ground_truth_category,
        ))
        obs = env.step(IncidentAction(
            action_type="execute_remediation",
            remediation_action=gt.ground_truth_remediation,
            remediation_target=gt.ground_truth_target,
        ))

        assert obs.done is True
        score = grade_episode(env.state)
        assert 0.9 <= score <= 1.0, f"{task_id}/seed={seed}: score={score}"

    def test_wrong_answers_low_score(self, env):
        """Wrong answers should produce a low score."""
        env.reset(task_id="easy", seed=0)

        env.step(IncidentAction(action_type="set_severity", severity="P4"))
        env.step(IncidentAction(
            action_type="identify_root_cause",
            root_cause="everything is fine",
            root_cause_category="security",
        ))
        obs = env.step(IncidentAction(
            action_type="execute_remediation",
            remediation_action="flush_cache",
            remediation_target="cache",
        ))

        assert obs.done is True
        score = grade_episode(env.state)
        assert score < 0.15

    def test_escalation_episode(self, env):
        """Escalation should terminate the episode."""
        env.reset(task_id="easy", seed=0)

        env.step(IncidentAction(action_type="check_alerts"))
        env.step(IncidentAction(action_type="set_severity", severity="P1"))
        obs = env.step(IncidentAction(
            action_type="escalate",
            escalation_team="database",
        ))

        assert obs.done is True
        score = grade_episode(env.state)
        # Score should be partial — got severity but no remediation
        assert 0.0 <= score <= 1.0

    def test_max_actions_episode(self, env):
        """Burning all actions should produce a valid score."""
        env.reset(task_id="easy", seed=0)

        for _ in range(35):
            obs = env.step(IncidentAction(action_type="check_alerts"))
            if obs.done:
                break

        assert obs.done is True
        score = grade_episode(env.state)
        assert 0.0 <= score <= 1.0

    def test_score_range_always_valid(self, env):
        """Score is in [0.0, 1.0] regardless of play pattern."""
        for task_id in ["easy", "medium", "hard"]:
            env.reset(task_id=task_id, seed=0)

            # Random-ish sequence of actions
            env.step(IncidentAction(action_type="check_alerts"))
            env.step(IncidentAction(
                action_type="query_logs",
                service_name="api-gw",
            ))
            env.step(IncidentAction(action_type="set_severity", severity="P3"))
            obs = env.step(IncidentAction(
                action_type="execute_remediation",
                remediation_action="restart_service",
                remediation_target="api-gw",
            ))

            assert obs.done is True
            score = grade_episode(env.state)
            assert 0.0 <= score <= 1.0, f"{task_id}: score={score}"


class TestRewardAccumulation:
    """Test that rewards accumulate correctly."""

    def test_rewards_are_floats(self, env):
        env.reset(task_id="easy", seed=0)
        obs = env.step(IncidentAction(action_type="check_alerts"))
        assert isinstance(obs.reward, (int, float)) or obs.reward is None

    def test_total_reward_tracked(self, env):
        env.reset(task_id="easy", seed=0)
        env.step(IncidentAction(action_type="set_severity", severity="P1"))
        env.step(IncidentAction(
            action_type="execute_remediation",
            remediation_action="restart_service",
            remediation_target="order-svc",
        ))
        # Total reward should be accumulated in state
        assert isinstance(env.state.total_reward, float)
