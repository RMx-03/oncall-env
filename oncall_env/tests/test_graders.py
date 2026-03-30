"""Unit tests for the deterministic grading system."""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from oncall_env.models import IncidentState
from oncall_env.tasks.graders import (
    grade_episode,
    grade_episode_breakdown,
    WEIGHT_SEVERITY,
    WEIGHT_ROOT_CAUSE_CATEGORY,
    WEIGHT_ROOT_CAUSE_DESCRIPTION,
    WEIGHT_REMEDIATION_ACTION,
    WEIGHT_REMEDIATION_TARGET,
    WEIGHT_EFFICIENCY_FAST,
    WEIGHT_EFFICIENCY_MEDIUM,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _make_state(**overrides) -> IncidentState:
    """Build an IncidentState with ground truth defaults."""
    defaults = {
        "episode_id": "test-ep",
        "step_count": 8,
        "task_id": "easy",
        "task_difficulty": "easy",
        "ground_truth_severity": "P2",
        "ground_truth_root_cause": "Database connection pool exhaustion on order-db due to slow queries from recent deploy",
        "ground_truth_category": "database",
        "ground_truth_remediation": "rollback_deploy",
        "ground_truth_target": "order-svc",
        "done": True,
    }
    defaults.update(overrides)
    return IncidentState(**defaults)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Perfect Score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPerfectScore:
    """Agent gets everything exactly right."""

    def test_perfect_score_near_1(self):
        state = _make_state(
            agent_severity="P2",
            agent_category="database",
            agent_root_cause="connection pool exhaustion order-db slow queries deploy",
            agent_remediation="rollback_deploy",
            agent_remediation_target="order-svc",
            step_count=8,
        )
        score = grade_episode(state)
        assert score >= 0.95
        assert score <= 1.0

    def test_perfect_breakdown(self):
        state = _make_state(
            agent_severity="P2",
            agent_category="database",
            agent_root_cause="Database connection pool exhaustion on order-db due to slow queries from recent deploy",
            agent_remediation="rollback_deploy",
            agent_remediation_target="order-svc",
            step_count=5,
        )
        breakdown = grade_episode_breakdown(state)
        assert breakdown["severity"] == WEIGHT_SEVERITY
        assert breakdown["root_cause_category"] == WEIGHT_ROOT_CAUSE_CATEGORY
        assert breakdown["remediation_action"] == WEIGHT_REMEDIATION_ACTION
        assert breakdown["remediation_target"] == WEIGHT_REMEDIATION_TARGET
        assert breakdown["efficiency"] == WEIGHT_EFFICIENCY_FAST
        assert breakdown["total"] == 1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Zero Score
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestZeroScore:
    """Agent provides no answers at all."""

    def test_zero_score(self):
        state = _make_state(
            agent_severity=None,
            agent_category=None,
            agent_root_cause=None,
            agent_remediation=None,
            agent_remediation_target=None,
            step_count=30,
        )
        score = grade_episode(state)
        assert score == 0.0

    def test_zero_breakdown(self):
        state = _make_state(
            agent_severity=None,
            agent_category=None,
            agent_root_cause=None,
            agent_remediation=None,
            agent_remediation_target=None,
            step_count=30,
        )
        breakdown = grade_episode_breakdown(state)
        assert breakdown["severity"] == 0.0
        assert breakdown["root_cause_category"] == 0.0
        assert breakdown["root_cause_description"] == 0.0
        assert breakdown["remediation_action"] == 0.0
        assert breakdown["remediation_target"] == 0.0
        assert breakdown["efficiency"] == 0.0
        assert breakdown["total"] == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Severity Off-by-One
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestSeverityScoring:
    """Tests for severity partial credit."""

    def test_exact_match(self):
        state = _make_state(agent_severity="P2")
        breakdown = grade_episode_breakdown(state)
        assert breakdown["severity"] == WEIGHT_SEVERITY

    def test_off_by_one_above(self):
        state = _make_state(agent_severity="P1")  # GT is P2
        breakdown = grade_episode_breakdown(state)
        assert breakdown["severity"] == 0.10  # Half credit

    def test_off_by_one_below(self):
        state = _make_state(agent_severity="P3")  # GT is P2
        breakdown = grade_episode_breakdown(state)
        assert breakdown["severity"] == 0.10

    def test_off_by_two(self):
        state = _make_state(agent_severity="P4")  # GT is P2
        breakdown = grade_episode_breakdown(state)
        assert breakdown["severity"] == 0.0

    def test_missing_severity(self):
        state = _make_state(agent_severity=None)
        breakdown = grade_episode_breakdown(state)
        assert breakdown["severity"] == 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Partial Credit Scenarios
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestPartialCredit:
    """Agent gets some things right, some wrong."""

    def test_right_action_wrong_target(self):
        state = _make_state(
            agent_severity="P2",
            agent_category="database",
            agent_root_cause="database issue",
            agent_remediation="rollback_deploy",
            agent_remediation_target="order-db",  # GT is order-svc
        )
        score = grade_episode(state)
        # Gets severity + category + action + some description + efficiency
        # Misses target (0.15)
        assert 0.5 < score < 0.95

    def test_right_severity_wrong_everything(self):
        state = _make_state(
            agent_severity="P2",
            agent_category="network",  # Wrong
            agent_root_cause="totally wrong cause",
            agent_remediation="flush_cache",  # Wrong
            agent_remediation_target="cache",  # Wrong
            step_count=25,
        )
        score = grade_episode(state)
        # Only severity (0.20) + maybe minimal description overlap
        assert 0.0 < score < 0.35

    def test_wrong_severity_right_everything_else(self):
        state = _make_state(
            agent_severity="P4",  # GT is P2 — off by 2
            agent_category="database",
            agent_root_cause="connection pool exhaustion order-db slow queries deploy",
            agent_remediation="rollback_deploy",
            agent_remediation_target="order-svc",
            step_count=8,
        )
        score = grade_episode(state)
        # Loses severity (0.20), gets everything else
        assert 0.7 < score < 0.85

    def test_all_wrong(self):
        state = _make_state(
            agent_severity="P4",  # GT is P2
            agent_category="security",  # GT is database
            agent_root_cause="nothing happened",
            agent_remediation="flush_cache",  # GT is rollback_deploy
            agent_remediation_target="cache",  # GT is order-svc
            step_count=30,
        )
        score = grade_episode(state)
        assert score < 0.05


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Efficiency
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEfficiency:
    """Tests for the efficiency bonus."""

    def test_fast_bonus(self):
        state = _make_state(step_count=5)
        breakdown = grade_episode_breakdown(state)
        assert breakdown["efficiency"] == WEIGHT_EFFICIENCY_FAST

    def test_medium_bonus(self):
        state = _make_state(step_count=15)
        breakdown = grade_episode_breakdown(state)
        assert breakdown["efficiency"] == WEIGHT_EFFICIENCY_MEDIUM

    def test_slow_no_bonus(self):
        state = _make_state(step_count=25)
        breakdown = grade_episode_breakdown(state)
        assert breakdown["efficiency"] == 0.0

    def test_boundary_fast(self):
        state = _make_state(step_count=10)
        breakdown = grade_episode_breakdown(state)
        assert breakdown["efficiency"] == WEIGHT_EFFICIENCY_FAST

    def test_boundary_medium(self):
        state = _make_state(step_count=20)
        breakdown = grade_episode_breakdown(state)
        assert breakdown["efficiency"] == WEIGHT_EFFICIENCY_MEDIUM


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Score Range
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestScoreRange:
    """Ensure scores are always in [0.0, 1.0]."""

    @pytest.mark.parametrize("severity", ["P1", "P2", "P3", "P4", None])
    @pytest.mark.parametrize("category", ["database", "network", "config", "resource", None])
    def test_score_in_range(self, severity, category):
        state = _make_state(
            agent_severity=severity,
            agent_category=category,
            step_count=10,
        )
        score = grade_episode(state)
        assert 0.0 <= score <= 1.0

    def test_weights_sum_to_1(self):
        total = (
            WEIGHT_SEVERITY
            + WEIGHT_ROOT_CAUSE_CATEGORY
            + WEIGHT_ROOT_CAUSE_DESCRIPTION
            + WEIGHT_REMEDIATION_ACTION
            + WEIGHT_REMEDIATION_TARGET
            + WEIGHT_EFFICIENCY_FAST
        )
        assert abs(total - 1.0) < 1e-10

    def test_grade_returns_float(self):
        state = _make_state(agent_severity="P2")
        score = grade_episode(state)
        assert isinstance(score, float)

    def test_breakdown_total_matches_grade(self):
        state = _make_state(
            agent_severity="P2",
            agent_category="database",
            agent_root_cause="database pool exhaustion",
            agent_remediation="rollback_deploy",
            agent_remediation_target="order-svc",
            step_count=10,
        )
        score = grade_episode(state)
        breakdown = grade_episode_breakdown(state)
        assert abs(score - breakdown["total"]) < 1e-10


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Root Cause Description
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRootCauseDescription:
    """Tests for keyword-overlap scoring."""

    def test_identical_description(self):
        state = _make_state(
            agent_root_cause="Database connection pool exhaustion on order-db due to slow queries from recent deploy",
        )
        breakdown = grade_episode_breakdown(state)
        assert breakdown["root_cause_description"] == WEIGHT_ROOT_CAUSE_DESCRIPTION

    def test_no_overlap_description(self):
        state = _make_state(
            agent_root_cause="totally unrelated random words xyz abc",
        )
        breakdown = grade_episode_breakdown(state)
        assert breakdown["root_cause_description"] < 0.02

    def test_partial_overlap(self):
        state = _make_state(
            agent_root_cause="connection pool order-db",
        )
        breakdown = grade_episode_breakdown(state)
        assert 0.0 < breakdown["root_cause_description"] < WEIGHT_ROOT_CAUSE_DESCRIPTION

    def test_empty_description(self):
        state = _make_state(agent_root_cause=None)
        breakdown = grade_episode_breakdown(state)
        assert breakdown["root_cause_description"] == 0.0
