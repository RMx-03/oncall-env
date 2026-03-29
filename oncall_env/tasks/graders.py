"""
Deterministic grading functions for the Incident Triage Environment.

Grades an agent's episode performance on a 0.0–1.0 scale.

Scoring breakdown (adds up to 1.00):
    Severity classification:    0.20
    Root cause category:        0.25
    Root cause description:     0.10 (keyword overlap)
    Remediation action:         0.25
    Remediation target:         0.15
    Efficiency bonus:           0.05

All functions are **pure and deterministic** — no randomness,
no LLM-as-judge, no network calls.  Imports ONLY from models.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set

from oncall_env.models import IncidentState

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Scoring weight constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WEIGHT_SEVERITY: float = 0.20
WEIGHT_SEVERITY_OFF_BY_ONE: float = 0.10

WEIGHT_ROOT_CAUSE_CATEGORY: float = 0.25

WEIGHT_ROOT_CAUSE_DESCRIPTION: float = 0.10

WEIGHT_REMEDIATION_ACTION: float = 0.25

WEIGHT_REMEDIATION_TARGET: float = 0.15

WEIGHT_EFFICIENCY_FAST: float = 0.05  # ≤10 steps
WEIGHT_EFFICIENCY_MEDIUM: float = 0.02  # ≤20 steps

EFFICIENCY_THRESHOLD_FAST: int = 10
EFFICIENCY_THRESHOLD_MEDIUM: int = 20

# Severity ordinal map for distance computation
SEVERITY_ORDINAL: Dict[str, int] = {"P1": 1, "P2": 2, "P3": 3, "P4": 4}

# Common stop words to exclude from keyword overlap
STOP_WORDS: Set[str] = frozenset({
    "the", "a", "an", "is", "was", "are", "were", "in", "on", "at",
    "to", "for", "of", "and", "or", "by", "with", "from", "due",
    "that", "this", "it", "its", "has", "had", "have", "be", "been",
})


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main grading function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def grade_episode(state: IncidentState) -> float:
    """Grade an agent's performance on a single incident episode.

    Computes a deterministic score from 0.0 to 1.0 based on:
      - Severity classification (0.20)
      - Root cause identification (0.25 category + 0.10 description)
      - Remediation correctness (0.25 action + 0.15 target)
      - Efficiency bonus (0.05)

    Args:
        state: Final episode state containing the agent's answers
               and ground truth values.

    Returns:
        Float score in [0.0, 1.0], rounded to 4 decimal places.
    """
    score = 0.0

    score += _score_severity(
        agent=state.agent_severity,
        ground_truth=state.ground_truth_severity,
    )

    score += _score_root_cause_category(
        agent=state.agent_category,
        ground_truth=state.ground_truth_category,
    )

    score += _score_root_cause_description(
        agent=state.agent_root_cause,
        ground_truth=state.ground_truth_root_cause,
    )

    score += _score_remediation_action(
        agent=state.agent_remediation,
        ground_truth=state.ground_truth_remediation,
    )

    score += _score_remediation_target(
        agent=state.agent_remediation_target,
        ground_truth=state.ground_truth_target,
    )

    score += _score_efficiency(step_count=state.step_count)

    return round(min(max(score, 0.0), 1.0), 4)


def grade_episode_breakdown(state: IncidentState) -> Dict[str, float]:
    """Grade an episode and return a per-component breakdown.

    Useful for debugging and reporting.  Returns the same total
    as ``grade_episode`` but split into named components.

    Args:
        state: Final episode state.

    Returns:
        Dict with keys: ``severity``, ``root_cause_category``,
        ``root_cause_description``, ``remediation_action``,
        ``remediation_target``, ``efficiency``, ``total``.
    """
    severity = _score_severity(
        agent=state.agent_severity,
        ground_truth=state.ground_truth_severity,
    )
    category = _score_root_cause_category(
        agent=state.agent_category,
        ground_truth=state.ground_truth_category,
    )
    description = _score_root_cause_description(
        agent=state.agent_root_cause,
        ground_truth=state.ground_truth_root_cause,
    )
    action = _score_remediation_action(
        agent=state.agent_remediation,
        ground_truth=state.ground_truth_remediation,
    )
    target = _score_remediation_target(
        agent=state.agent_remediation_target,
        ground_truth=state.ground_truth_target,
    )
    efficiency = _score_efficiency(step_count=state.step_count)

    total = severity + category + description + action + target + efficiency
    total = round(min(max(total, 0.0), 1.0), 4)

    return {
        "severity": round(severity, 4),
        "root_cause_category": round(category, 4),
        "root_cause_description": round(description, 4),
        "remediation_action": round(action, 4),
        "remediation_target": round(target, 4),
        "efficiency": round(efficiency, 4),
        "total": total,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Component scoring functions (pure, stateless)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _score_severity(
    agent: Optional[str],
    ground_truth: str,
) -> float:
    """Score severity classification (max 0.20).

    - Exact match → 0.20
    - Off by one level → 0.10
    - Otherwise → 0.00
    """
    if not agent or not ground_truth:
        return 0.0

    if agent == ground_truth:
        return WEIGHT_SEVERITY

    agent_ord = SEVERITY_ORDINAL.get(agent, 0)
    gt_ord = SEVERITY_ORDINAL.get(ground_truth, 0)

    if agent_ord == 0 or gt_ord == 0:
        return 0.0

    if abs(agent_ord - gt_ord) == 1:
        return WEIGHT_SEVERITY_OFF_BY_ONE

    return 0.0


def _score_root_cause_category(
    agent: Optional[str],
    ground_truth: str,
) -> float:
    """Score root cause category match (max 0.25).

    - Exact match → 0.25
    - Otherwise → 0.00
    """
    if not agent or not ground_truth:
        return 0.0

    if agent == ground_truth:
        return WEIGHT_ROOT_CAUSE_CATEGORY

    return 0.0


def _score_root_cause_description(
    agent: Optional[str],
    ground_truth: str,
) -> float:
    """Score root cause description via keyword overlap (max 0.10).

    Computes the ratio of overlapping meaningful words between
    the agent's description and the ground truth, then scales
    by the weight.

    Args:
        agent: Agent's free-text root cause description.
        ground_truth: Ground truth root cause description.

    Returns:
        Float in [0.0, 0.10].
    """
    if not agent or not ground_truth:
        return 0.0

    gt_words = set(ground_truth.lower().split()) - STOP_WORDS
    agent_words = set(agent.lower().split()) - STOP_WORDS

    if not gt_words:
        return 0.0

    overlap = len(gt_words & agent_words)
    ratio = overlap / len(gt_words)

    return round(ratio * WEIGHT_ROOT_CAUSE_DESCRIPTION, 4)


def _score_remediation_action(
    agent: Optional[str],
    ground_truth: str,
) -> float:
    """Score remediation action match (max 0.25).

    - Exact match → 0.25
    - Otherwise → 0.00
    """
    if not agent or not ground_truth:
        return 0.0

    if agent == ground_truth:
        return WEIGHT_REMEDIATION_ACTION

    return 0.0


def _score_remediation_target(
    agent: Optional[str],
    ground_truth: str,
) -> float:
    """Score remediation target match (max 0.15).

    - Exact match → 0.15
    - Otherwise → 0.00
    """
    if not agent or not ground_truth:
        return 0.0

    if agent == ground_truth:
        return WEIGHT_REMEDIATION_TARGET

    return 0.0


def _score_efficiency(step_count: int) -> float:
    """Score efficiency (max 0.05).

    - ≤10 steps → 0.05
    - ≤20 steps → 0.02
    - >20 steps → 0.00
    """
    if step_count <= EFFICIENCY_THRESHOLD_FAST:
        return WEIGHT_EFFICIENCY_FAST

    if step_count <= EFFICIENCY_THRESHOLD_MEDIUM:
        return WEIGHT_EFFICIENCY_MEDIUM

    return 0.0
