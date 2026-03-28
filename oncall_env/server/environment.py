"""
Core environment implementation for the Incident Triage Environment.

Implements the OpenEnv ``Environment`` interface:
  - ``reset()``  → initial observation
  - ``step()``   → observation after action
  - ``state``    → current episode state (property)

All business logic lives here.  Generators and scenario engine
are delegated to their dedicated modules.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server import Environment

from oncall_env.models import (
    IncidentAction,
    IncidentObservation,
    IncidentState,
    ServiceStatus,
)

from .alert_generator import generate_alerts
from .log_generator import generate_log_entries
from .metrics_generator import generate_metric_data
from .scenario_engine import AVAILABLE_SERVICES, ScenarioEngine

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Constants — named, not magic
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MAX_ACTIONS: int = 30

# Severity levels in order (for off-by-one distance)
SEVERITY_ORDER = ["P1", "P2", "P3", "P4"]

# Reward constants
REWARD_SEVERITY_EXACT: float = 0.15
REWARD_SEVERITY_OFF_BY_ONE: float = 0.07
REWARD_SEVERITY_WRONG: float = -0.05

REWARD_ROOT_CAUSE_CATEGORY_CORRECT: float = 0.15
REWARD_ROOT_CAUSE_CATEGORY_WRONG: float = -0.05
REWARD_ROOT_CAUSE_KEYWORD_MAX: float = 0.10

REWARD_REMEDIATION_ACTION_CORRECT: float = 0.25
REWARD_REMEDIATION_ACTION_WRONG: float = -0.10
REWARD_REMEDIATION_TARGET_CORRECT: float = 0.15
REWARD_REMEDIATION_TARGET_WRONG: float = -0.05

REWARD_ESCALATION_CORRECT_TEAM: float = 0.10

REWARD_EFFICIENCY_FAST: float = 0.10  # ≤8 steps
REWARD_EFFICIENCY_MEDIUM: float = 0.05  # ≤15 steps
REWARD_MAX_ACTIONS_EXCEEDED: float = -0.10
REWARD_INVALID_ACTION: float = -0.05

# Default goal text
GOAL_TEXT = (
    "You are an on-call SRE engineer responding to a production incident. "
    "Investigate the affected services by querying logs, metrics, and alerts. "
    "Then classify the severity, identify the root cause, and execute the "
    "correct remediation action. Work efficiently — you have a limited "
    "number of actions."
)


class IncidentTriageEnvironment(Environment):
    """Simulated production incident response environment.

    The agent acts as an on-call SRE engineer and must:
      1. **Triage** — Check alerts and classify severity
      2. **Investigate** — Query logs and metrics across services
      3. **Diagnose** — Identify the root cause category and description
      4. **Remediate** — Execute the correct fix on the right service

    Supports three difficulty levels (``easy``, ``medium``, ``hard``)
    with increasingly complex multi-service failures.
    """

    def __init__(self) -> None:
        """Initialise the environment with a scenario engine."""
        self._engine = ScenarioEngine()
        self._scenario: dict[str, Any] = {}
        self._state = IncidentState()
        self._seed: int = 0

    # ━━━━━━━━━━━━━━━━━━━━━━  reset  ━━━━━━━━━━━━━━━━━━━━━━

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        """Reset the environment to a fresh episode.

        Args:
            seed: Random seed for deterministic scenario selection.
            episode_id: Optional episode identifier.
            **kwargs: Must contain ``task_id`` (``easy`` | ``medium`` | ``hard``).

        Returns:
            Initial ``IncidentObservation`` with the incident summary,
            goal description, and list of available services.
        """
        task_id: str = kwargs.get("task_id", "easy")
        self._seed = seed if seed is not None else 0

        # Select scenario
        self._scenario = self._engine.select_scenario(task_id, seed)
        gt = self._scenario["ground_truth"]

        # Initialise state
        self._state = IncidentState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            task_difficulty=self._scenario["difficulty"],
            ground_truth_severity=gt["severity"],
            ground_truth_root_cause=gt["root_cause"],
            ground_truth_category=gt["root_cause_category"],
            ground_truth_remediation=gt["remediation_action"],
            ground_truth_target=gt["remediation_target"],
        )

        return IncidentObservation(
            done=False,
            reward=0.0,
            goal=GOAL_TEXT,
            incident_summary=self._scenario["incident_summary"],
            action_result="Environment reset — incident loaded.",
            action_success=True,
            available_services=AVAILABLE_SERVICES,
            investigation_actions_taken=0,
            max_actions=MAX_ACTIONS,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━  step  ━━━━━━━━━━━━━━━━━━━━━━

    def step(
        self,
        action: IncidentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        """Execute one step in the environment.

        Dispatches to the appropriate handler based on ``action.action_type``,
        computes reward, updates state, and returns the observation.

        Args:
            action: The agent's action.
            timeout_s: Unused (kept for interface compat).
            **kwargs: Unused.

        Returns:
            ``IncidentObservation`` with the action result.
        """
        # Guard: episode already done
        if self._state.done:
            return self._make_observation(
                action_result="Episode already ended. Call reset() to start a new one.",
                action_success=False,
                reward=0.0,
                done=True,
            )

        self._state.step_count += 1

        # Dispatch
        handlers = {
            "query_logs": self._handle_query_logs,
            "query_metrics": self._handle_query_metrics,
            "check_alerts": self._handle_check_alerts,
            "check_service_status": self._handle_check_status,
            "set_severity": self._handle_set_severity,
            "identify_root_cause": self._handle_identify_root_cause,
            "execute_remediation": self._handle_execute_remediation,
            "escalate": self._handle_escalate,
        }

        handler = handlers.get(action.action_type)
        if handler is None:
            return self._handle_invalid_action(action)

        try:
            obs = handler(action)
        except Exception as e:
            obs = self._make_observation(
                action_result=f"Internal error processing action: {e}",
                action_success=False,
                reward=REWARD_INVALID_ACTION,
            )

        # Check max actions termination
        if not obs.done and self._state.step_count >= MAX_ACTIONS:
            self._state.done = True
            self._state.total_reward += REWARD_MAX_ACTIONS_EXCEEDED
            return self._make_observation(
                action_result=(
                    f"Maximum actions ({MAX_ACTIONS}) exceeded. "
                    "Episode terminated."
                ),
                action_success=False,
                reward=REWARD_MAX_ACTIONS_EXCEEDED,
                done=True,
            )

        return obs

    # ━━━━━━━━━━━━━━━━━━━━━━  state  ━━━━━━━━━━━━━━━━━━━━━━

    @property
    def state(self) -> IncidentState:
        """Return the current episode state (for grading)."""
        return self._state

    # ━━━━━━━━━━━━━━━━━━━━━━  Handlers  ━━━━━━━━━━━━━━━━━━━

    def _handle_query_logs(self, action: IncidentAction) -> IncidentObservation:
        """Handle ``query_logs``: return log entries for a service."""
        service = action.service_name
        if not service:
            return self._error_obs("query_logs requires 'service_name'.")
        if service not in AVAILABLE_SERVICES:
            return self._error_obs(
                f"Unknown service '{service}'. "
                f"Available: {AVAILABLE_SERVICES}"
            )

        self._track_investigation(service)

        time_range = action.time_range or "last_15m"
        logs = generate_log_entries(
            service=service,
            scenario_config=self._scenario,
            time_range=time_range,
            seed=self._seed + hash(service),
        )

        return self._make_observation(
            action_result=(
                f"Retrieved {len(logs)} log entries from {service} "
                f"({time_range})."
            ),
            action_success=True,
            reward=0.0,
            log_entries=logs,
        )

    def _handle_query_metrics(self, action: IncidentAction) -> IncidentObservation:
        """Handle ``query_metrics``: return metric data for a service."""
        service = action.service_name
        if not service:
            return self._error_obs("query_metrics requires 'service_name'.")
        if service not in AVAILABLE_SERVICES:
            return self._error_obs(
                f"Unknown service '{service}'. "
                f"Available: {AVAILABLE_SERVICES}"
            )

        self._track_investigation(service)

        time_range = action.time_range or "last_15m"
        metrics = generate_metric_data(
            service=service,
            scenario_config=self._scenario,
            metric_name=action.metric_name,
            time_range=time_range,
            seed=self._seed + hash(service),
        )

        metric_names = list(metrics.keys())
        total_points = sum(len(v) for v in metrics.values())
        return self._make_observation(
            action_result=(
                f"Retrieved metrics for {service} ({time_range}): "
                f"{metric_names} — {total_points} data points total."
            ),
            action_success=True,
            reward=0.0,
            metric_data=metrics,
        )

    def _handle_check_alerts(self, action: IncidentAction) -> IncidentObservation:
        """Handle ``check_alerts``: return active monitoring alerts."""
        alerts = generate_alerts(self._scenario)

        return self._make_observation(
            action_result=f"Found {len(alerts)} active alert(s).",
            action_success=True,
            reward=0.0,
            alerts=alerts,
        )

    def _handle_check_status(self, action: IncidentAction) -> IncidentObservation:
        """Handle ``check_service_status``: return health status."""
        service_states = self._scenario.get("service_states", {})
        statuses = []

        # If a specific service is requested, return just that one
        services_to_check = (
            [action.service_name] if action.service_name
            else AVAILABLE_SERVICES
        )

        for svc in services_to_check:
            if svc not in AVAILABLE_SERVICES:
                continue
            raw_status = service_states.get(svc, "healthy")
            statuses.append(ServiceStatus(
                service=svc,
                status=raw_status,
                uptime=self._uptime_for_status(raw_status),
                last_deploy="2026-01-15T09:00:00Z",
            ))

        if action.service_name:
            self._track_investigation(action.service_name)

        return self._make_observation(
            action_result=(
                f"Retrieved status for {len(statuses)} service(s)."
            ),
            action_success=True,
            reward=0.0,
            service_statuses=statuses,
        )

    def _handle_set_severity(self, action: IncidentAction) -> IncidentObservation:
        """Handle ``set_severity``: classify the incident severity."""
        if not action.severity:
            return self._error_obs("set_severity requires 'severity' (P1–P4).")

        self._state.agent_severity = action.severity
        gt_severity = self._state.ground_truth_severity

        # Compute reward based on distance
        reward = self._compute_severity_reward(action.severity, gt_severity)
        self._state.total_reward += reward

        return self._make_observation(
            action_result=(
                f"Severity set to {action.severity}."
            ),
            action_success=True,
            reward=reward,
        )

    def _handle_identify_root_cause(
        self, action: IncidentAction,
    ) -> IncidentObservation:
        """Handle ``identify_root_cause``: declare root cause."""
        if not action.root_cause_category:
            return self._error_obs(
                "identify_root_cause requires 'root_cause_category'."
            )

        self._state.agent_category = action.root_cause_category
        self._state.agent_root_cause = action.root_cause or ""

        gt_category = self._state.ground_truth_category
        gt_description = self._state.ground_truth_root_cause

        # Category reward
        reward = 0.0
        if action.root_cause_category == gt_category:
            reward += REWARD_ROOT_CAUSE_CATEGORY_CORRECT
        else:
            reward += REWARD_ROOT_CAUSE_CATEGORY_WRONG

        # Keyword overlap reward (for free-text description)
        if action.root_cause:
            keyword_score = self._compute_keyword_overlap(
                action.root_cause, gt_description,
            )
            reward += keyword_score

        self._state.total_reward += reward

        return self._make_observation(
            action_result=(
                f"Root cause identified: [{action.root_cause_category}] "
                f"{action.root_cause or '(no description)'}"
            ),
            action_success=True,
            reward=reward,
        )

    def _handle_execute_remediation(
        self, action: IncidentAction,
    ) -> IncidentObservation:
        """Handle ``execute_remediation``: execute fix (TERMINAL)."""
        if not action.remediation_action:
            return self._error_obs(
                "execute_remediation requires 'remediation_action'."
            )
        if not action.remediation_target:
            return self._error_obs(
                "execute_remediation requires 'remediation_target'."
            )

        self._state.agent_remediation = action.remediation_action
        self._state.agent_remediation_target = action.remediation_target

        gt_action = self._state.ground_truth_remediation
        gt_target = self._state.ground_truth_target

        # Action reward
        reward = 0.0
        if action.remediation_action == gt_action:
            reward += REWARD_REMEDIATION_ACTION_CORRECT
        else:
            reward += REWARD_REMEDIATION_ACTION_WRONG

        # Target reward
        if action.remediation_target == gt_target:
            reward += REWARD_REMEDIATION_TARGET_CORRECT
        else:
            reward += REWARD_REMEDIATION_TARGET_WRONG

        # Efficiency bonus
        reward += self._compute_efficiency_bonus()

        self._state.total_reward += reward
        self._state.done = True

        return self._make_observation(
            action_result=(
                f"Remediation executed: {action.remediation_action} "
                f"on {action.remediation_target}. Episode complete."
            ),
            action_success=True,
            reward=reward,
            done=True,
        )

    def _handle_escalate(self, action: IncidentAction) -> IncidentObservation:
        """Handle ``escalate``: escalate to a team (TERMINAL)."""
        if not action.escalation_team:
            return self._error_obs(
                "escalate requires 'escalation_team'."
            )

        # Map ground truth category to expected escalation team
        gt_category = self._state.ground_truth_category
        expected_team = self._category_to_team(gt_category)

        reward = 0.0
        if action.escalation_team == expected_team:
            reward += REWARD_ESCALATION_CORRECT_TEAM
        else:
            reward += REWARD_INVALID_ACTION

        self._state.total_reward += reward
        self._state.done = True

        return self._make_observation(
            action_result=(
                f"Incident escalated to {action.escalation_team} team. "
                "Episode complete."
            ),
            action_success=True,
            reward=reward,
            done=True,
        )

    def _handle_invalid_action(
        self, action: IncidentAction,
    ) -> IncidentObservation:
        """Handle unknown action types."""
        self._state.total_reward += REWARD_INVALID_ACTION
        return self._make_observation(
            action_result=(
                f"Unknown action type: '{action.action_type}'. "
                f"Valid types: query_logs, query_metrics, check_alerts, "
                f"check_service_status, set_severity, identify_root_cause, "
                f"execute_remediation, escalate."
            ),
            action_success=False,
            reward=REWARD_INVALID_ACTION,
        )

    # ━━━━━━━━━━━━━━━━━━━━  Reward helpers  ━━━━━━━━━━━━━━━━━

    @staticmethod
    def _compute_severity_reward(agent: str, ground_truth: str) -> float:
        """Compute reward for severity classification.

        Returns:
            Positive for exact or off-by-one match, negative otherwise.
        """
        if agent == ground_truth:
            return REWARD_SEVERITY_EXACT

        try:
            agent_idx = SEVERITY_ORDER.index(agent)
            gt_idx = SEVERITY_ORDER.index(ground_truth)
        except ValueError:
            return REWARD_SEVERITY_WRONG

        if abs(agent_idx - gt_idx) == 1:
            return REWARD_SEVERITY_OFF_BY_ONE

        return REWARD_SEVERITY_WRONG

    @staticmethod
    def _compute_keyword_overlap(agent_text: str, ground_truth: str) -> float:
        """Compute keyword overlap reward between agent's root cause
        description and the ground truth.

        Returns:
            Float in [0.0, REWARD_ROOT_CAUSE_KEYWORD_MAX].
        """
        # Normalise to lowercase word sets
        agent_words = set(agent_text.lower().split())
        gt_words = set(ground_truth.lower().split())

        # Remove common stop words
        stop_words = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of",
            "is", "was", "are", "and", "or", "by", "with", "from",
            "that", "this", "it", "its", "has", "had", "have",
        }
        agent_words -= stop_words
        gt_words -= stop_words

        if not gt_words:
            return 0.0

        overlap = len(agent_words & gt_words)
        ratio = overlap / len(gt_words)

        return round(min(ratio * REWARD_ROOT_CAUSE_KEYWORD_MAX, REWARD_ROOT_CAUSE_KEYWORD_MAX), 4)

    def _compute_efficiency_bonus(self) -> float:
        """Compute efficiency bonus based on number of steps taken."""
        steps = self._state.step_count
        if steps <= 8:
            return REWARD_EFFICIENCY_FAST
        if steps <= 15:
            return REWARD_EFFICIENCY_MEDIUM
        return 0.0

    @staticmethod
    def _category_to_team(category: str) -> str:
        """Map root cause category to expected escalation team."""
        mapping = {
            "database": "database",
            "network": "network",
            "config": "platform",
            "resource": "platform",
            "dependency": "platform",
            "security": "security",
        }
        return mapping.get(category, "platform")

    @staticmethod
    def _uptime_for_status(status: str) -> str:
        """Generate a realistic uptime string based on service status."""
        if status == "down":
            return "0m (restarting)"
        if status == "degraded":
            return "2h 15m"
        return "14d 3h 22m"

    # ━━━━━━━━━━━━━━━━━━  Observation helpers  ━━━━━━━━━━━━━━

    def _track_investigation(self, service: str) -> None:
        """Track that the agent has investigated a service."""
        if service not in self._state.services_investigated:
            self._state.services_investigated.append(service)

    def _make_observation(self, **kwargs: Any) -> IncidentObservation:
        """Build an observation with common fields auto-filled."""
        return IncidentObservation(
            goal=GOAL_TEXT,
            incident_summary=self._scenario.get("incident_summary", ""),
            investigation_actions_taken=self._state.step_count,
            max_actions=MAX_ACTIONS,
            services_investigated=list(self._state.services_investigated),
            severity_set=self._state.agent_severity is not None,
            root_cause_identified=self._state.agent_category is not None,
            remediation_executed=self._state.agent_remediation is not None,
            available_services=AVAILABLE_SERVICES,
            **kwargs,
        )

    def _error_obs(self, message: str) -> IncidentObservation:
        """Build an error observation for invalid inputs."""
        return self._make_observation(
            action_result=message,
            action_success=False,
            last_action_error=message,
            reward=0.0,
        )
