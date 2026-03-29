"""
Typed EnvClient for the Incident Triage Environment.

Provides a strongly-typed wrapper around the OpenEnv WebSocket API.
Converts ``IncidentAction`` objects to wire-format dicts and parses
responses into ``StepResult[IncidentObservation]`` and ``IncidentState``.

Usage (sync)::

    from oncall_env.client import IncidentTriageEnv

    with IncidentTriageEnv(base_url="http://localhost:8000").sync() as env:
        result = env.reset(task_id="easy", seed=42)
        print(result.observation.incident_summary)
        result = env.step(IncidentAction(action_type="check_alerts"))
        print(result.observation.alerts)

Usage (async)::

    async with IncidentTriageEnv(base_url="http://localhost:8000") as env:
        result = await env.reset(task_id="easy")
        result = await env.step(IncidentAction(action_type="check_alerts"))

Dependency direction: imports ONLY from ``models.py`` (never ``server/``).
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient, StepResult

from oncall_env.models import (
    IncidentAction,
    IncidentObservation,
    IncidentState,
)


class IncidentTriageEnv(EnvClient[IncidentAction, IncidentObservation, IncidentState]):
    """Typed environment client for the incident triage environment.

    Wraps the OpenEnv ``EnvClient`` with proper type serialization
    for actions, observations, and state.

    Methods:
        reset(**kwargs) → StepResult[IncidentObservation]
        step(action)    → StepResult[IncidentObservation]
        state()         → IncidentState
        close()         → None
        sync()          → SyncEnvClient (synchronous wrapper)
    """

    def _step_payload(self, action: IncidentAction) -> Dict[str, Any]:
        """Serialize an ``IncidentAction`` to the JSON dict the server expects.

        Uses Pydantic's ``model_dump`` with ``exclude_none=True`` so
        only the relevant fields for each action type are sent.

        Args:
            action: Typed action to serialize.

        Returns:
            Dict with only non-None fields.
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[IncidentObservation]:
        """Parse a server response into a typed ``StepResult``.

        Extracts ``observation``, ``reward``, and ``done`` from the
        wire-format response and constructs an ``IncidentObservation``.

        Args:
            payload: Raw JSON dict from the WebSocket response.

        Returns:
            StepResult containing a typed IncidentObservation.
        """
        obs_data = payload.get("observation", {})
        observation = IncidentObservation.model_validate(obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> IncidentState:
        """Parse a state response into a typed ``IncidentState``.

        Args:
            payload: Raw JSON dict from the state endpoint.

        Returns:
            Typed IncidentState object.
        """
        return IncidentState.model_validate(payload)
