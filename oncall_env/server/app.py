"""
FastAPI application for the Incident Triage Environment.

Creates an HTTP server exposing the environment via the OpenEnv
``create_app`` helper.  Supports HTTP and WebSocket endpoints.

Usage:
    # Development (with auto-reload):
    uvicorn oncall_env.server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn oncall_env.server.app:app --host 0.0.0.0 --port 8000
"""

from openenv.core.env_server.http_server import create_app

from oncall_env.models import IncidentAction, IncidentObservation

from .environment import IncidentTriageEnvironment

# create_app expects a callable (factory) that returns an Environment instance.
# It wires up /reset, /step, /state, /health endpoints automatically.
app = create_app(
    IncidentTriageEnvironment,
    IncidentAction,
    IncidentObservation,
    env_name="oncall_env",
)


def main() -> None:
    """Entry point for direct execution.

    Usage::

        python -m oncall_env.server.app
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
