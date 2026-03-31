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
from fastapi.responses import RedirectResponse

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


@app.get("/")
def root() -> RedirectResponse:
    """Provide a root route so platform probes receive a valid response."""
    return RedirectResponse(url="/web", status_code=307)


def main() -> None:
    """Entry point for direct execution.

    Respects the ``PORT`` environment variable (set by HF Spaces).
    Defaults to 7860 if not set.

    Usage::

        python -m oncall_env.server.app
    """
    import os

    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
