# ─────────────────────────────────────────────────
# OnCall Incident Triage Environment — Dockerfile
# ─────────────────────────────────────────────────
# Runs the OpenEnv-compliant FastAPI server.
# Compatible with Hugging Face Spaces (Docker SDK).
#
# Build:  docker build -t oncall-env .
# Run:    docker run -p 7860:7860 oncall-env
# Health: curl http://localhost:7860/health
# ─────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY oncall_env/ ./oncall_env/
COPY inference.py .

# Copy root-level config files
COPY openenv.yaml .

# Health check (HF Spaces default port is 7860)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# HF Spaces expects port 7860
EXPOSE 7860

# Run the FastAPI server
# Using 0.0.0.0 to accept connections from outside the container
CMD ["uvicorn", "oncall_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
