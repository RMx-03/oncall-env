"""Phase 6 validation: Dockerfile, deployment configs, and HTTP API tests."""
import json
import os
import sys
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, workspace_root)

print("=" * 60)
print("PHASE 6 VALIDATION - Docker & Deployment Configs")
print("=" * 60)
print()

errors = []
all_pass = True

def check(name: str, condition: bool, detail: str = "") -> None:
    global all_pass
    status = "PASS" if condition else "FAIL"
    if not condition:
        all_pass = False
        errors.append(name)
    suffix = f" ({detail})" if detail else ""
    print(f"  {status}: {name}{suffix}")

# ── 1. File existence checks ──
print("1. DEPLOYMENT FILES")
print("-" * 40)

dockerfile_path = os.path.join(workspace_root, "Dockerfile")
dockerignore_path = os.path.join(workspace_root, ".dockerignore")
requirements_path = os.path.join(workspace_root, "requirements.txt")
openenv_yaml_path = os.path.join(workspace_root, "openenv.yaml")
inference_path = os.path.join(workspace_root, "inference.py")

check("Dockerfile exists", os.path.exists(dockerfile_path))
check(".dockerignore exists", os.path.exists(dockerignore_path))
check("requirements.txt (root) exists", os.path.exists(requirements_path))
check("openenv.yaml (root) exists", os.path.exists(openenv_yaml_path))
check("inference.py (root) exists", os.path.exists(inference_path))
print()

# ── 2. Dockerfile content checks ──
print("2. DOCKERFILE CONTENT")
print("-" * 40)

with open(dockerfile_path, "r", encoding="utf-8") as f:
    dockerfile = f.read()

check("Base image: python:3.11-slim", "python:3.11-slim" in dockerfile)
check("WORKDIR /app", "WORKDIR /app" in dockerfile)
check("COPY requirements.txt", "COPY requirements.txt" in dockerfile)
check("pip install", "pip install" in dockerfile)
check("COPY oncall_env/", "COPY oncall_env/" in dockerfile)
check("HEALTHCHECK defined", "HEALTHCHECK" in dockerfile)
check("EXPOSE port", "EXPOSE" in dockerfile)
check("CMD uvicorn", "uvicorn" in dockerfile)
check("oncall_env.server.app:app in CMD", "oncall_env.server.app:app" in dockerfile)
check("Port 7860 (HF Spaces)", "7860" in dockerfile)
print()

# ── 3. .dockerignore checks ──
print("3. DOCKERIGNORE CONTENT")
print("-" * 40)

with open(dockerignore_path, "r", encoding="utf-8") as f:
    dockerignore = f.read()

check(".git excluded", ".git" in dockerignore)
check(".venv excluded", ".venv" in dockerignore)
check("__pycache__ excluded", "__pycache__" in dockerignore)
check("tests excluded", "tests" in dockerignore)
check(".plans excluded", ".plans" in dockerignore)
print()

# ── 4. requirements.txt checks ──
print("4. REQUIREMENTS.TXT")
print("-" * 40)

with open(requirements_path, "r", encoding="utf-8") as f:
    requirements = f.read()

check("openenv-core dependency", "openenv-core" in requirements)
check("fastapi dependency", "fastapi" in requirements)
check("uvicorn dependency", "uvicorn" in requirements)
check("pydantic dependency", "pydantic" in requirements)
check("openai dependency", "openai" in requirements)
print()

# ── 5. openenv.yaml checks ──
print("5. OPENENV.YAML")
print("-" * 40)

import yaml  # noqa: E402

with open(openenv_yaml_path, "r", encoding="utf-8") as f:
    openenv_config = yaml.safe_load(f)

check("spec_version = 1", openenv_config.get("spec_version") == 1)
check("name = oncall_env", openenv_config.get("name") == "oncall_env")
check("type = space", openenv_config.get("type") == "space")
check("runtime = fastapi", openenv_config.get("runtime") == "fastapi")
check("app module path set", "oncall_env.server.app:app" in str(openenv_config.get("app", "")))
check("port = 7860", openenv_config.get("port") == 7860)
print()

# ── 6. app.py PORT env var support ──
print("6. APP.PY PORT SUPPORT")
print("-" * 40)

app_path = os.path.join(workspace_root, "oncall_env", "server", "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    app_source = f.read()

check("Reads PORT env var", 'os.environ.get("PORT"' in app_source or "os.environ.get('PORT'" in app_source)
check("Default port 7860", "7860" in app_source)
print()

# ── 7. inference.py wraps action for HTTP API ──
print("7. INFERENCE.PY HTTP API WRAPPING")
print("-" * 40)

with open(inference_path, "r", encoding="utf-8") as f:
    inference_source = f.read()

check("Step wraps action: json={'action': action}", '{"action": action}' in inference_source)
print()

# ── 8. Project structure completeness ──
print("8. PROJECT STRUCTURE COMPLETENESS")
print("-" * 40)

required_files = [
    "Dockerfile",
    ".dockerignore",
    "requirements.txt",
    "openenv.yaml",
    "inference.py",
    "oncall_env/__init__.py",
    "oncall_env/models.py",
    "oncall_env/client.py",
    "oncall_env/openenv.yaml",
    "oncall_env/server/__init__.py",
    "oncall_env/server/app.py",
    "oncall_env/server/environment.py",
    "oncall_env/server/scenario_engine.py",
    "oncall_env/server/log_generator.py",
    "oncall_env/server/metrics_generator.py",
    "oncall_env/server/alert_generator.py",
    "oncall_env/tasks/__init__.py",
    "oncall_env/tasks/task_definitions.py",
    "oncall_env/tasks/graders.py",
    "oncall_env/scenarios/easy_scenarios.json",
    "oncall_env/scenarios/medium_scenarios.json",
    "oncall_env/scenarios/hard_scenarios.json",
]

for fname in required_files:
    fpath = os.path.join(workspace_root, fname)
    check(f"  {fname}", os.path.exists(fpath))

print()

# ── Results ──
print("=" * 60)
if all_pass:
    print("ALL PHASE 6 VALIDATION TESTS PASSED")
else:
    print(f"FAILED ({len(errors)} errors):")
    for e in errors:
        print(f"  X {e}")
print("=" * 60)
