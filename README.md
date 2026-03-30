# OnCall Incident Triage Environment

> An [OpenEnv](https://github.com/huggingface/open-env) environment that simulates production incident response.
> An AI agent acts as an on-call SRE engineer — triaging alerts, investigating
> logs and metrics across a microservices system, diagnosing root cause,
> and executing remediation.

---

## Motivation

Every engineering team handles production incidents. When a service goes down at 3 AM,
an on-call engineer must quickly triage alerts, investigate logs and metrics,
identify the root cause, and execute the correct fix — all under time pressure.

This environment models that real-world workflow:
**Triage → Investigate → Diagnose → Remediate**.

It fills a gap in the OpenEnv ecosystem — no incident response environment exists today.
This is directly useful for training and evaluating AI agents that could assist
(or eventually replace) humans in on-call incident response.

### Simulated Infrastructure

The environment simulates an 8-service microservices architecture:

```
┌─────────┐     ┌───────────┐     ┌──────────┐
│  api-gw │────▶│ order-svc │────▶│ order-db │
└─────────┘     └───────────┘     └──────────┘
     │               │
     │          ┌────────────┐     ┌───────────────────┐
     ├─────────▶│payment-svc │     │ notification-svc  │
     │          └────────────┘     └───────────────────┘
     │
     ├─────────▶┌──────────┐
     │          │ auth-svc │
     │          └──────────┘
     │
     ├─────────▶┌──────────┐      ┌────────────┐
     │          │  cache   │      │ search-svc │
     └─────────▶└──────────┘      └────────────┘
```

Each service produces realistic logs, metrics, and alerts based on the active incident scenario.

---

## Action Space

The agent interacts with the environment through 8 action types:

| Action Type | Parameters | Description |
|------------|------------|-------------|
| `query_logs` | `service_name`, `time_range` | Fetch log entries from a service |
| `query_metrics` | `service_name`, `metric_name` | Fetch CPU/memory/latency/error metrics |
| `check_alerts` | `service_name` (optional) | View active monitoring alerts |
| `check_service_status` | `service_name` (optional) | Check service health (healthy/degraded/down) |
| `set_severity` | `severity` (P1–P4) | Classify the incident severity |
| `identify_root_cause` | `root_cause`, `root_cause_category` | Declare the root cause |
| `execute_remediation` | `remediation_action`, `remediation_target` | Execute a fix (**terminal**) |
| `escalate` | `escalation_team` | Escalate to another team (**terminal**) |

**Terminal actions**: `execute_remediation` and `escalate` end the episode.
The episode also ends if the agent exceeds 30 actions.

### Root Cause Categories

`database` · `network` · `config` · `resource` · `dependency` · `security`

### Remediation Actions

`restart_service` · `scale_up` · `rollback_deploy` · `flush_cache` · `fix_config` · `failover_db`

---

## Observation Space

Each observation (returned after every action) contains:

| Field | Type | Description |
|-------|------|-------------|
| `incident_summary` | `str` | Brief description of the ongoing incident |
| `goal` | `str` | What the agent needs to accomplish |
| `action_result` | `str` | Human-readable result of the last action |
| `action_success` | `bool` | Whether the last action succeeded |
| `log_entries` | `List[LogEntry]` | Log data (when querying logs) |
| `metric_data` | `Dict[str, List[MetricDataPoint]]` | Time-series metrics (when querying metrics) |
| `alerts` | `List[Alert]` | Active alerts (when checking alerts) |
| `service_statuses` | `List[ServiceStatus]` | Service health (when checking status) |
| `severity_set` | `bool` | Progress: has the agent classified severity? |
| `root_cause_identified` | `bool` | Progress: has the agent identified root cause? |
| `remediation_executed` | `bool` | Progress: has the agent executed remediation? |
| `available_services` | `List[str]` | Services the agent can query |
| `investigation_actions_taken` | `int` | Actions taken so far |
| `max_actions` | `int` | Max allowed actions (30) |

---

## Tasks

Three difficulty levels with 3 scenarios each (9 total):

| Task | Difficulty | Scenarios | Key Challenge |
|------|-----------|-----------|---------------|
| **Easy** | ⭐ | OOM crash, Disk full, Auth OOM | Single service failure, obvious error logs |
| **Medium** | ⭐⭐ | DB pool exhaustion, Bad deploy, Cache storm | Cascading multi-service failure, requires correlation |
| **Hard** | ⭐⭐⭐ | DNS failures, TLS expiry, Config drift | Infrastructure-level cause disguised as app issues, red herrings |

### Grading System

Episodes are graded on a deterministic 0.0–1.0 scale:

| Component | Weight | Description |
|-----------|--------|-------------|
| Severity classification | 0.20 | Exact match = full credit, off-by-one = half |
| Root cause category | 0.25 | Exact match only |
| Root cause description | 0.10 | Keyword overlap with ground truth |
| Remediation action | 0.25 | Exact match only |
| Remediation target | 0.15 | Exact match only |
| Efficiency bonus | 0.05 | ≤10 steps = full, ≤20 steps = partial |

> **No LLM-as-judge** — grading is purely deterministic and reproducible.

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- pip

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/oncall-env.git
cd oncall-env

# Create virtual environment
python -m venv oncall_env/.venv
oncall_env/.venv/Scripts/activate  # Windows
# source oncall_env/.venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install -e oncall_env/

# Start the environment server
uvicorn oncall_env.server.app:app --host 0.0.0.0 --port 8000
```

### Using the Typed Client

```python
from oncall_env import IncidentTriageEnv, IncidentAction

# Synchronous usage
with IncidentTriageEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(task_id="easy", seed=42)
    print(result.observation.incident_summary)

    result = env.step(IncidentAction(action_type="check_alerts"))
    for alert in result.observation.alerts:
        print(f"[{alert.severity}] {alert.title}")
```

### Docker

```bash
# Build
docker build -t oncall-env .

# Run
docker run -p 7860:7860 oncall-env

# Test
curl http://localhost:7860/health
# → {"status": "healthy"}
```

### Running Inference

The baseline agent uses an LLM (via OpenAI-compatible API) to autonomously
investigate and resolve incidents:

```bash
# Set your API key
set OPENAI_API_KEY=sk-...          # Windows
# export OPENAI_API_KEY=sk-...     # Linux/Mac

# Run against local server
python inference.py

# Run with a different provider
set LLM_PROVIDER=groq
set GROQ_API_KEY=gsk_...
python inference.py

# Run against a deployed HF Space
set ENV_URL=https://your-space.hf.space
python inference.py
```

**Supported providers**: `openai`, `openrouter`, `groq`, `gemini`, `custom`

---

## Running Tests

```bash
# Run all tests
python -m pytest oncall_env/tests/ -v

# Run specific test suites
python -m pytest oncall_env/tests/test_environment.py -v   # Environment tests
python -m pytest oncall_env/tests/test_graders.py -v       # Grader tests
python -m pytest oncall_env/tests/test_integration.py -v   # Integration tests
```

---

## Baseline Scores

*Scores from baseline inference agent (to be filled after running `python inference.py`):*

| Task | Score | Steps | Time |
|------|-------|-------|------|
| Easy | — | — | — |
| Medium | — | — | — |
| Hard | — | — | — |
| **Average** | **—** | | |

---

## Project Structure

```
oncall-env/
├── inference.py                  # Baseline LLM agent
├── Dockerfile                    # HF Spaces deployment
├── requirements.txt              # Python dependencies
├── openenv.yaml                  # OpenEnv spec config
│
└── oncall_env/                   # Main package
    ├── __init__.py               # Public API re-exports
    ├── models.py                 # Pydantic models (Action, Observation, State)
    ├── client.py                 # Typed EnvClient subclass
    ├── openenv.yaml              # Package-level OpenEnv config
    │
    ├── server/                   # Environment server
    │   ├── app.py                # FastAPI app (create_app)
    │   ├── environment.py        # Core environment logic
    │   ├── scenario_engine.py    # Scenario loading & selection
    │   ├── log_generator.py      # Synthetic log generation
    │   ├── metrics_generator.py  # Synthetic metric generation
    │   └── alert_generator.py    # Alert generation
    │
    ├── tasks/                    # Task definitions & grading
    │   ├── task_definitions.py   # Task configs (easy/medium/hard)
    │   └── graders.py            # Deterministic grading (0.0–1.0)
    │
    ├── scenarios/                # Incident scenario data (JSON)
    │   ├── easy_scenarios.json   # 3 easy scenarios
    │   ├── medium_scenarios.json # 3 medium scenarios
    │   └── hard_scenarios.json   # 3 hard scenarios
    │
    └── tests/                    # Test suite
        ├── test_environment.py   # Environment unit tests
        ├── test_graders.py       # Grader unit tests
        └── test_integration.py   # Full episode integration tests
```

---

## Technical Highlights

- **OpenEnv compliant**: Typed Pydantic models, `openenv.yaml`, `create_app`, Docker, HF Spaces
- **Deterministic grading**: No randomness, no LLM-as-judge — reproducible scores every time
- **9 realistic scenarios**: From obvious OOM crashes to subtle TLS cert expirations
- **Dense reward signal**: Reward at every decision point (severity, root cause, remediation)
- **Multi-provider inference**: Works with OpenAI, Groq, OpenRouter, Gemini, or any OpenAI-compatible API
- **Modular architecture**: Clean separation of models, server, tasks, and scenarios

---

## License

MIT
