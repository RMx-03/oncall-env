"""
Task definitions and deterministic grading functions.

Public API:
    TASKS                   — Frozen dict of all task configurations
    get_task(task_id)        — Get a single task config by ID
    list_task_ids()          — List all valid task IDs
    grade_episode(state)     — Grade an episode → [0.0, 1.0]
    grade_episode_breakdown  — Grade with per-component breakdown
"""

from .graders import grade_episode, grade_episode_breakdown
from .task_definitions import TASKS, get_task, list_task_ids

__all__ = [
    "TASKS",
    "get_task",
    "list_task_ids",
    "grade_episode",
    "grade_episode_breakdown",
]
