"""
main.py — FastAPI application exposing the OpenEnv API.

Endpoints:
  GET  /             → health check
  POST /reset        → initialize a task episode
  POST /step         → take an action
  GET  /state        → read current state
  GET  /tasks        → list all available tasks
"""

from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import database as db
from environment import get_env
from models import (
    ActionRequest, ActionResult,
    ResetRequest, ResetResult,
    StateResult, HealthResponse,
    TaskID,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("quest-openenv")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB on startup."""
    db.init_db()
    logger.info("MetaQuestOSEnv initialized. Database ready.")
    yield
    logger.info("MetaQuestOSEnv shutting down.")


app = FastAPI(
    title="MetaQuestOSEnv",
    description=(
        "An OpenEnv-compliant simulation of the system reliability layer of a "
        "Meta Quest-class Mixed Reality device. Trains AI agents to handle real "
        "operational failure modes: thermal throttling, sensor drift, and kernel recovery."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health / Root
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Health check — must return 200 for hackathon validator."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        environment="MetaQuestOSEnv",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


# ---------------------------------------------------------------------------
# Core OpenEnv API
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=ResetResult)
async def reset(request: ResetRequest) -> ResetResult:
    """
    Reset the environment to a fresh task scenario.
    Returns the initial observation, task description, and available actions.
    """
    try:
        env = get_env()
        result = env.reset(task_id=request.task_id, seed=request.seed)
        logger.info(f"Reset: task={request.task_id.value} seed={request.seed}")
        return result
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=ActionResult)
async def step(request: ActionRequest) -> ActionResult:
    """
    Execute one action in the environment.
    Returns updated observation, reward (0.0-1.0), and done flag.
    """
    try:
        env = get_env()
        result = env.step(action=request.action, task_id=request.task_id)
        logger.info(
            f"Step: action='{request.action}' "
            f"reward={result.reward:.3f} done={result.done}"
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Step failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state", response_model=StateResult)
async def state() -> StateResult:
    """
    Return the current environment state without advancing the episode.
    """
    try:
        env = get_env()
        return env.state()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"State failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Task listing
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    """List all available tasks with descriptions and available actions."""
    env = get_env()
    tasks = env.list_tasks()
    return {
        "tasks": [t.model_dump() for t in tasks],
        "count": len(tasks),
    }


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    """Get details for a specific task."""
    try:
        tid = TaskID(task_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_id}' not found. Valid tasks: {[t.value for t in TaskID]}"
        )
    env = get_env()
    tasks = {t.task_id: t for t in env.list_tasks()}
    return tasks[tid].model_dump()


# ---------------------------------------------------------------------------
# Action listing (convenience endpoint)
# ---------------------------------------------------------------------------

@app.get("/actions/{task_id}")
async def get_actions(task_id: str) -> Dict[str, Any]:
    """List all valid actions for a given task."""
    try:
        tid = TaskID(task_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    from tasks import TASK_ACTIONS
    return {
        "task_id": task_id,
        "actions": TASK_ACTIONS[tid],
        "count": len(TASK_ACTIONS[tid]),
    }


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=7860, reload=False)