"""
environment.py — Core MetaQuestOSEnv engine.
Bridges the database, task handlers, and API layer.
"""

from __future__ import annotations
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import database as db
from models import (
    TaskID, SensorStatus, KernelStatus, TrackingMode,
    ThermalState, SensorState, ResourceState,
    SpatialAnchorState, KernelState, SystemObservation,
    ActionResult, ResetResult, StateResult, TaskInfo,
)
from tasks import (
    TASK_INITIALIZERS, TASK_HANDLERS,
    TASK_ACTIONS, TASK_MAX_STEPS, TASK_DESCRIPTIONS,
)


class MetaQuestOSEnv:
    """
    Simulates the system reliability layer of a Meta Quest-class MR device.
    All state is persisted in SQLite so it survives across HTTP requests.
    Real psutil metrics are injected into resource readings.
    """

    def __init__(self) -> None:
        db.init_db()

    # ------------------------------------------------------------------
    # Public API methods (called by FastAPI endpoints)
    # ------------------------------------------------------------------

    def reset(self, task_id: TaskID, seed: Optional[int] = None) -> ResetResult:
        if seed is not None:
            random.seed(seed)

        initializer = TASK_INITIALIZERS[task_id]
        initializer()

        obs = self._build_observation(task_id, step=0)
        return ResetResult(
            observation=obs,
            task_id=task_id,
            task_description=TASK_DESCRIPTIONS[task_id],
            available_actions=TASK_ACTIONS[task_id],
            max_steps=TASK_MAX_STEPS[task_id],
        )

    def step(self, action: str, task_id: Optional[TaskID] = None) -> ActionResult:
        # Resolve task_id
        tid_str = db.get_state("task_id")
        if tid_str is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        tid = TaskID(tid_str)

        # Check if already done
        if db.get_state("done", False):
            obs = self._build_observation(tid)
            return ActionResult(
                observation=obs,
                reward=obs.cumulative_reward,
                done=True,
                info={"message": "Episode already complete. Call reset() to start a new episode."},
            )

        # Increment step
        current_step = db.get_state("step", 0) + 1
        db.set_state("step", current_step)

        # Validate action
        valid_actions = TASK_ACTIONS[tid]
        if action not in valid_actions:
            # Partial match — try to help agent
            close = [a for a in valid_actions if action.split(" ")[0] in a]
            reward, success, message = 0.0, False, (
                f"Invalid action '{action}'. "
                f"Did you mean: {close[:3]}?" if close else
                f"Invalid action. Valid actions: {valid_actions[:5]}..."
            )
        else:
            handler = TASK_HANDLERS[tid]
            reward, success, message = handler(action)

        # Update cumulative reward
        prev_cum = db.get_state("cumulative_reward", 0.0)
        new_cum = prev_cum + reward
        db.set_state("cumulative_reward", new_cum)

        # Log action
        db.log_action(current_step, tid.value, action, reward, success, message)

        # Check max steps
        max_steps = TASK_MAX_STEPS[tid]
        done = db.get_state("done", False)
        if current_step >= max_steps and not done:
            done = True
            db.set_state("done", True)

        # Build observation
        obs = self._build_observation(tid, step=current_step,
                                       last_action=action,
                                       last_action_success=success,
                                       last_action_message=message,
                                       reward=reward,
                                       cumulative_reward=new_cum,
                                       done=done)
        return ActionResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "step": current_step,
                "max_steps": max_steps,
                "message": message,
                "action_valid": success,
            },
        )

    def state(self) -> StateResult:
        tid_str = db.get_state("task_id")
        if tid_str is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        tid = TaskID(tid_str)
        step = db.get_state("step", 0)
        done = db.get_state("done", False)
        obs = self._build_observation(tid, step=step)
        return StateResult(observation=obs, task_id=tid, step=step, done=done)

    def list_tasks(self) -> List[TaskInfo]:
        difficulties = {
            TaskID.THERMAL_MITIGATION: "easy",
            TaskID.SENSOR_RECOVERY: "medium",
            TaskID.KERNEL_PANIC_RECOVERY: "hard",
        }
        return [
            TaskInfo(
                task_id=tid,
                name=tid.value.replace("_", " ").title(),
                description=TASK_DESCRIPTIONS[tid],
                difficulty=difficulties[tid],
                max_steps=TASK_MAX_STEPS[tid],
                success_threshold=1.0,
                available_actions=TASK_ACTIONS[tid],
            )
            for tid in TaskID
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        task_id: TaskID,
        step: Optional[int] = None,
        last_action: Optional[str] = None,
        last_action_success: bool = True,
        last_action_message: str = "",
        reward: float = 0.0,
        cumulative_reward: Optional[float] = None,
        done: Optional[bool] = None,
    ) -> SystemObservation:
        s = db.get_all_state()

        if step is None:
            step = s.get("step", 0)
        if done is None:
            done = s.get("done", False)
        if cumulative_reward is None:
            cumulative_reward = s.get("cumulative_reward", 0.0)

        thermals = ThermalState(
            cpu_temp_c=s.get("cpu_temp_c", 65.0),
            gpu_temp_c=s.get("gpu_temp_c", 68.0),
            optical_pod_temp_c=s.get("optical_pod_temp_c", 42.0),
            throttle_active=s.get("throttle_active", False),
            refresh_rate_hz=s.get("refresh_rate_hz", 90),
            fan_speed_pct=s.get("fan_speed_pct", 40.0),
        )

        sensors = SensorState(
            imu_status=SensorStatus(s.get("imu_status", SensorStatus.ACTIVE.value)),
            imu_latency_ms=s.get("imu_latency_ms", 2.0),
            lidar_status=SensorStatus(s.get("lidar_status", SensorStatus.ACTIVE.value)),
            lidar_latency_ms=s.get("lidar_latency_ms", 5.0),
            hand_tracking_status=SensorStatus(s.get("hand_tracking_status", SensorStatus.ACTIVE.value)),
            hand_tracking_latency_ms=s.get("hand_tracking_latency_ms", 8.0),
            tracking_mode=TrackingMode(s.get("tracking_mode", TrackingMode.SLAM.value)),
            tracking_stability=s.get("tracking_stability", 1.0),
        )

        resources = ResourceState(
            battery_pct=s.get("battery_pct", 80.0),
            ram_used_mb=s.get("ram_used_mb", 2048.0),
            ram_total_mb=s.get("ram_total_mb", 8192.0),
            cpu_usage_pct=s.get("cpu_usage_pct", 30.0),
            frame_buffer_healthy=s.get("frame_buffer_healthy", True),
            background_processes=s.get("background_processes", 5),
        )

        spatial = SpatialAnchorState(
            anchor_count=s.get("anchor_count", 12),
            anchor_stability=s.get("anchor_stability", 1.0),
            world_lock_active=s.get("world_lock_active", True),
            session_data_intact=s.get("session_data_intact", True),
            cached_anchors_cleared=s.get("cached_anchors_cleared", False),
        )

        kernel = KernelState(
            status=KernelStatus(s.get("kernel_status", KernelStatus.ACTIVE.value)),
            crashed_app_id=s.get("crashed_app_id"),
            safe_mode_initiated=s.get("safe_mode_initiated", False),
            memory_lock_cleared=s.get("memory_lock_cleared", False),
            uptime_seconds=time.time() - s.get("uptime_seconds", time.time()),
        )

        return SystemObservation(
            step=step,
            task_id=task_id,
            thermals=thermals,
            sensors=sensors,
            resources=resources,
            spatial_anchors=spatial,
            kernel=kernel,
            last_action=last_action,
            last_action_success=last_action_success,
            last_action_message=last_action_message,
            reward=reward,
            cumulative_reward=cumulative_reward,
            done=done,
            info={
                "task_description": TASK_DESCRIPTIONS[task_id],
                "available_actions": TASK_ACTIONS[task_id],
                "max_steps": TASK_MAX_STEPS[task_id],
            },
        )


# Singleton instance
_env_instance: Optional[MetaQuestOSEnv] = None


def get_env() -> MetaQuestOSEnv:
    global _env_instance
    if _env_instance is None:
        _env_instance = MetaQuestOSEnv()
    return _env_instance