"""
models.py — Typed Pydantic models for the Meta Quest OpenEnv environment.
All state, actions, observations, and responses are defined here.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SensorStatus(str, Enum):
    ACTIVE = "Active"
    DRIFTING = "Drifting"
    FAILED = "Failed"
    RECALIBRATING = "Recalibrating"


class KernelStatus(str, Enum):
    ACTIVE = "Active"
    DEGRADED = "Degraded"
    PANIC = "Panic"
    SAFE_MODE = "SafeMode"
    RECOVERING = "Recovering"


class TrackingMode(str, Enum):
    SLAM = "SLAM"
    VIO = "VIO"           # Visual-Inertial Odometry
    IMU_ONLY = "IMU_ONLY"
    DEGRADED = "Degraded"


class TaskID(str, Enum):
    THERMAL_MITIGATION = "thermal_mitigation"
    SENSOR_RECOVERY = "sensor_recovery"
    KERNEL_PANIC_RECOVERY = "kernel_panic_recovery"


# ---------------------------------------------------------------------------
# Sub-models: Hardware Components
# ---------------------------------------------------------------------------

class ThermalState(BaseModel):
    cpu_temp_c: float = Field(..., description="CPU temperature in Celsius")
    gpu_temp_c: float = Field(..., description="GPU temperature in Celsius")
    optical_pod_temp_c: float = Field(..., description="Optical pod temperature in Celsius")
    throttle_active: bool = Field(False, description="Whether thermal throttle is engaged")
    refresh_rate_hz: int = Field(90, description="Current display refresh rate (72/90/120 Hz)")
    fan_speed_pct: float = Field(0.0, description="Fan speed as percentage (0-100)")


class SensorState(BaseModel):
    imu_status: SensorStatus = Field(SensorStatus.ACTIVE, description="IMU sensor status")
    imu_latency_ms: float = Field(2.0, description="IMU read latency in ms")
    lidar_status: SensorStatus = Field(SensorStatus.ACTIVE, description="LiDAR sensor status")
    lidar_latency_ms: float = Field(5.0, description="LiDAR read latency in ms")
    hand_tracking_status: SensorStatus = Field(SensorStatus.ACTIVE, description="Hand tracking status")
    hand_tracking_latency_ms: float = Field(8.0, description="Hand tracking latency in ms")
    tracking_mode: TrackingMode = Field(TrackingMode.SLAM, description="Active tracking mode")
    tracking_stability: float = Field(1.0, ge=0.0, le=1.0, description="Stability score 0-1")


class ResourceState(BaseModel):
    battery_pct: float = Field(80.0, ge=0.0, le=100.0, description="Battery percentage")
    ram_used_mb: float = Field(..., description="RAM usage in MB")
    ram_total_mb: float = Field(8192.0, description="Total RAM in MB")
    cpu_usage_pct: float = Field(..., description="CPU usage percentage from host")
    frame_buffer_healthy: bool = Field(True, description="Frame buffer health status")
    background_processes: int = Field(5, description="Number of non-essential background processes")


class SpatialAnchorState(BaseModel):
    anchor_count: int = Field(12, description="Number of active spatial anchors")
    anchor_stability: float = Field(1.0, ge=0.0, le=1.0, description="Average anchor stability score")
    world_lock_active: bool = Field(True, description="Whether world anchor memory lock is active")
    session_data_intact: bool = Field(True, description="Whether previous session data is intact")
    cached_anchors_cleared: bool = Field(False, description="Whether cached anchors have been cleared")


class KernelState(BaseModel):
    status: KernelStatus = Field(KernelStatus.ACTIVE, description="Kernel operational status")
    crashed_app_id: Optional[int] = Field(None, description="App ID that caused crash (if any)")
    safe_mode_initiated: bool = Field(False, description="Whether safe mode boot was initiated")
    memory_lock_cleared: bool = Field(False, description="Whether memory lock has been cleared")
    uptime_seconds: float = Field(0.0, description="System uptime in seconds")


# ---------------------------------------------------------------------------
# Full System Observation
# ---------------------------------------------------------------------------

class SystemObservation(BaseModel):
    """Complete observable state of the Meta Quest OS simulation."""
    step: int = Field(0, description="Current step number")
    task_id: TaskID = Field(..., description="Active task identifier")
    thermals: ThermalState
    sensors: SensorState
    resources: ResourceState
    spatial_anchors: SpatialAnchorState
    kernel: KernelState
    last_action: Optional[str] = Field(None, description="Last action taken by agent")
    last_action_success: bool = Field(True, description="Whether last action succeeded")
    last_action_message: str = Field("", description="Human-readable result of last action")
    reward: float = Field(0.0, ge=0.0, le=1.0, description="Current step reward")
    cumulative_reward: float = Field(0.0, description="Total reward accumulated")
    done: bool = Field(False, description="Whether episode is complete")
    info: Dict[str, Any] = Field(default_factory=dict, description="Extra diagnostic info")


# ---------------------------------------------------------------------------
# Action Models
# ---------------------------------------------------------------------------

class ActionRequest(BaseModel):
    """Agent action submitted to the environment."""
    action: str = Field(..., description="Action command string")
    task_id: Optional[TaskID] = Field(None, description="Task context (optional)")


class ActionResult(BaseModel):
    """Result returned after executing an action."""
    observation: SystemObservation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reset & State Models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Request to reset the environment to a task scenario."""
    task_id: TaskID = Field(TaskID.THERMAL_MITIGATION, description="Which task scenario to initialize")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class ResetResult(BaseModel):
    """Result of a reset call."""
    observation: SystemObservation
    task_id: TaskID
    task_description: str
    available_actions: List[str]
    max_steps: int


class StateResult(BaseModel):
    """Current environment state (read-only snapshot)."""
    observation: SystemObservation
    task_id: TaskID
    step: int
    done: bool


# ---------------------------------------------------------------------------
# Task Info
# ---------------------------------------------------------------------------

class TaskInfo(BaseModel):
    task_id: TaskID
    name: str
    description: str
    difficulty: str
    max_steps: int
    success_threshold: float
    available_actions: List[str]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    environment: str = "MetaQuestOSEnv"