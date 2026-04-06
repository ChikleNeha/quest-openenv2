"""
tasks.py — Task definitions, action handlers, and graders for MetaQuestOSEnv.

Task 1: Thermal Mitigation        (Easy)   — GPU overheating scenario
Task 2: Sensor Fusion Recovery    (Medium) — LiDAR drift + SLAM failure
Task 3: Kernel Panic Recovery     (Hard)   — App crash + world-lock memory corruption
"""

from __future__ import annotations
import time
import random
from typing import Dict, List, Tuple, Any

from models import (
    TaskID, SensorStatus, KernelStatus, TrackingMode,
    ThermalState, SensorState, ResourceState, SpatialAnchorState, KernelState,
)
import database as db

# ---------------------------------------------------------------------------
# Available actions per task (publicly exposed to the agent)
# ---------------------------------------------------------------------------

ACTIONS_TASK1: List[str] = [
    "EXEC_THERMAL_THROTTLE --enable",
    "EXEC_THERMAL_THROTTLE --disable",
    "EXEC_DISPLAY_REFRESH --rate=72",
    "EXEC_DISPLAY_REFRESH --rate=90",
    "EXEC_DISPLAY_REFRESH --rate=120",
    "EXEC_KILL_BACKGROUND_PROCS",
    "EXEC_GPU_POWER_LIMIT --level=low",
    "EXEC_GPU_POWER_LIMIT --level=medium",
    "EXEC_GPU_POWER_LIMIT --level=high",
    "EXEC_FAN_OVERRIDE --speed=max",
    "EXEC_SUBSYSTEM_SUSPEND --target=hand_tracking",
    "EXEC_SUBSYSTEM_SUSPEND --target=lidar",
    "QUERY_THERMAL_STATUS",
    "QUERY_RESOURCE_STATUS",
    "NOOP",
]

ACTIONS_TASK2: List[str] = [
    "EXEC_SENSOR_RESET --target=lidar",
    "EXEC_SENSOR_RESET --target=imu",
    "EXEC_SENSOR_RESET --target=hand_tracking",
    "EXEC_SENSOR_RECALIBRATE --target=lidar",
    "EXEC_TRACKING_SWITCH --mode=VIO",
    "EXEC_TRACKING_SWITCH --mode=SLAM",
    "EXEC_TRACKING_SWITCH --mode=IMU_ONLY",
    "EXEC_VIO_RESET --priority=imu",
    "EXEC_SLAM_REINIT --low_light_mode=true",
    "EXEC_ANCHOR_RECOMPUTE",
    "EXEC_SUBSYSTEM_RESTART --target=tracking_engine",
    "QUERY_SENSOR_STATUS",
    "QUERY_TRACKING_STATUS",
    "NOOP",
]

ACTIONS_TASK3: List[str] = [
    "EXEC_SAFE_MODE_INIT --stage=1",
    "EXEC_SAFE_MODE_INIT --stage=2",
    "EXEC_SAFE_MODE_INIT --stage=3",
    "EXEC_MEMORY_LOCK_CLEAR --target=world_anchor",
    "EXEC_MEMORY_LOCK_CLEAR --target=app_cache",
    "EXEC_ANCHOR_CACHE_FLUSH",
    "EXEC_SESSION_RESTORE --source=backup",
    "EXEC_KERNEL_RESTART --mode=safe",
    "EXEC_KERNEL_RESTART --mode=full",
    "EXEC_APP_FORCE_KILL --id=crashed",
    "EXEC_FRAME_BUFFER_RESET",
    "QUERY_KERNEL_STATUS",
    "QUERY_ANCHOR_STATUS",
    "QUERY_SESSION_STATUS",
    "NOOP",
]

TASK_ACTIONS: Dict[TaskID, List[str]] = {
    TaskID.THERMAL_MITIGATION: ACTIONS_TASK1,
    TaskID.SENSOR_RECOVERY: ACTIONS_TASK2,
    TaskID.KERNEL_PANIC_RECOVERY: ACTIONS_TASK3,
}

TASK_MAX_STEPS: Dict[TaskID, int] = {
    TaskID.THERMAL_MITIGATION: 10,
    TaskID.SENSOR_RECOVERY: 12,
    TaskID.KERNEL_PANIC_RECOVERY: 15,
}

TASK_DESCRIPTIONS: Dict[TaskID, str] = {
    TaskID.THERMAL_MITIGATION: (
        "SCENARIO: A high-intensity MR experience is running. GPU temperature has reached 85°C "
        "and is climbing. The device is at risk of emergency shutdown. "
        "GOAL: Bring GPU temperature below 70°C without crashing the OS. "
        "SUCCESS: reward=1.0 if gpu_temp_c < 70 and kernel.status == Active within max_steps."
    ),
    TaskID.SENSOR_RECOVERY: (
        "SCENARIO: Low-light environmental conditions have caused the LiDAR sensor to enter "
        "Drift state, destabilizing SLAM tracking. Virtual objects are floating away from their anchors. "
        "GOAL: Restore tracking_stability above 0.9 using sensor reset and tracking mode switching. "
        "SUCCESS: reward=1.0 if tracking_stability > 0.9 and lidar_status != Failed."
    ),
    TaskID.KERNEL_PANIC_RECOVERY: (
        "SCENARIO: A spatial application (App ID 57) has crashed and corrupted the World Anchor "
        "memory lock. The kernel is in PANIC state. The system is unresponsive to normal commands. "
        "GOAL: Execute a multi-step Safe Mode recovery — clear memory lock, flush anchor cache, "
        "restore session data, and bring kernel back to Active without factory reset. "
        "SUCCESS: reward=1.0 if kernel.status == Active and session_data_intact == True."
    ),
}


# ---------------------------------------------------------------------------
# Initial state builders
# ---------------------------------------------------------------------------

def _real_system_metrics() -> Tuple[float, float]:
    """Pull real CPU and RAM metrics from host machine."""
    try:
        import psutil
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        return cpu, ram.used / (1024 * 1024)  # CPU%, RAM in MB
    except Exception:
        return 30.0, 2048.0  # safe fallback


def init_task1_state() -> None:
    """GPU overheating scenario."""
    cpu_pct, ram_mb = _real_system_metrics()
    db.clear_all()
    db.set_state_bulk({
        "task_id": TaskID.THERMAL_MITIGATION.value,
        "step": 0,
        "done": False,
        # Thermals — GPU is dangerously hot
        "cpu_temp_c": 72.0 + random.uniform(-2, 2),
        "gpu_temp_c": 85.0 + random.uniform(0, 5),
        "optical_pod_temp_c": 48.0,
        "throttle_active": False,
        "refresh_rate_hz": 120,
        "fan_speed_pct": 60.0,
        # Sensors — all healthy
        "imu_status": SensorStatus.ACTIVE.value,
        "imu_latency_ms": 2.0,
        "lidar_status": SensorStatus.ACTIVE.value,
        "lidar_latency_ms": 5.0,
        "hand_tracking_status": SensorStatus.ACTIVE.value,
        "hand_tracking_latency_ms": 8.0,
        "tracking_mode": TrackingMode.SLAM.value,
        "tracking_stability": 0.95,
        # Resources — real system data
        "battery_pct": 78.0,
        "ram_used_mb": ram_mb,
        "ram_total_mb": 8192.0,
        "cpu_usage_pct": cpu_pct,
        "frame_buffer_healthy": True,
        "background_processes": 7,
        # Spatial
        "anchor_count": 12,
        "anchor_stability": 0.92,
        "world_lock_active": True,
        "session_data_intact": True,
        "cached_anchors_cleared": False,
        # Kernel
        "kernel_status": KernelStatus.ACTIVE.value,
        "crashed_app_id": None,
        "safe_mode_initiated": False,
        "memory_lock_cleared": False,
        "uptime_seconds": time.time(),
        # Tracking
        "gpu_power_level": "high",
        "hand_suspended": False,
        "lidar_suspended": False,
        "cumulative_reward": 0.0,
    })


def init_task2_state() -> None:
    """LiDAR drift + SLAM failure scenario."""
    cpu_pct, ram_mb = _real_system_metrics()
    db.clear_all()
    db.set_state_bulk({
        "task_id": TaskID.SENSOR_RECOVERY.value,
        "step": 0,
        "done": False,
        "cpu_temp_c": 65.0,
        "gpu_temp_c": 68.0,
        "optical_pod_temp_c": 42.0,
        "throttle_active": False,
        "refresh_rate_hz": 90,
        "fan_speed_pct": 40.0,
        # LiDAR is Drifting — this is the problem
        "imu_status": SensorStatus.ACTIVE.value,
        "imu_latency_ms": 2.5,
        "lidar_status": SensorStatus.DRIFTING.value,
        "lidar_latency_ms": 45.0,        # very high latency = drift indicator
        "hand_tracking_status": SensorStatus.ACTIVE.value,
        "hand_tracking_latency_ms": 8.0,
        "tracking_mode": TrackingMode.SLAM.value,
        "tracking_stability": 0.31,      # very low — objects are floating
        "battery_pct": 65.0,
        "ram_used_mb": ram_mb,
        "ram_total_mb": 8192.0,
        "cpu_usage_pct": cpu_pct,
        "frame_buffer_healthy": True,
        "background_processes": 4,
        "anchor_count": 12,
        "anchor_stability": 0.28,
        "world_lock_active": True,
        "session_data_intact": True,
        "cached_anchors_cleared": False,
        "kernel_status": KernelStatus.ACTIVE.value,
        "crashed_app_id": None,
        "safe_mode_initiated": False,
        "memory_lock_cleared": False,
        "uptime_seconds": time.time(),
        "gpu_power_level": "medium",
        "hand_suspended": False,
        "lidar_suspended": False,
        "vio_reset_done": False,
        "slam_reinit_done": False,
        "cumulative_reward": 0.0,
    })


def init_task3_state() -> None:
    """Kernel panic + world-lock corruption scenario."""
    cpu_pct, ram_mb = _real_system_metrics()
    db.clear_all()
    # Save a "good" session backup before the crash
    db.backup_session({
        "workspace_layout": "dual_panel_code_editor",
        "anchor_positions": {"desk": [0.1, 0.0, -0.5], "monitor": [0.0, 0.3, -1.2]},
        "user_preferences": {"theme": "dark", "hand_dominance": "right"},
        "open_apps": ["CodeEditor", "Terminal", "BrowserOverlay"],
    })
    db.set_state_bulk({
        "task_id": TaskID.KERNEL_PANIC_RECOVERY.value,
        "step": 0,
        "done": False,
        "cpu_temp_c": 78.0,
        "gpu_temp_c": 74.0,
        "optical_pod_temp_c": 50.0,
        "throttle_active": True,
        "refresh_rate_hz": 72,
        "fan_speed_pct": 90.0,
        # Sensors degraded from the crash
        "imu_status": SensorStatus.ACTIVE.value,
        "imu_latency_ms": 4.0,
        "lidar_status": SensorStatus.FAILED.value,
        "lidar_latency_ms": 999.0,
        "hand_tracking_status": SensorStatus.FAILED.value,
        "hand_tracking_latency_ms": 999.0,
        "tracking_mode": TrackingMode.DEGRADED.value,
        "tracking_stability": 0.0,
        "battery_pct": 55.0,
        "ram_used_mb": min(ram_mb + 2000, 7800),    # simulated memory leak
        "ram_total_mb": 8192.0,
        "cpu_usage_pct": min(cpu_pct + 40, 98.0),
        "frame_buffer_healthy": False,
        "background_processes": 0,
        # World-lock corruption
        "anchor_count": 12,
        "anchor_stability": 0.0,
        "world_lock_active": True,          # locked and broken
        "session_data_intact": False,       # lost due to crash
        "cached_anchors_cleared": False,
        # Kernel in PANIC
        "kernel_status": KernelStatus.PANIC.value,
        "crashed_app_id": 57,
        "safe_mode_initiated": False,
        "memory_lock_cleared": False,
        "safe_mode_stage": 0,               # must progress 1→2→3
        "uptime_seconds": time.time(),
        "gpu_power_level": "low",
        "hand_suspended": True,
        "lidar_suspended": True,
        "session_restored": False,
        "cumulative_reward": 0.0,
    })


TASK_INITIALIZERS = {
    TaskID.THERMAL_MITIGATION: init_task1_state,
    TaskID.SENSOR_RECOVERY: init_task2_state,
    TaskID.KERNEL_PANIC_RECOVERY: init_task3_state,
}


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def handle_action_task1(action: str) -> Tuple[float, bool, str]:
    """Returns (reward, success, message)."""
    s = db.get_all_state()
    gpu = s.get("gpu_temp_c", 85.0)
    cpu = s.get("cpu_temp_c", 72.0)
    bg = s.get("background_processes", 7)
    rr = s.get("refresh_rate_hz", 120)
    kernel = s.get("kernel_status", KernelStatus.ACTIVE.value)
    throttle = s.get("throttle_active", False)

    updates: Dict[str, Any] = {}
    success = True
    message = ""

    if action == "EXEC_THERMAL_THROTTLE --enable":
        updates["throttle_active"] = True
        updates["gpu_temp_c"] = _clamp(gpu - random.uniform(4, 8), 50, 95)
        updates["cpu_temp_c"] = _clamp(cpu - random.uniform(2, 4), 50, 90)
        message = f"Thermal throttle engaged. GPU: {updates['gpu_temp_c']:.1f}°C"

    elif action == "EXEC_THERMAL_THROTTLE --disable":
        updates["throttle_active"] = False
        updates["gpu_temp_c"] = _clamp(gpu + random.uniform(3, 6), 50, 98)
        message = "Thermal throttle disabled. GPU temperature rising."

    elif action == "EXEC_DISPLAY_REFRESH --rate=72":
        updates["refresh_rate_hz"] = 72
        updates["gpu_temp_c"] = _clamp(gpu - random.uniform(5, 10), 50, 95)
        message = f"Refresh rate set to 72Hz. GPU load reduced. Temp: {updates['gpu_temp_c']:.1f}°C"

    elif action == "EXEC_DISPLAY_REFRESH --rate=90":
        delta = -2 if rr == 120 else 3
        updates["refresh_rate_hz"] = 90
        updates["gpu_temp_c"] = _clamp(gpu + delta, 50, 98)
        message = f"Refresh rate set to 90Hz."

    elif action == "EXEC_DISPLAY_REFRESH --rate=120":
        updates["refresh_rate_hz"] = 120
        updates["gpu_temp_c"] = _clamp(gpu + random.uniform(4, 8), 50, 100)
        message = f"Refresh rate set to 120Hz. GPU load increased. Temp: {updates['gpu_temp_c']:.1f}°C"
        if updates["gpu_temp_c"] > 95:
            updates["kernel_status"] = KernelStatus.DEGRADED.value
            message += " WARNING: Approaching emergency shutdown!"

    elif action == "EXEC_KILL_BACKGROUND_PROCS":
        if bg > 0:
            killed = min(bg, random.randint(2, 4))
            updates["background_processes"] = bg - killed
            updates["gpu_temp_c"] = _clamp(gpu - random.uniform(3, 7), 50, 95)
            updates["cpu_temp_c"] = _clamp(cpu - random.uniform(2, 5), 50, 90)
            message = f"Killed {killed} background processes. GPU: {updates['gpu_temp_c']:.1f}°C"
        else:
            success = False
            message = "No background processes to kill."

    elif action == "EXEC_GPU_POWER_LIMIT --level=low":
        updates["gpu_power_level"] = "low"
        updates["gpu_temp_c"] = _clamp(gpu - random.uniform(8, 14), 50, 95)
        message = f"GPU power limited to LOW. Temp: {updates['gpu_temp_c']:.1f}°C"

    elif action == "EXEC_GPU_POWER_LIMIT --level=medium":
        updates["gpu_power_level"] = "medium"
        updates["gpu_temp_c"] = _clamp(gpu - random.uniform(3, 6), 50, 95)
        message = f"GPU power limited to MEDIUM."

    elif action == "EXEC_GPU_POWER_LIMIT --level=high":
        updates["gpu_power_level"] = "high"
        updates["gpu_temp_c"] = _clamp(gpu + random.uniform(2, 5), 50, 100)
        message = "GPU power set to HIGH."

    elif action == "EXEC_FAN_OVERRIDE --speed=max":
        updates["fan_speed_pct"] = 100.0
        updates["gpu_temp_c"] = _clamp(gpu - random.uniform(3, 6), 50, 95)
        updates["cpu_temp_c"] = _clamp(cpu - random.uniform(2, 4), 50, 90)
        message = f"Fan override to MAX. Passive cooling engaged. GPU: {updates.get('gpu_temp_c', gpu):.1f}°C"

    elif action == "EXEC_SUBSYSTEM_SUSPEND --target=hand_tracking":
        updates["hand_suspended"] = True
        updates["gpu_temp_c"] = _clamp(gpu - random.uniform(2, 4), 50, 95)
        message = "Hand tracking suspended. Minor GPU relief."

    elif action == "EXEC_SUBSYSTEM_SUSPEND --target=lidar":
        updates["lidar_suspended"] = True
        updates["gpu_temp_c"] = _clamp(gpu - random.uniform(1, 3), 50, 95)
        message = "LiDAR suspended. Minor GPU relief."

    elif action in ("QUERY_THERMAL_STATUS", "QUERY_RESOURCE_STATUS"):
        message = (f"GPU={gpu:.1f}°C CPU={cpu:.1f}°C "
                   f"throttle={throttle} refresh={rr}Hz bg_procs={bg}")

    elif action == "NOOP":
        # Passive heat creep when doing nothing
        updates["gpu_temp_c"] = _clamp(gpu + random.uniform(1, 3), 50, 100)
        message = f"No action. GPU temperature rising: {updates['gpu_temp_c']:.1f}°C"
        if updates["gpu_temp_c"] > 95:
            updates["kernel_status"] = KernelStatus.DEGRADED.value

    else:
        success = False
        message = f"Unknown action: {action}"

    if updates:
        db.set_state_bulk(updates)

    # --- Compute reward ---
    final_gpu = db.get_state("gpu_temp_c", gpu)
    final_kernel = db.get_state("kernel_status", KernelStatus.ACTIVE.value)

    if final_kernel == KernelStatus.PANIC.value:
        reward = 0.0  # OS crashed — episode failure
        db.set_state("done", True)
        message += " CRITICAL: Kernel panic! OS crashed."
    elif final_gpu < 70.0 and final_kernel == KernelStatus.ACTIVE.value:
        reward = 1.0
        db.set_state("done", True)
    elif final_gpu < 73.0:
        reward = 0.85
    elif final_gpu < 77.0:
        reward = 0.65
    elif final_gpu < 81.0:
        reward = 0.45
    elif final_gpu < 85.0:
        reward = 0.3
    else:
        reward = 0.1

    return reward, success, message


def handle_action_task2(action: str) -> Tuple[float, bool, str]:
    s = db.get_all_state()
    lidar = s.get("lidar_status", SensorStatus.DRIFTING.value)
    imu = s.get("imu_status", SensorStatus.ACTIVE.value)
    stability = s.get("tracking_stability", 0.31)
    mode = s.get("tracking_mode", TrackingMode.SLAM.value)
    vio_done = s.get("vio_reset_done", False)

    updates: Dict[str, Any] = {}
    success = True
    message = ""

    if action == "EXEC_SENSOR_RESET --target=lidar":
        if lidar == SensorStatus.FAILED.value:
            updates["lidar_status"] = SensorStatus.RECALIBRATING.value
            updates["lidar_latency_ms"] = 30.0
            message = "LiDAR reset initiated. Status: Recalibrating."
        elif lidar in (SensorStatus.DRIFTING.value, SensorStatus.RECALIBRATING.value):
            updates["lidar_status"] = SensorStatus.RECALIBRATING.value
            updates["lidar_latency_ms"] = 20.0
            updates["tracking_stability"] = _clamp(stability + 0.15, 0, 1)
            message = "LiDAR reset applied. Recalibrating..."
        else:
            message = "LiDAR already Active."

    elif action == "EXEC_SENSOR_RECALIBRATE --target=lidar":
        if lidar == SensorStatus.RECALIBRATING.value:
            updates["lidar_status"] = SensorStatus.ACTIVE.value
            updates["lidar_latency_ms"] = 5.0
            updates["tracking_stability"] = _clamp(stability + 0.2, 0, 1)
            message = "LiDAR calibration complete. Status: Active."
        elif lidar == SensorStatus.DRIFTING.value:
            updates["lidar_latency_ms"] = s.get("lidar_latency_ms", 45) - 10
            updates["tracking_stability"] = _clamp(stability + 0.05, 0, 1)
            message = "Calibration attempted on drifting sensor. Partial improvement."
        else:
            message = "Calibrate when sensor is in Recalibrating state for best results."

    elif action == "EXEC_SENSOR_RESET --target=imu":
        updates["imu_status"] = SensorStatus.RECALIBRATING.value
        updates["imu_latency_ms"] = 3.0
        message = "IMU reset initiated."

    elif action == "EXEC_SENSOR_RESET --target=hand_tracking":
        updates["hand_tracking_status"] = SensorStatus.RECALIBRATING.value
        message = "Hand tracking reset initiated."

    elif action == "EXEC_TRACKING_SWITCH --mode=VIO":
        updates["tracking_mode"] = TrackingMode.VIO.value
        updates["tracking_stability"] = _clamp(stability + 0.25, 0, 1)
        message = f"Tracking switched to VIO. Stability: {updates['tracking_stability']:.2f}"

    elif action == "EXEC_TRACKING_SWITCH --mode=IMU_ONLY":
        updates["tracking_mode"] = TrackingMode.IMU_ONLY.value
        updates["tracking_stability"] = _clamp(stability + 0.15, 0, 1)
        message = f"Tracking switched to IMU_ONLY. Stability: {updates['tracking_stability']:.2f}"

    elif action == "EXEC_TRACKING_SWITCH --mode=SLAM":
        if lidar != SensorStatus.ACTIVE.value:
            updates["tracking_stability"] = _clamp(stability - 0.1, 0, 1)
            message = "Cannot switch to SLAM — LiDAR not Active. Stability degraded."
            success = False
        else:
            updates["tracking_mode"] = TrackingMode.SLAM.value
            updates["tracking_stability"] = _clamp(stability + 0.1, 0, 1)
            message = "Tracking restored to SLAM."

    elif action == "EXEC_VIO_RESET --priority=imu":
        updates["vio_reset_done"] = True
        updates["tracking_mode"] = TrackingMode.VIO.value
        updates["tracking_stability"] = _clamp(stability + 0.3, 0, 1)
        updates["anchor_stability"] = _clamp(s.get("anchor_stability", 0.28) + 0.2, 0, 1)
        message = f"VIO reset with IMU priority. Stability: {updates['tracking_stability']:.2f}"

    elif action == "EXEC_SLAM_REINIT --low_light_mode=true":
        if lidar == SensorStatus.ACTIVE.value:
            updates["tracking_mode"] = TrackingMode.SLAM.value
            updates["tracking_stability"] = _clamp(stability + 0.25, 0, 1)
            message = "SLAM reinitialized in low-light mode."
        else:
            message = "SLAM reinit requires Active LiDAR. Use VIO or reset LiDAR first."
            success = False

    elif action == "EXEC_ANCHOR_RECOMPUTE":
        new_stab = _clamp(stability + 0.1, 0, 1)
        updates["anchor_stability"] = new_stab
        updates["tracking_stability"] = _clamp(stability + 0.05, 0, 1)
        message = f"Spatial anchors recomputed. Anchor stability: {new_stab:.2f}"

    elif action == "EXEC_SUBSYSTEM_RESTART --target=tracking_engine":
        updates["tracking_stability"] = _clamp(stability + 0.1, 0, 1)
        updates["lidar_status"] = SensorStatus.RECALIBRATING.value
        message = "Tracking engine restarted. All sensors recalibrating."

    elif action in ("QUERY_SENSOR_STATUS", "QUERY_TRACKING_STATUS"):
        message = (f"LiDAR={lidar} IMU={imu} mode={mode} "
                   f"stability={stability:.2f}")

    elif action == "NOOP":
        # Drift worsens over time
        updates["tracking_stability"] = _clamp(stability - 0.05, 0, 1)
        message = f"No action. Drift worsening. Stability: {updates['tracking_stability']:.2f}"

    else:
        success = False
        message = f"Unknown action: {action}"

    if updates:
        db.set_state_bulk(updates)

    # --- Compute reward ---
    final_stability = db.get_state("tracking_stability", stability)
    final_lidar = db.get_state("lidar_status", lidar)

    if final_stability > 0.9 and final_lidar != SensorStatus.FAILED.value:
        reward = 1.0
        db.set_state("done", True)
    elif final_stability > 0.75:
        reward = 0.8
    elif final_stability > 0.6:
        reward = 0.65
    elif final_stability > 0.45:
        reward = 0.5
    elif final_stability > 0.3:
        reward = 0.35
    else:
        reward = 0.15

    return reward, success, message


def handle_action_task3(action: str) -> Tuple[float, bool, str]:
    s = db.get_all_state()
    kernel = s.get("kernel_status", KernelStatus.PANIC.value)
    safe_stage = s.get("safe_mode_stage", 0)
    mem_cleared = s.get("memory_lock_cleared", False)
    anchors_cleared = s.get("cached_anchors_cleared", False)
    session_restored = s.get("session_restored", False)
    world_lock = s.get("world_lock_active", True)
    frame_ok = s.get("frame_buffer_healthy", False)

    updates: Dict[str, Any] = {}
    success = True
    message = ""

    if action == "EXEC_SAFE_MODE_INIT --stage=1":
        if kernel not in (KernelStatus.PANIC.value, KernelStatus.DEGRADED.value):
            message = "Safe mode only needed when kernel is in PANIC or DEGRADED state."
            success = False
        else:
            updates["kernel_status"] = KernelStatus.SAFE_MODE.value
            updates["safe_mode_stage"] = 1
            updates["safe_mode_initiated"] = True
            message = "Safe Mode Stage 1 initiated. Kernel entering controlled safe state."

    elif action == "EXEC_SAFE_MODE_INIT --stage=2":
        if safe_stage < 1:
            message = "Must complete Stage 1 before Stage 2."
            success = False
        else:
            updates["safe_mode_stage"] = 2
            updates["cpu_usage_pct"] = _clamp(s.get("cpu_usage_pct", 90) - 30, 10, 100)
            updates["ram_used_mb"] = _clamp(s.get("ram_used_mb", 7800) - 1500, 1000, 8000)
            message = "Safe Mode Stage 2: Resource isolation complete. CPU/RAM stabilizing."

    elif action == "EXEC_SAFE_MODE_INIT --stage=3":
        if safe_stage < 2:
            message = "Must complete Stage 2 before Stage 3."
            success = False
        else:
            updates["safe_mode_stage"] = 3
            updates["crashed_app_id"] = None
            message = "Safe Mode Stage 3: Crash isolation complete. Preparing recovery environment."

    elif action == "EXEC_MEMORY_LOCK_CLEAR --target=world_anchor":
        if kernel not in (KernelStatus.SAFE_MODE.value, KernelStatus.RECOVERING.value):
            message = "Memory lock can only be cleared in Safe Mode. Initiate safe mode first."
            success = False
        else:
            updates["memory_lock_cleared"] = True
            updates["world_lock_active"] = False
            updates["kernel_status"] = KernelStatus.RECOVERING.value
            message = "World Anchor memory lock cleared. Kernel entering Recovery state."

    elif action == "EXEC_MEMORY_LOCK_CLEAR --target=app_cache":
        updates["ram_used_mb"] = _clamp(s.get("ram_used_mb", 5000) - 800, 1000, 8000)
        message = "App cache memory cleared."

    elif action == "EXEC_ANCHOR_CACHE_FLUSH":
        if not mem_cleared:
            message = "Clear memory lock before flushing anchor cache."
            success = False
        else:
            updates["cached_anchors_cleared"] = True
            updates["anchor_stability"] = 0.0   # reset — will be rebuilt on restore
            updates["anchor_count"] = 0
            message = "Spatial anchor cache flushed. Ready for session restore."

    elif action == "EXEC_SESSION_RESTORE --source=backup":
        if not anchors_cleared:
            message = "Flush anchor cache before restoring session."
            success = False
        else:
            backup = db.restore_session()
            if backup:
                updates["session_data_intact"] = True
                updates["session_restored"] = True
                updates["anchor_count"] = 12
                updates["anchor_stability"] = 0.85
                updates["tracking_stability"] = 0.8
                message = f"Session restored from backup. Workspace: {backup.get('workspace_layout', 'unknown')}"
            else:
                success = False
                message = "No backup found."

    elif action == "EXEC_KERNEL_RESTART --mode=safe":
        if not session_restored:
            message = "Restore session data before restarting kernel."
            success = False
        elif safe_stage < 3:
            message = "Complete all Safe Mode stages before kernel restart."
            success = False
        else:
            updates["kernel_status"] = KernelStatus.ACTIVE.value
            updates["frame_buffer_healthy"] = True
            updates["lidar_status"] = SensorStatus.RECALIBRATING.value
            updates["hand_tracking_status"] = SensorStatus.RECALIBRATING.value
            updates["tracking_mode"] = TrackingMode.VIO.value
            updates["tracking_stability"] = 0.75
            message = "Kernel restarted in Safe Mode. System recovering. All sensors recalibrating."

    elif action == "EXEC_KERNEL_RESTART --mode=full":
        # Full restart = factory reset = BAD outcome
        updates["kernel_status"] = KernelStatus.ACTIVE.value
        updates["session_data_intact"] = False
        updates["session_restored"] = False
        updates["anchor_count"] = 0
        updates["anchor_stability"] = 0.0
        message = "FULL RESTART performed. Session data LOST. Factory reset equivalent."

    elif action == "EXEC_APP_FORCE_KILL --id=crashed":
        updates["crashed_app_id"] = None
        updates["ram_used_mb"] = _clamp(s.get("ram_used_mb", 7000) - 500, 1000, 8000)
        message = "Crashed application process terminated."

    elif action == "EXEC_FRAME_BUFFER_RESET":
        if kernel in (KernelStatus.ACTIVE.value, KernelStatus.RECOVERING.value):
            updates["frame_buffer_healthy"] = True
            message = "Frame buffer reset successful."
        else:
            message = "Frame buffer reset requires kernel in Active or Recovering state."
            success = False

    elif action in ("QUERY_KERNEL_STATUS", "QUERY_ANCHOR_STATUS", "QUERY_SESSION_STATUS"):
        message = (f"kernel={kernel} stage={safe_stage} mem_cleared={mem_cleared} "
                   f"anchors_cleared={anchors_cleared} session_restored={session_restored} "
                   f"world_lock={world_lock}")

    elif action == "NOOP":
        message = "No action taken. System remains in current state."

    else:
        success = False
        message = f"Unknown action: {action}"

    if updates:
        db.set_state_bulk(updates)

    # --- Multi-criteria reward ---
    final_kernel = db.get_state("kernel_status", kernel)
    final_session = db.get_state("session_data_intact", False)
    final_restored = db.get_state("session_restored", False)
    final_mem = db.get_state("memory_lock_cleared", False)
    final_stage = db.get_state("safe_mode_stage", safe_stage)

    if final_kernel == KernelStatus.ACTIVE.value and final_session and final_restored:
        reward = 1.0
        db.set_state("done", True)
    elif final_kernel == KernelStatus.ACTIVE.value and not final_session:
        reward = 0.3  # recovered but lost session data
    elif final_kernel == KernelStatus.RECOVERING.value and final_restored:
        reward = 0.75
    elif final_kernel == KernelStatus.RECOVERING.value:
        reward = 0.5
    elif final_kernel == KernelStatus.SAFE_MODE.value:
        # Partial credit for stage progression — higher floor
        reward = 0.3 + (final_stage * 0.1) + (0.1 if final_mem else 0)
    else:
        reward = 0.0

    return reward, success, message


TASK_HANDLERS = {
    TaskID.THERMAL_MITIGATION: handle_action_task1,
    TaskID.SENSOR_RECOVERY: handle_action_task2,
    TaskID.KERNEL_PANIC_RECOVERY: handle_action_task3,
}