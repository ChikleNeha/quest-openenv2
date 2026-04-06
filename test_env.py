"""
test_env.py — Local smoke tests for MetaQuestOSEnv.

Tests the environment directly (no HTTP server needed).
Run this before deploying to catch any issues early.

Usage:
    python test_env.py
"""

from __future__ import annotations
import sys
import traceback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "✓"
FAIL = "✗"
results = []


def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"       → {e}")
        traceback.print_exc()
        results.append((name, False))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_imports():
    import models, database, tasks, environment
    from models import TaskID, SensorStatus, KernelStatus
    assert hasattr(models, "SystemObservation")
    assert hasattr(models, "ActionRequest")
    assert hasattr(models, "ResetRequest")


def test_db_init():
    import database as db
    db.init_db()
    db.set_state("test_key", {"hello": "world"})
    val = db.get_state("test_key")
    assert val == {"hello": "world"}, f"Got {val}"
    db.set_state_bulk({"a": 1, "b": 2})
    assert db.get_state("a") == 1
    assert db.get_state("b") == 2


def test_task1_reset_and_step():
    from environment import MetaQuestOSEnv
    from models import TaskID
    env = MetaQuestOSEnv()
    r = env.reset(TaskID.THERMAL_MITIGATION, seed=42)
    assert r.task_id == TaskID.THERMAL_MITIGATION
    assert len(r.available_actions) > 0
    assert r.observation.thermals.gpu_temp_c >= 80.0, "GPU should start hot"
    assert r.observation.kernel.status.value == "Active"
    assert r.max_steps == 10


def test_task1_reward_range():
    from environment import MetaQuestOSEnv
    from models import TaskID
    env = MetaQuestOSEnv()
    env.reset(TaskID.THERMAL_MITIGATION, seed=42)
    actions = [
        "EXEC_THERMAL_THROTTLE --enable",
        "EXEC_DISPLAY_REFRESH --rate=72",
        "EXEC_GPU_POWER_LIMIT --level=low",
    ]
    for action in actions:
        result = env.step(action)
        assert 0.0 <= result.reward <= 1.0, f"Reward {result.reward} out of range"


def test_task1_solvable():
    from environment import MetaQuestOSEnv
    from models import TaskID
    env = MetaQuestOSEnv()
    env.reset(TaskID.THERMAL_MITIGATION, seed=42)
    policy = [
        "EXEC_THERMAL_THROTTLE --enable",
        "EXEC_DISPLAY_REFRESH --rate=72",
        "EXEC_GPU_POWER_LIMIT --level=low",
        "EXEC_KILL_BACKGROUND_PROCS",
        "EXEC_FAN_OVERRIDE --speed=max",
    ]
    final = None
    for action in policy:
        final = env.step(action)
        if final.done:
            break
    assert final is not None
    assert final.observation.thermals.gpu_temp_c < 70.0, (
        f"GPU still too hot: {final.observation.thermals.gpu_temp_c}"
    )
    assert final.reward == 1.0


def test_task2_reset_and_step():
    from environment import MetaQuestOSEnv
    from models import TaskID, SensorStatus
    env = MetaQuestOSEnv()
    r = env.reset(TaskID.SENSOR_RECOVERY, seed=42)
    assert r.observation.sensors.lidar_status == SensorStatus.DRIFTING
    assert r.observation.sensors.tracking_stability < 0.5, "Should start unstable"
    assert r.max_steps == 12


def test_task2_solvable():
    from environment import MetaQuestOSEnv
    from models import TaskID
    env = MetaQuestOSEnv()
    env.reset(TaskID.SENSOR_RECOVERY, seed=42)
    policy = [
        "EXEC_SENSOR_RESET --target=lidar",
        "EXEC_VIO_RESET --priority=imu",
        "EXEC_SENSOR_RECALIBRATE --target=lidar",
        "EXEC_ANCHOR_RECOMPUTE",
    ]
    final = None
    for action in policy:
        final = env.step(action)
        if final.done:
            break
    assert final is not None
    assert final.observation.sensors.tracking_stability > 0.9, (
        f"Stability still low: {final.observation.sensors.tracking_stability}"
    )
    assert final.reward == 1.0


def test_task3_reset_and_step():
    from environment import MetaQuestOSEnv
    from models import TaskID, KernelStatus
    env = MetaQuestOSEnv()
    r = env.reset(TaskID.KERNEL_PANIC_RECOVERY, seed=42)
    assert r.observation.kernel.status == KernelStatus.PANIC
    assert r.observation.spatial_anchors.session_data_intact == False
    assert r.max_steps == 15


def test_task3_solvable():
    from environment import MetaQuestOSEnv
    from models import TaskID, KernelStatus
    env = MetaQuestOSEnv()
    env.reset(TaskID.KERNEL_PANIC_RECOVERY, seed=42)
    policy = [
        "EXEC_SAFE_MODE_INIT --stage=1",
        "EXEC_SAFE_MODE_INIT --stage=2",
        "EXEC_SAFE_MODE_INIT --stage=3",
        "EXEC_MEMORY_LOCK_CLEAR --target=world_anchor",
        "EXEC_ANCHOR_CACHE_FLUSH",
        "EXEC_SESSION_RESTORE --source=backup",
        "EXEC_KERNEL_RESTART --mode=safe",
    ]
    final = None
    for action in policy:
        final = env.step(action)
        if final.done:
            break
    assert final is not None
    assert final.observation.kernel.status == KernelStatus.ACTIVE
    assert final.observation.spatial_anchors.session_data_intact == True
    assert final.reward == 1.0


def test_task3_partial_rewards():
    """Verify partial reward signals exist between 0 and 1."""
    from environment import MetaQuestOSEnv
    from models import TaskID
    env = MetaQuestOSEnv()
    env.reset(TaskID.KERNEL_PANIC_RECOVERY, seed=42)
    # Only do stage 1 — should get partial reward
    r = env.step("EXEC_SAFE_MODE_INIT --stage=1")
    assert 0.0 < r.reward < 1.0, f"Expected partial reward, got {r.reward}"
    assert not r.done


def test_invalid_action():
    from environment import MetaQuestOSEnv
    from models import TaskID
    env = MetaQuestOSEnv()
    env.reset(TaskID.THERMAL_MITIGATION, seed=42)
    r = env.step("TOTALLY_INVALID_COMMAND")
    assert r.reward == 0.0
    assert not r.observation.last_action_success


def test_max_steps_terminates():
    from environment import MetaQuestOSEnv
    from models import TaskID
    env = MetaQuestOSEnv()
    env.reset(TaskID.THERMAL_MITIGATION, seed=42)
    for _ in range(10):  # max_steps for task 1
        result = env.step("NOOP")
    assert result.done, "Episode should end at max_steps"


def test_state_endpoint():
    from environment import MetaQuestOSEnv
    from models import TaskID
    env = MetaQuestOSEnv()
    env.reset(TaskID.SENSOR_RECOVERY, seed=42)
    env.step("EXEC_SENSOR_RESET --target=lidar")
    state = env.state()
    assert state.step == 1
    assert state.task_id == TaskID.SENSOR_RECOVERY
    assert not state.done


def test_pydantic_serialization():
    from environment import MetaQuestOSEnv
    from models import TaskID
    import json
    env = MetaQuestOSEnv()
    r = env.reset(TaskID.THERMAL_MITIGATION, seed=42)
    # Must be JSON-serializable (required for HTTP API)
    serialized = r.model_dump_json()
    parsed = json.loads(serialized)
    assert "observation" in parsed
    assert "thermals" in parsed["observation"]
    assert "gpu_temp_c" in parsed["observation"]["thermals"]


def test_psutil_integration():
    """Verify real system metrics are being injected."""
    from environment import MetaQuestOSEnv
    from models import TaskID
    env = MetaQuestOSEnv()
    r = env.reset(TaskID.THERMAL_MITIGATION, seed=99)
    # RAM and CPU should be real values, not zeros
    assert r.observation.resources.ram_used_mb > 0, "RAM should be > 0"
    assert r.observation.resources.cpu_usage_pct >= 0, "CPU should be >= 0"
    assert r.observation.resources.ram_used_mb < 8192, "RAM should be < 8GB total"


def test_list_tasks():
    from environment import MetaQuestOSEnv
    env = MetaQuestOSEnv()
    tasks = env.list_tasks()
    assert len(tasks) == 3
    task_ids = {t.task_id.value for t in tasks}
    assert "thermal_mitigation" in task_ids
    assert "sensor_recovery" in task_ids
    assert "kernel_panic_recovery" in task_ids
    for t in tasks:
        assert 0.0 <= t.success_threshold <= 1.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'='*60}")
    print("  MetaQuestOSEnv — Local Smoke Tests")
    print(f"{'='*60}\n")

    check("Imports work", test_imports)
    check("Database init and read/write", test_db_init)
    check("Task 1: reset initializes correctly", test_task1_reset_and_step)
    check("Task 1: rewards in [0.0, 1.0]", test_task1_reward_range)
    check("Task 1: solvable (reward=1.0 achievable)", test_task1_solvable)
    check("Task 2: reset initializes correctly", test_task2_reset_and_step)
    check("Task 2: solvable (reward=1.0 achievable)", test_task2_solvable)
    check("Task 3: reset initializes correctly", test_task3_reset_and_step)
    check("Task 3: solvable (reward=1.0 achievable)", test_task3_solvable)
    check("Task 3: partial rewards exist", test_task3_partial_rewards)
    check("Invalid action handled gracefully", test_invalid_action)
    check("Max steps terminates episode", test_max_steps_terminates)
    check("State endpoint works", test_state_endpoint)
    check("Pydantic serialization to JSON", test_pydantic_serialization)
    check("psutil real metrics injected", test_psutil_integration)
    check("List tasks returns all 3", test_list_tasks)

    print(f"\n{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print(f"  Results: {passed}/{total} tests passed")

    failed = [(n, ok) for n, ok in results if not ok]
    if failed:
        print(f"\n  Failed tests:")
        for name, _ in failed:
            print(f"    ✗ {name}")
        print(f"{'='*60}\n")
        sys.exit(1)
    else:
        print(f"  ✓ All tests passed — ready to submit!")
        print(f"{'='*60}\n")
        sys.exit(0)


if __name__ == "__main__":
    main()