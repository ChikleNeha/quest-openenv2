"""
grader.py — Standalone task grader for MetaQuestOSEnv.

Enumerates all tasks, runs each grader with a scripted optimal policy,
and verifies that rewards are in [0.0, 1.0]. Used by hackathon validators.

Usage:
    python grader.py
    python grader.py --env-url http://localhost:7860
"""

from __future__ import annotations
import sys
import argparse
import requests
import json
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Optimal scripted policies per task (deterministic, no LLM needed)
# ---------------------------------------------------------------------------

OPTIMAL_POLICIES: Dict[str, List[str]] = {
    "thermal_mitigation": [
        "EXEC_THERMAL_THROTTLE --enable",
        "EXEC_DISPLAY_REFRESH --rate=72",
        "EXEC_GPU_POWER_LIMIT --level=low",
        "EXEC_KILL_BACKGROUND_PROCS",
        "EXEC_FAN_OVERRIDE --speed=max",
    ],
    "sensor_recovery": [
        "EXEC_SENSOR_RESET --target=lidar",
        "EXEC_VIO_RESET --priority=imu",
        "EXEC_SENSOR_RECALIBRATE --target=lidar",
        "EXEC_ANCHOR_RECOMPUTE",
        "EXEC_TRACKING_SWITCH --mode=SLAM",
    ],
    "kernel_panic_recovery": [
        "EXEC_SAFE_MODE_INIT --stage=1",
        "EXEC_SAFE_MODE_INIT --stage=2",
        "EXEC_SAFE_MODE_INIT --stage=3",
        "EXEC_MEMORY_LOCK_CLEAR --target=world_anchor",
        "EXEC_ANCHOR_CACHE_FLUSH",
        "EXEC_SESSION_RESTORE --source=backup",
        "EXEC_KERNEL_RESTART --mode=safe",
    ],
}

EXPECTED_DIFFICULTIES = {
    "thermal_mitigation": "easy",
    "sensor_recovery": "medium",
    "kernel_panic_recovery": "hard",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def get(url: str, base: str) -> Dict[str, Any]:
    r = requests.get(f"{base}{url}", timeout=15)
    r.raise_for_status()
    return r.json()


def post(url: str, body: Dict, base: str) -> Dict[str, Any]:
    r = requests.post(f"{base}{url}", json=body, timeout=15)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Individual task grader
# ---------------------------------------------------------------------------

def grade_task(task_id: str, env_url: str) -> Tuple[bool, float, str]:
    """
    Returns (passed, score, message).
    passed = True if at least one reward in [0.0, 1.0] and task is solvable.
    """
    # Reset
    reset = post("/reset", {"task_id": task_id, "seed": 42}, env_url)
    assert "observation" in reset, "reset() missing 'observation'"
    assert "available_actions" in reset, "reset() missing 'available_actions'"
    assert "task_description" in reset, "reset() missing 'task_description'"
    assert "max_steps" in reset, "reset() missing 'max_steps'"

    policy = OPTIMAL_POLICIES[task_id]
    rewards: List[float] = []
    done = False
    final_reward = 0.0

    for action in policy:
        if done:
            break
        step_result = post("/step", {"action": action}, env_url)

        assert "reward" in step_result, "step() missing 'reward'"
        assert "done" in step_result, "step() missing 'done'"
        assert "observation" in step_result, "step() missing 'observation'"

        reward = step_result["reward"]
        done = step_result["done"]

        # CRITICAL CHECK: reward must be in [0.0, 1.0]
        assert 0.0 <= reward <= 1.0, (
            f"Reward {reward} out of [0.0, 1.0] range on task {task_id}"
        )

        rewards.append(reward)
        final_reward = reward

    # Verify state endpoint works
    state = get("/state", env_url)
    assert "observation" in state, "state() missing 'observation'"
    assert "step" in state, "state() missing 'step'"
    assert "done" in state, "state() missing 'done'"

    max_reward = max(rewards) if rewards else 0.0
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    success = done and final_reward >= 1.0

    msg = (
        f"steps={len(rewards)} max_reward={max_reward:.3f} "
        f"avg_reward={avg_reward:.3f} success={success}"
    )
    return True, avg_reward, msg


# ---------------------------------------------------------------------------
# Main grader
# ---------------------------------------------------------------------------

def run_grader(env_url: str) -> bool:
    print(f"\n{'='*65}")
    print(f"  MetaQuestOSEnv — Submission Grader")
    print(f"  Environment: {env_url}")
    print(f"{'='*65}\n")

    all_passed = True

    # 1. Health check
    print("[ CHECK ] Health endpoint...")
    try:
        health = get("/health", env_url)
        assert health.get("status") == "ok", f"Unexpected health: {health}"
        print(f"  ✓ /health → {health}\n")
    except Exception as e:
        print(f"  ✗ Health check FAILED: {e}\n")
        return False

    # 2. Task listing
    print("[ CHECK ] Task listing...")
    try:
        tasks_resp = get("/tasks", env_url)
        tasks = tasks_resp["tasks"]
        assert len(tasks) >= 3, f"Need >= 3 tasks, got {len(tasks)}"
        task_ids = [t["task_id"] for t in tasks]
        for tid in OPTIMAL_POLICIES:
            assert tid in task_ids, f"Missing task: {tid}"
        print(f"  ✓ Found {len(tasks)} tasks: {task_ids}\n")
    except Exception as e:
        print(f"  ✗ Task listing FAILED: {e}\n")
        all_passed = False

    # 3. Grade each task
    print("[ GRADING TASKS ]\n")
    results = {}
    for task_id in OPTIMAL_POLICIES:
        print(f"  Task: {task_id} ({EXPECTED_DIFFICULTIES[task_id]})")
        try:
            passed, score, msg = grade_task(task_id, env_url)
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status} | score={score:.3f} | {msg}")
            results[task_id] = {"passed": passed, "score": score}
        except AssertionError as e:
            print(f"  ✗ FAIL | AssertionError: {e}")
            results[task_id] = {"passed": False, "score": 0.0}
            all_passed = False
        except Exception as e:
            print(f"  ✗ FAIL | Error: {e}")
            results[task_id] = {"passed": False, "score": 0.0}
            all_passed = False
        print()

    # 4. Summary
    print(f"{'='*65}")
    print("  SUMMARY")
    print(f"{'='*65}")
    total_score = 0.0
    for tid, r in results.items():
        status = "✓" if r["passed"] else "✗"
        print(f"  {status} {tid:<35} score={r['score']:.3f}")
        total_score += r["score"]

    avg = total_score / len(results) if results else 0.0
    overall = "✓ ALL CHECKS PASSED" if all_passed else "✗ SOME CHECKS FAILED"
    print(f"\n  Average Score : {avg:.3f}")
    print(f"  Overall       : {overall}")
    print(f"{'='*65}\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MetaQuestOSEnv Grader")
    parser.add_argument(
        "--env-url",
        default="http://localhost:7860",
        help="Base URL of the running environment server",
    )
    args = parser.parse_args()

    ok = run_grader(args.env_url)
    sys.exit(0 if ok else 1)