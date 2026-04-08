---
title: Quest Openenv
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# MetaQuestOSEnv 🥽

> **OpenEnv-compliant AI benchmark** simulating the system reliability layer of a Meta Quest-class Mixed Reality device.

AI agents act as **Autonomous System Reliability Engineers**, diagnosing and recovering from real operational failure modes documented in Meta's Quest developer ecosystem.

---

## What This Is

This environment trains LLM agents to manage a simulated MR operating system. The agent receives a JSON observation of the device state and must issue text commands to restore stability — exactly like a real SRE would.

Real host metrics (CPU load, RAM usage) are injected into observations via `psutil`, making the resource layer genuinely grounded in real hardware data.

---

## Architecture

```
┌─────────────────────────────────────────┐
│           FastAPI Layer                  │
│    /reset  /step  /state  /tasks         │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│         Core Environment Engine          │
│  Task Handler → Reward → State Builder   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│     SQLite Hardware Abstraction Layer    │
│  Thermals | Sensors | Kernel | Anchors   │
│  + psutil real CPU/RAM injection         │
└─────────────────────────────────────────┘
```

**Stack:** Python 3.11 · FastAPI · SQLite3 · psutil · Pydantic v2 · openenv-core
**Constraints:** Runs on 2 vCPU / 8 GB RAM (Hugging Face Spaces)

---

## The Three Tasks

### Task 1 — Thermal Mitigation `[Easy]` · Max 10 steps

**Scenario:** A high-intensity MR experience pushes GPU temperature to 85°C. Emergency shutdown is imminent.

**Goal:** Bring `gpu_temp_c` below 70°C without triggering a kernel panic.

| Condition | Reward |
|-----------|--------|
| `gpu_temp_c` < 70°C, kernel Active | **1.0** |
| `gpu_temp_c` < 73°C | 0.85 |
| `gpu_temp_c` < 77°C | 0.65 |
| `gpu_temp_c` < 81°C | 0.45 |
| `gpu_temp_c` < 85°C | 0.3 |
| No progress | 0.1 |
| Kernel panic | 0.0 |

---

### Task 2 — Sensor Fusion & SLAM Recovery `[Medium]` · Max 12 steps

**Scenario:** Low-light conditions cause LiDAR to enter Drift state. SLAM tracking collapses (`tracking_stability = 0.31`). Virtual objects are floating.

**Goal:** Restore `tracking_stability > 0.9` using sensor resets and tracking mode switching.

| Condition | Reward |
|-----------|--------|
| `stability` > 0.9, LiDAR not Failed | **1.0** |
| `stability` > 0.75 | 0.8 |
| `stability` > 0.60 | 0.65 |
| `stability` > 0.45 | 0.5 |
| `stability` > 0.30 | 0.35 |
| `stability` <= 0.30 | 0.15 |

---

### Task 3 — Kernel Panic Recovery `[Hard]` · Max 15 steps

**Scenario:** App ID 57 crashes and corrupts the World Anchor memory lock. Kernel enters PANIC. Session data is lost.

**Goal:** Multi-step Safe Mode recovery without factory reset:
1. Safe Mode stages 1 -> 2 -> 3
2. Clear World Anchor memory lock
3. Flush corrupted anchor cache
4. Restore session from backup
5. Restart kernel safely

| Condition | Reward |
|-----------|--------|
| Kernel Active + session restored | **1.0** |
| Kernel Recovering + session restored | 0.75 |
| Kernel Recovering | 0.5 |
| Kernel Active + session lost | 0.3 |
| Safe Mode active (per stage) | 0.3-0.6 |
| Kernel still in PANIC | 0.0 |

---

## Scoring Formula

```
score = final_reward * max(0.5, 1.0 - 0.04 * (steps_taken - 1))
```

Efficiency matters — solving in 3 steps scores higher than solving in 10. This makes the benchmark genuinely challenging to optimize.

**Baseline scores (DeepSeek-V3):**
```
Task 1 - thermal_mitigation:    0.920  (3 steps)
Task 2 - sensor_recovery:       0.880  (4 steps)
Task 3 - kernel_panic_recovery: 0.760  (7 steps)
Overall Average:                0.853
```

---

## Observation Space

```json
{
  "step": 3,
  "task_id": "thermal_mitigation",
  "thermals": {
    "cpu_temp_c": 72.1,
    "gpu_temp_c": 83.4,
    "throttle_active": false,
    "refresh_rate_hz": 120
  },
  "sensors": {
    "lidar_status": "Drifting",
    "tracking_stability": 0.31,
    "tracking_mode": "SLAM"
  },
  "resources": {
    "ram_used_mb": 3241.5,
    "cpu_usage_pct": 34.2
  },
  "spatial_anchors": {
    "world_lock_active": true,
    "session_data_intact": true
  },
  "kernel": {
    "status": "Active",
    "memory_lock_cleared": false
  },
  "reward": 0.45,
  "done": false
}
```

`ram_used_mb` and `cpu_usage_pct` are real host metrics from `psutil`.

---

## Action Space

Text-based commands:
```
EXEC_<SUBSYSTEM>_<OPERATION> [--flag=value]   # Mutates state
QUERY_<SUBSYSTEM>                              # Read-only diagnostic
NOOP                                           # No operation
```

Task 1 example actions:
```
EXEC_THERMAL_THROTTLE --enable
EXEC_DISPLAY_REFRESH --rate=72
EXEC_GPU_POWER_LIMIT --level=low
EXEC_KILL_BACKGROUND_PROCS
EXEC_FAN_OVERRIDE --speed=max
```

Task 3 required sequence:
```
EXEC_SAFE_MODE_INIT --stage=1
EXEC_SAFE_MODE_INIT --stage=2
EXEC_SAFE_MODE_INIT --stage=3
EXEC_MEMORY_LOCK_CLEAR --target=world_anchor
EXEC_ANCHOR_CACHE_FLUSH
EXEC_SESSION_RESTORE --source=backup
EXEC_KERNEL_RESTART --mode=safe
```

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/health` | GET | Health check |
| `/reset` | POST | Initialize task (accepts empty `{}`) |
| `/step` | POST | Execute one action |
| `/state` | GET | Read current state |
| `/tasks` | GET | List all tasks |
| `/actions/{task_id}` | GET | Valid actions for task |

Quick test:
```bash
# Health
curl https://your-space.hf.space/health

# Reset with empty body (defaults to thermal_mitigation)
curl -X POST https://your-space.hf.space/reset \
  -H "Content-Type: application/json" -d '{}'

# Reset specific task
curl -X POST https://your-space.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "kernel_panic_recovery", "seed": 42}'

# Step
curl -X POST https://your-space.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": "EXEC_THERMAL_THROTTLE --enable"}'
```

---

## Setup

### Local
```bash
git clone <your-repo>
cd quest-openenv
pip install -r requirements.txt
python main.py
# Visit http://localhost:7860/docs
```

### Docker
```bash
docker build -t quest-openenv .
docker run -p 7860:7860 quest-openenv
```

---

## Running Inference

```bash
# Terminal 1 - start server
python main.py

# Terminal 2 - run inference
set API_BASE_URL=https://router.huggingface.co/v1
set MODEL_NAME=deepseek-ai/DeepSeek-V3:fastest
set HF_TOKEN=hf_your_token_here
set ENV_BASE_URL=http://localhost:7860
python inference.py
```

The inference script includes model fallback chain, action fuzzy matching, state-aware scripted fallback, and retry logic — it will always produce valid scores regardless of which LLM is used.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | HuggingFace API key |
| `ENV_BASE_URL` | No | Server URL (default: http://localhost:7860) |

---

## File Structure

```
quest-openenv/
├── main.py           # FastAPI app
├── models.py         # Pydantic typed models
├── environment.py    # Core simulation engine
├── tasks.py          # Task definitions + reward functions
├── database.py       # SQLite hardware abstraction layer
├── inference.py      # LLM baseline inference script
├── grader.py         # Standalone task grader
├── test_env.py       # Local smoke tests
├── server/
│   └── app.py        # openenv-core entry point
├── openenv.yaml      # OpenEnv specification
├── pyproject.toml    # Project config with openenv-core dependency
├── uv.lock           # Locked dependencies
├── Dockerfile        # Container for HF Spaces
├── requirements.txt  # Python dependencies
└── README.md
```

---

## License

MIT