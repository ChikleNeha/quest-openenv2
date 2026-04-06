---
title: Quest Openenv
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---



# MetaQuestOSEnv

> An OpenEnv-compliant AI benchmark simulating the system reliability layer of a Meta Quest-class Mixed Reality device.

---

## Overview

This environment trains AI agents to act as **Autonomous System Reliability Engineers** for next-generation spatial computing hardware. Inspired by real failure modes documented in Meta's Quest developer ecosystem, the benchmark challenges agents to diagnose and recover from hardware emergencies using text-based system commands.

Real host system metrics (CPU load, RAM usage) are injected into observations via `psutil`, making the resource layer genuinely grounded in real-world data.

---

## Environment Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FastAPI Layer                      │
│         /reset  /step  /state  /tasks                │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│              Core Environment Engine                 │
│   Task Handler → Reward Function → State Builder     │
└───────────────────┬─────────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────────┐
│         SQLite Hardware Abstraction Layer            │
│   Thermals │ Sensors │ Kernel │ Anchors │ Resources  │
│   + psutil real CPU/RAM metrics injection            │
└─────────────────────────────────────────────────────┘
```

**Stack:** Python 3.11 · FastAPI · SQLite3 · psutil · Pydantic v2  
**Constraints:** Optimized for 2 vCPU / 8 GB RAM (Hugging Face Spaces)

---

## Tasks

### Task 1: Thermal Mitigation `[Easy]`
| Property | Value |
|----------|-------|
| Max Steps | 10 |
| Difficulty | Easy |
| Success Threshold | reward = 1.0 |

**Scenario:** A high-intensity MR experience pushes GPU temperature to 85°C+. Emergency shutdown is imminent.

**Goal:** Bring `gpu_temp_c` below 70°C without triggering a kernel panic.

**Partial Rewards:**
| Condition | Reward |
|-----------|--------|
| `gpu_temp_c` < 70°C, kernel Active | **1.0** |
| `gpu_temp_c` < 75°C | 0.7 |
| `gpu_temp_c` < 80°C | 0.45 |
| `gpu_temp_c` < 85°C | 0.2 |
| No progress | 0.05 |
| Kernel panic | 0.0 |

---

### Task 2: Sensor Fusion & SLAM Recovery `[Medium]`
| Property | Value |
|----------|-------|
| Max Steps | 12 |
| Difficulty | Medium |
| Success Threshold | reward = 1.0 |

**Scenario:** Low-light conditions cause LiDAR to enter Drift state. SLAM tracking collapses (`tracking_stability = 0.31`). Virtual objects float away from their anchors.

**Goal:** Restore `tracking_stability > 0.9` using sensor resets and tracking mode switching.

**Partial Rewards:**
| Condition | Reward |
|-----------|--------|
| `stability` > 0.9, LiDAR not Failed | **1.0** |
| `stability` > 0.75 | 0.7 |
| `stability` > 0.60 | 0.5 |
| `stability` > 0.45 | 0.3 |
| `stability` > 0.30 | 0.1 |
| `stability` ≤ 0.30 | 0.0 |

---

### Task 3: Kernel Panic Recovery `[Hard]`
| Property | Value |
|----------|-------|
| Max Steps | 15 |
| Difficulty | Hard |
| Success Threshold | reward = 1.0 |

**Scenario:** App ID 57 crashes, corrupting the World Anchor memory lock. Kernel enters PANIC state. Session data is lost.

**Goal:** Execute a multi-step Safe Mode recovery:
1. Initiate Safe Mode (Stages 1 → 2 → 3)
2. Clear the World Anchor memory lock
3. Flush corrupted spatial anchor cache
4. Restore session data from backup
5. Restart kernel in Safe Mode (NOT full factory reset)

**Partial Rewards:**
| Condition | Reward |
|-----------|--------|
| Kernel Active + session restored | **1.0** |
| Kernel Recovering + session restored | 0.75 |
| Kernel Recovering | 0.5 |
| Kernel Active + session lost (factory reset) | 0.3 |
| Safe Mode active (stage × 0.1) | 0.1–0.4 |
| Kernel still in PANIC | 0.0 |

---

## Observation Space

The agent receives a JSON observation with these key fields:

```json
{
  "step": 3,
  "task_id": "thermal_mitigation",
  "thermals": {
    "cpu_temp_c": 72.1,
    "gpu_temp_c": 83.4,
    "throttle_active": false,
    "refresh_rate_hz": 120,
    "fan_speed_pct": 60.0
  },
  "sensors": {
    "imu_status": "Active",
    "lidar_status": "Drifting",
    "tracking_mode": "SLAM",
    "tracking_stability": 0.31
  },
  "resources": {
    "battery_pct": 78.0,
    "ram_used_mb": 3241.5,
    "cpu_usage_pct": 34.2
  },
  "spatial_anchors": {
    "anchor_stability": 0.92,
    "world_lock_active": true,
    "session_data_intact": true
  },
  "kernel": {
    "status": "Active",
    "crashed_app_id": null
  },
  "reward": 0.2,
  "done": false
}
```

> `resources.ram_used_mb` and `resources.cpu_usage_pct` are **real host metrics** from `psutil`.

---

## Action Space

Each task exposes a discrete text-based action set. Actions follow this schema:

```
EXEC_<SUBSYSTEM>_<OPERATION> [--flag=value]   # Mutates state
QUERY_<SUBSYSTEM>                              # Read-only diagnostic
NOOP                                           # No-op (passive heat/drift creep)
```

**Example actions (Task 1):**
```
EXEC_THERMAL_THROTTLE --enable
EXEC_DISPLAY_REFRESH --rate=72
EXEC_KILL_BACKGROUND_PROCS
EXEC_GPU_POWER_LIMIT --level=low
EXEC_FAN_OVERRIDE --speed=max
QUERY_THERMAL_STATUS
```

See `/actions/{task_id}` endpoint for the full list per task.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check (returns 200) |
| `/health` | GET | Health check |
| `/reset` | POST | Initialize a task episode |
| `/step` | POST | Execute one action |
| `/state` | GET | Read current state |
| `/tasks` | GET | List all tasks |
| `/tasks/{task_id}` | GET | Task details |
| `/actions/{task_id}` | GET | Valid actions for task |

### Reset
```bash
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "thermal_mitigation", "seed": 42}'
```

### Step
```bash
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": "EXEC_THERMAL_THROTTLE --enable"}'
```

### State
```bash
curl http://localhost:7860/state
```

---

## Setup & Running

### Local

```bash
# Clone repo
git clone <your-repo-url>
cd quest-openenv

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
# → Server starts at http://localhost:7860
```

### Docker

```bash
docker build -t quest-openenv .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api-inference.huggingface.co/v1/" \
  -e MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="your_token_here" \
  quest-openenv
```

---

## Running Inference

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1/"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="your_hf_token"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

**Expected output:**
```
[START] task=thermal_mitigation env=MetaQuestOSEnv model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action=EXEC_THERMAL_THROTTLE --enable reward=0.45 done=false error=none
[STEP] step=2 action=EXEC_DISPLAY_REFRESH --rate=72 reward=0.70 done=false error=none
[STEP] step=3 action=EXEC_GPU_POWER_LIMIT --level=low reward=1.00 done=true error=none
[END] success=true steps=3 score=0.717 rewards=0.45,0.70,1.00
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint base URL |
| `MODEL_NAME` | Yes | Model identifier for inference |
| `HF_TOKEN` | Yes | Hugging Face / API authentication key |
| `ENV_BASE_URL` | No | Environment server URL (default: `http://localhost:7860`) |

---

## File Structure

```
quest-openenv/
├── main.py           # FastAPI app (step/reset/state endpoints)
├── models.py         # Pydantic typed models (OpenEnv spec)
├── environment.py    # Core simulation engine
├── tasks.py          # Task definitions, action handlers, graders
├── database.py       # SQLite hardware abstraction layer
├── inference.py      # LLM baseline inference script
├── openenv.yaml      # OpenEnv specification
├── Dockerfile        # Container for HF Spaces deployment
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

---

## Pre-Submission Checklist

- [x] HF Space deploys and returns 200 on `/`
- [x] `reset()` responds correctly
- [x] `openenv.yaml` validates
- [x] Typed Pydantic models in `models.py`
- [x] `step()` / `reset()` / `state()` endpoints implemented
- [x] Dockerfile builds and runs
- [x] `inference.py` in root directory
- [x] Uses OpenAI client with `API_BASE_URL` and `MODEL_NAME`
- [x] Emits `[START]`, `[STEP]`, `[END]` structured logs
- [x] 3 tasks with graders, rewards in `[0.0, 1.0]`
- [x] Partial reward signals on all tasks
- [x] Inference runtime < 20 minutes
- [x] Runs on 2 vCPU / 8 GB RAM

---

## License

MIT