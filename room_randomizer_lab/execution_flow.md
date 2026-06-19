# Execution Flow: Moment by Moment

This traces **exactly** what happens, in order, when you type the command and press Enter.

---

## Moment 0: You type the command

```bash
./isaaclab.sh -p /path/to/room_randomizer_lab/run_randomizer.py --num_envs 4
```

The `isaaclab.sh` wrapper activates the correct Python environment (with Omniverse + PhysX + torch + isaaclab all pre-loaded), then calls `python run_randomizer.py --num_envs 4`.

---

## Moment 1: Isaac Sim boots up

**File:** [run_randomizer.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/run_randomizer.py) — lines 5–24

```python
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(...)
parser.add_argument("--num_envs", type=int, default=4, ...)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)       # ← THIS BOOTS ISAAC SIM
simulation_app = app_launcher.app
```

**What happens:**
- `AppLauncher(args_cli)` starts the full Omniverse Kit / Isaac Sim runtime.
- A viewer window appears on your screen (black at first).
- The physics engine (PhysX) initializes.
- This takes ~10–30 seconds depending on your hardware.

> [!IMPORTANT]
> Nothing else can be imported until this completes. That's why `import torch` and all isaaclab imports come **after** the `AppLauncher` call.

---

## Moment 2: Configuration objects are created

**File:** [run_randomizer.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/run_randomizer.py) — lines 59–66

```python
env_cfg = RoomEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs   # overrides default 20 → 4
env_cfg.actions = DummyActionsCfg()
env_cfg.observations = DummyObservationsCfg()
```

**What happens:**
- Python instantiates `RoomEnvCfg` from [room_env_cfg.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/room_env_cfg.py).
- `__post_init__` runs, setting:
  - `dt = 1/120`, `decimation = 2`
  - `sim.device = "cpu"` (forces CPU PhysX to avoid GPU tensor view crashes)
  - `sim.use_fabric = False` (USD-backed transforms for correct rendering)
- Inside it, `RoomSceneCfg` from [room_scene_cfg.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/room_scene_cfg.py) is instantiated with all 14 asset fields (ground, light, room_shell, 3 table-group proxies, 8 wall-prop proxies, 1 camera).
- `RoomEventCfg` is instantiated, registering `randomize_room_layout` from [room_events.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/room_events.py) with `mode="reset"`.
- **No USD is loaded yet.** These are just Python dataclass objects describing *what* to build.

---

## Moment 3: The environment is constructed

**File:** [run_randomizer.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/run_randomizer.py) — line 70

```python
env = ManagerBasedEnv(cfg=env_cfg)
```

This single line triggers an enormous amount of work. Here's what happens inside Isaac Lab:

### 3a. Scene construction — USD stage is built

Isaac Lab reads `RoomSceneCfg` and processes each field **in declaration order**:

| Order | Field | What Isaac Lab does |
|---|---|---|
| 1 | `ground` | Creates `/World/ground`, spawns a flat GroundPlane primitive |
| 2 | `dome_light` | Creates `/World/light`, spawns a dome light at intensity 3000 |
| 3 | `room_shell` | **Loads `new_base_room.usda`** → creates prim at `{ENV}/RoomShell`. The room (walls, floor, ceiling, visual props) appears in the viewer |
| 4 | `desk` | **Spawns an invisible proxy cuboid** at `{ENV}/desk_proxy`. Size matches `DESK_BBOX` (1.40×0.90×0.08m). Gravity disabled, high damping. |
| 5 | `chair` | Spawns proxy cuboid at `{ENV}/chair_proxy` (0.80×0.80×0.08m) |
| 6 | `ridgeback` | Spawns a **visible** proxy cuboid at `{ENV}/RidgebackProxy` (1.30×1.00×0.35m, dark blue) |
| 7–14 | 8 wall props | Each spawns an invisible proxy cuboid at `{ENV}/<name>_proxy`, sized from `WALL_PROP_META` |

> [!IMPORTANT]
> **Dual-body architecture:** The physics simulation controls the **proxy cuboids** (simple shapes Isaac Lab tensor views can handle). The **visual USD props** (detailed furniture meshes) live inside the room shell and are synced to the proxy positions via `_sync_visual_props()` on each reset.

### 3b. Environment cloning — 4 copies are made

Isaac Lab sees `num_envs=4` and `env_spacing=16.0`. It:

1. Takes everything under `/World/envs/env_0/` (the room shell + all proxy bodies).
2. Clones it 3 more times → `/World/envs/env_1/`, `env_2/`, `env_3/`.
3. Arranges them in a 2×2 grid, spaced 16 meters apart.
4. Records each environment's world-space origin in `env.scene.env_origins` → a `(4, 3)` tensor.

**In the viewer, you now see 4 identical hospital rooms arranged in a grid.**

### 3c. Physics handles are acquired

For each `RigidObjectCfg` field (desk, chair, ridgeback, 8 wall props), Isaac Lab:

1. Finds the proxy cuboid prim across all 4 environments.
2. Acquires a **PhysX rigid body handle** — this is the CPU-side physics pointer that allows `write_root_state_to_sim` to work.
3. Reads the `default_root_state` (position + orientation + velocities) from the current transforms.

> [!NOTE]
> Because `sim.device = "cpu"`, physics runs on the CPU. The proxy cuboids have `rigid_body_enabled=True` with `disable_gravity=True` and high damping, so they stay where placed and don't fall or drift.

### 3d. Event manager is initialized

Isaac Lab reads `RoomEventCfg` and registers:
- `randomize_room_layout` as a reset-mode event term.
- It stores the `params` dict: `wall_prop_names` (all 8), `table_prop_names` (empty), `min_table_objects` (0).

---

## Moment 4: First reset — randomization fires!

**File:** [run_randomizer.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/run_randomizer.py) — line 83

```python
env.reset()
```

Inside `env.reset()`, Isaac Lab calls every event term registered with `mode="reset"`. That means it calls:

**File:** [room_events.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/room_events.py) — `randomize_room_layout()`

```python
randomize_room_layout(
    env=env,
    env_ids=torch.tensor([0, 1, 2, 3]),   # all 4 envs reset
    wall_prop_names=["medical_cabinet", "shelf_set", ...],  # 8 names
    table_prop_names=[],           # disabled
    min_table_objects=0,
)
```

### 4a. Phase 1: Wall props are placed (`_place_wall_props`)

For each of the 8 wall props, across all 4 environments:

1. Sorts props by priority (tall props first — medical cabinet, shelf set, supply cabinet).
2. For each env, tries up to 100 random positions along allowed wall zones:
   - Randomly picks a wall zone (back or right, filtered by `allowed_walls`).
   - Samples a random position along the wall strip (`sample_min` to `sample_max`).
   - Computes the OBB at that position using the prop's `bbox` + wall zone's `base_yaw` + prop's `yaw_offset`.
   - Checks OBB room bounds (all 4 corners inside) ✓
   - Checks OBB overlap with already-placed props (SAT collision with margin) ✓
3. Calls `build_root_state()` to construct a `(4, 13)` tensor.
4. Calls `asset.write_root_state_to_sim(root_state, env_ids=env_ids)`.
5. Calls `_sync_visual_props()` — moves the corresponding USD mesh inside the room shell to match the proxy position.

**→ Each wall prop's physics proxy teleports to its new position, and the visual USD mesh follows.**

Props that can't find a valid position after 100 tries get moved to `z = -100` (underground, invisible).

### 4b. Phase 2: Table group is placed (`_place_table_group`)

1. Samples random `(x, y, yaw)` from the room interior zone (x ∈ [−10, −5], y ∈ [−9, −6]).
2. Computes chair position via orbit offset `(0.0, −1.65)` rotated by desk yaw.
3. Computes robot position via orbit offset `(−1.95, +1.10)` rotated by desk yaw.
4. Validates all 3 OBBs (desk, chair, ridgeback) don't overlap each other or any wall prop.
5. Up to 300 tries. Falls back to fixed position `(−7.5, −7.5)` with random yaw.
6. Calls `write_root_state_to_sim` for desk, chair, and ridgeback.
7. Calls `_sync_visual_props()` for desk and chair (ridgeback has no visual mapping — the blue cuboid IS the visual).

**→ The desk, chair, and ridgeback proxy teleport to new positions. The visual desk and chair meshes follow.**

### 4c. Phase 3: Tabletop objects (currently disabled)

With `table_prop_names = []`, this phase is skipped entirely. No coffee cup, desk lamp, or portable box is placed.

**At this point, the viewer shows 4 hospital rooms, each with a unique random layout.**

---

## Moment 5: Simulation steps forward

**File:** [run_randomizer.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/run_randomizer.py) — line 86

```python
env.step(action=torch.empty(env.num_envs, 0, device=env.device))
```

Each call to `env.step()`:

1. Applies actions (empty in our case — no robot control yet).
2. Steps CPU PhysX by `dt × decimation` = `(1/120) × 2` = 0.0167 seconds of sim time.
3. Proxy cuboids have gravity disabled and high damping — they stay in place.
4. Updates all physics state tensors.

This repeats 150 times (~1.25 seconds of sim time), then...

---

## Moment 6: Reset fires again (step 150)

```python
if step_count % reset_interval == 0:
    env.reset()   # → randomize_room_layout fires again
```

**→ All furniture proxy bodies teleport to new positions, visual props sync, the viewer shows rooms "snap" into new configurations.**

This loop continues indefinitely: 150 steps of physics → reset → new layout → 150 steps → reset → ...

---

## Moment 7: You press Ctrl+C

```python
except KeyboardInterrupt:
    print("Simulation stopped by user.")
finally:
    simulation_app.close()    # cleanly shuts down Isaac Sim
```

---

## Summary: File execution order

```mermaid
sequenceDiagram
    participant User
    participant run_randomizer.py
    participant room_env_cfg.py
    participant room_scene_cfg.py
    participant room_events.py
    participant placement_utils.py
    participant constants.py
    participant Isaac Lab
    participant CPU PhysX

    User->>run_randomizer.py: ./isaaclab.sh -p run_randomizer.py
    run_randomizer.py->>Isaac Lab: AppLauncher() — boot Isaac Sim
    run_randomizer.py->>room_env_cfg.py: RoomEnvCfg()
    room_env_cfg.py->>room_scene_cfg.py: RoomSceneCfg()
    room_scene_cfg.py->>constants.py: import DESK_BBOX, WALL_PROP_META, FLOOR_Z
    room_env_cfg.py->>room_events.py: import randomize_room_layout
    room_events.py->>constants.py: import OBB sizes, wall zones, orbits
    room_events.py->>placement_utils.py: import SAT math, build_root_state
    run_randomizer.py->>Isaac Lab: ManagerBasedEnv(cfg)
    Isaac Lab->>CPU PhysX: Load USD shell, spawn proxy cuboids, clone 4 envs
    run_randomizer.py->>Isaac Lab: env.reset()
    Isaac Lab->>room_events.py: randomize_room_layout(env, env_ids)
    room_events.py->>placement_utils.py: OBB collision, build_root_state()
    room_events.py->>CPU PhysX: write_root_state_to_sim() × 11 assets
    room_events.py->>room_events.py: _sync_visual_props() — move USD meshes
    loop Every 150 steps
        run_randomizer.py->>Isaac Lab: env.step()
        Isaac Lab->>CPU PhysX: simulate dt × decimation
        run_randomizer.py->>Isaac Lab: env.reset()
        Isaac Lab->>room_events.py: randomize_room_layout()
    end
    User->>run_randomizer.py: Ctrl+C
    run_randomizer.py->>Isaac Lab: simulation_app.close()
```
