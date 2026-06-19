# Transition Reference: `nbr_gen.py` → Isaac Lab `ManagerBasedEnv`

> [!NOTE]
> This document records the architectural decisions and mappings from the original procedural `nbr_gen.py` script to the current Isaac Lab implementation. The migration is **complete** — this serves as a reference for understanding the final design.

---

## 1. Architecture Summary

The original `nbr_gen.py` created rooms procedurally by directly manipulating USD prims. The Isaac Lab version uses a declarative configuration + event-driven reset pattern.

| Aspect | Original `nbr_gen.py` | Current Isaac Lab |
|---|---|---|
| **Scene creation** | Imperative: `define_xform()`, `add_payload_prim()`, `set_xform()` | Declarative: `@configclass RoomSceneCfg` with `RigidObjectCfg` fields |
| **Environment cloning** | Manual grid loop with `ROOM_SPACING_X/Y` | Automatic: `num_envs=20`, `env_spacing=16.0` |
| **Randomization timing** | Once at build (baked into USD) | Every reset via event term |
| **Collision detection** | Circle-packing + predefined slots | OBB + SAT + continuous zone sampling |
| **Math backend** | Python `random` + scalar math | Hybrid: Python `random` for sampling, `torch` for state writes |
| **State representation** | USD xformOps | Root state tensors: `(N, 13)` = pos + quat + velocities |
| **Physics bodies** | Static USD meshes | **Proxy cuboids** (invisible) + visual sync |
| **Physics device** | N/A (no simulation) | CPU PhysX, Fabric disabled |

---

## 2. What Got Deleted

All of these elements from `nbr_gen.py` were replaced by Isaac Lab's built-in mechanisms:

- `define_xform()`, `add_payload_prim()`, `set_xform()`, `add_internal_reference_prim()`, `add_internal_reference_xform()` — replaced by `RigidObjectCfg` declarations
- Grid layout loop (`create_room_instance()`, `ROOM_SPACING_X/Y`, `GENERATED_START_OFFSET`) — replaced by `num_envs` + `env_spacing`
- `hide_*` functions — replaced by `z = -100` despawning pattern
- `build_ui()` / `main()` — replaced by Isaac Lab's `AppLauncher`
- `PlacementSlot` and predefined slot lists — replaced by continuous `WallZone` sampling
- Circle-packing collision (`is_free()`, spacing_radius) — replaced by OBB + SAT

---

## 3. Component Mapping (Final)

### 3.1 Scene Definition

Each randomizable object is a named field in [RoomSceneCfg](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/room_scene_cfg.py#L59-L182):

| Original `nbr_gen.py` Object | Scene Cfg Field | Asset Type | Prim Path |
|---|---|---|---|
| Room structure (`/World/Environment`) | `room_shell` | `AssetBaseCfg` + `UsdFileCfg` | `{ENV}/RoomShell` |
| Ground/lights | `ground`, `dome_light` | `AssetBaseCfg` | `/World/ground`, `/World/light` |
| SM_Desk_04a | `desk` | `RigidObjectCfg` + proxy cuboid | `{ENV}/desk_proxy` |
| SM_Chair_04a | `chair` | `RigidObjectCfg` + proxy cuboid | `{ENV}/chair_proxy` |
| ridgeback_03 | `ridgeback` | `RigidObjectCfg` + visible cuboid | `{ENV}/RidgebackProxy` |
| SM_MedicalCabinet_01a | `medical_cabinet` | `RigidObjectCfg` + proxy cuboid | `{ENV}/medical_cabinet_proxy` |
| SM_ShelfSet_01a | `shelf_set` | `RigidObjectCfg` + proxy cuboid | `{ENV}/shelf_set_proxy` |
| SM_SupplyCabinet_01c | `supply_cabinet` | `RigidObjectCfg` + proxy cuboid | `{ENV}/supply_cabinet_proxy` |
| SM_SupplyCart_02a | `supply_cart_a` | `RigidObjectCfg` + proxy cuboid | `{ENV}/supply_cart_a_proxy` |
| SM_SupplyCart_03a | `supply_cart_b` | `RigidObjectCfg` + proxy cuboid | `{ENV}/supply_cart_b_proxy` |
| SM_TrashCan | `trash_can` | `RigidObjectCfg` + proxy cuboid | `{ENV}/trash_can_proxy` |
| SM_Plant01 | `plant_a` | `RigidObjectCfg` + proxy cuboid | `{ENV}/plant_a_proxy` |
| SM_Plant02 | `plant_b` | `RigidObjectCfg` + proxy cuboid | `{ENV}/plant_b_proxy` |
| Top-down camera | `top_down_camera` | `AssetBaseCfg` + `PinholeCameraCfg` | `{ENV}/TopDownCamera` |

> [!IMPORTANT]
> Tabletop objects (SM_CoffeeToGo, SM_Lamp02, SM_BoxPortableC) are **not currently in the scene config**. They were removed during the proxy migration. Their metadata still exists in `constants.py` for future re-enablement.

### 3.2 State Writing: `set_xform()` → `write_root_state_to_sim()`

```python
# ORIGINAL (imperative USD)
set_xform(path, translate=(x, y, z), yaw_deg=90.0)

# CURRENT (Isaac Lab tensor API)
root_state = build_root_state(pos, yaw_rad, env_origins, env_ids, default_state)
asset.write_root_state_to_sim(root_state, env_ids=env_ids)
```

The [build_root_state()](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/placement_utils.py#L232-L251) helper:
1. Clones the asset's default state for the given env IDs.
2. Adds `env_origins` offsets to positions (local → world).
3. Converts yaw to `(w, x, y, z)` quaternion via [yaw_to_quat()](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/placement_utils.py#L148-L161).
4. Zeroes velocities.

### 3.3 Visual Sync (New — No Equivalent in nbr_gen.py)

Because physics proxies are invisible cuboids, the visual furniture meshes inside the room shell must be moved manually:

```python
# After writing proxy state to sim:
_sync_visual_props(env, env_ids, "desk", desk_positions, desk_yaws)
```

This calls [_set_visual_prop_pose()](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/room_events.py#L69-L93) which:
1. Looks up the visual prim path via `_VISUAL_PROP_REL_PATHS` (e.g., `"desk"` → `"RoomShell/Environment/props/room_props/SM_Desk_04a"`)
2. Sets `xformOp:translate` and `xformOp:rotateZYX` via `pxr` API
3. Handles `float3` vs `double3` attribute type differences

### 3.4 Coordinate System

The original room coordinates were kept as-is (no re-centering):

```
Room bounds: x ∈ [-13.0, -2.5],  y ∈ [-11.25, -5.0]
Floor: z = 0.0
```

All placement math operates in room-local coordinates. The `env_origins` offset is applied only at the final `build_root_state()` call.

### 3.5 Dynamic Object Count

The "despawn underground" pattern from `nbr_gen.py` was carried over:

```python
# Hide an object by moving it far below ground
pos[:, 2] = -100.0  # DESPAWN_Z
asset.write_root_state_to_sim(state, env_ids=env_ids)
```

All environments have the same set of prims. Objects that shouldn't appear in a particular env are teleported to `z = -100`.

---

## 4. Resolved Design Questions

These were open questions during the migration, now resolved:

| Question | Resolution |
|---|---|
| **Room shell loading strategy?** | Load the full `new_base_room.usda` as a single `AssetBaseCfg`. All imported rigid body APIs were **stripped** from the USD. Visual props remain as static meshes, synced to proxy positions at reset. |
| **Kinematic vs. dynamic objects?** | All objects use `rigid_body_enabled=True` with `disable_gravity=True` and high damping (effectively kinematic but GPU-tensor-compatible). No objects are `kinematic_enabled=True`. |
| **USD asset hosting?** | The room USD is loaded from a local path. Individual prop assets are no longer loaded separately — they're embedded in the room shell USD. |
| **RL or data generation?** | Currently configured as a `ManagerBasedEnv` for future RL training. Still has dummy action/observation managers. |
| **CPU vs GPU PhysX?** | **CPU PhysX** forced due to GPU tensor view crashes. Fabric disabled for correct USD-backed rendering. |

---

## 5. Resulting File Structure (Current)

```
room_randomizer_lab/
├── __init__.py               # Exports RoomEnvCfg, RoomSceneCfg
├── constants.py              # OBB sizes, wall zones, orbit offsets, asset paths
├── placement_utils.py        # SAT collision, quaternion math, build_root_state
├── room_scene_cfg.py         # @configclass RoomSceneCfg — 14 asset fields (proxy cuboids)
├── room_events.py            # 3-phase placement + visual prop sync
├── room_env_cfg.py           # @configclass RoomEnvCfg — master config (CPU PhysX)
├── run_randomizer.py         # Launcher script (AppLauncher + sim loop)
├── test_placement.py         # Standalone matplotlib OBB test (no Isaac Sim)
├── placement_test.png        # Output from test_placement.py
├── execution_flow.md         # Step-by-step runtime trace
├── implementation_plan.md    # OBB placement algorithm documentation
├── project_architecture.md   # Architecture overview + RL roadmap
└── transition_guide.md       # This file — migration reference
```
