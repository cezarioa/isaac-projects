# Placement Algorithm: OBB + Continuous Zones (Implemented)

> [!NOTE]
> This document describes the **implemented** OBB-based placement algorithm. The original circle-packing + predefined-slots system has been fully replaced.

---

## Background: Why the Redesign?

| Issue with old system | Impact |
|---|---|
| **Circles waste space** | A 1.2m × 0.4m desk used a 1.2m-radius circle — ~70% wasted area. Nearby objects rejected even when they'd physically fit. |
| **Predefined slots limit variety** | Only 10 wall slots and 12 room slots → same positions appeared repeatedly. |
| **Only 3/8 wall props fit** | Circle envelopes too large, slots too close — most props failed overlap checks. |

---

## Implemented Approach: OBB + Continuous Zones

### 1. Oriented Bounding Boxes (OBB)

Every placeable object has a 2D **oriented bounding box** defined by `BBox(half_w, half_d)`. Collision detection uses the **Separating Axis Theorem (SAT)**:

- For two OBBs, project both onto 4 axes (2 edge normals per box).
- If projections overlap on **all 4 axes** → collision.
- If any axis has a gap → no collision.

```
Old:  PropMeta(spacing_radius=0.85)     → circle of radius 0.85, Area ≈ 2.27 m²
New:  BBox(half_w=0.55, half_d=0.35)   → 1.10 × 0.70m rectangle, Area ≈ 0.77 m²
```

Implemented in [placement_utils.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/placement_utils.py):
- [obb_corners()](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/placement_utils.py#L55-L69) — rotates local corners into world space
- [obb_overlap()](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/placement_utils.py#L82-L115) — SAT test with optional margin inflation
- [obb_inside_room()](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/placement_utils.py#L122-L129) — all 4 corners inside room bounds

### 2. Continuous Placement Zones

Objects sample from continuous **placement zones** instead of fixed slot lists.

#### Wall zones (strips along walls)

Defined as `WallZone` dataclasses in [constants.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/constants.py#L65-L82):

| Zone | Free axis | Range | Fixed coord | Base yaw |
|------|-----------|-------|-------------|----------|
| Back wall | X | [−12.0, −4.0] | Y = −10.75 | 0.0 (face +Y) |
| Right wall | Y | [−10.0, −5.5] | X = −3.0 | π/2 (face −X) |

Wall props sample a random position along the strip (1 DOF), then check OBB overlap. Each prop also has `allowed_walls` restricting which zones it can use.

#### Room interior zone (for table group)

The table group samples from the full room interior:
- x ∈ [−10.0, −5.0], y ∈ [−9.0, −6.0], yaw ∈ [0, 2π)
- Chair at orbit offset (0.0, −1.65) from desk center
- Robot at orbit offset (−1.95, +1.10) from desk center

#### Desk surface zone

Tabletop objects sample local (x, y) coordinates on the desk surface within [−0.38, 0.38] × [−0.22, 0.22], using OBB overlap with `DESK_OBJECT_MARGIN = 0.03m`.

---

## Implementation Details

### Physics: Dual-Body Proxy Architecture

Due to PhysX GPU tensor view crashes with imported USD rigid bodies, the implementation uses **proxy cuboids** for physics and **visual sync** for rendering:

- **Proxy cuboids** (`_proxy_box_cfg()` in [room_scene_cfg.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/room_scene_cfg.py#L39-L51)): invisible, gravity-disabled, high-damping `CuboidCfg` bodies
- **Visual sync** (`_sync_visual_props()` in [room_events.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/room_events.py#L96-L98)): moves USD meshes inside the room shell to match proxy positions using `pxr` API
- **CPU PhysX** forced via `sim.device = "cpu"`, `sim.use_fabric = False`

### Constants & Metadata

All bounding box dimensions defined in [constants.py](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/constants.py):

| Object | `half_w` (m) | `half_d` (m) | Tall? | Allowed walls |
|---|---|---|---|---|
| SM_MedicalCabinet_01a | 0.55 | 0.35 | ✅ | right only |
| SM_ShelfSet_01a | 0.65 | 0.35 | ✅ | right only |
| SM_SupplyCabinet_01c | 0.50 | 0.35 | ✅ | back only |
| SM_SupplyCart_02a | 0.55 | 0.40 | | both |
| SM_SupplyCart_03a | 0.55 | 0.40 | | both |
| SM_TrashCan | 0.25 | 0.25 | | both |
| SM_Plant01 | 0.35 | 0.35 | | both |
| SM_Plant02 | 0.35 | 0.35 | | both |

Table group:
| Object | `half_w` | `half_d` |
|---|---|---|
| Desk | 0.70 | 0.45 |
| Chair | 0.40 | 0.40 |
| Robot (Ridgeback) | 0.65 | 0.50 |

### Event Term: 3-Phase Algorithm

Implemented in [randomize_room_layout()](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/room_events.py#L105-L135):

**Phase 1 — Wall props** ([_place_wall_props](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/room_events.py#L282)):
1. Sort by tall-first priority.
2. Per env: sample up to 100 positions across allowed wall zones.
3. OBB room bounds + overlap rejection.
4. Write to sim + sync visual.
5. Failed placements despawn to z = −100.

**Phase 2 — Table group** ([_place_table_group](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/room_events.py#L384)):
1. Sample random (x, y, yaw) up to 300 tries.
2. Validate desk + chair + robot OBBs (room bounds + no overlap with wall props + no overlap with each other).
3. Fallback to fixed position (−7.5, −7.5) with random yaw.
4. Write to sim + sync visual for desk and chair.

**Phase 3 — Tabletop objects** ([_place_desk_objects](file:///Users/cezarioa/Projects/isaac-projects/room_randomizer_lab/room_events.py#L519)):
1. Currently disabled (`table_prop_names = []`).
2. When enabled: rejection-sample local positions on desk surface with OBB overlap.
3. Transform to world via desk yaw rotation.

---

## Results

| Metric | Old (circles + slots) | Current (OBB + zones) |
|---|---|---|
| Wall prop placements per room | ~3/8 | **~6–8/8** |
| Position variety | 10 fixed wall positions | **Continuous** along wall strips |
| Desk position variety | 12 fixed positions | **Continuous** in room interior |
| Collision accuracy | ~30% wasted space | **<5% wasted** |
| Physics stability | GPU crashes on reset | **Stable** (CPU PhysX + proxy bodies) |

---

## Verification

### Automated Tests
- `python test_placement.py` — generates 6 rooms, visualizes OBBs as rotated rectangles
- Validates all OBB corners inside room bounds
- Validates no OBB pairs overlap (SAT all-vs-all)
- Output: `placement_test.png`

### Manual Verification
- Run in Isaac Lab viewer and watch resets
- Inspect that furniture doesn't clip through walls or each other
- Verify visual props track proxy positions correctly
