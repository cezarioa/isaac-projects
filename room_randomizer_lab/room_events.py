# room_events.py
# Event term functions for room randomization.
# Uses OBB collision + continuous wall-zone sampling.

from __future__ import annotations

import math
import random
from typing import List

import torch

from isaaclab.envs import ManagerBasedEnv

from .constants import (
    CHAIR_BBOX,
    CHAIR_ORBIT_OFFSET,
    DESK_BBOX,
    DESK_LOCAL_X_MAX,
    DESK_LOCAL_Y_MAX,
    DESK_OBJECT_MARGIN,
    DESK_OBJECT_Z,
    DESPAWN_Z,
    FLOOR_Z,
    OBB_PLACEMENT_MARGIN,
    ROBOT_BBOX,
    ROBOT_ORBIT_OFFSET,
    TABLE_FALLBACK_X,
    TABLE_FALLBACK_Y,
    TABLE_GROUP_MAX_TRIES,
    TABLE_PROP_META,
    TABLE_SAMPLE_X_MAX,
    TABLE_SAMPLE_X_MIN,
    TABLE_SAMPLE_Y_MAX,
    TABLE_SAMPLE_Y_MIN,
    WALL_PROP_META,
    WALL_PROP_YAW_BY_WALL,
    WALL_ZONES,
    WallZone,
)
from .placement_utils import (
    OBB,
    build_root_state,
    local_to_world_xy,
    make_obb,
    obb_inside_room,
    obb_overlap,
    obb_overlap_any,
    offset_from_yaw,
    offset_from_yaw_batched,
)


# ======================================================================
# Combined event term
# ======================================================================

def randomize_room_layout(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    wall_prop_names: list[str],
    table_prop_names: list[str],
    min_table_objects: int = 2,
):
    """Randomize the full room layout for the given environments.

    Uses OBB collision detection and continuous zone sampling.
    """
    M = len(env_ids)
    device = env.device

    # Per-environment list of placed OBBs.
    all_placed: List[List[OBB]] = [[] for _ in range(M)]

    # Per-environment placement results for later phases.
    desk_positions = torch.zeros(M, 3, device=device)
    desk_yaws = torch.zeros(M, device=device)

    # --- Phase 1: Wall props ------------------------------------------
    _place_wall_props(env, env_ids, wall_prop_names, all_placed)

    # --- Phase 2: Table group (desk + chair + robot) ------------------
    _place_table_group(env, env_ids, all_placed, desk_positions, desk_yaws)

    # --- Phase 3: Tabletop objects ------------------------------------
    _place_desk_objects(env, env_ids, table_prop_names, desk_positions, desk_yaws, min_table_objects)


# ======================================================================
# Phase 1: Wall prop placement — continuous zone sampling
# ======================================================================

def _sample_wall_position(zone: WallZone, meta, rng: random.Random) -> tuple[float, float, float]:
    """Sample a random (cx, cy, yaw) along a wall zone strip.

    Returns:
        (cx, cy, yaw_rad) for the prop centre.
    """
    pos_along_wall = rng.uniform(zone.sample_min, zone.sample_max)

    # Apply per-prop wall offset (push away from wall surface).
    offset = meta.wall_offset

    if zone.wall == "back":
        cx = pos_along_wall
        cy = zone.fixed_coord + offset  # push into room (+Y)
    else:  # "right"
        cx = zone.fixed_coord - offset  # push into room (-X)
        cy = pos_along_wall

    yaw = zone.base_yaw + meta.yaw_offset
    return cx, cy, yaw


def _place_wall_props(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    wall_prop_names: list[str],
    all_placed: List[List[OBB]],
):
    """Place wall props using continuous zone sampling + OBB collision."""
    M = len(env_ids)
    device = env.device
    rng = random.Random()

    # Shuffle order: tall props first for best placement priority.
    sorted_names = sorted(
        wall_prop_names,
        key=lambda n: not WALL_PROP_META[n].tall,
    )

    for name in sorted_names:
        meta = WALL_PROP_META[name]
        asset = env.scene[name]

        pos_local = torch.zeros(M, 3, device=device)
        yaw_rad = torch.zeros(M, device=device)
        placed_mask = torch.zeros(M, dtype=torch.bool, device=device)

        for env_idx in range(M):
            # Try up to 50 random positions across allowed walls.
            success = False

            # Shuffle which walls to try.
            allowed_zones = [z for z in WALL_ZONES if z.wall in meta.allowed_walls]
            rng.shuffle(allowed_zones)

            for _ in range(50):
                # Pick a random wall zone.
                zone = rng.choice(allowed_zones)
                cx, cy, yaw = _sample_wall_position(zone, meta, rng)

                candidate = make_obb(cx, cy, meta.bbox, yaw)

                # Check room bounds.
                if not obb_inside_room(candidate):
                    continue

                # Check overlap with already-placed props.
                if obb_overlap_any(candidate, all_placed[env_idx], margin=OBB_PLACEMENT_MARGIN):
                    continue

                # Valid placement!
                pos_local[env_idx] = torch.tensor([cx, cy, FLOOR_Z], device=device)
                yaw_rad[env_idx] = yaw
                placed_mask[env_idx] = True
                all_placed[env_idx].append(candidate)
                success = True
                break

            if not success:
                pos_local[env_idx, 2] = DESPAWN_Z

        # Write to simulation.
        root_state = build_root_state(
            pos_local, yaw_rad,
            env.scene.env_origins, env_ids,
            asset.data.default_root_state,
        )
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)


# ======================================================================
# Phase 2: Table group — continuous interior sampling
# ======================================================================

def _place_table_group(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    all_placed: List[List[OBB]],
    desk_positions: torch.Tensor,
    desk_yaws: torch.Tensor,
):
    """Place desk + chair + robot with continuous sampling and OBB collision."""
    M = len(env_ids)
    device = env.device
    rng = random.Random()
    env_origins = env.scene.env_origins

    for env_idx in range(M):
        success = False

        for _ in range(TABLE_GROUP_MAX_TRIES):
            # Random position and yaw in room interior.
            dx = rng.uniform(TABLE_SAMPLE_X_MIN, TABLE_SAMPLE_X_MAX)
            dy = rng.uniform(TABLE_SAMPLE_Y_MIN, TABLE_SAMPLE_Y_MAX)
            dyaw = rng.uniform(0, 2 * math.pi)

            # Compute satellite positions.
            cx, cy = offset_from_yaw(dx, dy, dyaw, CHAIR_ORBIT_OFFSET[0], CHAIR_ORBIT_OFFSET[1])
            rx, ry = offset_from_yaw(dx, dy, dyaw, ROBOT_ORBIT_OFFSET[0], ROBOT_ORBIT_OFFSET[1])

            chair_yaw = dyaw + math.pi
            robot_yaw = dyaw - math.pi / 2

            # Build OBBs.
            desk_obb = make_obb(dx, dy, DESK_BBOX, dyaw)
            chair_obb = make_obb(cx, cy, CHAIR_BBOX, chair_yaw)
            robot_obb = make_obb(rx, ry, ROBOT_BBOX, robot_yaw)

            # All three must be inside room.
            if not (obb_inside_room(desk_obb) and obb_inside_room(chair_obb) and obb_inside_room(robot_obb)):
                continue

            # None must overlap wall props.
            placed = all_placed[env_idx]
            if (obb_overlap_any(desk_obb, placed) or
                obb_overlap_any(chair_obb, placed) or
                obb_overlap_any(robot_obb, placed)):
                continue

            # The three must not overlap each other.
            if (obb_overlap(desk_obb, chair_obb, margin=OBB_PLACEMENT_MARGIN) or
                obb_overlap(desk_obb, robot_obb, margin=OBB_PLACEMENT_MARGIN) or
                obb_overlap(chair_obb, robot_obb, margin=OBB_PLACEMENT_MARGIN)):
                continue

            # Valid!
            desk_positions[env_idx] = torch.tensor([dx, dy, FLOOR_Z], device=device)
            desk_yaws[env_idx] = dyaw
            all_placed[env_idx].extend([desk_obb, chair_obb, robot_obb])
            success = True
            break

        if not success:
            # Fallback.
            dx, dy = TABLE_FALLBACK_X, TABLE_FALLBACK_Y
            dyaw = rng.uniform(0, 2 * math.pi)
            desk_positions[env_idx] = torch.tensor([dx, dy, FLOOR_Z], device=device)
            desk_yaws[env_idx] = dyaw

    # --- Write desk, chair, ridgeback to sim --------------------------
    for env_idx in range(M):
        pass  # positions already stored in desk_positions/desk_yaws

    # Build batched positions for satellites.
    chair_pos = offset_from_yaw_batched(
        desk_positions, desk_yaws,
        CHAIR_ORBIT_OFFSET[0], CHAIR_ORBIT_OFFSET[1], FLOOR_Z,
    )
    chair_yaw = desk_yaws + math.pi

    robot_pos = offset_from_yaw_batched(
        desk_positions, desk_yaws,
        ROBOT_ORBIT_OFFSET[0], ROBOT_ORBIT_OFFSET[1], FLOOR_Z,
    )
    robot_yaw = desk_yaws - math.pi / 2

    desk_asset = env.scene["desk"]
    desk_state = build_root_state(desk_positions, desk_yaws, env_origins, env_ids, desk_asset.data.default_root_state)
    desk_asset.write_root_state_to_sim(desk_state, env_ids=env_ids)

    chair_asset = env.scene["chair"]
    chair_state = build_root_state(chair_pos, chair_yaw, env_origins, env_ids, chair_asset.data.default_root_state)
    chair_asset.write_root_state_to_sim(chair_state, env_ids=env_ids)

    robot_asset = env.scene["ridgeback"]
    robot_state = build_root_state(robot_pos, robot_yaw, env_origins, env_ids, robot_asset.data.default_root_state)
    robot_asset.write_root_state_to_sim(robot_state, env_ids=env_ids)


# ======================================================================
# Phase 3: Tabletop objects — OBB collision on desk surface
# ======================================================================

def _place_desk_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    table_prop_names: list[str],
    desk_pos: torch.Tensor,
    desk_yaw_rad: torch.Tensor,
    min_table_objects: int = 2,
):
    """Place tabletop objects on the desk surface with OBB collision."""
    M = len(env_ids)
    device = env.device
    env_origins = env.scene.env_origins
    rng = random.Random()

    num_total = len(table_prop_names)

    for env_idx in range(M):
        # How many objects in this env (2 or 3).
        count = rng.randint(min_table_objects, num_total)
        desk_placed: List[OBB] = []

        for i, name in enumerate(table_prop_names):
            asset = env.scene[name]
            meta = TABLE_PROP_META[name]
            visible = i < count

            if visible:
                # Rejection-sample local (x, y) on desk surface.
                placed = False
                for _ in range(100):
                    lx = rng.uniform(-DESK_LOCAL_X_MAX, DESK_LOCAL_X_MAX)
                    ly = rng.uniform(-DESK_LOCAL_Y_MAX, DESK_LOCAL_Y_MAX)
                    obj_yaw = rng.uniform(0, 2 * math.pi)

                    candidate = make_obb(lx, ly, meta.bbox, obj_yaw)
                    if not obb_overlap_any(candidate, desk_placed, margin=DESK_OBJECT_MARGIN):
                        desk_placed.append(candidate)

                        # Transform to world.
                        wx, wy = offset_from_yaw(
                            desk_pos[env_idx, 0].item(),
                            desk_pos[env_idx, 1].item(),
                            desk_yaw_rad[env_idx].item(),
                            lx, ly,
                        )
                        world_yaw = desk_yaw_rad[env_idx].item() + obj_yaw

                        pos = torch.tensor([wx, wy, DESK_OBJECT_Z], device=device).unsqueeze(0)
                        yaw = torch.tensor([world_yaw], device=device)
                        eid = env_ids[env_idx:env_idx+1]

                        root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                        asset.write_root_state_to_sim(root_state, env_ids=eid)
                        placed = True
                        break

                if not placed:
                    # Couldn't fit — place at desk centre.
                    wx, wy = desk_pos[env_idx, 0].item(), desk_pos[env_idx, 1].item()
                    pos = torch.tensor([wx, wy, DESK_OBJECT_Z], device=device).unsqueeze(0)
                    yaw = torch.tensor([0.0], device=device)
                    eid = env_ids[env_idx:env_idx+1]
                    root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                    asset.write_root_state_to_sim(root_state, env_ids=eid)
            else:
                # Despawn.
                pos = torch.tensor([0.0, 0.0, DESPAWN_Z], device=device).unsqueeze(0)
                yaw = torch.tensor([0.0], device=device)
                eid = env_ids[env_idx:env_idx+1]
                root_state = build_root_state(pos, yaw, env_origins, eid, asset.data.default_root_state)
                asset.write_root_state_to_sim(root_state, env_ids=eid)
