# room_events.py
# Event term functions for room randomization.
# Called on every episode reset by the Isaac Lab EventManager.
#
# All placement logic operates in parallel across env_ids using torch
# tensors.  The rejection-sampling pattern uses a `remaining` bool mask
# to track which environments still need a valid placement.

from __future__ import annotations

import math
import random
from typing import Sequence

import torch

from isaaclab.envs import ManagerBasedEnv

from .constants import (
    BACK_WALL_LINE_Y,
    CHAIR_ORBIT_OFFSET,
    CHAIR_RADIUS,
    DESK_LOCAL_X_MAX,
    DESK_LOCAL_X_MIN,
    DESK_LOCAL_Y_MAX,
    DESK_LOCAL_Y_MIN,
    DESK_OBJECT_MARGIN,
    DESK_OBJECT_Z,
    DESK_RADIUS,
    DESPAWN_Z,
    FLOOR_Z,
    RIGHT_WALL_LINE_X,
    ROBOT_ORBIT_OFFSET,
    ROBOT_RADIUS,
    ROOM_PROP_PLACEMENT_SLOTS,
    TABLE_FALLBACK_X,
    TABLE_FALLBACK_Y,
    TABLE_GROUP_FIT_MARGIN,
    TABLE_GROUP_MAX_RANDOM_TRIES,
    TABLE_GROUP_MAX_SLOT_TRIES,
    TABLE_SAMPLE_X_MAX,
    TABLE_SAMPLE_X_MIN,
    TABLE_SAMPLE_Y_MAX,
    TABLE_SAMPLE_Y_MIN,
    TALL_PROP_CORNER_CLEARANCE,
    WALL_PLACEMENT_SLOTS,
    WALL_PROP_META,
    WALL_PROP_YAW_BY_WALL,
    WALL_YAWS,
    TABLE_PROP_META,
    PlacementSlot,
    WallPropMeta,
)
from .placement_utils import (
    build_root_state,
    is_free_batched,
    local_to_world_xy,
    offset_from_yaw_batched,
    point_inside_room_batched,
    table_group_fits_batched,
    yaw_to_quat,
    wall_yaw_for_prop,
    tall_prop_corner_ok,
)


# ======================================================================
# Combined event term: wall props → table group → desk objects
# ======================================================================

def randomize_room_layout(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    wall_prop_names: list[str],
    table_prop_names: list[str],
    min_table_objects: int = 2,
):
    """Randomize the full room layout for the given environments.

    This is registered as a single event term with ``mode="reset"`` so that
    placement dependencies (wall props reserve space before the table group)
    are handled naturally.

    Args:
        env:               The Isaac Lab environment.
        env_ids:           (M,) indices of environments being reset.
        wall_prop_names:   Scene field names for wall props (order matters for priority).
        table_prop_names:  Scene field names for tabletop objects.
        min_table_objects: Minimum number of tabletop objects visible (rest despawned).
    """
    M = len(env_ids)
    device = env.device

    # Shared occupied-floor tensor.  Grows as we place objects.
    # Shape: (M, K, 3) where K increases.  Start empty.
    occupied = torch.zeros(M, 0, 3, device=device)

    # --- Phase 1: Wall props ------------------------------------------
    occupied = _place_wall_props(env, env_ids, wall_prop_names, occupied)

    # --- Phase 2: Table group (desk + chair + robot) ------------------
    desk_pos, desk_yaw_rad, occupied = _place_table_group(env, env_ids, occupied)

    # --- Phase 3: Tabletop objects ------------------------------------
    _place_desk_objects(env, env_ids, table_prop_names, desk_pos, desk_yaw_rad, min_table_objects)


# ======================================================================
# Phase 1: Wall prop placement
# ======================================================================

def _place_wall_props(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    wall_prop_names: list[str],
    occupied: torch.Tensor,
) -> torch.Tensor:
    """Place wall props into shuffled wall slots, per environment.

    Tall props are placed first to get the best slots.  Props that don't
    fit any remaining slot are despawned (moved underground).

    Returns the updated ``occupied`` tensor.
    """
    M = len(env_ids)
    device = env.device

    slots = list(WALL_PLACEMENT_SLOTS)
    num_slots = len(slots)

    # Sort names: tall props first (same priority as nbr_gen.py).
    sorted_names = sorted(
        wall_prop_names,
        key=lambda n: not WALL_PROP_META[n].tall,
    )

    for name in sorted_names:
        meta = WALL_PROP_META[name]
        asset = env.scene[name]

        # For each env, try slots in a random order.
        # Since slot count is small (~10), we do this per-env with a CPU loop
        # and batch the final write.
        pos_local = torch.zeros(M, 3, device=device)
        yaw_rad = torch.zeros(M, device=device)
        placed = torch.zeros(M, dtype=torch.bool, device=device)

        for env_idx in range(M):
            shuffled_slots = slots.copy()
            random.shuffle(shuffled_slots)

            # Filter to walls that have a yaw entry.
            allowed_walls = {w for (n, w) in WALL_PROP_YAW_BY_WALL if n == meta.usd_name}
            if allowed_walls:
                compatible = [s for s in shuffled_slots if s.wall in allowed_walls]
            else:
                compatible = shuffled_slots

            for slot in compatible:
                # Room bounds check.
                r = meta.spacing_radius
                if not (
                    -13.0 + r <= slot.x <= -1.0 - r
                    and -11.0 + r <= slot.y <= -5.0 - r
                ):
                    continue

                # Tall prop corner clearance.
                if meta.tall and not tall_prop_corner_ok(
                    slot.x, slot.y, slot.wall,
                    BACK_WALL_LINE_Y, RIGHT_WALL_LINE_X, TALL_PROP_CORNER_CLEARANCE,
                ):
                    continue

                # Circle overlap check against occupied floor (for this env).
                candidate = torch.tensor([[slot.x, slot.y]], device=device)
                occ_env = occupied[env_idx:env_idx+1]  # (1, K, 3)
                if is_free_batched(candidate, meta.spacing_radius, occ_env, margin=0.15).item():
                    # Tall-vs-tall spacing.
                    if meta.tall and occ_env.shape[1] > 0:
                        tall_mask = occ_env[0, :, 2] >= 0.80
                        if tall_mask.any():
                            tall_occ = occ_env[0, tall_mask, :2]
                            dists = torch.norm(tall_occ - candidate, dim=-1)
                            if (dists < 2.0).any():
                                continue

                    # Apply wall clearance offset.
                    sx, sy = slot.x, slot.y
                    if meta.wall_clearance > 0:
                        if slot.wall == "back":
                            sy += meta.wall_clearance
                        elif slot.wall == "right":
                            sx -= meta.wall_clearance

                    pos_local[env_idx] = torch.tensor([sx, sy, FLOOR_Z], device=device)
                    yaw_deg = wall_yaw_for_prop(meta.usd_name, slot.wall, WALL_YAWS, WALL_PROP_YAW_BY_WALL)
                    yaw_rad[env_idx] = math.radians(yaw_deg)
                    placed[env_idx] = True
                    break

        # Despawn props that couldn't be placed.
        pos_local[~placed, 2] = DESPAWN_Z

        # Write to simulation.
        root_state = build_root_state(
            pos_local, yaw_rad,
            env.scene.env_origins, env_ids,
            asset.data.default_root_state,
        )
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)

        # Add placed props to the occupied floor.
        new_occ = torch.zeros(M, 1, 3, device=device)
        new_occ[:, 0, 0] = pos_local[:, 0]
        new_occ[:, 0, 1] = pos_local[:, 1]
        new_occ[:, 0, 2] = meta.spacing_radius
        # Only count actually placed props.
        new_occ[~placed, 2] = 0.0
        occupied = torch.cat([occupied, new_occ], dim=1)

    return occupied


# ======================================================================
# Phase 2: Table group (desk + chair + ridgeback)
# ======================================================================

def _place_table_group(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    occupied: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Place the table group (desk + chair + robot) with rejection sampling.

    Returns:
        desk_pos:     (M, 3) local desk positions.
        desk_yaw_rad: (M,) desk yaw in radians.
        occupied:     updated occupied tensor.
    """
    M = len(env_ids)
    device = env.device

    desk_xy = torch.zeros(M, 2, device=device)
    desk_yaw_rad = torch.zeros(M, device=device)
    remaining = torch.ones(M, dtype=torch.bool, device=device)

    # Strategy 1: Try pre-approved room slots with random yaw.
    slots = list(ROOM_PROP_PLACEMENT_SLOTS)
    random.shuffle(slots)

    for slot in slots:
        if not remaining.any():
            break
        r_mask = remaining
        n_remaining = r_mask.sum().item()

        candidate_xy = torch.tensor([[slot.x, slot.y]], device=device).expand(n_remaining, -1)

        for _ in range(TABLE_GROUP_MAX_SLOT_TRIES):
            if not r_mask.any():
                break

            yaw = torch.rand(M, device=device) * 2 * math.pi
            ok = table_group_fits_batched(
                desk_xy=candidate_xy.new_tensor([[slot.x, slot.y]]).expand(M, -1),
                desk_yaw_rad=yaw,
                chair_offset=CHAIR_ORBIT_OFFSET,
                robot_offset=ROBOT_ORBIT_OFFSET,
                desk_radius=DESK_RADIUS,
                chair_radius=CHAIR_RADIUS,
                robot_radius=ROBOT_RADIUS,
                occupied=occupied,
                margin=TABLE_GROUP_FIT_MARGIN,
            )
            accept = remaining & ok
            desk_xy[accept, 0] = slot.x
            desk_xy[accept, 1] = slot.y
            desk_yaw_rad[accept] = yaw[accept]
            remaining = remaining & ~accept

    # Strategy 2: Random positions in the room interior.
    for _ in range(TABLE_GROUP_MAX_RANDOM_TRIES):
        if not remaining.any():
            break

        rand_xy = torch.zeros(M, 2, device=device)
        rand_xy[:, 0] = torch.rand(M, device=device) * (TABLE_SAMPLE_X_MAX - TABLE_SAMPLE_X_MIN) + TABLE_SAMPLE_X_MIN
        rand_xy[:, 1] = torch.rand(M, device=device) * (TABLE_SAMPLE_Y_MAX - TABLE_SAMPLE_Y_MIN) + TABLE_SAMPLE_Y_MIN

        yaw = torch.rand(M, device=device) * 2 * math.pi

        ok = table_group_fits_batched(
            desk_xy=rand_xy,
            desk_yaw_rad=yaw,
            chair_offset=CHAIR_ORBIT_OFFSET,
            robot_offset=ROBOT_ORBIT_OFFSET,
            desk_radius=DESK_RADIUS,
            chair_radius=CHAIR_RADIUS,
            robot_radius=ROBOT_RADIUS,
            occupied=occupied,
            margin=TABLE_GROUP_FIT_MARGIN,
        )
        accept = remaining & ok
        desk_xy[accept] = rand_xy[accept]
        desk_yaw_rad[accept] = yaw[accept]
        remaining = remaining & ~accept

    # Strategy 3: Fallback to room center with random yaw.
    if remaining.any():
        desk_xy[remaining, 0] = TABLE_FALLBACK_X
        desk_xy[remaining, 1] = TABLE_FALLBACK_Y
        desk_yaw_rad[remaining] = torch.rand(remaining.sum(), device=device) * 2 * math.pi

    # --- Compute satellite positions ----------------------------------
    desk_pos = torch.zeros(M, 3, device=device)
    desk_pos[:, :2] = desk_xy
    desk_pos[:, 2] = FLOOR_Z

    chair_pos = offset_from_yaw_batched(
        desk_pos, desk_yaw_rad,
        CHAIR_ORBIT_OFFSET[0], CHAIR_ORBIT_OFFSET[1], FLOOR_Z,
    )
    chair_yaw_rad = desk_yaw_rad + math.pi  # face the desk

    robot_pos = offset_from_yaw_batched(
        desk_pos, desk_yaw_rad,
        ROBOT_ORBIT_OFFSET[0], ROBOT_ORBIT_OFFSET[1], FLOOR_Z,
    )
    robot_yaw_rad = desk_yaw_rad - math.pi / 2

    # --- Write desk, chair, ridgeback to sim --------------------------
    env_origins = env.scene.env_origins

    desk_asset = env.scene["desk"]
    desk_state = build_root_state(desk_pos, desk_yaw_rad, env_origins, env_ids, desk_asset.data.default_root_state)
    desk_asset.write_root_state_to_sim(desk_state, env_ids=env_ids)

    chair_asset = env.scene["chair"]
    chair_state = build_root_state(chair_pos, chair_yaw_rad, env_origins, env_ids, chair_asset.data.default_root_state)
    chair_asset.write_root_state_to_sim(chair_state, env_ids=env_ids)

    robot_asset = env.scene["ridgeback"]
    robot_state = build_root_state(robot_pos, robot_yaw_rad, env_origins, env_ids, robot_asset.data.default_root_state)
    robot_asset.write_root_state_to_sim(robot_state, env_ids=env_ids)

    # --- Update occupied floor ----------------------------------------
    desk_occ = torch.zeros(M, 1, 3, device=device)
    desk_occ[:, 0, :2] = desk_xy
    desk_occ[:, 0, 2] = DESK_RADIUS

    chair_occ = torch.zeros(M, 1, 3, device=device)
    chair_occ[:, 0, :2] = chair_pos[:, :2]
    chair_occ[:, 0, 2] = CHAIR_RADIUS

    robot_occ = torch.zeros(M, 1, 3, device=device)
    robot_occ[:, 0, :2] = robot_pos[:, :2]
    robot_occ[:, 0, 2] = ROBOT_RADIUS

    occupied = torch.cat([occupied, desk_occ, chair_occ, robot_occ], dim=1)

    return desk_pos, desk_yaw_rad, occupied


# ======================================================================
# Phase 3: Tabletop object placement
# ======================================================================

def _place_desk_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    table_prop_names: list[str],
    desk_pos: torch.Tensor,
    desk_yaw_rad: torch.Tensor,
    min_table_objects: int = 2,
):
    """Place tabletop objects on the desk surface with rejection sampling.

    Randomly despawns 0–1 objects to achieve variable count (min_table_objects
    to len(table_prop_names)).
    """
    M = len(env_ids)
    device = env.device
    env_origins = env.scene.env_origins

    # How many objects per env (2 or 3).
    num_total = len(table_prop_names)
    counts = torch.randint(min_table_objects, num_total + 1, (M,), device=device)

    # Track occupied positions on the desk surface (local coords).
    occupied_desk = torch.zeros(M, 0, 3, device=device)

    for i, name in enumerate(table_prop_names):
        asset = env.scene[name]
        meta = TABLE_PROP_META[name]

        # Should this object be visible in this env?
        visible = i < counts  # (M,) bool

        # Rejection-sample local (x, y) on the desk surface.
        local_xy = torch.zeros(M, 2, device=device)
        remaining = visible.clone()

        for _ in range(100):
            if not remaining.any():
                break

            prop_x = (torch.rand(M, device=device) * 2 - 1) * (DESK_LOCAL_X_MAX)
            prop_y = (torch.rand(M, device=device) * 2 - 1) * (DESK_LOCAL_Y_MAX)
            prop_xy = torch.stack([prop_x, prop_y], dim=-1)

            ok = is_free_batched(
                prop_xy,
                meta.radius,
                occupied_desk,
                margin=DESK_OBJECT_MARGIN,
            )

            accept = remaining & ok
            local_xy[accept] = prop_xy[accept]
            remaining = remaining & ~accept

        # Track placed objects.
        new_occ = torch.zeros(M, 1, 3, device=device)
        new_occ[:, 0, :2] = local_xy
        new_occ[:, 0, 2] = meta.radius
        occupied_desk = torch.cat([occupied_desk, new_occ], dim=1)

        # Compute world position from local desk coordinates.
        world_xy = local_to_world_xy(desk_pos, desk_yaw_rad, local_xy)

        pos = torch.zeros(M, 3, device=device)
        pos[:, 0] = world_xy[:, 0]
        pos[:, 1] = world_xy[:, 1]
        pos[:, 2] = DESK_OBJECT_Z

        # Despawn invisible objects.
        pos[~visible, 2] = DESPAWN_Z

        # Random yaw for each object.
        obj_yaw_rad = torch.rand(M, device=device) * 2 * math.pi

        root_state = build_root_state(pos, obj_yaw_rad, env_origins, env_ids, asset.data.default_root_state)
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)
