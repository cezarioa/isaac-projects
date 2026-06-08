# placement_utils.py
# Torch-batched geometry utilities for room randomization.
# Every function operates on (N, ...) tensors so all environments
# are processed in parallel.

from __future__ import annotations

import math

import torch

from .constants import (
    ROOM_X_MIN,
    ROOM_X_MAX,
    ROOM_Y_MIN,
    ROOM_Y_MAX,
)


# ------------------------------------------------------------------
# Quaternion helpers
# ------------------------------------------------------------------

def yaw_to_quat(yaw_rad: torch.Tensor) -> torch.Tensor:
    """Convert Z-axis yaw angles to (w, x, y, z) quaternions.

    Args:
        yaw_rad: (N,) yaw angles in radians.

    Returns:
        (N, 4) quaternion tensor in (w, x, y, z) order.
    """
    half = yaw_rad * 0.5
    quat = torch.zeros(yaw_rad.shape[0], 4, device=yaw_rad.device, dtype=yaw_rad.dtype)
    quat[:, 0] = torch.cos(half)   # w
    # x, y stay 0 (Z-up yaw-only rotation)
    quat[:, 3] = torch.sin(half)   # z
    return quat


# ------------------------------------------------------------------
# Coordinate transforms
# ------------------------------------------------------------------

def offset_from_yaw_batched(
    origin: torch.Tensor,
    yaw_rad: torch.Tensor,
    local_x: float,
    local_y: float,
    z: float,
) -> torch.Tensor:
    """Apply a local offset rotated by yaw around an origin.

    Args:
        origin:  (N, 3) world-space origin positions.
        yaw_rad: (N,) yaw angles in radians.
        local_x: scalar X offset in the local frame.
        local_y: scalar Y offset in the local frame.
        z:       scalar Z for the result.

    Returns:
        (N, 3) world-space positions.
    """
    cos_yaw = torch.cos(yaw_rad)
    sin_yaw = torch.sin(yaw_rad)

    result = torch.zeros_like(origin)
    result[:, 0] = origin[:, 0] + local_x * cos_yaw - local_y * sin_yaw
    result[:, 1] = origin[:, 1] + local_x * sin_yaw + local_y * cos_yaw
    result[:, 2] = z
    return result


def local_to_world_xy(
    desk_pos: torch.Tensor,
    desk_yaw_rad: torch.Tensor,
    local_xy: torch.Tensor,
) -> torch.Tensor:
    """Transform local desk-surface (x, y) to world (x, y).

    Args:
        desk_pos:     (N, 3) desk world position.
        desk_yaw_rad: (N,) desk yaw in radians.
        local_xy:     (N, 2) local coordinates on the desk surface.

    Returns:
        (N, 2) world-space XY positions.
    """
    cos_yaw = torch.cos(desk_yaw_rad)
    sin_yaw = torch.sin(desk_yaw_rad)

    world_x = desk_pos[:, 0] + local_xy[:, 0] * cos_yaw - local_xy[:, 1] * sin_yaw
    world_y = desk_pos[:, 1] + local_xy[:, 0] * sin_yaw + local_xy[:, 1] * cos_yaw

    return torch.stack([world_x, world_y], dim=-1)


# ------------------------------------------------------------------
# Bounds & collision checks
# ------------------------------------------------------------------

def point_inside_room_batched(
    xy: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Check whether (x, y) ± radius fits inside the room bounds.

    Args:
        xy:     (N, 2) candidate positions.
        radius: scalar clearance radius.

    Returns:
        (N,) bool tensor — True if the circle fits inside the room.
    """
    return (
        (xy[:, 0] >= ROOM_X_MIN + radius)
        & (xy[:, 0] <= ROOM_X_MAX - radius)
        & (xy[:, 1] >= ROOM_Y_MIN + radius)
        & (xy[:, 1] <= ROOM_Y_MAX - radius)
    )


def is_free_batched(
    candidates: torch.Tensor,
    radius: float,
    occupied: torch.Tensor,
    margin: float = 0.15,
) -> torch.Tensor:
    """Vectorised circle-packing check.

    Args:
        candidates: (N, 2) proposed XY positions.
        radius:     scalar radius of the proposed object.
        occupied:   (N, K, 3) — x, y, spacing_radius of K already-placed objects.
                    Use K=0 (empty) when nothing is placed yet.
        margin:     extra spacing between circles.

    Returns:
        (N,) bool tensor — True if the candidate doesn't overlap any occupied circle.
    """
    if occupied.shape[1] == 0:
        return torch.ones(candidates.shape[0], dtype=torch.bool, device=candidates.device)

    # (N, 1, 2) - (N, K, 2) -> (N, K)
    diff = candidates.unsqueeze(1) - occupied[:, :, :2]
    dist = torch.norm(diff, dim=-1)  # (N, K)
    min_sep = radius + occupied[:, :, 2] + margin  # (N, K)

    return (dist >= min_sep).all(dim=1)  # (N,)


def table_group_fits_batched(
    desk_xy: torch.Tensor,
    desk_yaw_rad: torch.Tensor,
    chair_offset: tuple[float, float],
    robot_offset: tuple[float, float],
    desk_radius: float,
    chair_radius: float,
    robot_radius: float,
    occupied: torch.Tensor,
    margin: float = 0.45,
) -> torch.Tensor:
    """Check whether desk + chair + robot all fit without overlapping occupied circles.

    Args:
        desk_xy:       (N, 2) proposed desk positions.
        desk_yaw_rad:  (N,) proposed desk yaw angles.
        chair_offset:  (local_x, local_y) chair orbit offset.
        robot_offset:  (local_x, local_y) robot orbit offset.
        desk_radius:   desk spacing radius.
        chair_radius:  chair spacing radius.
        robot_radius:  robot spacing radius.
        occupied:      (N, K, 3) already-placed objects.
        margin:        extra spacing.

    Returns:
        (N,) bool — True if the entire table group fits.
    """
    N = desk_xy.shape[0]
    device = desk_xy.device

    desk_pos_3d = torch.zeros(N, 3, device=device)
    desk_pos_3d[:, :2] = desk_xy

    chair_pos = offset_from_yaw_batched(desk_pos_3d, desk_yaw_rad, chair_offset[0], chair_offset[1], 0.0)
    robot_pos = offset_from_yaw_batched(desk_pos_3d, desk_yaw_rad, robot_offset[0], robot_offset[1], 0.0)

    # All three must be inside the room.
    ok = (
        point_inside_room_batched(desk_xy, desk_radius)
        & point_inside_room_batched(chair_pos[:, :2], chair_radius)
        & point_inside_room_batched(robot_pos[:, :2], robot_radius)
    )

    # All three must be free of occupied circles.
    ok = ok & is_free_batched(desk_xy, desk_radius, occupied, margin=margin)
    ok = ok & is_free_batched(chair_pos[:, :2], chair_radius, occupied, margin=margin)
    ok = ok & is_free_batched(robot_pos[:, :2], robot_radius, occupied, margin=margin)

    return ok


# ------------------------------------------------------------------
# Wall-prop helpers
# ------------------------------------------------------------------

def wall_yaw_for_prop(
    usd_name: str,
    wall: str,
    wall_yaws: dict[str, float],
    yaw_by_wall: dict[tuple[str, str], float],
) -> float:
    """Compute the yaw for a wall prop on a given wall (scalar, not batched)."""
    base = wall_yaws.get(wall, 0.0)
    offset = yaw_by_wall.get((usd_name, wall), 0.0)
    return base + offset


def tall_prop_corner_ok(
    slot_x: float,
    slot_y: float,
    wall: str,
    back_wall_y: float,
    right_wall_x: float,
    clearance: float,
) -> bool:
    """Scalar check: is a tall prop far enough from the perpendicular wall corner?"""
    if wall == "right" and slot_y < back_wall_y + clearance:
        return False
    if wall == "back" and slot_x > right_wall_x - clearance:
        return False
    return True


# ------------------------------------------------------------------
# Root-state builder
# ------------------------------------------------------------------

def build_root_state(
    pos: torch.Tensor,
    yaw_rad: torch.Tensor,
    env_origins: torch.Tensor,
    env_ids: torch.Tensor,
    default_state: torch.Tensor,
) -> torch.Tensor:
    """Build a (len(env_ids), 13) root-state tensor for write_root_state_to_sim.

    Args:
        pos:           (len(env_ids), 3) local positions (room frame).
        yaw_rad:       (len(env_ids),) yaw angles in radians.
        env_origins:   (total_envs, 3) from env.scene.env_origins.
        env_ids:       (M,) indices of environments being reset.
        default_state: (total_envs, 13) from asset.data.default_root_state.

    Returns:
        (M, 13) root state: pos + quat + zero velocities.
    """
    state = default_state[env_ids].clone()

    # Position = local + env origin offset.
    state[:, 0] = pos[:, 0] + env_origins[env_ids, 0]
    state[:, 1] = pos[:, 1] + env_origins[env_ids, 1]
    state[:, 2] = pos[:, 2] + env_origins[env_ids, 2]

    # Quaternion from yaw (w, x, y, z).
    quat = yaw_to_quat(yaw_rad)
    state[:, 3:7] = quat

    # Zero velocities.
    state[:, 7:] = 0.0

    return state
