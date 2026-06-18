# room_scene_cfg.py
# Declarative scene configuration for the hospital room environment.
# Each randomizable object is a separate named field so the event term
# can access it via env.scene["field_name"].

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from .constants import CHAIR_BBOX, DESK_BBOX, FLOOR_Z, ROBOT_BBOX, WALL_PROP_META


# ------------------------------------------------------------------
# Helper to reduce boilerplate for kinematic wall-prop configs
# ------------------------------------------------------------------

def _kinematic_usd_cfg(usd_path: str) -> sim_utils.UsdFileCfg:
    """UsdFileCfg with kinematic rigid-body properties (won't fall)."""
    return sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )


def _dynamic_usd_cfg(usd_path: str, mass: float = 0.05) -> sim_utils.UsdFileCfg:
    """UsdFileCfg with dynamic rigid-body properties (affected by physics)."""
    return sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )


def _proxy_box_cfg(half_w: float, half_d: float, height: float = 0.08) -> sim_utils.CuboidCfg:
    """Invisible GPU-safe kinematic proxy used by Isaac Lab tensor views."""
    return sim_utils.CuboidCfg(
        size=(2.0 * half_w, 2.0 * half_d, height),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=True,
            linear_damping=10.0,
            angular_damping=10.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visible=False,
    )


# ------------------------------------------------------------------
# Scene configuration
# ------------------------------------------------------------------

@configclass
class RoomSceneCfg(InteractiveSceneCfg):
    """Hospital room scene with all randomizable objects as named fields.

    The event term repositions every object on each episode reset.
    Default ``init_state`` positions are safe starting points; the randomizer
    overwrites them immediately.
    """

    # --- static environment -------------------------------------------

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    # Room shell: the full template USD with walls, floor, ceiling.
    # Original authored props inside it should have their visibility
    # disabled in the USD file itself (pre-processing step).
    room_shell = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RoomShell",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/cezar/isaac-sim/isaac-projects/new_base_room.usda",
        ),
    )

    # --- table group (kinematic) --------------------------------------

    desk = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/desk_proxy",
        spawn=_proxy_box_cfg(DESK_BBOX.half_w, DESK_BBOX.half_d),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.0, -7.5, FLOOR_Z)),
    )

    chair = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/chair_proxy",
        spawn=_proxy_box_cfg(CHAIR_BBOX.half_w, CHAIR_BBOX.half_d),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.0, -9.15, FLOOR_Z)),
    )

    ridgeback = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RidgebackProxy",
        spawn=sim_utils.CuboidCfg(
            size=(2.0 * ROBOT_BBOX.half_w, 2.0 * ROBOT_BBOX.half_d, 0.35),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=True,
                linear_damping=10.0,
                angular_damping=10.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.18, 0.28)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-6.1, -5.95, 0.18)),
    )

    # --- wall props (kinematic) ---------------------------------------

    medical_cabinet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/medical_cabinet_proxy",
        spawn=_proxy_box_cfg(WALL_PROP_META["medical_cabinet"].bbox.half_w, WALL_PROP_META["medical_cabinet"].bbox.half_d),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, -10.0, FLOOR_Z)),
    )

    shelf_set = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/shelf_set_proxy",
        spawn=_proxy_box_cfg(WALL_PROP_META["shelf_set"].bbox.half_w, WALL_PROP_META["shelf_set"].bbox.half_d),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-4.32, -10.76, FLOOR_Z)),
    )

    supply_cabinet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/supply_cabinet_proxy",
        spawn=_proxy_box_cfg(WALL_PROP_META["supply_cabinet"].bbox.half_w, WALL_PROP_META["supply_cabinet"].bbox.half_d),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.78, -10.91, FLOOR_Z)),
    )

    supply_cart_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/supply_cart_a_proxy",
        spawn=_proxy_box_cfg(WALL_PROP_META["supply_cart_a"].bbox.half_w, WALL_PROP_META["supply_cart_a"].bbox.half_d),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-6.50, -10.95, FLOOR_Z)),
    )

    supply_cart_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/supply_cart_b_proxy",
        spawn=_proxy_box_cfg(WALL_PROP_META["supply_cart_b"].bbox.half_w, WALL_PROP_META["supply_cart_b"].bbox.half_d),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.20, -11.0, FLOOR_Z)),
    )

    trash_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/trash_can_proxy",
        spawn=_proxy_box_cfg(WALL_PROP_META["trash_can"].bbox.half_w, WALL_PROP_META["trash_can"].bbox.half_d),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.10, -10.91, FLOOR_Z)),
    )

    plant_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/plant_a_proxy",
        spawn=_proxy_box_cfg(WALL_PROP_META["plant_a"].bbox.half_w, WALL_PROP_META["plant_a"].bbox.half_d),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.80, -5.50, FLOOR_Z)),
    )

    plant_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/plant_b_proxy",
        spawn=_proxy_box_cfg(WALL_PROP_META["plant_b"].bbox.half_w, WALL_PROP_META["plant_b"].bbox.half_d),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.89, -6.35, FLOOR_Z)),
    )

    # --- cameras ------------------------------------------------------

    top_down_camera = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TopDownCamera",
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-7.1, -8.6, 14.8),
            rot=(0.7071068, 0.0, 0.7071068, 0.0) # Look down
        ),
    )

