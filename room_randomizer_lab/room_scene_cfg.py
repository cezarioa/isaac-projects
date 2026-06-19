# room_scene_cfg.py
# Declarative scene configuration for the hospital room environment.
# Each randomizable object is a separate named field so the event term
# can access it via env.scene["field_name"].

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from .constants import ASSET_PATHS, CHAIR_BBOX, DESK_BBOX, DESK_OBJECT_Z, FLOOR_Z, ROBOT_BBOX, TABLE_PROP_META, WALL_PROP_META


# ------------------------------------------------------------------
# Helper to reduce boilerplate for kinematic wall-prop configs
# ------------------------------------------------------------------

def _kinematic_usd_cfg(usd_path: str) -> sim_utils.UsdFileCfg:
    """UsdFileCfg with kinematic rigid-body properties (won't fall)."""
    return sim_utils.UsdFileCfg(
        usd_path=usd_path,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=True,
            linear_damping=10.0,
            angular_damping=10.0,
        ),
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


def _desk_object_cfg(
    half_w: float, half_d: float, height: float = 0.06,
    color: tuple = (0.8, 0.2, 0.2),
) -> sim_utils.CuboidCfg:
    """Small visible proxy cuboid for tabletop objects."""
    return sim_utils.CuboidCfg(
        size=(2.0 * half_w, 2.0 * half_d, height),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=True,
            linear_damping=10.0,
            angular_damping=10.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
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
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_Desk_04a"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.0, -7.5, FLOOR_Z)),
    )

    chair = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Chair",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_Chair_04a"]),
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
        prim_path="{ENV_REGEX_NS}/MedicalCabinet",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_MedicalCabinet_01a"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, -10.0, FLOOR_Z)),
    )

    shelf_set = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ShelfSet",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_ShelfSet_01a"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-4.32, -10.76, FLOOR_Z)),
    )

    supply_cabinet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SupplyCabinet",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_SupplyCabinet_01c"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.78, -10.91, FLOOR_Z)),
    )

    supply_cart_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SupplyCartA",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_SupplyCart_02a"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-6.50, -10.95, FLOOR_Z)),
    )

    supply_cart_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SupplyCartB",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_SupplyCart_03a"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.20, -11.0, FLOOR_Z)),
    )

    trash_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TrashCan",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_TrashCan"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.10, -10.91, FLOOR_Z)),
    )

    plant_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlantA",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_Plant01"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.80, -5.50, FLOOR_Z)),
    )

    plant_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/PlantB",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_Plant02"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.89, -6.35, FLOOR_Z)),
    )

    # --- tabletop objects (visible proxies on desk surface) ------------

    coffee_cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CoffeeCup",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_CoffeeToGo"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.0, -7.5, DESK_OBJECT_Z)),
    )

    desk_lamp = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/DeskLamp",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_Lamp02"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.2, -7.3, DESK_OBJECT_Z)),
    )

    box_portable = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BoxPortable",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_BoxPortableC"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-6.8, -7.7, DESK_OBJECT_Z)),
    )

    # --- cameras ------------------------------------------------------

    top_down_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopDownCamera",
        update_period=0.0,
        height=720,
        width=1280,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(-5.5, -9.1, 16.8),
            rot=(0.7071068, 0.0, 0.7071068, 0.0),  # look down
            convention="world",
        ),
    )

