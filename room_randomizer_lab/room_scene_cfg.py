# room_scene_cfg.py
# Declarative scene configuration for the hospital room environment.
# Each randomizable object is a separate named field so the event term
# can access it via env.scene["field_name"].

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .constants import ASSET_PATHS, DESK_OBJECT_Z, DESK_TOP_Z, FLOOR_Z


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
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(),
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
    # TODO: replace with the actual path to your room-shell USD
    # room_shell = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/RoomShell",
    #     spawn=sim_utils.UsdFileCfg(usd_path="<path_to_room_shell.usd>"),
    # )

    # --- table group (kinematic) --------------------------------------

    desk = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RoomProps/Desk",
        spawn=_kinematic_usd_cfg(ASSET_PATHS.get("SM_Desk_04a", "")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.0, -7.5, FLOOR_Z)),
    )

    chair = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RoomProps/Chair",
        spawn=_kinematic_usd_cfg(ASSET_PATHS.get("SM_Chair_04a", "")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.0, -9.15, FLOOR_Z)),
    )

    ridgeback = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RoomProps/Ridgeback",
        spawn=_kinematic_usd_cfg(ASSET_PATHS.get("ridgeback_03", "")),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.95, -6.4, FLOOR_Z)),
    )

    # --- wall props (kinematic) ---------------------------------------

    medical_cabinet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallProps/MedicalCabinet",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_MedicalCabinet_01a"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.0, -10.0, FLOOR_Z)),
    )

    shelf_set = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallProps/ShelfSet",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_ShelfSet_01a"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-4.32, -10.76, FLOOR_Z)),
    )

    supply_cabinet = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallProps/SupplyCabinet",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_SupplyCabinet_01c"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.78, -10.91, FLOOR_Z)),
    )

    supply_cart_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallProps/SupplyCartA",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_SupplyCart_02a"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-6.50, -10.95, FLOOR_Z)),
    )

    supply_cart_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallProps/SupplyCartB",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_SupplyCart_03a"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.20, -11.0, FLOOR_Z)),
    )

    trash_can = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallProps/TrashCan",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_TrashCan"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-8.10, -10.91, FLOOR_Z)),
    )

    plant_a = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallProps/PlantA",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_Plant01"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.80, -5.50, FLOOR_Z)),
    )

    plant_b = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WallProps/PlantB",
        spawn=_kinematic_usd_cfg(ASSET_PATHS["SM_Plant02"]),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.89, -6.35, FLOOR_Z)),
    )

    # --- tabletop objects (dynamic — affected by physics) -------------

    coffee_cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableProps/CoffeeCup",
        spawn=_dynamic_usd_cfg(ASSET_PATHS["SM_CoffeeToGo"], mass=0.03),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.0, -7.5, DESK_OBJECT_Z)),
    )

    desk_lamp = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableProps/DeskLamp",
        spawn=_dynamic_usd_cfg(ASSET_PATHS["SM_Lamp02"], mass=0.15),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-6.8, -7.5, DESK_OBJECT_Z)),
    )

    box_portable = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TableProps/BoxPortable",
        spawn=_dynamic_usd_cfg(ASSET_PATHS["SM_BoxPortableC"], mass=0.10),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-7.2, -7.5, DESK_OBJECT_Z)),
    )
