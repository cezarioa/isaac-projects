# constants.py
# Room placement constants carried over from nbr_gen.py.
# These define the physical layout of the hospital room environment
# and the spatial rules for object placement.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

# ============================================================
# Room geometry
# ============================================================

# Local floor bounds of the template room (/World/Environment).
# All slot positions live in this coordinate frame.
ROOM_X_MIN = -13.0
ROOM_X_MAX = -1.0
ROOM_Y_MIN = -11.0
ROOM_Y_MAX = -5.0
FLOOR_Z = 0.0

# Random table-group sampling bounds (interior, away from walls).
TABLE_SAMPLE_X_MIN = -9.50
TABLE_SAMPLE_X_MAX = -5.00
TABLE_SAMPLE_Y_MIN = -8.50
TABLE_SAMPLE_Y_MAX = -5.50

# Fallback center if rejection sampling fails.
TABLE_FALLBACK_X = -7.00
TABLE_FALLBACK_Y = -7.25

# ============================================================
# Object sizes (radius = 2-D bounding circle used for spacing)
# ============================================================

DESK_RADIUS = 2.60
DESK_TOP_Z = 0.78
DESK_OBJECT_Z = DESK_TOP_Z + 0.04
CHAIR_RADIUS = 0.75
ROBOT_RADIUS = 1.00

# ============================================================
# Source transforms from the authored template room.
# Used only to compute the "source offset" when building
# internal-reference xforms (kept for reference, but Isaac Lab
# uses `init_state` positions instead).
# ============================================================

DESK_SOURCE_CENTER = (-4.40, -7.10, 0.0)
CHAIR_SOURCE_CENTER = (-5.80, -7.30, 0.0)
ROBOT_SOURCE_CENTER = (-6.1, -5.95, 0.0)

# ============================================================
# Orbital offsets (local-frame displacement from desk center)
# ============================================================

CHAIR_ORBIT_OFFSET = (0.0, -1.65)   # (local_x, local_y)
ROBOT_ORBIT_OFFSET = (-1.95, 1.10)

# ============================================================
# Desk-surface object placement bounds (local to desk prim)
# ============================================================

DESK_LOCAL_X_MIN = -0.38
DESK_LOCAL_X_MAX = 0.38
DESK_LOCAL_Y_MIN = -0.22
DESK_LOCAL_Y_MAX = 0.22
DESK_OBJECT_MARGIN = 0.05

# ============================================================
# Table-group rejection sampling
# ============================================================

TABLE_GROUP_FIT_MARGIN = 0.45      # margin used in table_group_fits
TABLE_GROUP_MAX_SLOT_TRIES = 24    # yaw samples per slot
TABLE_GROUP_MAX_RANDOM_TRIES = 200 # random (x,y,yaw) attempts

# ============================================================
# Wall geometry for corner clearance
# ============================================================

BACK_WALL_LINE_Y = -10.0       # Y of the back wall's room-facing edge
RIGHT_WALL_LINE_X = -2.5       # X of the right wall's room-facing edge
TALL_PROP_CORNER_CLEARANCE = 1.5

# Despawn height: objects moved here are effectively invisible.
DESPAWN_Z = -100.0

# ============================================================
# Placement slot: a pre-approved (x, y) anchor for an object
# ============================================================


@dataclass(frozen=True)
class PlacementSlot:
    x: float
    y: float
    z: float
    yaw: float      # default yaw in degrees
    radius: float   # spacing radius for collision checks
    wall: str        # "back", "right", or "room"


# ============================================================
# Wall prop placement slots
# ============================================================

WALL_PLACEMENT_SLOTS: List[PlacementSlot] = [
    # Back wall lane
    PlacementSlot(x=-3.00, y=-10.00, z=0.0, yaw=90.0,  radius=0.65, wall="back"),
    PlacementSlot(x=-4.32, y=-10.76, z=0.0, yaw=90.0,  radius=0.65, wall="back"),
    PlacementSlot(x=-5.78, y=-10.91, z=0.0, yaw=0.0,   radius=0.65, wall="back"),
    PlacementSlot(x=-6.50, y=-10.95, z=0.0, yaw=0.0,   radius=0.65, wall="back"),
    PlacementSlot(x=-7.20, y=-11.00, z=0.0, yaw=0.0,   radius=0.65, wall="back"),
    PlacementSlot(x=-8.10, y=-10.91, z=0.0, yaw=0.0,   radius=0.65, wall="back"),
    # Right wall lane
    PlacementSlot(x=-2.80, y=-5.50,  z=0.0, yaw=90.0,  radius=0.65, wall="right"),
    PlacementSlot(x=-2.89, y=-6.35,  z=0.0, yaw=90.0,  radius=0.65, wall="right"),
    PlacementSlot(x=-3.29, y=-7.20,  z=0.0, yaw=90.0,  radius=0.65, wall="right"),
    PlacementSlot(x=-2.95, y=-8.60,  z=0.0, yaw=90.0,  radius=0.65, wall="right"),
]

# ============================================================
# Room-interior prop placement slots (used for table group)
# ============================================================

ROOM_PROP_PLACEMENT_SLOTS: List[PlacementSlot] = [
    PlacementSlot(x=-7.00, y=-7.50, z=0.0, yaw=0.0,    radius=DESK_RADIUS, wall="room"),
    PlacementSlot(x=-8.00, y=-7.00, z=0.0, yaw=90.0,   radius=DESK_RADIUS, wall="room"),
    PlacementSlot(x=-6.00, y=-7.00, z=0.0, yaw=180.0,  radius=DESK_RADIUS, wall="room"),
    PlacementSlot(x=-7.50, y=-8.00, z=0.0, yaw=-90.0,  radius=DESK_RADIUS, wall="room"),
    PlacementSlot(x=-8.50, y=-7.50, z=0.0, yaw=45.0,   radius=DESK_RADIUS, wall="room"),
    PlacementSlot(x=-6.50, y=-8.00, z=0.0, yaw=-45.0,  radius=DESK_RADIUS, wall="room"),
    PlacementSlot(x=-5.50, y=-7.50, z=0.0, yaw=0.0,    radius=DESK_RADIUS, wall="room"),
    PlacementSlot(x=-9.00, y=-7.50, z=0.0, yaw=135.0,  radius=DESK_RADIUS, wall="room"),
    PlacementSlot(x=-7.00, y=-6.50, z=0.0, yaw=90.0,   radius=DESK_RADIUS, wall="room"),
    PlacementSlot(x=-8.00, y=-8.00, z=0.0, yaw=0.0,    radius=DESK_RADIUS, wall="room"),
    PlacementSlot(x=-6.00, y=-6.50, z=0.0, yaw=-90.0,  radius=DESK_RADIUS, wall="room"),
    PlacementSlot(x=-7.50, y=-7.00, z=0.0, yaw=180.0,  radius=DESK_RADIUS, wall="room"),
]

# ============================================================
# Wall yaw lookup tables
# ============================================================

# Base yaw per wall direction.
WALL_YAWS: Dict[str, float] = {
    "back": 0.0,
    "right": 90.0,
}

# Per-(prop, wall) yaw offsets so objects face flush against the wall.
WALL_PROP_YAW_BY_WALL: Dict[Tuple[str, str], float] = {
    ("SM_MedicalCabinet_01a", "back"):  180.0,
    ("SM_MedicalCabinet_01a", "right"): 180.0,
    ("SM_ShelfSet_01a",       "back"):  180.0,
    ("SM_ShelfSet_01a",       "right"): 180.0,
    ("SM_SupplyCabinet_01c",  "back"):  90.0,
    ("SM_SupplyCabinet_01c",  "right"): 90.0,
    ("SM_TrashCan",           "back"):  0.0,
    ("SM_TrashCan",           "right"): 0.0,
    ("SM_Plant01",            "back"):  0.0,
    ("SM_Plant01",            "right"): 0.0,
    ("SM_Plant02",            "back"):  0.0,
    ("SM_Plant02",            "right"): 0.0,
    ("SM_SupplyCart_02a",     "back"):  0.0,
    ("SM_SupplyCart_02a",     "right"): 0.0,
    ("SM_SupplyCart_03a",     "back"):  0.0,
    ("SM_SupplyCart_03a",     "right"): 0.0,
}

# ============================================================
# Wall-prop metadata: spacing radius, tall flag, wall clearance
# Keyed by the Isaac Lab scene-config field name.
# ============================================================


@dataclass(frozen=True)
class WallPropMeta:
    """Placement metadata for a wall prop (used by the event term)."""
    usd_name: str             # original USD prim name (e.g. "SM_MedicalCabinet_01a")
    spacing_radius: float     # 2-D spacing circle for collision checks
    tall: bool = False        # tall furniture needs extra corner clearance
    wall_clearance: float = 0.0  # extra push away from the wall surface (metres)


# Maps Isaac Lab scene field name → placement metadata.
WALL_PROP_META: Dict[str, WallPropMeta] = {
    "medical_cabinet": WallPropMeta("SM_MedicalCabinet_01a", spacing_radius=0.85, tall=True, wall_clearance=0.20),
    "shelf_set":       WallPropMeta("SM_ShelfSet_01a",       spacing_radius=0.85, tall=True),
    "supply_cabinet":  WallPropMeta("SM_SupplyCabinet_01c",  spacing_radius=0.85, tall=True),
    "trash_can":       WallPropMeta("SM_TrashCan",           spacing_radius=0.45),
    "plant_a":         WallPropMeta("SM_Plant01",            spacing_radius=0.55),
    "plant_b":         WallPropMeta("SM_Plant02",            spacing_radius=0.55),
    "supply_cart_a":   WallPropMeta("SM_SupplyCart_02a",     spacing_radius=0.75),
    "supply_cart_b":   WallPropMeta("SM_SupplyCart_03a",     spacing_radius=0.75),
}

# ============================================================
# Table-object metadata
# ============================================================


@dataclass(frozen=True)
class TablePropMeta:
    """Placement metadata for a tabletop object."""
    radius: float    # 2-D collision radius on the desk surface


TABLE_PROP_META: Dict[str, TablePropMeta] = {
    "coffee_cup":   TablePropMeta(radius=0.12),
    "desk_lamp":    TablePropMeta(radius=0.18),
    "box_portable": TablePropMeta(radius=0.18),
}

# ============================================================
# Asset USD paths (Omniverse S3 CDN)
# ============================================================

_HOSPITAL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Hospital/Props"
_OFFICE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Office/Props"
_WAREHOUSE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/Props"

ASSET_PATHS: Dict[str, str] = {
    # Wall props
    "SM_MedicalCabinet_01a": f"{_HOSPITAL}/SM_MedicalCabinet_01a.usd",
    "SM_ShelfSet_01a":       f"{_HOSPITAL}/SM_ShelfSet_01a.usd",
    "SM_SupplyCabinet_01c":  f"{_HOSPITAL}/SM_SupplyCabinet_01c.usd",
    "SM_TrashCan":           f"{_HOSPITAL}/SM_TrashCan.usd",
    "SM_SupplyCart_02a":     f"{_HOSPITAL}/SM_SupplyCart_02a.usd",
    "SM_SupplyCart_03a":     f"{_HOSPITAL}/SM_SupplyCart_03a.usd",
    "SM_Plant01":            f"{_OFFICE}/SM_Plant01.usd",
    "SM_Plant02":            f"{_OFFICE}/SM_Plant02.usd",
    # Tabletop props
    "SM_CoffeeToGo":         f"{_OFFICE}/SM_CoffeeToGo.usd",
    "SM_Lamp02":             f"{_OFFICE}/SM_Lamp02.usd",
    "SM_BoxPortableC":       f"{_OFFICE}/SM_BoxPortableC.usd",
    # Room props
    "SM_CratePlastic_D_01":  f"{_WAREHOUSE}/SM_CratePlastic_D_01.usd",
}

# The template room USD — adjust this to your local/Nucleus path.
TEMPLATE_ROOM_USD = "/World/Environment"
