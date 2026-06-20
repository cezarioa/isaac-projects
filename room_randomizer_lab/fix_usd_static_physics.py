"""Strip PhysicsRigidBodyAPI from static geometry in new_base_room.usda.
Static walls and floor should NOT be rigid bodies ? they only need
collision APIs (which are on child meshes). Having RigidBodyAPI on them
causes PhysX GPU simulation to crash because their ConvexMesh collision
shapes aren't GPU-compatible.
This script:
1. Backs up the original USD to new_base_room.usda.bak
2. Removes PhysicsRigidBodyAPI from all prims under room_static/
3. Verifies the props (desk, chair, etc.) still have RigidBodyAPI
4. Saves the modified USD
"""
import shutil

from pxr import Usd, UsdPhysics

try:
    from .paths import ROOM_SHELL_USD
except ImportError:
    from paths import ROOM_SHELL_USD

USD_PATH = str(ROOM_SHELL_USD)
BACKUP_PATH = USD_PATH + ".bak"
# Prims under this path are static geometry ? strip RigidBodyAPI
STATIC_ROOT = "/World/Environment/room_static"
# These props MUST keep RigidBodyAPI ? verify they're untouched
PROP_PRIMS = [
    "/World/Environment/props/room_props/SM_Desk_04a",
    "/World/Environment/props/room_props/SM_Chair_04a",
    "/World/Environment/props/wall_props/SM_MedicalCabinet_01a",
    "/World/Environment/props/wall_props/SM_ShelfSet_01a",
    "/World/Environment/props/wall_props/SM_SupplyCabinet_01c",
    "/World/Environment/props/wall_props/SM_SupplyCart_02a",
    "/World/Environment/props/wall_props/SM_SupplyCart_03a",
    "/World/Environment/props/wall_props/SM_TrashCan",
    "/World/Environment/props/wall_props/SM_Plant01",
    "/World/Environment/props/wall_props/SM_Plant02",
]
# 1. Backup
print(f"Backing up: {USD_PATH} -> {BACKUP_PATH}")
shutil.copy2(USD_PATH, BACKUP_PATH)
# 2. Open stage
stage = Usd.Stage.Open(USD_PATH)
if not stage:
    print(f"ERROR: Could not open {USD_PATH}")
    exit(1)
# 3. Strip RigidBodyAPI from static geometry
stripped = []
for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    if prim_path.startswith(STATIC_ROOT) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        stripped.append(prim_path)
        print(f"  STRIPPED RigidBodyAPI: {prim_path}")
if not stripped:
    print("WARNING: No static prims found with RigidBodyAPI. Check STATIC_ROOT path.")
else:
    print(f"\nStripped RigidBodyAPI from {len(stripped)} static prims.")
# 4. Verify props still have RigidBodyAPI
print("\nVerifying prop prims still have RigidBodyAPI:")
all_ok = True
for path in PROP_PRIMS:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        print(f"  MISSING: {path}")
        all_ok = False
    elif not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"  ERROR - RigidBodyAPI missing: {path}")
        all_ok = False
    else:
        print(f"  OK: {path}")
if not all_ok:
    print("\nERROR: Some prop prims are missing or lost RigidBodyAPI!")
    print("Restoring backup...")
    shutil.copy2(BACKUP_PATH, USD_PATH)
    exit(1)
# 5. Save
stage.GetRootLayer().Save()
print(f"\nSaved modified USD: {USD_PATH}")
print(f"Backup at: {BACKUP_PATH}")
# 6. Final verification
print("\n" + "=" * 60)
print("FINAL STATE ? all prims with RigidBodyAPI:")
print("=" * 60)
stage2 = Usd.Stage.Open(USD_PATH)
for prim in stage2.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"  {prim.GetPath()}")
