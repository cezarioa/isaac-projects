"""Remove remaining PhysX rigid-body authoring from static room geometry."""

from pxr import Sdf, Usd, UsdPhysics

try:
    from .paths import ROOM_SHELL_USD
except ImportError:
    from paths import ROOM_SHELL_USD

USD_PATH = str(ROOM_SHELL_USD)
STATIC_ROOT = "/World/Environment/room_static"
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


stage = Usd.Stage.Open(USD_PATH)
if not stage:
    raise SystemExit(f"ERROR: Could not open {USD_PATH}")

removed = []
cleared_attrs = []
for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    if not prim_path.startswith(STATIC_ROOT):
        continue

    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        removed.append((prim_path, "PhysicsRigidBodyAPI"))

    api_schemas = prim.GetMetadata("apiSchemas")
    if api_schemas:
        edited = Sdf.TokenListOp()
        edited.explicitItems = [schema for schema in api_schemas.GetAppliedItems() if schema != "PhysxRigidBodyAPI"]
        if edited.explicitItems != api_schemas.GetAppliedItems():
            prim.SetMetadata("apiSchemas", edited)
            removed.append((prim_path, "PhysxRigidBodyAPI"))

    for attr_name in ("physics:rigidBodyEnabled", "physics:kinematicEnabled"):
        attr = prim.GetAttribute(attr_name)
        if attr:
            prim.RemoveProperty(attr_name)
            cleared_attrs.append((prim_path, attr_name))

print("Removed rigid-body authoring from static geometry:")
for prim_path, schema in removed:
    print(f"  REMOVED {schema}: {prim_path}")
for prim_path, attr_name in cleared_attrs:
    print(f"  REMOVED attr {attr_name}: {prim_path}")

if not removed and not cleared_attrs:
    print("  No remaining static rigid-body authoring found.")

print("\nVerifying static room prims no longer have rigid-body APIs:")
static_ok = True
for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    if not prim_path.startswith(STATIC_ROOT):
        continue
    applied = prim.GetAppliedSchemas()
    if "PhysicsRigidBodyAPI" in applied or "PhysxRigidBodyAPI" in applied:
        print(f"  ERROR: {prim_path} APIs={applied}")
        static_ok = False

print("\nVerifying props still have RigidBodyAPI:")
props_ok = True
for path in PROP_PRIMS:
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        print(f"  MISSING: {path}")
        props_ok = False
    elif not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"  ERROR - RigidBodyAPI missing: {path}")
        props_ok = False
    else:
        print(f"  OK: {path}")

if not static_ok or not props_ok:
    raise SystemExit("ERROR: Verification failed; not saving.")

stage.GetRootLayer().Save()
print(f"\nSaved modified USD: {USD_PATH}")

print("\nFinal rigid-body APIs under room_static:")
found_static = False
stage2 = Usd.Stage.Open(USD_PATH)
for prim in stage2.Traverse():
    prim_path = str(prim.GetPath())
    if prim_path.startswith(STATIC_ROOT):
        applied = prim.GetAppliedSchemas()
        if "PhysicsRigidBodyAPI" in applied or "PhysxRigidBodyAPI" in applied:
            found_static = True
            print(f"  {prim_path}: {applied}")
if not found_static:
    print("  None")
