"""Dump the prim hierarchy of new_base_room.usda and check for RigidBodyAPI."""
from pxr import Usd, UsdPhysics
USD_PATH = "/home/cezar/isaac-sim/isaac-projects/new_base_room.usda"
EXPECTED_PRIMS = [
    "Environment/props/room_props/SM_Desk_04a",
    "Environment/props/room_props/SM_Chair_04a",
    "Environment/props/wall_props/SM_MedicalCabinet_01a",
    "Environment/props/wall_props/SM_ShelfSet_01a",
    "Environment/props/wall_props/SM_SupplyCabinet_01c",
    "Environment/props/wall_props/SM_SupplyCart_02a",
    "Environment/props/wall_props/SM_SupplyCart_03a",
    "Environment/props/wall_props/SM_TrashCan",
    "Environment/props/wall_props/SM_Plant01",
    "Environment/props/wall_props/SM_Plant02",
]
stage = Usd.Stage.Open(USD_PATH)
if not stage:
    print(f"ERROR: Could not open USD file: {USD_PATH}")
    exit(1)
# 1. Print full prim hierarchy (first 200 prims to avoid flooding)
print("=" * 80)
print("FULL PRIM HIERARCHY (first 200 prims):")
print("=" * 80)
for i, prim in enumerate(stage.Traverse()):
    if i >= 200:
        print(f"... (truncated, {i}+ prims total)")
        break
    apis = prim.GetAppliedSchemas()
    api_str = f"  APIs: {apis}" if apis else ""
    print(f"  {prim.GetPath()}{api_str}")
# 2. Check for PhysicsScene prims
print("\n" + "=" * 80)
print("PHYSICS SCENE CHECK:")
print("=" * 80)
found_physics_scene = False
for prim in stage.Traverse():
    if prim.GetTypeName() == "PhysicsScene" or UsdPhysics.Scene(prim):
        print(f"  FOUND PhysicsScene: {prim.GetPath()}")
        found_physics_scene = True
if not found_physics_scene:
    print("  No PhysicsScene found in USD (good ? Isaac Lab creates its own)")
# 3. Check expected prim paths
print("\n" + "=" * 80)
print("EXPECTED PRIM PATH CHECK:")
print("=" * 80)
root_path = stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else ""
print(f"  Default prim: {root_path}")
for rel_path in EXPECTED_PRIMS:
    # Try with and without default prim prefix
    candidates = [
        f"/{rel_path}",
        f"{root_path}/{rel_path}",
    ]
    found = False
    for full_path in candidates:
        prim = stage.GetPrimAtPath(full_path)
        if prim and prim.IsValid():
            has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
            prim_type = prim.GetTypeName()
            print(
                f"  OK   {rel_path}\n"
                f"         path={full_path}  type={prim_type}\n"
                f"         RigidBodyAPI={has_rigid}  CollisionAPI={has_collision}"
            )
            found = True
            break
    if not found:
        print(f"  MISS {rel_path}  (tried: {candidates})")
# 4. Check for any external references (S3/HTTP URLs)
print("\n" + "=" * 80)
print("EXTERNAL REFERENCE CHECK:")
print("=" * 80)
ext_refs = []
for prim in stage.Traverse():
    refs = prim.GetMetadata("references")
    if refs:
        for ref_list in [refs.prependedItems, refs.appendedItems]:
            for ref in ref_list:
                asset_path = ref.assetPath
                if asset_path and ("http" in asset_path or "s3:" in asset_path):
                    ext_refs.append((str(prim.GetPath()), asset_path))
if ext_refs:
    for prim_path, ref_path in ext_refs[:20]:
        print(f"  {prim_path} -> {ref_path}")
else:
    print("  No external (HTTP/S3) references found in prim references")
