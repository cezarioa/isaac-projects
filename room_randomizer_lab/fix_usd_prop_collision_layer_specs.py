"""Remove imported prop mesh collision opinions from new_base_room.usda.

The room randomizer moves these props with Isaac Lab RigidObject views and
uses OBB placement checks for layout validity. The imported convex-hull mesh
colliders are not needed for that workflow and can trigger PhysX GPU tensor
view failures. Keep the rigid-body APIs on the expected prop roots, but strip
collision schemas from their payload children in the root USDA layer.
"""

from pxr import Sdf

try:
    from .paths import ROOM_SHELL_USD
except ImportError:
    from paths import ROOM_SHELL_USD

USD_PATH = str(ROOM_SHELL_USD)
PROPS_ROOT = "/World/Environment/props"

EXPECTED_RIGID_ROOTS = [
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

REMOVE_COLLISION_SCHEMAS = {
    "PhysicsCollisionAPI",
    "PhysxCollisionAPI",
    "PhysicsMeshCollisionAPI",
    "PhysxConvexHullCollisionAPI",
    "PhysxTriangleMeshCollisionAPI",
    "PhysxCookedDataAPI:triangleMesh",
}

REMOVE_COLLISION_ATTRS = {
    "physics:approximation",
    "physics:collisionEnabled",
}


def iter_specs(spec):
    yield spec
    for child in spec.nameChildren:
        yield from iter_specs(child)


def token_list_without(items, remove):
    return [item for item in items if item not in remove]


layer = Sdf.Layer.FindOrOpen(USD_PATH)
if layer is None:
    raise SystemExit(f"ERROR: Could not open layer {USD_PATH}")

removed_schemas = []
removed_attrs = []

for root in layer.rootPrims:
    for prim in iter_specs(root):
        path = str(prim.path)
        if not path.startswith(PROPS_ROOT):
            continue

        if prim.HasInfo("apiSchemas"):
            op = prim.GetInfo("apiSchemas")
            explicit = list(op.explicitItems)
            prepended = list(op.prependedItems)
            appended = list(op.appendedItems)
            deleted = list(op.deletedItems)

            new_explicit = token_list_without(explicit, REMOVE_COLLISION_SCHEMAS)
            new_prepended = token_list_without(prepended, REMOVE_COLLISION_SCHEMAS)
            new_appended = token_list_without(appended, REMOVE_COLLISION_SCHEMAS)
            new_deleted = token_list_without(deleted, REMOVE_COLLISION_SCHEMAS)

            for item in explicit + prepended + appended:
                if item in REMOVE_COLLISION_SCHEMAS:
                    removed_schemas.append((path, item))

            if (
                new_explicit != explicit
                or new_prepended != prepended
                or new_appended != appended
                or new_deleted != deleted
            ):
                new_op = Sdf.TokenListOp()
                new_op.explicitItems = new_explicit
                new_op.prependedItems = new_prepended
                new_op.appendedItems = new_appended
                new_op.deletedItems = new_deleted
                prim.SetInfo("apiSchemas", new_op)

        for prop in list(prim.properties):
            if prop.name in REMOVE_COLLISION_ATTRS:
                prop_name = prop.name
                prim.RemoveProperty(prop)
                removed_attrs.append((path, prop_name))

print("Layer-spec prop collision cleanup:")
for path, schema in removed_schemas:
    print(f"  REMOVED schema {schema}: {path}")
for path, attr in removed_attrs:
    print(f"  REMOVED attr {attr}: {path}")
if not removed_schemas and not removed_attrs:
    print("  No prop collision opinions found in root layer.")

print("\nVerifying expected prop rigid-body roots are preserved:")
all_ok = True
for expected_path in EXPECTED_RIGID_ROOTS:
    spec = layer.GetPrimAtPath(expected_path)
    if spec is None:
        print(f"  MISSING: {expected_path}")
        all_ok = False
        continue
    op = spec.GetInfo("apiSchemas") if spec.HasInfo("apiSchemas") else None
    schemas = set()
    if op is not None:
        schemas.update(op.explicitItems)
        schemas.update(op.prependedItems)
        schemas.update(op.appendedItems)
    if "PhysicsRigidBodyAPI" not in schemas:
        print(f"  ERROR missing PhysicsRigidBodyAPI: {expected_path}")
        all_ok = False
    else:
        print(f"  OK: {expected_path}")

if not all_ok:
    raise SystemExit("ERROR: Rigid-body root verification failed; not saving.")

layer.Save()
print(f"\nSaved layer: {USD_PATH}")
