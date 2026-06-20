"""Remove authored collision APIs from static room-shell geometry.

The room shell remains visual-only. The Isaac Lab ground plane supplies floor
collision, and room placement uses OBB checks rather than wall physics contacts.
"""

from pxr import Sdf, Usd

try:
    from .paths import ROOM_SHELL_USD
except ImportError:
    from paths import ROOM_SHELL_USD

USD_PATH = str(ROOM_SHELL_USD)
STATIC_ROOT = "/World/Environment/room_static"
COLLISION_SCHEMAS = {
    "PhysicsCollisionAPI",
    "PhysxCollisionAPI",
    "PhysicsMeshCollisionAPI",
    "PhysxConvexHullCollisionAPI",
    "PhysxTriangleMeshCollisionAPI",
    "PhysxCookedDataAPI:triangleMesh",
}
COLLISION_ATTRS = (
    "physics:approximation",
    "physics:collisionEnabled",
    "physxCollision:contactOffset",
    "physxCollision:restOffset",
)


stage = Usd.Stage.Open(USD_PATH)
if not stage:
    raise SystemExit(f"ERROR: Could not open {USD_PATH}")

removed_schemas = []
removed_attrs = []
for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    if not prim_path.startswith(STATIC_ROOT):
        continue

    api_schemas = prim.GetMetadata("apiSchemas")
    if api_schemas:
        applied = api_schemas.GetAppliedItems()
        kept = [schema for schema in applied if schema not in COLLISION_SCHEMAS]
        if kept != applied:
            edited = Sdf.TokenListOp()
            edited.explicitItems = kept
            prim.SetMetadata("apiSchemas", edited)
            for schema in applied:
                if schema in COLLISION_SCHEMAS:
                    removed_schemas.append((prim_path, schema))

    for attr_name in COLLISION_ATTRS:
        attr = prim.GetAttribute(attr_name)
        if attr:
            prim.RemoveProperty(attr_name)
            removed_attrs.append((prim_path, attr_name))

print("Removed static room collision authoring:")
for prim_path, schema in removed_schemas:
    print(f"  REMOVED {schema}: {prim_path}")
for prim_path, attr_name in removed_attrs:
    print(f"  REMOVED attr {attr_name}: {prim_path}")
if not removed_schemas and not removed_attrs:
    print("  No static room collision authoring found.")

print("\nVerifying no collision APIs remain under room_static:")
ok = True
for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    if not prim_path.startswith(STATIC_ROOT):
        continue
    applied = set(prim.GetAppliedSchemas())
    remaining = sorted(applied.intersection(COLLISION_SCHEMAS))
    if remaining:
        ok = False
        print(f"  ERROR: {prim_path} APIs={remaining}")

if not ok:
    raise SystemExit("ERROR: Static collision cleanup verification failed; not saving.")

stage.GetRootLayer().Save()
print(f"\nSaved modified USD: {USD_PATH}")
