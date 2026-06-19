"""Remove all prop rigid-body authoring from the visual room USD.

The room USD is kept as visual geometry. Isaac Lab should create and manage
separate simple proxy rigid objects, so the imported room shell should not
contribute any rigid bodies to PhysX.
"""

from pxr import Sdf

USD_PATH = "/home/cezar/isaac-sim/isaac-projects/new_base_room.usda"
PROPS_ROOT = "/World/Environment/props"
REMOVE_SCHEMAS = {"PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"}
REMOVE_ATTRS = {"physics:rigidBodyEnabled", "physics:kinematicEnabled"}


def iter_specs(spec):
    yield spec
    for child in spec.nameChildren:
        yield from iter_specs(child)


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

            new_explicit = [item for item in explicit if item not in REMOVE_SCHEMAS]
            new_prepended = [item for item in prepended if item not in REMOVE_SCHEMAS]
            new_appended = [item for item in appended if item not in REMOVE_SCHEMAS]
            new_deleted = [item for item in deleted if item not in REMOVE_SCHEMAS]

            for item in explicit + prepended + appended:
                if item in REMOVE_SCHEMAS:
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
            if prop.name in REMOVE_ATTRS:
                prop_name = prop.name
                prim.RemoveProperty(prop)
                removed_attrs.append((path, prop_name))

print("Remove all prop rigid bodies from room USD:")
for path, schema in removed_schemas:
    print(f"  REMOVED schema {schema}: {path}")
for path, attr in removed_attrs:
    print(f"  REMOVED attr {attr}: {path}")
if not removed_schemas and not removed_attrs:
    print("  No prop rigid-body authoring found.")

layer.Save()
print(f"\nSaved layer: {USD_PATH}")
