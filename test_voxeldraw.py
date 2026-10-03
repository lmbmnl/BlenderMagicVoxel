# Run: blender -b --factory-startup --python test_voxeldraw.py
import importlib.util
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

spec = importlib.util.spec_from_file_location(
    "voxel_draw", os.path.join(os.path.dirname(os.path.abspath(__file__)), "__init__.py"))
vd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vd)


def area(verts, faces):
    q = verts[faces]
    return float(np.abs(np.cross(q[:, 1] - q[:, 0], q[:, 3] - q[:, 0])).sum())


def outward(cells, verts, faces):
    """Every quad normal points away from the solid (probe half a voxel out is empty)."""
    q = verts[faces]
    n = np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0])
    n /= np.linalg.norm(n, axis=1)[:, None]
    probe = np.floor(q.mean(1) + n * 0.5).astype(int)
    return all(tuple(p) not in cells for p in probe.tolist())


# --- meshing
one = {(0, 0, 0): 1}
v, f, c = vd.build_geometry(one, 1.0)
assert (len(v), len(f)) == (8, 6) and outward(one, v, f)
assert v.min() == 0 and v.max() == 1  # voxel sits on the floor

block = {(x, y, z): 5 for x in range(3) for y in range(3) for z in range(2)}
v, f, c = vd.build_geometry(block, 1.0)
vg, fg, cg = vd.build_geometry(block, 1.0, greedy=True)
assert len(fg) == 6 and area(v, f) == area(vg, fg) and outward(block, vg, fg)

two_col = dict(block)
two_col[(0, 0, 1)] = 7
vg, fg, cg = vd.build_geometry(two_col, 1.0, greedy=True)
assert len(fg) > 6 and area(v, f) == area(vg, fg) and set(cg.tolist()) == {5, 7}

ell = {(0, 0, 0): 1, (1, 0, 0): 1, (0, 1, 0): 1}
vg, fg, _ = vd.build_geometry(ell, 2.0, greedy=True)
v, f, _ = vd.build_geometry(ell, 2.0)
assert area(v, f) == area(vg, fg) and outward(ell, vg / 2, fg)

# --- brush shapes
assert len(vd.shape_offsets('POINT', 1, 0)) == 1
assert len(vd.shape_offsets('POINT', 3, 0)) == 9
assert len(vd.shape_offsets('SQUARE', 4, 0)) == 12
for s in ('POINT', 'SQUARE', 'TRIANGLE', 'HEXAGON'):
    assert vd.shape_offsets(s, 1, 0) == {(0, 0)}
    for n in range(1, 12):
        pts = vd.shape_offsets(s, n, 0)
        assert vd.shape_offsets(s, n, 4) == pts
        if s in ('TRIANGLE', 'HEXAGON'):
            lo, hi = min(u for u, _ in pts), max(u for u, _ in pts)
            assert {(lo + hi - u, w) for u, w in pts} == pts, (s, n)  # left/right symmetric
assert len(vd.shape_offsets('TRIANGLE', 2, 0)) == 4   # .#. / ###
assert len(vd.shape_offsets('TRIANGLE', 3, 0)) == 8   # ..#.. / .#.#. / #####
assert len(vd.shape_offsets('HEXAGON', 2, 0)) == 6    # .##. / #..# / .##.
assert len(vd.shape_offsets('HEXAGON', 3, 0)) == 12
hexa = vd.shape_offsets('HEXAGON', 4, 0)
assert {(u, -w) for u, w in hexa} == hexa  # top/bottom symmetric too (odd height)
tri = vd.shape_offsets('TRIANGLE', 5, 0)
assert (0, 2) in tri  # apex up, centred
assert (2, 0) in vd.shape_offsets('TRIANGLE', 5, 1)  # rotated CW: apex points right
assert sorted(vd.stamp_cells((0, 0, 0), 2, 'POINT', 3, 0))[0] == (-1, -1, 0)
assert len(vd.shape_offsets('TRIANGLE', 3, 0, True)) == 9       # filled: 5 + 3 + 1
assert len(vd.shape_offsets('SQUARE', 4, 0, True)) == 16
assert len(vd.shape_offsets('CIRCLE', 3, 0, True)) == 5          # plus sign
assert len(vd.shape_offsets('CIRCLE', 6, 0)) < len(vd.shape_offsets('CIRCLE', 6, 0, True))
assert len(vd.stamp_cells((0, 0, 0), 2, 'SPHERE', 3, 0, True)) == 7
assert len(vd.stamp_cells((0, 0, 0), 2, 'SPHERE', 5, 0)) < len(vd.stamp_cells((0, 0, 0), 2, 'SPHERE', 5, 0, True))

# --- mirror / line / box / flood / face
assert vd.mirror_cells([(0, 0, 0)], [0]) == [(0, 0, 0)]  # the first voxel is on the mirror plane
assert sorted(vd.mirror_cells([(2, 0, 0)], [0])) == [(-2, 0, 0), (2, 0, 0)]
assert len(vd.mirror_cells([(1, 1, 1)], [0, 1, 2])) == 8
assert vd.line_cells((0, 0, 0), (4, 2, 0)) == [(0, 0, 0), (1, 1, 0), (2, 1, 0), (3, 2, 0), (4, 2, 0)]
assert len(vd.box_cells((0, 0, 0), (3, 2, 0), 2, True)) == 12
assert len(vd.box_cells((0, 0, 0), (3, 2, 0), 2, False)) == 10
two = {(0, 0, 0): 1, (1, 0, 0): 2, (5, 0, 0): 1}
assert vd.flood(two, (0, 0, 0)) == {(0, 0, 0), (1, 0, 0)}
assert vd.flood(two, (0, 0, 0), same_color=True) == {(0, 0, 0)}
slab = {(x, y, 0): 1 for x in range(4) for y in range(4)}
slab[(0, 0, 1)] = 1  # covers one top face
assert len(vd.face_region(slab, (3, 3, 0), (0, 0, 1))) == 15

# --- selection transforms stay on the grid and are exact
shape = {(0, 0, 0): 1, (1, 0, 0): 2, (2, 0, 0): 3, (2, 1, 0): 4}
sel = set(shape)
for axis in range(3):
    for ccw in (False, True):
        fn = vd.rotate_fn(sel, axis, ccw)
        cur = dict(shape)
        for _ in range(4):
            cur = {fn(c): k for c, k in cur.items()}
            fn = vd.rotate_fn(set(cur), axis, ccw)
        assert cur == shape, (axis, ccw)
fn = vd.rotate_fn(sel, 2, True)
assert len({fn(c) for c in sel}) == 4 and vd.bounds_of({fn(c) for c in sel}) == ((0, 0, 0), (1, 2, 0))
back = vd.rotate_fn({fn(c) for c in sel}, 2, False)
assert {back(fn(c)): k for c, k in shape.items()} == shape  # CW undoes CCW exactly
ff = vd.flip_fn(sel, 0)
assert {ff(c): k for c, k in shape.items()} == {(2, 0, 0): 1, (1, 0, 0): 2, (0, 0, 0): 3, (0, 1, 0): 4}
changes, new_sel = vd.transform(shape, {(0, 0, 0)}, lambda c: (c[0], c[1], c[2] + 1))
assert changes == {(0, 0, 0): [1, 0], (0, 0, 1): [0, 1]} and new_sel == {(0, 0, 1)}
changes, _ = vd.transform(shape, {(0, 0, 0)}, lambda c: (c[0] + 1, c[1], c[2]), keep=True)
assert changes == {(1, 0, 0): [2, 1]}  # duplicate overwrites the target, keeps the source


class VS:
    color_index = 9


vd._sel.clear()
vd._sel.update({(0, 0, 0), (1, 0, 0)})
assert vd.selection_action(VS, shape, 'PAINT') == {(0, 0, 0): [1, 9], (1, 0, 0): [2, 9]}
assert vd.selection_action(VS, shape, 'COPY') == {} and vd._clip == {(0, 0, 0): 1, (1, 0, 0): 2}
ch = vd.selection_action(VS, shape, 'MOVE', delta=(0, 0, 1))
assert ch[(0, 0, 1)] == [0, 1] and vd._sel == {(0, 0, 1), (1, 0, 1)}
assert vd.selection_action(VS, shape, 'DELETE') == {}  # moved cells are not in `shape`
vd._sel.clear()
vd._sel.update({(1, 0, 0), (2, 0, 0)})
ch = vd.selection_action(VS, shape, 'MIRROR', axis=0)  # copy across the first-voxel plane
assert ch == {(-1, 0, 0): [0, 2], (-2, 0, 0): [0, 3]} and len(vd._sel) == 4

# --- gizmo geometry
assert abs(vd.ray_axis_param((5, 3, 10), (0, 0, -1), (0, 0, 0), 0) - 5) < 1e-9
assert vd.ray_axis_param((5, 3, 10), (1, 0, 0), (0, 0, 0), 0) is None  # parallel
assert vd.seg_dist((5, 3), (0, 0), (10, 0)) == 3 and vd.seg_dist((-4, 3), (0, 0), (10, 0)) == 5

# --- palette removal shifts the indices above it
assert vd.remap_removed_color({(0, 0, 0): 3, (1, 0, 0): 7, (2, 0, 0): 5}, 5) == \
    {(0, 0, 0): 3, (1, 0, 0): 6, (2, 0, 0): 5}

# --- box select projection (identity camera: x, y in -1..1 map to the region)
picked = vd.cells_in_rect({(0, 0, 0): 1, (5, 5, 0): 1}, np.eye(4), 0.1, 100, 100, (40, 40, 60, 60))
assert picked == {(0, 0, 0)}

# --- .vox round trip
pal = np.array(vd.default_palette(), np.float32)
pal[7] = (1, 0, 0)
cells, rpal = vd.read_vox(vd.vox_bytes(two_col, pal))
lo = vd.bounds_of(two_col)[0]
assert cells == {(c[0] - lo[0], c[1] - lo[1], c[2] - lo[2]): k for c, k in two_col.items()}
assert rpal[7] == (1.0, 0.0, 0.0) and len(rpal) == 256
try:
    vd.vox_bytes({(0, 0, 0): 1, (300, 0, 0): 1}, pal)
    raise AssertionError("too big must fail")
except ValueError:
    pass

# --- picking
hit = vd.pick_cell((0.5, 0.5, 10), (0, 0, -1), one, vd.bounds_of(one), 1.0, 2, 0.0, False)
assert hit == ((0, 0, 1), 2)
hit = vd.pick_cell((3.5, 0.5, 10), (0, 0, -1), one, vd.bounds_of(one), 1.0, 2, 0.0, False)
assert hit == ((3, 0, 0), 2)  # floor: cell above the plane
assert vd.pick_cell((3.5, 0.5, 10), (0, 0, -1), one, vd.bounds_of(one), 1.0, 2, 0.0, True) is None

# --- palette
pal = vd.default_palette()
assert len(pal) == 256 and pal[1] == (1, 1, 1) and pal[2] == (1, 1, 0.8)

# --- Blender integration
vd.register()
scene = bpy.context.scene
vs = scene.voxel_settings
vd.ensure_palette(vs)
assert np.allclose(vd.palette_array(scene)[2], (1, 1, 0.8))
obj = vd.ensure_session_object(bpy.context)
vd.ensure_material(obj)
scene.cursor.location = (1, 2, 3)
vs.voxel_size = 0.5
vd.place_first_voxel(bpy.context, obj, vs)
bpy.context.view_layer.update()
assert vd.load_cells(obj) == {(0, 0, 0): vs.color_index}
centre_bottom = obj.matrix_world @ Vector((0.25, 0.25, 0))  # voxel 0 bottom centre
assert (centre_bottom - scene.cursor.location).length < 1e-6
assert vs.bl_rna.properties["voxel_size"].subtype == 'DISTANCE'  # shown in scene units
vs.voxel_size = 1.0
vd.save_cells(obj, two_col, 1.0)
assert vd.load_cells(obj) == two_col
vd.rebuild_mesh(obj, two_col, 1.0, vd.palette_array(scene))
assert len(obj.data.polygons) == 42 and not obj.data.validate()

bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
vd.rebuild_mesh(obj, block, 1.0, vd.palette_array(scene))
bm = bmesh.from_edit_mesh(obj.data)
assert not any(v.select for v in bm.verts) and not any(f.select for f in bm.faces)
vs.palette[5].color = (1, 0, 0)  # update callback rebuilds from the stored cells
bpy.ops.object.mode_set(mode='OBJECT')
me = obj.data
assert len(me.polygons) == 42
col = me.color_attributes["vd_color"].data
red = sum(1 for d in col if tuple(round(x, 2) for x in d.color_srgb[:3]) == (1, 0, 0))
assert red == 39 * 4, red  # 42 faces, the corner voxel (3 faces) is colour 7

# panel operators: selection, replace colour, .vox export / import (all undoable history)
vd._sel.clear()
vd._sel.add((0, 0, 1))
assert bpy.ops.voxeldraw.selection(action='FLIP', axis=0) == {'FINISHED'}
assert vd.load_cells(obj) == two_col  # a 1-voxel flip is a no-op
assert bpy.ops.voxeldraw.selection(action='DELETE') == {'FINISHED'}
assert (0, 0, 1) not in vd.load_cells(obj) and len(vd._hist["undo"]) == 1
vs.replace_from, vs.color_index = 5, 3
bpy.ops.voxeldraw.replace_color()
assert set(vd.load_cells(obj).values()) == {3}
path = os.path.join(bpy.app.tempdir, "vd_test.vox")
assert bpy.ops.voxeldraw.export_vox(filepath=path) == {'FINISHED'}
bpy.ops.voxeldraw.clear()
assert not vd.load_cells(obj)
assert bpy.ops.voxeldraw.import_vox(filepath=path) == {'FINISHED'}
assert len(vd.load_cells(obj)) == len(two_col) - 1
assert tuple(round(x, 2) for x in vs.palette[5].color) == (1, 0, 0)  # palette came back too

# palette editing: used colours can't be removed, removing shifts the voxels' indices
assert set(vd.load_cells(obj).values()) == {3} and vs.color_index == 3
try:
    bpy.ops.voxeldraw.palette_remove(index=3)
    raise AssertionError("removing a used colour must fail")
except RuntimeError as e:
    assert "used by 17 voxels" in str(e)
rgb3 = tuple(vs.palette[3].color)
vs.palette[3].name = "Skin"
assert bpy.ops.voxeldraw.palette_remove(index=2) == {'FINISHED'}
assert len(vs.palette) == 255 and vs.palette[2].name == "Skin" and tuple(vs.palette[2].color) == rgb3
assert set(vd.load_cells(obj).values()) == {2} and vs.color_index == 2
assert bpy.ops.voxeldraw.palette_add() == {'FINISHED'} and len(vs.palette) == 256
assert vs.color_index == 255 and vd.palette_array(scene).shape == (256, 3)

vs.greedy = True
assert bpy.ops.voxeldraw.confirm() == {'FINISHED'}
assert len(me.polygons) < 42 and "_vd_cells" not in obj and vs.target is None
assert me.materials[0].name == "VoxelDraw"
vd.unregister()
print("ALL TESTS PASSED")
sys.exit(0)
