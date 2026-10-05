# Run: blender -b --factory-startup --python test_voxeldraw.py
import importlib.util
import os
import struct
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "voxel_draw", os.path.join(HERE, "__init__.py"), submodule_search_locations=[HERE])
vd = importlib.util.module_from_spec(spec)
sys.modules["voxel_draw"] = vd  # package: the add-on imports its vd_ui module
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
# with the size limit the plane is the model centre: 0..39 maps onto itself
assert sorted(vd.mirror_cells([(3, 0, 0)], [0], (39, 0, 0))) == [(3, 0, 0), (36, 0, 0)]
assert sorted(vd.mirror_cells([(19, 0, 0)], [0], (39, 0, 0))) == [(19, 0, 0), (20, 0, 0)]


class MV:
    mirror_live, mirror, use_limit, model_size = True, (True, False, True), False, (40, 40, 40)


assert vd.live_mirror_axes(MV) == [0, 2] and vd.mirror_sums(MV) == (0, 0, 0)
assert len(vd.mirrored(MV, [(2, 0, 3)])) == 4
MV.use_limit = True
assert vd.mirror_sums(MV) == (39, 39, 39)
assert all(0 <= c[i] < 40 for c in vd.mirrored(MV, [(2, 5, 3)]) for i in range(3))  # nothing clipped
(ax, quad), (ax2, _) = vd.mirror_planes(MV, None)
assert (ax, ax2) == (0, 2) and all(p[0] == 20 for p in quad)  # plane in the middle of 0..40
MV.use_limit = False
assert all(p[0] == 0.5 for p in vd.mirror_planes(MV, ((0, 0, 0), (3, 3, 3)))[0][1])  # middle of voxel 0
MV.mirror_live = False
assert vd.live_mirror_axes(MV) == [] and vd.mirrored(MV, [(2, 0, 3)]) == [(2, 0, 3)]
assert vd.mirror_planes(MV, None) == []
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
# 45 deg gizmo steps: multiples of 90 are the exact rotations, odd steps resample
fn = vd.rotate_fn(sel, 2, True)
assert vd.rotate_cells(shape, 2, 2) == {fn(c): k for c, k in shape.items()}
cw = vd.rotate_fn(sel, 2, False)
assert vd.rotate_cells(shape, 2, -2) == vd.rotate_cells(shape, 2, 6) == {cw(c): k for c, k in shape.items()}
assert vd.rotate_cells(shape, 0, 8) == shape and vd.rotate_cells({(4, 5, 6): 7}, 0, 1) == {(4, 5, 6): 7}
sq = {(x, y, 3): 1 + (x + y) % 2 for x in range(5) for y in range(5)}
dia = vd.rotate_cells(sq, 2, 1)
assert len(dia) == 25 and vd.bounds_of(dia) == ((-1, -1, 3), (5, 5, 3)) and set(dia.values()) == {1, 2}
assert {(4 - x, y, z) for x, y, z in dia} == set(dia)  # diamond, symmetric
assert len(vd.rotate_cells({(x, 0, 0): 1 for x in range(6)}, 2, 1)) == 8  # line -> staircase
assert abs(vd.wrap_angle(7.0) - (7.0 - 2 * np.pi)) < 1e-9 and vd.wrap_angle(np.pi) == np.pi
ff = vd.flip_fn(sel, 0)
assert {ff(c): k for c, k in shape.items()} == {(2, 0, 0): 1, (1, 0, 0): 2, (0, 0, 0): 3, (0, 1, 0): 4}
changes, new_sel = vd.transform(shape, {(0, 0, 0)}, lambda c: (c[0], c[1], c[2] + 1))
assert changes == {(0, 0, 0): [1, 0], (0, 0, 1): [0, 1]} and new_sel == {(0, 0, 1)}
changes, _ = vd.transform(shape, {(0, 0, 0)}, lambda c: (c[0] + 1, c[1], c[2]), keep=True)
assert changes == {(1, 0, 0): [2, 1]}  # duplicate overwrites the target, keeps the source


class VS:
    color_index = 9
    use_limit, model_size = False, (40, 40, 40)


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

# --- brush ghost: outer faces only (pushed out), unique edges
tris, segs = vd.ghost_geometry([(0, 0, 0)])
assert tris.shape == (36, 3) and segs.shape == (24, 3)
assert tris.min() < 0 and tris.max() > 1  # inflated past the voxel
tris, segs = vd.ghost_geometry([(0, 0, 0), (1, 0, 0), (1, 0, 0)])
assert tris.shape == (60, 3) and segs.shape == (40, 3)  # 10 faces; 8 + 8 unit edges + 4 seam edges
edges = {frozenset(map(tuple, e)) for e in segs.reshape(-1, 2, 3).tolist()}
assert len(edges) == 20 and frozenset({(1, 0, 0), (1, 1, 0)}) in edges  # seam kept once

# --- gizmo geometry
assert abs(vd.ray_axis_param((5, 3, 10), (0, 0, -1), (0, 0, 0), 0) - 5) < 1e-9
assert vd.ray_axis_param((5, 3, 10), (1, 0, 0), (0, 0, 0), 0) is None  # parallel
assert vd.seg_dist((5, 3), (0, 0), (10, 0)) == 3 and vd.seg_dist((-4, 3), (0, 0), (10, 0)) == 5

# --- erase stroke: the voxel behind a just-erased one is not picked (no digging)
F = type("F", (), {k: v for k, v in vd.VOXELDRAW_OT_start.__dict__.items() if callable(v)})


class BS:
    brush, shape, brush_size, rotation, fill, axis, z = 'SHAPE', 'POINT', 1, 0, False, 'XY', 0.0
    mirror_live, mirror, use_limit, model_size, tentacle = False, (True, False, False), False, (40, 40, 40), False


wall = {(x, 0, z): 1 for x in range(3) for z in range(3)} | {(x, 1, z): 1 for x in range(3) for z in range(3)}
op = F()
op.cells, op.size, op.face_key, op.drag, op.stroke = dict(wall), 1.0, None, None, True
op.snapshot, op.snap_bounds = set(wall), vd.bounds_of(wall)
del op.cells[(1, 0, 1)]  # erased by the click
op.bounds = vd.bounds_of(op.cells)
ray = ((1.5, -10.0, 1.5), (0.0, 1.0, 0.0))  # same spot, mouse jittered
assert op._brush_cells(BS, 'ERASE', *ray) == [(1, 0, 1)]  # already gone: nothing more to erase
assert op._brush_cells(BS, 'ERASE', (0.5, -10.0, 1.5), (0.0, 1.0, 0.0)) == [(0, 0, 1)]  # next block
BS.tentacle = True  # Extrude Mode keeps the old digging behaviour
assert op._brush_cells(BS, 'ERASE', *ray) == [(1, 1, 1)]
op.stroke = BS.tentacle = False
assert op._brush_cells(BS, 'ERASE', *ray) == [(1, 1, 1)]  # hover after the stroke: the one behind

# --- proportional editing: same falloffs as Blender's own transform
def native_falloff(kind, R=10.0):
    me = bpy.data.meshes.new("pe")
    me.from_pydata([(x * 0.25, 0, 0) for x in range(46)], [], [])
    ob = bpy.data.objects.new("pe", me)
    bpy.context.collection.objects.link(ob)
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(me)
    for v in bm.verts:
        v.select = v.index == 0
    bm.select_flush(True)
    bmesh.update_edit_mesh(me)
    bpy.ops.transform.translate(value=(0, 0, 1), use_proportional_edit=True,
                                proportional_edit_falloff=kind, proportional_size=R)
    bpy.ops.object.mode_set(mode='OBJECT')
    out = [(v.co.x, v.co.z) for v in me.vertices]
    bpy.data.objects.remove(ob)
    bpy.data.meshes.remove(me)
    return out


for kind in vd.FALLOFF:
    for x, z in native_falloff(kind):
        ours = vd.falloff_weight(kind, 1 - x / 10) if x <= 10 else 0.0
        assert abs(z - ours) < 1e-5, (kind, x, z, ours)
assert 0 <= vd.falloff_weight('RANDOM', 0.5, (3, 4, 5), 1) <= 0.5
assert vd.falloff_weight('RANDOM', 0.5, (3, 4, 5), 1) == vd.falloff_weight('RANDOM', 0.5, (3, 4, 5), 1)

row = {(x, 0, 0): 1 for x in range(8)}
w = vd.proportional_weights(row, {(0, 0, 0)}, 4, 'LINEAR')
assert {c: round(v, 6) for c, v in w.items()} == {(1, 0, 0): 0.75, (2, 0, 0): 0.5, (3, 0, 0): 0.25}
gap = {(0, 0, 0): 1, (2, 0, 0): 1}  # not touching
assert (2, 0, 0) in vd.proportional_weights(gap, {(0, 0, 0)}, 3, 'LINEAR')
assert vd.proportional_weights(gap, {(0, 0, 0)}, 3, 'LINEAR', connected=True) == {}
bent = {(0, 0, 0): 1, (0, 1, 0): 1, (1, 1, 0): 1, (2, 1, 0): 1, (2, 0, 0): 1}  # U shape
wc = vd.proportional_weights(bent, {(0, 0, 0)}, 5, 'LINEAR', connected=True)
assert round(wc[(2, 0, 0)], 6) == round(1 - 4 / 5, 6)  # 4 steps round the U, not 2 across
deep = {(0, 0, 0): 1, (0, 0, 5): 1}
assert vd.proportional_weights(deep, {(0, 0, 0)}, 2, 'CONSTANT') == {}
assert vd.proportional_weights(deep, {(0, 0, 0)}, 2, 'CONSTANT', view=(0, 0, 1)) == {(0, 0, 5): 1.0}  # projected

# pull a slab up 3 with radius 3: a solid hill (stretch) vs loose voxels (no stretch)
slab = {(x, y, z): 1 for x in range(-4, 5) for y in range(-4, 5) for z in range(-2, 1)}
wl = vd.proportional_weights(slab, {(0, 0, 0)}, 3, 'LINEAR')
ch, new_sel = vd.proportional_move(slab, {(0, 0, 0)}, wl, (0, 0, 3), stretch=True)
hill = dict(slab)
for c, (_old, new) in ch.items():
    if new:
        hill[c] = new
    else:
        hill.pop(c, None)
assert new_sel == {(0, 0, 3)} and (0, 0, 3) in hill
for x in range(-4, 5):
    for y in range(-4, 5):
        col = sorted(z for (a, b, z) in hill if (a, b) == (x, y))
        assert col == list(range(col[0], col[-1] + 1)), (x, y, col)  # no holes in any column
assert hill.get((1, 0, 2)) == 1 and (1, 0, 3) not in hill  # neighbour: round(0.67 * 3) = 2
assert sorted(z for (a, b, z) in hill if (a, b) == (0, 0)) == [-1, 0, 1, 2, 3]  # bottom rises too, as in Blender
assert sorted(z for (a, b, z) in hill if (a, b) == (4, 0)) == [-2, -1, 0]  # out of reach: untouched
ch2, _ = vd.proportional_move(slab, {(0, 0, 0)}, wl, (0, 0, 3), stretch=False)
assert any(not new for _old, new in ch2.values())  # plain move leaves gaps behind
# push down 2: a crater, the voxels above the new floor are carved
ch3, _ = vd.proportional_move(slab, {(0, 0, 0)}, wl, (0, 0, -2), stretch=True)
crater = dict(slab)
for c, (_old, new) in ch3.items():
    if new:
        crater[c] = new
    else:
        crater.pop(c, None)
assert sorted(z for (a, b, z) in crater if (a, b) == (0, 0)) == [-3, -2]  # dent 2, bottom pushed out
# sliding inside a solid row keeps it solid, colours slide along
bar = {(x, 0, 0): 1 + (x == 0) for x in range(-4, 5)}
chs, _ = vd.proportional_move(bar, {(0, 0, 0)}, vd.proportional_weights(bar, {(0, 0, 0)}, 3, 'LINEAR'), (2, 0, 0))
slid = {c: chs.get(c, [0, k])[1] for c, k in bar.items()}
assert set(c for c, k in slid.items() if k) == set(bar) and slid[(2, 0, 0)] == 2
# diagonal move: no holes either
chd, sel_d = vd.proportional_move(slab, {(0, 0, 0)}, wl, (2, 0, 2))
assert sel_d == {(2, 0, 2)} and chd[(2, 0, 2)][1] == 1
assert vd.proportional_move(slab, {(0, 0, 0)}, wl, (0, 0, 0)) == ({}, {(0, 0, 0)})
assert vd._round_away(-1.5) == -2 and vd._round_away(1.5) == 2 and vd._round_away(0.49) == 0

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


def chunk(cid, content, children=b""):
    return cid + struct.pack("<ii", len(content), len(children)) + content + children


def vox(*chunks):
    return b"VOX " + struct.pack("<i", 150) + chunk(b"MAIN", b"", b"".join(chunks))


d0 = struct.pack("<i", 0)  # empty dict
empty_vox = vox(chunk(b"SIZE", struct.pack("<3i", 1, 1, 1)), chunk(b"XYZI", d0))
assert vd.read_vox(empty_vox) == ({}, None)
good = vd.vox_bytes(two_col, pal)
bad = [good[:n] for n in (20, 40, 60, len(good) - 1)]
bad.append(vox(chunk(b"nTRN", struct.pack("<i", 0) + d0 + struct.pack("<4i", 1, -1, -1, 1) + d0),
               chunk(b"nSHP", struct.pack("<i", 1) + d0 + struct.pack("<2i", 1, 5) + d0)))  # no model 5
bad.append(vox(chunk(b"nTRN", struct.pack("<i", 0) + d0 + struct.pack("<4i", 1, -1, -1, 1) + d0),
               chunk(b"nGRP", struct.pack("<i", 1) + d0 + struct.pack("<2i", 1, 0))))  # loop 0 -> 1 -> 0
for data in bad:
    try:
        vd.read_vox(data)
        raise AssertionError("damaged file must fail")
    except ValueError as e:
        assert "Damaged" in str(e), e

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
assert vd.history_valid(obj)  # Start keeps this history: Ctrl+Z brings the voxels back
assert sum(1 for _old, new in vd._hist["undo"][-1][0].values() if new == 0) == len(two_col) - 1
for name, data, msg in (("vd_empty.vox", empty_vox, "no voxels"), ("vd_bad.vox", good[:60], "Damaged")):
    p = os.path.join(bpy.app.tempdir, name)
    with open(p, "wb") as f:
        f.write(data)
    try:
        bpy.ops.voxeldraw.import_vox(filepath=p)
        raise AssertionError("must be refused")
    except RuntimeError as e:
        assert msg in str(e), e
    assert vd.history_valid(obj)  # session and its undo untouched
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

# Confirm refused (session object in Edit Mode, another object active) leaves the session alone
other = bpy.data.objects.new("other", bpy.data.meshes.new("other"))
scene.collection.objects.link(other)
for o in (obj, other):
    o.select_set(True)
bpy.context.view_layer.objects.active = other
bpy.ops.object.mode_set(mode='EDIT')
assert obj.mode == 'EDIT'
vd._drawing = True
undo_len = len(vd._hist["undo"])
try:
    bpy.ops.voxeldraw.confirm()
    raise AssertionError("must be refused")
except RuntimeError as e:
    assert "Leave Edit Mode first" in str(e)
assert vd._drawing and len(vd._hist["undo"]) == undo_len and "_vd_cells" in obj
bpy.ops.object.mode_set(mode='OBJECT')
vd._drawing = False
bpy.data.objects.remove(other)
bpy.context.view_layer.objects.active = obj

vs.greedy = True
assert bpy.ops.voxeldraw.confirm() == {'FINISHED'}
assert len(me.polygons) < 42 and "_vd_cells" not in obj and vs.target is None
assert me.materials[0].name == "VoxelDraw"

# --- more sessions: pause one, start another, come back to it (voxels and undo kept)
import types  # noqa: E402
ctx = bpy.context
assert vd.open_sessions(scene) == []  # the confirmed one is not a session any more
a = vd.new_session_object(ctx)
vd.place_first_voxel(ctx, a, vs)
vd.ensure_material(a)
vd.activate_object(ctx, a)
bpy.ops.object.mode_set(mode='EDIT')
vd._hist.update(undo=[], redo=[], rev=int(a["_vd_rev"]), obj=a.name)
cells_a = vd.load_cells(a)
vd.commit(scene, a, cells_a, {(1, 0, 0): [0, 3]}, 1.0)
assert len(vd._hist["undo"]) == 1 and vd.history_valid(a)

F = type("F", (), {k: v for k, v in vd.VOXELDRAW_OT_start.__dict__.items() if callable(v)})
op = F()
op.grab = op.spin = op.sel_press = op.drag = None
op.stroke = op.deleting = op.dirty = False
op.editing, op.cells, op.rev, op.size = True, cells_a, int(a["_vd_rev"]), 1.0
op.changes, op.last_build, op.face_key, op.face_cells, op.exit_requested = {}, 0.0, None, None, False
op._timer = None
vd._drawing = True
scene.cursor.location = (10, 0, 0)
op._switch(ctx, a, "")  # New Session
b = vs.target
assert b is not a and b.mode == 'EDIT' and a.mode == 'OBJECT'
assert vd.load_cells(b) == {(0, 0, 0): vs.color_index}  # first voxel at the cursor
assert vd.load_cells(a) == cells_a  # the paused one keeps its voxels...
assert a.name in vd._hist_stash and vd._hist["obj"] == b.name and vd._hist["undo"] == []  # ...and its undo
assert op.editing is None  # the modal syncs the new session on its next event
vd.commit(scene, b, vd.load_cells(b), {(0, 0, 1): [0, 2]}, 1.0)
assert [o.name for o in vd.open_sessions(scene)] == sorted([a.name, b.name])

op._switch(ctx, b, a.name)  # back to the first one
assert vs.target is a and a.mode == 'EDIT' and b.mode == 'OBJECT'
assert len(vd._hist["undo"]) == 1 and vd.history_valid(a)  # Ctrl+Z still works on it
assert b.name in vd._hist_stash and len(vd._hist_stash[b.name][0]) == 1

# the request goes through the running modal (it owns strokes, history, Edit Mode)
vd._state["area"] = 777
area = types.SimpleNamespace(as_pointer=lambda: 777, tag_redraw=lambda: None)


class Ctx:
    def __getattr__(self, name):
        return area if name == "area" else getattr(bpy.context, name)


timer = types.SimpleNamespace(type='TIMER', value='NOTHING')
assert vd.request_session(ctx, b.name) and vd._state["switch"] == b.name
assert vd.VOXELDRAW_OT_start._modal(op, Ctx(), timer) == {'PASS_THROUGH'}
assert vs.target is b and b.mode == 'EDIT' and vd._state["switch"] is None
assert op.cells == vd.load_cells(b) and op.editing  # synced and drawing on b
assert not vd.request_session(ctx, "nope")  # not a session: nothing happens
assert vd._state["switch"] is None

# Blender's navigation gizmo (top right) gets its clicks while the tool runs
win = types.SimpleNamespace(type='WINDOW', x=0, y=0, width=1000, height=800, data=None)
area.regions = [win]
area.spaces = types.SimpleNamespace(active=types.SimpleNamespace(show_gizmo=True, show_gizmo_navigate=True))
pressed = []
op._press = lambda *a: pressed.append(a)


def click_at(x, y):
    ev = types.SimpleNamespace(type='LEFTMOUSE', value='PRESS', mouse_x=x, mouse_y=y,
                               shift=False, ctrl=False, alt=False, oskey=False)
    return vd.VOXELDRAW_OT_start._modal(op, Ctx(), ev)


assert click_at(950, 750) == {'PASS_THROUGH'} and not pressed  # the orbit ball
assert click_at(978, 665) == {'PASS_THROUGH'} and not pressed  # the pan button
assert click_at(500, 400) == {'RUNNING_MODAL'} and len(pressed) == 1  # elsewhere: draw
del op._press

# Pause / Resume (sidebar): Object Mode and back; with the tool stopped it starts it
pr = vd.VOXELDRAW_OT_pause_resume
assert b.mode == 'EDIT'
assert pr.execute(None, ctx) == {'FINISHED'} and b.mode == 'OBJECT'
assert pr.execute(None, ctx) == {'FINISHED'} and b.mode == 'EDIT'
vd._drawing = False
began = []
vd._start_tool = lambda: began.append(vs.target.name)
assert pr.execute(None, ctx) == {'FINISHED'} and began == [b.name]
del vd._start_tool
vd._drawing = True

# delete: a paused session yes, the one in use no
try:
    bpy.ops.voxeldraw.session_delete(name=b.name)
    raise AssertionError("must be refused")
except RuntimeError as e:
    assert "session in use" in str(e)
a_name = a.name
assert bpy.ops.voxeldraw.session_delete(name=a_name) == {'FINISHED'}
assert a_name not in [o.name for o in vd.open_sessions(scene)] and not vd._hist_stash

# with the tool stopped, resuming sets the target, restores its undo and starts the tool
started = []
vd._start_tool = lambda: started.append(vs.target.name)
bpy.ops.object.mode_set(mode='OBJECT')
vd._drawing = False
c = vd.new_session_object(ctx)  # current: c; b paused with its history
vd.place_first_voxel(ctx, c, vs)
vd._hist_stash[b.name] = (vd._hist["undo"], vd._hist["redo"], vd._hist["rev"])
vd._hist.update(undo=[], redo=[], rev=None, obj=None)
assert vd.request_session(ctx, b.name) and started == [b.name] and vs.target is b
assert vd._hist["obj"] == b.name and len(vd._hist["undo"]) == 1
assert vd.request_session(ctx, "") and vs.target not in (b, c) and len(started) == 2
del vd._start_tool

# Confirm finishes only the current session
vd.save_cells(vs.target, {(0, 0, 0): 1}, 1.0)
assert bpy.ops.voxeldraw.confirm() == {'FINISHED'}
assert sorted(o.name for o in vd.open_sessions(scene)) == sorted([b.name, c.name])
vd.unregister()
print("ALL TESTS PASSED")
sys.exit(0)
