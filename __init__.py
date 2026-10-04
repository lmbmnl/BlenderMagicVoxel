# SPDX-License-Identifier: MIT
# BlenderMagicVoxel by lmbmnl - https://github.com/lmbmnl/BlenderMagicVoxel
# MagicaVoxel-style voxel editor for Blender 5.x (sidebar: Start Voxel / Clear / Confirm).
# Inspired by VoxelDraw by Sreeraj R (2020, MIT): https://github.com/theunnecessarythings/VoxelDraw
#
# Data model: the drawing is a dict {integer grid cell: palette index 1..255}
# stored on the object (custom props "_vd_*"), MagicaVoxel style. The mesh is
# regenerated from it (numpy), so it is always a single welded body with no
# hidden faces, and editing keeps working across mode switches (Edit Mode =
# drawing, Object Mode = paused) and save/load, until "Confirm". Undo/redo are
# handled by the tool itself (delta history). Selection, move, rotate and flip
# work on whole cells, so everything always stays on the voxel grid.

import struct
import time
from functools import lru_cache
from itertools import product
from math import floor, inf

import blf
import bmesh
import bpy
import gpu
import numpy as np
from bpy.app.handlers import persistent
from bpy_extras.io_utils import ExportHelper, ImportHelper
from bpy_extras.view3d_utils import (location_3d_to_region_2d, region_2d_to_origin_3d,
                                     region_2d_to_vector_3d)
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

# ---------------------------------------------------------------- pure logic start
AXIS_ITEMS = [
    ("XY", "XY", "Floor plane XY (normal Z)"),
    ("YZ", "YZ", "Floor plane YZ (normal X)"),
    ("XZ", "XZ", "Floor plane XZ (normal Y)"),
]
AXIS_INDEX = {"XY": 2, "YZ": 0, "XZ": 1}  # index of the plane normal
MODE_ITEMS = [
    ("ATTACH", "Attach", "Add voxels with the active colour (1)"),
    ("ERASE", "Erase", "Remove voxels (2)"),
    ("MOVE", "Move", "Move the selected voxels with the gizmo, snapped to the grid (3)"),
    ("PAINT", "Paint", "Recolour existing voxels (4)"),
    ("SELECT", "Select", "Select voxels to move / rotate / flip / copy (5)"),
]
MOVE_AXIS_ITEMS = [
    ("FREE", "Free", "Drag moves on the plane facing the view"),
    ("X", "X", "Drag moves along X (X key)"),
    ("Y", "Y", "Drag moves along Y (Y key)"),
    ("Z", "Z", "Drag moves along Z (Z key)"),
]
BRUSH_ITEMS = [
    ("SHAPE", "Shape", "Stamp the brush shape, drag to paint"),
    ("BOX", "Box", "Drag a rectangle on the plane of the first voxel"),
    ("LINE", "Line", "Drag a line on the plane of the first voxel"),
    ("FACE", "Face", "Whole flat face region: extrude / remove / paint it"),
    ("FILL", "Fill", "Connected voxels of the same colour: recolour / remove them"),
]
SHAPE_ITEMS = [
    ("POINT", "Point", "Filled n x n square"),
    ("SQUARE", "Square", "Square"),
    ("TRIANGLE", "Triangle", "Triangle"),
    ("HEXAGON", "Hexagon", "Hexagon"),
    ("CIRCLE", "Circle", "Circle"),
    ("SPHERE", "Sphere", "Sphere (3D, size up to 64)"),
]
COLOR_ATTR = "vd_color"
TMP_MESH = ".vd_tmp"
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
N6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

# (neighbour offset, quad corners CCW seen from outside). Cell (i,j,k) occupies
# the lattice box [i,i+1]x[j,j+1]x[k,k+1]; world = lattice * size, like
# MagicaVoxel (voxels sit ON the floor plane). Index // 2 = axis.
FACES = (
    ((1, 0, 0), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))),
    ((-1, 0, 0), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0))),
    ((0, 1, 0), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0))),
    ((0, -1, 0), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1))),
    ((0, 0, 1), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))),
    ((0, 0, -1), ((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0))),
)

_B, _OFF = 1 << 21, 1 << 20  # int64 cell keys; coordinates are capped at +-100000


def _keys(p):
    p = p + _OFF
    return (p[:, 0] * _B + p[:, 1]) * _B + p[:, 2]


def default_palette():
    """MagicaVoxel default palette as sRGB floats; index 0 = empty."""
    steps = (0xff, 0xcc, 0x99, 0x66, 0x33, 0x00)
    ramp = (0xee, 0xdd, 0xbb, 0xaa, 0x88, 0x77, 0x55, 0x44, 0x22, 0x11)
    cols = [(0, 0, 0)]
    cols += [(r, g, b) for r in steps for g in steps for b in steps][:-1]
    cols += [(v, 0, 0) for v in ramp] + [(0, v, 0) for v in ramp]
    cols += [(0, 0, v) for v in ramp] + [(v, v, v) for v in ramp]
    return [(r / 255, g / 255, b / 255) for r, g, b in cols]


def exposed_faces(cells):
    """Per FACES direction: (cells (n,3), colours (n,)) whose face there is visible."""
    pos = np.array(list(cells), np.int64).reshape(-1, 3)
    col = np.fromiter(cells.values(), np.int64, len(cells))
    sk = np.sort(_keys(pos))
    out = []
    for normal, _quad in FACES:
        nk = _keys(pos + normal)
        idx = np.minimum(np.searchsorted(sk, nk), len(sk) - 1)
        m = sk[idx] != nk
        out.append((pos[m], col[m]))
    return out


def greedy_rects(pos, col, axis):
    """Merge coplanar same-colour faces of one direction into rectangles.
    Returns (lowest cells (n,3), extents (n,3), colours (n,))."""
    b, c = [n for n in range(3) if n != axis]
    left = {(p[axis], p[b], p[c]): k for p, k in zip(pos.tolist(), col.tolist())}
    lows, exts, cols = [], [], []
    for key in sorted(left):
        k = left.get(key)
        if k is None:  # already merged into a rectangle
            continue
        lay, u, v = key
        h = 1
        while left.get((lay, u, v + h)) == k:
            h += 1
        w = 1
        while all(left.get((lay, u + w, v + t)) == k for t in range(h)):
            w += 1
        for du in range(w):
            for dv in range(h):
                del left[(lay, u + du, v + dv)]
        low, ext = [0, 0, 0], [1, 1, 1]
        low[axis], low[b], low[c] = lay, u, v
        ext[b], ext[c] = w, h
        lows.append(low)
        exts.append(ext)
        cols.append(k)
    return (np.array(lows, np.int64).reshape(-1, 3),
            np.array(exts, np.int64).reshape(-1, 3), np.array(cols, np.int64))


def build_geometry(cells, size, greedy=False):
    """Welded surface of the voxel dict: (verts (m,3), quads (f,4), face colours (f,)).
    greedy=True merges same-colour coplanar faces (leaves T-junctions)."""
    if not cells:
        return np.zeros((0, 3), np.float32), np.zeros((0, 4), np.int64), np.zeros(0, np.int64)
    corners, cols = [], []
    for n, ((_normal, quad), (pos, col)) in enumerate(zip(FACES, exposed_faces(cells))):
        ext = np.ones_like(pos)
        if greedy:
            pos, ext, col = greedy_rects(pos, col, n // 2)
        corners.append(pos[:, None, :] + np.array(quad) * ext[:, None, :])
        cols.append(col)
    corners = np.concatenate(corners).reshape(-1, 3)
    _, first, inv = np.unique(_keys(corners), return_index=True, return_inverse=True)
    verts = (corners[first] * size).astype(np.float32)
    return verts, inv.reshape(-1, 4), np.concatenate(cols)


def ghost_geometry(cells, inflate=0.02):
    """Brush ghost in lattice units: (triangles (3t,3) of its outer faces pushed out by
    `inflate` so they never z-fight the voxels they cover, unique edges (2e,3) of
    those faces). Inner faces / edges between ghost voxels are left out."""
    tris, quads = [], []
    for (normal, quad), (pos, _col) in zip(FACES, exposed_faces(dict.fromkeys(cells, 1))):
        q = pos[:, None, :] + np.array(quad)
        quads.append(q)
        tris.append((q + np.array(normal) * inflate)[:, (0, 1, 2, 0, 2, 3)])
    seg = np.concatenate(quads)[:, ((0, 1), (1, 2), (2, 3), (3, 0))].reshape(-1, 2, 3)
    k = _keys(seg.reshape(-1, 3)).reshape(-1, 2)
    _, first = np.unique(np.sort(k, axis=1), axis=0, return_index=True)
    return (np.concatenate(tris).reshape(-1, 3).astype(np.float32),
            seg[np.sort(first)].reshape(-1, 3).astype(np.float32))


def mirror_planes(vs, bnds):
    """Live mirror planes as quads (4 corners, lattice units) + axis: [(axis, corners)].
    They span the size limit box, or the model bounds plus a margin."""
    sums = mirror_sums(vs)
    if vs.use_limit:
        lo, hi = (0, 0, 0), tuple(vs.model_size)
    else:
        lo, hi = bnds or ((0, 0, 0), (0, 0, 0))
        lo = [min(lo[i], 0) - 2 for i in range(3)]
        hi = [max(hi[i], 0) + 3 for i in range(3)]
    out = []
    for a in live_mirror_axes(vs):
        b, c = [i for i in range(3) if i != a]
        corners = []
        for u, v in ((lo[b], lo[c]), (hi[b], lo[c]), (hi[b], hi[c]), (lo[b], hi[c])):
            p = [0.0, 0.0, 0.0]
            p[a], p[b], p[c] = (sums[a] + 1) / 2, u, v
            corners.append(p)
        out.append((a, corners))
    return out


def _outline(pts, dirs):
    return {p for p in pts
            if any(tuple(p[i] + d[i] for i in range(len(p))) not in pts for d in dirs)}


@lru_cache(maxsize=64)
def shape_offsets(shape, n, rot, fill=False):
    """2D brush cells (u, v) centred on (0, 0), rotated rot x 90 deg clockwise.
    n = side length in cells. Slanted sides are 45 deg pixel steps, so every
    size is symmetric: triangle base 2n-1 / height n, hexagon width 3n-2 / height 2n-1.
    fill=False keeps only the outline (Point is always filled)."""
    m = n - 1
    if shape in ('POINT', 'SQUARE'):
        pts = {(u, v) for u in range(n) for v in range(n)}
    elif shape == 'CIRCLE':
        r2 = (n * n - n) / 4
        pts = {(u, v) for u in range(n) for v in range(n)
               if (u - m / 2) ** 2 + (v - m / 2) ** 2 <= r2}
    else:
        if shape == 'TRIANGLE':  # row v spans u in [a, b]
            rows = [(v, 2 * m - v) for v in range(n)]
        else:
            rows = [(abs(v - m), 3 * m - abs(v - m)) for v in range(2 * m + 1)]
        pts = {(u, v) for v, (a, b) in enumerate(rows) for u in range(a, b + 1)}
    if shape != 'POINT' and not fill:
        pts = _outline(pts, N4)
    for _ in range(rot % 4):
        pts = {(v, -u) for u, v in pts}
    us, vs = zip(*pts)
    cu, cv = (min(us) + max(us)) // 2, (min(vs) + max(vs)) // 2
    return frozenset((u - cu, v - cv) for u, v in pts)


@lru_cache(maxsize=16)
def sphere_offsets(n, fill=False):
    m = n - 1
    r2 = (n * n - n) / 4
    pts = {(x, y, z) for x, y, z in product(range(n), repeat=3)
           if (x - m / 2) ** 2 + (y - m / 2) ** 2 + (z - m / 2) ** 2 <= r2}
    if not fill:
        pts = _outline(pts, N6)
    h = m // 2
    return frozenset((x - h, y - h, z - h) for x, y, z in pts)


def stamp_cells(cell, axis, shape, n, rot, fill=False):
    """Brush cells centred on `cell`, lying in the plane with normal `axis`."""
    if shape == 'SPHERE':  # ponytail: capped at 64, a python-built ball is O(n^3)
        return [(cell[0] + x, cell[1] + y, cell[2] + z) for x, y, z in sphere_offsets(min(n, 64), fill)]
    b, c = [i for i in range(3) if i != axis]
    out = []
    for u, v in shape_offsets(shape, n, rot, fill):
        p = list(cell)
        p[b] += u
        p[c] += v
        out.append(tuple(p))
    return out


def mirror_cell(c, axis, s=0):
    """Mirror cell i to s - i. s = 0: plane through the middle of voxel (0,0,0), the
    first voxel spawned on the 3D cursor; s = size - 1: centre of a 0..size model."""
    return tuple(s - c[i] if i == axis else c[i] for i in range(3))


def mirror_cells(cells, axes, sums=(0, 0, 0)):
    out = set(cells)
    for a in axes:
        out |= {mirror_cell(c, a, sums[a]) for c in out}
    return list(out)


def live_mirror_axes(vs):
    return [i for i, on in enumerate(vs.mirror) if on] if vs.mirror_live else []


def mirrored(vs, cells):
    """Brush cells plus their live-mirror copies."""
    axes = live_mirror_axes(vs)
    return mirror_cells(cells, axes, mirror_sums(vs)) if axes else cells


def mirror_sums(vs):
    """Per-axis s of mirror_cell: the model centre when the size limit is on (the
    first voxel would sit on the limit edge), else the first voxel."""
    return tuple(n - 1 for n in vs.model_size) if vs.use_limit else (0, 0, 0)


def ray_axis_param(o, d, c, axis):
    """Parameter s of the point c + s*e_axis closest to the ray o + t*d (None if parallel)."""
    w = [c[i] - o[i] for i in range(3)]
    b, cc = d[axis], sum(x * x for x in d)
    den = cc - b * b
    if den < 1e-9 * cc:
        return None
    return (b * sum(d[i] * w[i] for i in range(3)) - cc * w[axis]) / den


def seg_dist(p, a, b):
    """2D distance from point p to segment ab."""
    ab = (b[0] - a[0], b[1] - a[1])
    L = ab[0] ** 2 + ab[1] ** 2
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / L))
    return ((p[0] - a[0] - t * ab[0]) ** 2 + (p[1] - a[1] - t * ab[1]) ** 2) ** 0.5


def line_cells(a, b):
    n = max(abs(b[i] - a[i]) for i in range(3))
    if n == 0:
        return [tuple(a)]
    return [tuple(a[i] + floor((b[i] - a[i]) * s / n + 0.5) for i in range(3)) for s in range(n + 1)]


def box_cells(a, b, axis, fill=True):
    """Rectangle between a and b (same layer along `axis`); fill=False = border only."""
    lo = [min(a[i], b[i]) for i in range(3)]
    hi = [max(a[i], b[i]) for i in range(3)]
    p, q = [i for i in range(3) if i != axis]
    return [c for c in product(*(range(lo[i], hi[i] + 1) for i in range(3)))
            if fill or c[p] in (lo[p], hi[p]) or c[q] in (lo[q], hi[q])]


def flood(cells, start, same_color=False):
    """6-connected voxels reachable from start (optionally only the same colour)."""
    col = cells[start]
    seen, stack = {start}, [start]
    while stack:
        x, y, z = stack.pop()
        for q in ((x + 1, y, z), (x - 1, y, z), (x, y + 1, z), (x, y - 1, z), (x, y, z + 1), (x, y, z - 1)):
            if q not in seen and q in cells and (not same_color or cells[q] == col):
                seen.add(q)
                stack.append(q)
    return seen


def face_region(cells, start, normal, limit=65536):
    """Voxels of the flat, exposed face region containing start (MagicaVoxel Face brush)."""
    a = [abs(v) for v in normal].index(1)
    b, c = [i for i in range(3) if i != a]
    seen, stack = {start}, [start]
    while stack and len(seen) < limit:
        p = stack.pop()
        for db, dc in N4:
            q = list(p)
            q[b] += db
            q[c] += dc
            q = tuple(q)
            if q in seen or q not in cells:
                continue
            if (q[0] + normal[0], q[1] + normal[1], q[2] + normal[2]) in cells:
                continue  # covered: not part of the visible face
            seen.add(q)
            stack.append(q)
    return seen


def rotate_fn(sel, axis, ccw):
    """90 deg rotation around `axis` (right-hand ccw), kept centred on the selection.
    Odd/even sides can't share an exact centre: the half-voxel is rounded toward
    zero, so rotating back (or 4 times) returns exactly to the start."""
    b, c = (axis + 1) % 3, (axis + 2) % 3

    def raw(cell):
        p = list(cell)
        p[b], p[c] = (-cell[c], cell[b]) if ccw else (cell[c], -cell[b])
        return p
    lo, hi = bounds_of(sel)
    rlo, rhi = bounds_of({tuple(raw(s)) for s in (lo, hi)})  # rotated box corners
    off = [lo[i] + int(((hi[i] - lo[i]) - (rhi[i] - rlo[i])) / 2) - rlo[i] for i in range(3)]
    return lambda cell: tuple(v + off[i] for i, v in enumerate(raw(cell)))


def flip_fn(sel, axis):
    lo, hi = bounds_of(sel)
    return lambda c: tuple(lo[i] + hi[i] - c[i] if i == axis else c[i] for i in range(3))


def transform(cells, sel, fn, keep=False):
    """Move the selected voxels through fn (overwriting what is there).
    Returns ({cell: [old, new]}, new selection)."""
    moved = {fn(c): cells[c] for c in sel}
    new = {} if keep else {c: 0 for c in sel}
    new.update(moved)
    return {c: [cells.get(c, 0), k] for c, k in new.items() if cells.get(c, 0) != k}, set(moved)


def cells_in_rect(cells, mvp, size, width, height, rect):
    """Voxels whose centre projects inside the screen rectangle (region pixels)."""
    if not cells:
        return set()
    keys = list(cells)
    p = (np.array(keys, np.float64) + 0.5) * size
    h = np.c_[p, np.ones(len(p))] @ np.array(mvp, np.float64).T
    w = h[:, 3]
    ok = w > 1e-6
    w = np.where(ok, w, 1)
    x = (h[:, 0] / w + 1) * width / 2
    y = (h[:, 1] / w + 1) * height / 2
    x0, x1 = sorted((rect[0], rect[2]))
    y0, y1 = sorted((rect[1], rect[3]))
    inside = ok & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    return {k for k, f in zip(keys, inside.tolist()) if f}


def vox_bytes(cells, pal):
    """MagicaVoxel .vox (one model, shifted to start at 0) from cells + (256,3) sRGB palette."""
    lo, hi = bounds_of(cells)
    dims = [hi[i] - lo[i] + 1 for i in range(3)]
    if max(dims) > 256:
        raise ValueError(f"Model is {dims[0]}x{dims[1]}x{dims[2]}: .vox allows 256 per axis")
    v = np.array([(x - lo[0], y - lo[1], z - lo[2], k) for (x, y, z), k in cells.items()], np.uint8)
    rgba = np.full((256, 4), 255, np.uint8)
    rgba[:255, :3] = np.clip(np.round(np.asarray(pal)[1:] * 255), 0, 255)  # file entry i = index i+1

    def chunk(cid, content, children=b""):
        return cid + struct.pack("<ii", len(content), len(children)) + content + children
    body = (chunk(b"SIZE", struct.pack("<3i", *dims))
            + chunk(b"XYZI", struct.pack("<i", len(v)) + v.tobytes())
            + chunk(b"RGBA", rgba.tobytes()))
    return b"VOX " + struct.pack("<i", 150) + chunk(b"MAIN", b"", body)


def _read_str(b, p):
    n, = struct.unpack_from("<i", b, p)
    return b[p + 4:p + 4 + n].decode("utf-8", "replace"), p + 4 + n


def _read_dict(b, p):
    n, = struct.unpack_from("<i", b, p)
    p += 4
    d = {}
    for _ in range(n):
        k, p = _read_str(b, p)
        d[k], p = _read_str(b, p)
    return d, p


def _vox_rot(r):
    i0, i1 = r & 3, (r >> 2) & 3
    m = np.zeros((3, 3), np.int64)
    for row, (i, bit) in enumerate(((i0, 16), (i1, 32), (3 - i0 - i1, 64))):
        m[row, i] = -1 if r & bit else 1
    return m


def read_vox(data):
    """-> (cells shifted to start at 0, palette list[256] or None). Merges every
    model of the scene graph with its translation/rotation. Any damage in the
    file raises ValueError."""
    if data[:4] != b"VOX ":
        raise ValueError("Not a MagicaVoxel .vox file")
    try:
        return _read_vox(data)
    except (struct.error, IndexError, KeyError, ValueError, RecursionError) as e:
        raise ValueError(f"Damaged .vox file: {e}") from e


def _read_vox(data):
    if data[8:12] != b"MAIN":
        raise ValueError("no MAIN chunk")
    p, size, models, nodes, rgba = 8, (0, 0, 0), [], {}, None
    while p + 12 <= len(data):
        cid = data[p:p + 4]
        n, m = struct.unpack_from("<ii", data, p + 4)
        c = p + 12
        if n < 0 or m < 0 or c + n + m > len(data):
            raise ValueError(f"chunk {cid!r} is truncated")
        if cid == b"SIZE":
            size = struct.unpack_from("<3i", data, c)
        elif cid == b"XYZI":
            k, = struct.unpack_from("<i", data, c)
            v = np.frombuffer(data, np.uint8, k * 4, c + 4).reshape(-1, 4).astype(np.int64)
            models.append((size, v))
        elif cid == b"RGBA":
            rgba = np.frombuffer(data, np.uint8, 1024, c).reshape(256, 4)
        elif cid in (b"nTRN", b"nGRP", b"nSHP"):
            nid, = struct.unpack_from("<i", data, c)
            _, q = _read_dict(data, c + 4)
            if cid == b"nTRN":
                child, _res, _layer, nf = struct.unpack_from("<4i", data, q)
                nodes[nid] = ("T", child, _read_dict(data, q + 16)[0] if nf else {})
            elif cid == b"nGRP":
                k, = struct.unpack_from("<i", data, q)
                nodes[nid] = ("G", struct.unpack_from(f"<{k}i", data, q + 4))
            else:
                k, = struct.unpack_from("<i", data, q)
                q, ids = q + 4, []
                for _ in range(k):
                    mid, = struct.unpack_from("<i", data, q)
                    _, q = _read_dict(data, q + 4)
                    ids.append(mid)
                nodes[nid] = ("S", ids)
        p = c + n if cid == b"MAIN" else c + n + m

    placed = []

    def walk(nid, rot, t, path):
        node = nodes.get(nid)
        if node is None:
            return
        if nid in path:
            raise ValueError("the scene graph has a loop")
        path = path | {nid}
        if node[0] == "T":
            fr = node[2]
            r = _vox_rot(int(fr["_r"])) if "_r" in fr else np.eye(3, dtype=np.int64)
            tn = np.array([int(x) for x in fr["_t"].split()], np.int64) if "_t" in fr else np.zeros(3, np.int64)
            if tn.shape != (3,):
                raise ValueError("bad translation")
            walk(node[1], rot @ r, rot @ tn + t, path)
        elif node[0] == "G":
            for ch in node[1]:
                walk(ch, rot, t, path)
        else:
            for mid in node[1]:
                if not 0 <= mid < len(models):
                    raise ValueError(f"shape refers to missing model {mid}")
                (sx, sy, sz), v = models[mid]
                # ponytail: pivot = size // 2; rotated even-sized models may land 1 voxel off
                placed.append(((v[:, :3] - (sx // 2, sy // 2, sz // 2)) @ rot.T + t, v[:, 3]))
    if nodes:
        walk(0, np.eye(3, dtype=np.int64), np.zeros(3, np.int64), frozenset())
    else:
        placed = [(v[:, :3], v[:, 3]) for _, v in models]
    cells = {}
    if sum(len(b) for _, b in placed):
        pos = np.concatenate([a for a, _ in placed])
        col = np.concatenate([b for _, b in placed])
        pos -= pos.min(0)
        cells = {tuple(c): k for c, k in zip(pos.tolist(), col.tolist()) if k}
    pal = None
    if rgba is not None:
        pal = [(0.0, 0.0, 0.0)] + [tuple(x / 255 for x in rgba[i, :3]) for i in range(255)]
    return cells, pal


def bounds_of(cells):
    if not cells:
        return None
    xs, ys, zs = zip(*cells)
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def grow(bnds, c):
    if bnds is None:
        return c, c
    lo, hi = bnds
    return (tuple(min(lo[n], c[n]) for n in range(3)),
            tuple(max(hi[n], c[n]) for n in range(3)))


def raycast_cells(o, d, cells, bnds, size):
    """Grid DDA. Ray in object-local space. Returns (cell, outward_normal) or None."""
    bmin, bmax = bnds
    p = [o[n] / size for n in range(3)]
    dd = [d[n] / size for n in range(3)]

    # clip the ray against the cells' bounding box (slab test)
    tmin, tmax, entry = 0.0, inf, None
    for n in range(3):
        lo, hi = bmin[n], bmax[n] + 1
        if abs(dd[n]) < 1e-12:
            if p[n] < lo or p[n] > hi:
                return None
            continue
        t1, t2 = (lo - p[n]) / dd[n], (hi - p[n]) / dd[n]
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin:
            tmin, entry = t1, n
        tmax = min(tmax, t2)
    if tmin > tmax:
        return None

    t = tmin + 1e-7
    start = [p[n] + dd[n] * t for n in range(3)]
    cell = [floor(s) for s in start]
    step = [1 if dd[n] > 0 else -1 for n in range(3)]
    tnext, tdelta = [inf] * 3, [inf] * 3
    for n in range(3):
        if abs(dd[n]) < 1e-12:
            continue
        boundary = cell[n] + (1 if step[n] > 0 else 0)
        tnext[n] = t + (boundary - start[n]) / dd[n]
        tdelta[n] = step[n] / dd[n]

    axis = entry if entry is not None else max(range(3), key=lambda n: abs(dd[n]))
    for _ in range(sum(bmax[n] - bmin[n] + 1 for n in range(3)) + 6):
        key = (cell[0], cell[1], cell[2])
        if key in cells:
            normal = [0, 0, 0]
            normal[axis] = -step[axis]
            return key, tuple(normal)
        n = tnext.index(min(tnext))
        if tnext[n] > tmax:
            return None
        cell[n] += step[n]
        tnext[n] += tdelta[n]
        axis = n
    return None


def pick_cell(o, d, cells, bnds, size, axis, offset, existing):
    """(cell, plane normal axis) under the ray, or None. existing=True targets the
    hit voxel (erase/paint), else the empty cell in front of it; falls back to
    the floor plane (voxels sit on top of it)."""
    if cells and bnds:
        hit = raycast_cells(o, d, cells, bnds, size)
        if hit:
            c, n = hit
            ax = [abs(v) for v in n].index(1)
            return (c if existing else (c[0] + n[0], c[1] + n[1], c[2] + n[2])), ax
    if existing or abs(d[axis]) < 1e-9:
        return None
    t = (offset - o[axis]) / d[axis]
    if t < 0:
        return None
    cell = [floor((o[i] + d[i] * t) / size) for i in range(3)]
    cell[axis] = floor(offset / size + 1e-6)
    if any(abs(c) > 100000 for c in cell):
        return None
    return tuple(cell), axis


def plane_cell(o, d, start, axis, size):
    """Cell under the ray on the layer of `start` (for Box / Line drags), max 256 away."""
    if abs(d[axis]) < 1e-9:
        return None
    t = ((start[axis] + 0.5) * size - o[axis]) / d[axis]
    if t < 0:
        return None
    c = [floor((o[i] + d[i] * t) / size) for i in range(3)]
    c[axis] = start[axis]
    return tuple(max(start[i] - 255, min(start[i] + 255, c[i])) for i in range(3))
# ---------------------------------------------------------------- pure logic end


_drawing = False
_state = {"area": 0, "hover": None, "size": 1.0, "mode": 'ATTACH', "float": None,
          "rect": None, "bounds": None, "pick": False, "gizmo_hover": None, "line": None}
# Own undo history: entries are ({cell: [old_colour, new_colour]}, voxel_size),
# colour 0 = empty. "rev" is the revision of object "obj" (name) produced by our
# own last change: while they match, the history applies (even between sessions).
_hist = {"undo": [], "redo": [], "rev": None, "obj": None}
_sel = set()   # selected cells
_clip = {}     # copied voxels, relative to their lowest corner
MAX_HISTORY = 100
CUBE_CORNERS = np.array([(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1)], np.float32)
CUBE_LINES = np.array(((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6),
                       (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)), np.int32)
MODE_KEYS = {'ONE': 'ATTACH', 'TWO': 'ERASE', 'THREE': 'MOVE', 'FOUR': 'PAINT', 'FIVE': 'SELECT'}
AXIS_KEYS = {'X': 0, 'Y': 1, 'Z': 2}
AXIS_COLORS = ((0.95, 0.25, 0.3, 1), (0.55, 0.85, 0.2, 1), (0.25, 0.55, 1, 1))
# key -> (screen axis: 0 right, 1 up, 2 toward the viewer, sign)
NUDGE = {'RIGHT_ARROW': (0, 1), 'LEFT_ARROW': (0, -1), 'UP_ARROW': (1, 1),
         'DOWN_ARROW': (1, -1), 'PAGE_UP': (2, 1), 'PAGE_DOWN': (2, -1)}
STATUS = "VoxelDraw: shortcuts in the legend at the bottom left (H hides it)    Tab: pause"
LEGEND = (
    "LMB draw    Ctrl+LMB erase    RMB pick colour    Shift+RMB replace colour",
    "1-5 Attach/Erase/Move/Paint/Select    B brush    Ctrl+RMB shape    Ctrl+Wheel size    "
    "Shift+LMB rotate    Ctrl+F fill    M live mirror",
    "Select: LMB click/drag (Shift add, Ctrl remove)    A all    Alt+A none    "
    "L linked    C same colour",
    "Move: drag a gizmo arrow (axis) or its centre (free)    X / Y / Z lock the drag axis",
    "Selection: G move    Shift+D duplicate    R rotate    F / Shift+F flip    "
    "Arrows PgUp PgDn nudge    X delete    P paint    Ctrl+C / Ctrl+V",
    "Ctrl+Z / Ctrl+Shift+Z undo / redo    Tab pause    H hide legend",
)


def tag_redraw_all(context):
    for win in context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


def load_cells(obj):
    raw = obj.get("_vd_cells")
    if raw is None:
        return {}
    it = iter(raw.to_list())
    pos = list(zip(it, it, it))
    cols = obj.get("_vd_cols")
    return dict(zip(pos, cols.to_list() if cols is not None else [1] * len(pos)))


def save_cells(obj, cells, size):
    if cells:
        obj["_vd_cells"] = [v for c in cells for v in c]
        obj["_vd_cols"] = list(cells.values())
        obj["_vd_size"] = size
    else:
        for key in ("_vd_cells", "_vd_cols", "_vd_size"):
            obj.pop(key, None)
    rev = int(obj.get("_vd_rev", 0)) + 1
    obj["_vd_rev"] = rev
    return rev


def session_size(obj, vs):
    return float(obj["_vd_size"]) if "_vd_size" in obj else vs.voxel_size


def history_valid(obj):
    """True if the undo history ends at the voxels currently stored on obj."""
    return _hist["obj"] == obj.name and _hist["rev"] == int(obj.get("_vd_rev", 0))


def push_history(obj, changes, size, rev):
    """Record a change that took obj to revision `rev`. Older steps are dropped
    if the voxels were changed by someone else (Blender undo, other object)."""
    if not (_hist["obj"] == obj.name and _hist["rev"] == rev - 1):
        _hist["undo"].clear()
    _hist["undo"].append((changes, size))
    del _hist["undo"][:-MAX_HISTORY]
    _hist["redo"].clear()
    _hist.update(rev=rev, obj=obj.name)


def commit(scene, obj, cells, changes, size):
    """Apply {cell: [old, new]} to `cells`, store it on obj, rebuild, push undo."""
    if not changes:
        return
    for c, (_old, new) in changes.items():
        if new:
            cells[c] = new
        else:
            cells.pop(c, None)
    rebuild_mesh(obj, cells, size, palette_array(scene))
    push_history(obj, changes, size, save_cells(obj, cells, size))


def selection_action(vs, cells, action, axis=0, ccw=False, delta=(0, 0, 0)):
    """Selection commands. Updates _sel / _clip, returns the voxel changes to commit."""
    sel = {c for c in _sel if c in cells}
    if action == 'ALL':
        _sel.clear()
        _sel.update(cells)
        return {}
    if action == 'NONE':
        _sel.clear()
        return {}
    if not sel:
        return {}
    if action == 'COPY':
        lo = bounds_of(sel)[0]
        _clip.clear()
        _clip.update({(c[0] - lo[0], c[1] - lo[1], c[2] - lo[2]): cells[c] for c in sel})
        return {}
    if action == 'DELETE':
        _sel.clear()
        return {c: [cells[c], 0] for c in sel}
    if action == 'PAINT':
        k = vs.color_index
        return {c: [cells[c], k] for c in sel if cells[c] != k}
    if action == 'MIRROR':  # mirrored copy across the live mirror plane
        s = mirror_sums(vs)[axis]
        changes, moved = transform(cells, sel, lambda c: mirror_cell(c, axis, s), keep=True)
        _sel.update(moved)
        return changes
    if action == 'ROTATE':
        fn = rotate_fn(sel, axis, ccw)
    elif action == 'FLIP':
        fn = flip_fn(sel, axis)
    else:  # MOVE
        fn = lambda c: (c[0] + delta[0], c[1] + delta[1], c[2] + delta[2])  # noqa: E731
    changes, new_sel = transform(cells, sel, fn)
    _sel.clear()
    _sel.update(new_sel)
    return changes


def view_axes(rv3d, obj):
    """Lattice axes closest to screen right, screen up and toward the viewer: [(axis, sign)] x3."""
    inv = obj.matrix_world.to_3x3().inverted_safe()
    used, out = set(), []
    for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        w = inv @ (rv3d.view_rotation @ Vector(v))
        a = max((i for i in range(3) if i not in used), key=lambda i: abs(w[i]))
        used.add(a)
        out.append((a, 1 if w[a] > 0 else -1))
    return out


def gizmo_screen(region, rv3d, obj, cells, size):
    """Move gizmo in region pixels: (centre, {axis: arrow end}) at the centre of `cells`,
    or None. Axes pointing at the viewer are hidden."""
    if not cells:
        return None
    lo, hi = bounds_of(cells)
    c_local = Vector([(lo[i] + hi[i] + 1) / 2 * size for i in range(3)])
    mw = obj.matrix_world
    c2 = location_3d_to_region_2d(region, rv3d, mw @ c_local)
    if c2 is None:
        return None
    dirs = {}
    for a in range(3):
        e = Vector((0.0, 0.0, 0.0))
        e[a] = size
        p2 = location_3d_to_region_2d(region, rv3d, mw @ (c_local + e))
        if p2 is not None:
            dirs[a] = p2 - c2
    longest = max((v.length for v in dirs.values()), default=0.0)
    if longest < 1e-6:
        return None
    px = 90 * bpy.context.preferences.system.ui_scale
    return c2, {a: c2 + v.normalized() * px for a, v in dirs.items() if v.length > 0.2 * longest}


def gizmo_hit(giz, mouse):
    """'C' (centre), an axis index, or None."""
    if giz is None:
        return None
    c2, ends = giz
    s = bpy.context.preferences.system.ui_scale
    if (mouse - c2).length < 12 * s:
        return 'C'
    best = min(ends, key=lambda a: seg_dist(mouse, c2, ends[a]), default=None)
    if best is not None and seg_dist(mouse, c2, ends[best]) < 9 * s:
        return best
    return None


def ensure_palette(vs):
    if len(vs.palette) < 2:
        set_palette(vs, default_palette())


def set_palette(vs, rgb):
    """Replace the palette (item 0 = empty voxel placeholder, hidden in the list)."""
    vs.palette.clear()
    for c in rgb:
        vs.palette.add()["color"] = c  # id-prop write: skips the update callback


def palette_array(scene):
    """(256, 3) sRGB colours by index; unused indices are grey."""
    pal = scene.voxel_settings.palette
    if len(pal) < 2:
        return np.array(default_palette(), np.float32)
    arr = np.empty(len(pal) * 3, np.float32)
    pal.foreach_get("color", arr)
    out = np.full((256, 3), 0.5, np.float32)
    out[:len(pal)] = arr.reshape(-1, 3)[:256]
    return out


def remap_removed_color(cells, k):
    """Indices above a removed palette entry k shift down by one."""
    return {c: (v - 1 if v > k else v) for c, v in cells.items()}


def fill_mesh(me, verts, faces, rgb):
    me.clear_geometry()
    nf = len(faces)
    me.vertices.add(len(verts))
    me.loops.add(nf * 4)
    me.polygons.add(nf)
    me.vertices.foreach_set("co", verts.ravel())
    me.loops.foreach_set("vertex_index", faces.ravel().astype(np.int32))
    me.polygons.foreach_set("loop_start", np.arange(0, nf * 4, 4, dtype=np.int32))
    me.update(calc_edges=True)
    me.shade_flat()
    for name in (".select_vert", ".select_edge", ".select_poly"):  # add() selects everything
        sel = me.attributes.get(name)
        if sel is not None:
            me.attributes.remove(sel)
    attr = me.color_attributes.get(COLOR_ATTR) or me.color_attributes.new(COLOR_ATTR, 'BYTE_COLOR', 'CORNER')
    rgba = np.ones((nf, 4, 4), np.float32)
    rgba[:, :, :3] = rgb[:, None, :]
    attr.data.foreach_set("color_srgb", rgba.ravel())
    me.color_attributes.active_color_name = me.color_attributes.default_color_name = COLOR_ATTR


def rebuild_mesh(obj, cells, size, pal, greedy=False):
    verts, faces, fcol = build_geometry(cells, size, greedy)
    me = obj.data
    if obj.mode != 'EDIT':
        fill_mesh(me, verts, faces, pal[fcol])
        return
    # Edit Mode: build with fast array access on a scratch mesh, then copy it in.
    tmp = bpy.data.meshes.get(TMP_MESH) or bpy.data.meshes.new(TMP_MESH)
    fill_mesh(tmp, verts, faces, pal[fcol])
    bm = bmesh.from_edit_mesh(me)
    bm.clear()
    bm.from_mesh(tmp)
    bmesh.update_edit_mesh(me)
    me.color_attributes.active_color_name = me.color_attributes.default_color_name = COLOR_ATTR


def ensure_material(obj):
    me = obj.data
    if me.materials:
        return
    mat = bpy.data.materials.get("VoxelDraw")
    if mat is None:
        mat = bpy.data.materials.new("VoxelDraw")
        nt = mat.node_tree
        attr = nt.nodes.new("ShaderNodeAttribute")
        attr.attribute_name = COLOR_ATTR
        attr.location = (-300, 300)
        nt.links.new(attr.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    me.materials.append(mat)


def activate_object(context, obj):
    active = context.view_layer.objects.active
    if active is not None and active == obj:
        return
    if active is not None and active.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj


def ensure_session_object(context):
    vs = context.scene.voxel_settings
    obj = vs.target
    if obj is not None and obj.name in context.scene.objects:
        return obj
    me = bpy.data.meshes.new("VoxelDraw")
    obj = bpy.data.objects.new("VoxelDraw", me)
    context.collection.objects.link(obj)
    obj["_vd_rev"] = 0
    vs.target = obj
    return obj


def place_first_voxel(context, obj, vs):
    """Empty session: move the object so voxel (0,0,0) sits centred on the 3D cursor
    (bottom face on it along the floor normal), and add it with the active colour."""
    half = [vs.voxel_size / 2] * 3
    half[AXIS_INDEX[vs.axis]] = 0.0
    obj.location = context.scene.cursor.location - obj.matrix_world.to_3x3() @ Vector(half)
    save_cells(obj, {(0, 0, 0): vs.color_index}, vs.voxel_size)


def region_under_mouse(area, event):
    """WINDOW region under the mouse, or None if the mouse is over sidebar/toolbar/header."""
    mx, my = event.mouse_x, event.mouse_y
    window = None
    for r in area.regions:
        inside = r.x <= mx < r.x + r.width and r.y <= my < r.y + r.height
        if not inside:
            continue
        if r.type == 'WINDOW':
            window = r
        elif r.width > 1 and r.height > 1:
            return None
    return window


def pick_end(context):
    _state["pick"] = False
    context.window.cursor_modal_restore()
    context.workspace.status_text_set(STATUS)
    tag_redraw_all(context)


def _cycle(items, value):
    ids = [s[0] for s in items]
    return ids[(ids.index(value) + 1) % len(ids)]


# ---------------------------------------------------------------- properties
def _redraw(self, context):
    tag_redraw_all(context)


def refresh_session(self, context):
    """Rebuild the session mesh from the stored cells (palette changed)."""
    obj = context.scene.voxel_settings.target
    if obj is not None and "_vd_cells" in obj:
        rebuild_mesh(obj, load_cells(obj), float(obj["_vd_size"]), palette_array(context.scene))
    tag_redraw_all(context)


class VoxelColor(bpy.types.PropertyGroup):
    color: bpy.props.FloatVectorProperty(
        name="Color", subtype='COLOR_GAMMA', size=3, min=0.0, max=1.0,
        update=refresh_session)


class VoxelSettings(bpy.types.PropertyGroup):
    voxel_size: bpy.props.FloatProperty(
        name="Voxel Size", default=1.0, min=0.001, precision=3, subtype='DISTANCE',
        description="Locked while the session contains voxels", update=_redraw)
    axis: bpy.props.EnumProperty(
        name="Floor", items=AXIS_ITEMS, default='XY', update=_redraw)
    z: bpy.props.FloatProperty(
        name="Offset", default=0.0, precision=3, subtype='DISTANCE',
        description="Floor plane offset along its normal (object space)", update=_redraw)
    show_floor: bpy.props.BoolProperty(name="Show Floor Grid", default=False, update=_redraw)
    show_bounds: bpy.props.BoolProperty(
        name="Show Bounds", default=True, update=_redraw,
        description="Draw the model bounding box (or the size limit box)")
    show_legend: bpy.props.BoolProperty(name="Show Legend", default=True, update=_redraw)
    tentacle: bpy.props.BoolProperty(
        name="Extrude Mode", default=False,
        description="Pick against voxels added during the same stroke "
                    "(original 'tentacle' behaviour)")
    mode: bpy.props.EnumProperty(name="Mode", items=MODE_ITEMS, default='ATTACH', update=_redraw)
    brush: bpy.props.EnumProperty(name="Brush", items=BRUSH_ITEMS, default='SHAPE', update=_redraw,
                                  description="Brush type (B cycles)")
    shape: bpy.props.EnumProperty(
        name="Shape", items=SHAPE_ITEMS, default='POINT', update=_redraw,
        description="Brush shape (Ctrl+RMB cycles)")
    brush_size: bpy.props.IntProperty(
        name="Size", default=1, min=1, max=256, soft_max=64, update=_redraw,
        description="Brush size in voxels (Ctrl+Wheel)")
    rotation: bpy.props.IntProperty(
        name="Rotation x90°", default=0, min=0, max=3, update=_redraw,
        description="Brush rotation, clockwise (Shift+LMB)")
    fill: bpy.props.BoolProperty(name="Fill", default=False, update=_redraw,
                                 description="Filled shapes / boxes instead of outlines (Ctrl+F)")
    mirror_live: bpy.props.BoolProperty(
        name="Live Mirror", default=False, update=_redraw,
        description="While drawing, every brush (Shape, Box, Line, Face, Fill) also "
                    "builds the mirrored part (M)")
    mirror: bpy.props.BoolVectorProperty(
        name="Mirror Axes", size=3, subtype='XYZ', default=(True, False, False), update=_redraw,
        description="Axes of the live mirror. The plane goes through the first voxel "
                    "(the one spawned on the 3D cursor), or through the model centre "
                    "when Limit Model Size is on")
    move_axis: bpy.props.EnumProperty(
        name="Axis", items=MOVE_AXIS_ITEMS, default='FREE', update=_redraw,
        description="Move mode: axis used when dragging outside the gizmo arrows")
    use_limit: bpy.props.BoolProperty(
        name="Limit Model Size", default=False, update=_redraw,
        description="Only add voxels inside 0..size on each axis, like a MagicaVoxel model")
    model_size: bpy.props.IntVectorProperty(
        name="Model Size", size=3, subtype='XYZ', default=(40, 40, 40), min=1, max=256,
        update=_redraw)
    color_index: bpy.props.IntProperty(name="Color", default=1, min=1, max=255, update=_redraw)
    replace_from: bpy.props.IntProperty(name="From", default=1, min=1, max=255,
                                        description="Palette index to replace with the active colour")
    palette: bpy.props.CollectionProperty(type=VoxelColor)
    greedy: bpy.props.BoolProperty(
        name="Optimize Mesh", default=True,
        description="On Confirm, merge same-colour coplanar faces into big quads "
                    "(greedy meshing, like MagicaVoxel export; leaves T-junctions)")
    target: bpy.props.PointerProperty(type=bpy.types.Object)


# ---------------------------------------------------------------- operators
class VOXELDRAW_OT_start(bpy.types.Operator):
    bl_idname = "voxeldraw.start"
    bl_label = "Start Voxel"
    bl_description = ("Start (or resume) a voxel session and enter Edit Mode. The shortcuts "
                      "are listed in the legend at the bottom left of the viewport. "
                      "Tab to Object Mode pauses, Tab back to Edit Mode resumes")

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == 'VIEW_3D'

    def invoke(self, context, event):
        global _drawing
        if _drawing:
            return {'CANCELLED'}
        vs = context.scene.voxel_settings
        ensure_palette(vs)
        obj = ensure_session_object(context)
        keep = history_valid(obj)  # e.g. Clear / Import done while no session ran
        if "_vd_cells" not in obj and not keep:  # new session (a cleared one stays undoable)
            place_first_voxel(context, obj, vs)
        ensure_material(obj)
        activate_object(context, obj)
        if obj.mode != 'EDIT':
            bpy.ops.object.mode_set(mode='EDIT')
        shading = context.space_data.shading
        if shading.type == 'SOLID':
            shading.color_type = 'VERTEX'  # show the voxel colours in Solid mode
        self.stroke = self.deleting = self.dirty = False
        self.changes = {}
        self.cells, self.bounds = {}, None
        self.snapshot, self.snap_bounds = set(), None
        self.drag = self.grab = self.sel_press = self.face_key = self.face_cells = None
        self.rev = -1
        self.size = 1.0
        self.last_build = 0.0
        self.editing = None  # forces the first mode transition
        _sel.clear()
        if not keep:
            _hist.update(undo=[], redo=[], rev=int(obj.get("_vd_rev", 0)), obj=obj.name)
        _state.update(area=context.area.as_pointer(), hover=None, size=1.0, float=None, rect=None)
        _drawing = True
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)
        tag_redraw_all(context)
        return {'RUNNING_MODAL'}

    # -- helpers
    def _sync(self, obj):
        rev = int(obj.get("_vd_rev", 0))
        if rev != self.rev:  # changed by a panel button, or externally (Blender undo, reload)
            self.cells = load_cells(obj)
            self.bounds = bounds_of(self.cells)
            self.rev = rev
            self.face_key = None
            _sel.intersection_update(self.cells)
            _state["bounds"] = self.bounds
            if not history_valid(obj):  # not one of our own changes: history is stale
                _hist.update(undo=[], redo=[], rev=rev, obj=obj.name)

    def _size_for(self, obj, vs):
        if self.cells and "_vd_size" in obj:
            return float(obj["_vd_size"])
        return vs.voxel_size

    def _rebuild(self, context, obj):
        rebuild_mesh(obj, self.cells, self.size, palette_array(context.scene))

    def _commit(self, context, obj, changes):
        commit(context.scene, obj, self.cells, changes, self.size)
        self.rev = int(obj.get("_vd_rev", 0))
        self.bounds = _state["bounds"] = bounds_of(self.cells)
        self.face_key = None

    def _mode(self, vs):
        if vs.mode in ('SELECT', 'MOVE'):
            return vs.mode
        return 'ERASE' if self.deleting else vs.mode

    def _set_editing(self, context, obj, editing):
        """Edit Mode = drawing, any other mode = paused."""
        self.editing = editing
        if editing:
            self._sync(obj)
            self.size = self._size_for(obj, context.scene.voxel_settings)
            self._rebuild(context, obj)  # cells are the source of truth
            _state["bounds"] = self.bounds
            msg = STATUS
        else:
            _state["hover"] = None
            msg = "VoxelDraw paused    Tab: back to Edit Mode to resume"
        context.workspace.status_text_set(msg)
        tag_redraw_all(context)

    def _ray(self, region, event, obj):
        pos = (event.mouse_x - region.x, event.mouse_y - region.y)
        rv3d = region.data
        origin = region_2d_to_origin_3d(region, rv3d, pos)
        direction = region_2d_to_vector_3d(region, rv3d, pos)
        inv = obj.matrix_world.inverted_safe()
        return inv @ origin, inv.to_3x3() @ direction

    def _hit(self, o, d):
        if not self.cells:
            return None
        return raycast_cells(o, d, self.cells, self.bounds, self.size)

    def _brush_cells(self, vs, mode, o, d):
        """Cells the brush would affect under the ray (the preview), or None."""
        if mode == 'MOVE':
            return None
        cells = self._stroke_cells(vs, mode, o, d)
        if not cells or mode == 'SELECT':
            return cells
        return mirrored(vs, cells)

    def _stroke_cells(self, vs, mode, o, d):
        """_brush_cells without the live mirror."""
        brush = vs.brush
        if mode == 'SELECT' or brush == 'FILL':
            hit = self._hit(o, d)
            return [hit[0]] if hit else None
        if brush == 'FACE':
            hit = self._hit(o, d)
            if hit is None:
                return None
            if (hit, mode) != self.face_key:
                cell, n = hit
                reg = face_region(self.cells, cell, n)
                if mode == 'ATTACH':
                    reg = [(c[0] + n[0], c[1] + n[1], c[2] + n[2]) for c in reg]
                self.face_key, self.face_cells = (hit, mode), list(reg)
            return self.face_cells
        if self.drag:
            start, axis = self.drag
            end = plane_cell(o, d, start, axis, self.size)
            if end is None:
                return _state["hover"]
            return line_cells(start, end) if brush == 'LINE' else box_cells(start, end, axis, vs.fill)
        else:
            snap = self.stroke and mode == 'ATTACH' and not vs.tentacle
            pool, bnds = (self.snapshot, self.snap_bounds) if snap else (self.cells, self.bounds)
            hit = pick_cell(o, d, pool, bnds, self.size, AXIS_INDEX[vs.axis], vs.z, mode != 'ATTACH')
            if hit is None:
                return None
            if brush in ('BOX', 'LINE'):
                return [hit[0]]
            return stamp_cells(*hit, vs.shape, vs.brush_size, vs.rotation, vs.fill)

    def _set_hover(self, context, cells, mode):
        if cells != _state["hover"] or self.size != _state["size"] or mode != _state["mode"]:
            _state.update(hover=cells, size=self.size, mode=mode)
            context.area.tag_redraw()

    def _apply(self, context, obj, vs, cells, mode):
        color = vs.color_index
        lim = tuple(vs.model_size) if vs.use_limit else None
        changed = False
        for cell in cells:
            old = self.cells.get(cell, 0)
            new = 0 if mode == 'ERASE' else color
            if old == new or (mode == 'ATTACH' and old) or (mode == 'PAINT' and not old):
                continue
            if lim and not old and not all(0 <= cell[i] < lim[i] for i in range(3)):
                continue
            self.changes.setdefault(cell, [old, new])[1] = new
            if new:
                self.cells[cell] = new
                self.bounds = grow(self.bounds, cell)
            else:
                del self.cells[cell]
            changed = True
        if not changed:
            return
        self.dirty = True
        self.face_key = None
        _state["bounds"] = self.bounds
        now = time.monotonic()
        if now - self.last_build > 0.03:  # throttle rebuilds during fast drags
            self._rebuild(context, obj)
            self.last_build = now

    def _step(self, context, region, event, obj, vs, deleting):
        o, d = self._ray(region, event, obj)
        mode = vs.mode if vs.mode in ('SELECT', 'MOVE') else ('ERASE' if deleting else vs.mode)
        if mode == 'MOVE':  # highlight the gizmo part under the mouse
            h = gizmo_hit(gizmo_screen(region, region.data, obj, _sel, self.size),
                          Vector((event.mouse_x - region.x, event.mouse_y - region.y)))
            if h != _state["gizmo_hover"]:
                _state["gizmo_hover"] = h
                context.area.tag_redraw()
        cells = self._brush_cells(vs, mode, o, d)
        self._set_hover(context, cells, mode)
        if self.stroke and cells and vs.brush == 'SHAPE' and mode != 'SELECT':
            self._apply(context, obj, vs, cells, mode)

    def _end_stroke(self, context, obj):
        self.stroke = False
        self.drag = None
        if not self.dirty:
            return
        self.dirty = False
        self._rebuild(context, obj)
        self.rev = save_cells(obj, self.cells, self.size)
        push_history(obj, self.changes, self.size, self.rev)

    def _history(self, context, obj, redo):
        self._sync(obj)
        src, dst = (_hist["redo"], _hist["undo"]) if redo else (_hist["undo"], _hist["redo"])
        if not src:
            return
        changes, size = src.pop()
        for cell, (old, new) in changes.items():
            v = new if redo else old
            if v:
                self.cells[cell] = v
            else:
                self.cells.pop(cell, None)
        dst.append((changes, size))
        self.bounds = _state["bounds"] = bounds_of(self.cells)
        self.size = size
        self.face_key = None
        _sel.intersection_update(self.cells)
        self._rebuild(context, obj)
        self.rev = save_cells(obj, self.cells, size)
        _hist["rev"] = self.rev
        context.area.tag_redraw()

    # -- mouse press / select
    def _press(self, context, region, event, obj, vs):
        self._sync(obj)
        self.size = self._size_for(obj, vs)
        if vs.mode == 'SELECT':
            op = 'ADD' if event.shift else 'SUB' if event.ctrl else 'SET'
            self.sel_press = (event.mouse_x - region.x, event.mouse_y - region.y, op, region)
            return
        if vs.mode == 'MOVE':  # arrow = that axis, centre = free, elsewhere = the panel axis
            floating = {c: self.cells[c] for c in _sel if c in self.cells}
            if floating:
                h = gizmo_hit(gizmo_screen(region, region.data, obj, floating, self.size),
                              Vector((event.mouse_x - region.x, event.mouse_y - region.y)))
                if h is None and vs.move_axis != 'FREE':
                    h = 'XYZ'.index(vs.move_axis)
                self._grab_start(context, region, event, obj, floating, set(floating),
                                 line=h if isinstance(h, int) else None, drag=True)
            return
        self.deleting = event.ctrl
        self.changes = {}
        self.snapshot, self.snap_bounds = set(self.cells), self.bounds
        self.stroke = True
        mode = self._mode(vs)
        o, d = self._ray(region, event, obj)
        if vs.brush in ('BOX', 'LINE'):
            self.drag = pick_cell(o, d, self.cells, self.bounds, self.size,
                                  AXIS_INDEX[vs.axis], vs.z, mode != 'ATTACH')
            self._step(context, region, event, obj, vs, self.deleting)
        elif vs.brush == 'FILL':
            hit = self._hit(o, d)
            if hit:
                region_cells = set()
                for start in mirrored(vs, [hit[0]]):
                    if start in self.cells and start not in region_cells:
                        region_cells |= flood(self.cells, start, same_color=True)
                self._apply(context, obj, vs, region_cells, 'ERASE' if mode == 'ERASE' else 'PAINT')
        elif vs.brush == 'FACE':
            cells = self._brush_cells(vs, mode, o, d)
            if cells:
                self._apply(context, obj, vs, cells, mode)
        else:
            self._step(context, region, event, obj, vs, self.deleting)

    def _select_end(self, context, event, obj):
        x0, y0, op, region = self.sel_press
        self.sel_press = None
        _state["rect"] = None
        self._sync(obj)
        x1, y1 = event.mouse_x - region.x, event.mouse_y - region.y
        if abs(x1 - x0) + abs(y1 - y0) > 4:
            mvp = region.data.perspective_matrix @ obj.matrix_world
            picked = cells_in_rect(self.cells, mvp, self.size, region.width, region.height,
                                   (x0, y0, x1, y1))
        else:
            hit = self._hit(*self._ray(region, event, obj))
            picked = {hit[0]} if hit else set()
        if op == 'SET':
            _sel.clear()
        if op == 'SUB':
            _sel.difference_update(picked)
        else:
            _sel.update(picked)
        tag_redraw_all(context)

    # -- grab (move / duplicate / paste)
    def _grab_start(self, context, region, event, obj, floating, remove, line=None, drag=False):
        """Float voxels with the mouse. line = axis lock; drag = confirm on LMB release."""
        if not floating:
            return
        a = view_axes(region.data, obj)[2][0]
        lo, hi = bounds_of(floating)
        centre = [(lo[i] + hi[i] + 1) / 2 * self.size for i in range(3)]
        self.grab = {"float": floating, "remove": remove, "axis": a, "plane": centre[a],
                     "centre": centre, "line": line, "drag": drag,
                     "start": None, "move": [0, 0, 0], "nudge": [0, 0, 0], "delta": (0, 0, 0)}
        _state.update(hover=None, line=line)
        self._grab_update(region, event, obj)
        context.workspace.status_text_set(
            ("Move: drag    release: confirm" if drag else
             "Move: mouse, Arrows, PgUp / PgDn    LMB / Enter: confirm") + "    RMB / Esc: cancel")
        tag_redraw_all(context)

    def _grab_update(self, region, event, obj):
        g = self.grab
        if region is not None and g["line"] is not None:
            o, d = self._ray(region, event, obj)
            s = ray_axis_param(o, d, g["centre"], g["line"])
            if s is not None:
                if g["start"] is None:
                    g["start"] = s
                g["move"] = [0, 0, 0]
                g["move"][g["line"]] = floor((s - g["start"]) / self.size + 0.5)
        elif region is not None:
            o, d = self._ray(region, event, obj)
            a = g["axis"]
            if abs(d[a]) > 1e-9:
                t = (g["plane"] - o[a]) / d[a]
                p = [o[i] + d[i] * t for i in range(3)]
                if g["start"] is None:
                    g["start"] = p
                g["move"] = [0 if i == a else floor((p[i] - g["start"][i]) / self.size + 0.5)
                             for i in range(3)]
        dx = g["delta"] = tuple(g["move"][i] + g["nudge"][i] for i in range(3))
        _state["float"] = [(c[0] + dx[0], c[1] + dx[1], c[2] + dx[2]) for c in g["float"]]

    def _grab_end(self, context, obj, ok):
        g, self.grab = self.grab, None
        _state.update(float=None, line=None)
        if ok:
            dx = g["delta"]
            moved = {(c[0] + dx[0], c[1] + dx[1], c[2] + dx[2]): k for c, k in g["float"].items()}
            new = {c: 0 for c in g["remove"]}
            new.update(moved)
            self._commit(context, obj, {c: [self.cells.get(c, 0), k] for c, k in new.items()
                                        if self.cells.get(c, 0) != k})
            _sel.clear()
            _sel.update(moved)
        context.workspace.status_text_set(STATUS)
        tag_redraw_all(context)

    def _grab_modal(self, context, event, obj):
        t, pressed = event.type, event.value == 'PRESS'
        region = region_under_mouse(context.area, event)
        if t == 'MOUSEMOVE':
            self._grab_update(region, event, obj)
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}
        if self.grab["drag"] and t == 'LEFTMOUSE' and event.value == 'RELEASE':
            self._grab_end(context, obj, True)
        elif pressed and t in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER', 'SPACE'}:
            self._grab_end(context, obj, True)
        elif pressed and t in {'RIGHTMOUSE', 'ESC'}:
            self._grab_end(context, obj, False)
        elif pressed and t in NUDGE and region is not None:
            which, sign = NUDGE[t]
            a, s = view_axes(region.data, obj)[which]
            self.grab["nudge"][a] += s * sign
            self._grab_update(None, event, obj)
            context.area.tag_redraw()
        elif t == 'MIDDLEMOUSE' or t.startswith(('WHEEL', 'NUMPAD', 'TRACKPAD', 'NDOF')):
            return {'PASS_THROUGH'}  # navigation keeps working
        return {'RUNNING_MODAL'}

    # -- keyboard
    def _key(self, context, region, event, obj, vs):
        """Keyboard shortcuts (PRESS). Returns True when handled."""
        t = event.type
        ctrl, shift, alt = event.ctrl or event.oskey, event.shift, event.alt
        if t in MODE_KEYS and not (ctrl or alt or shift):
            vs.mode = MODE_KEYS[t]
            return True
        if t == 'B' and not (ctrl or alt):
            vs.brush = _cycle(BRUSH_ITEMS, vs.brush)
            return True
        if t == 'H' and not (ctrl or alt):
            vs.show_legend = not vs.show_legend
            return True
        if t == 'F' and ctrl:
            vs.fill = not vs.fill
            return True
        if t == 'M' and not (ctrl or alt or shift):
            vs.mirror_live = not vs.mirror_live
            return True
        if t in AXIS_KEYS and vs.mode == 'MOVE' and not (ctrl or alt or shift):
            vs.move_axis = 'FREE' if vs.move_axis == t else t
            return True
        if self.stroke or self.sel_press:
            return False
        self._sync(obj)
        self.size = self._size_for(obj, vs)
        o, d = self._ray(region, event, obj)
        if t == 'A' and not ctrl:
            selection_action(vs, self.cells, 'NONE' if alt else 'ALL')
            return True
        if t in ('L', 'C') and not (ctrl or alt):
            hit = self._hit(o, d)
            if hit:
                if not shift:
                    _sel.clear()
                if t == 'L':
                    _sel.update(flood(self.cells, hit[0]))
                else:
                    col = self.cells[hit[0]]
                    _sel.update(c for c, k in self.cells.items() if k == col)
            return True
        if t == 'C' and ctrl:
            selection_action(vs, self.cells, 'COPY')
            return True
        if t == 'V' and ctrl:
            if _clip:
                hit = pick_cell(o, d, self.cells, self.bounds, self.size,
                                AXIS_INDEX[vs.axis], vs.z, False)
                b = hit[0] if hit else (0, 0, 0)
                floating = {(b[0] + c[0], b[1] + c[1], b[2] + c[2]): k for c, k in _clip.items()}
                self._grab_start(context, region, event, obj, floating, set())
            return True
        if (t == 'G' and not (ctrl or alt or shift)) or (t == 'D' and shift and not ctrl):
            floating = {c: self.cells[c] for c in _sel if c in self.cells}
            self._grab_start(context, region, event, obj, floating,
                             set() if t == 'D' else set(floating))
            return True
        axes = view_axes(region.data, obj)
        if t == 'R' and not (ctrl or alt):  # clockwise as seen on screen
            changes = selection_action(vs, self.cells, 'ROTATE', axes[2][0], ccw=axes[2][1] < 0)
        elif t == 'F' and not alt:
            changes = selection_action(vs, self.cells, 'FLIP', axes[1 if shift else 0][0])
        elif t in NUDGE and not (ctrl or alt):
            which, sign = NUDGE[t]
            a, s = axes[which]
            delta = [0, 0, 0]
            delta[a] = s * sign
            changes = selection_action(vs, self.cells, 'MOVE', delta=delta)
        elif t in ('DEL', 'X') and not ctrl:
            changes = selection_action(vs, self.cells, 'DELETE')
        elif t == 'P' and not ctrl:
            changes = selection_action(vs, self.cells, 'PAINT')
        else:
            return False
        self._commit(context, obj, changes)
        return True

    # -- eyedropper (palette button): next click on a voxel picks its colour
    def _pick_modal(self, context, event, obj, vs):
        t, pressed = event.type, event.value == 'PRESS'
        if pressed and t in {'RIGHTMOUSE', 'ESC'}:
            pick_end(context)
        elif pressed and t == 'LEFTMOUSE':
            region = region_under_mouse(context.area, event)
            if region is None:
                return {'PASS_THROUGH'}
            self._sync(obj)
            hit = self._hit(*self._ray(region, event, obj))
            if hit:
                vs.color_index = self.cells[hit[0]]
            pick_end(context)
        elif t in {'MOUSEMOVE', 'MIDDLEMOUSE'} or t.startswith(('WHEEL', 'NUMPAD', 'TRACKPAD', 'NDOF')):
            return {'PASS_THROUGH'}
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        global _drawing
        obj = context.scene.voxel_settings.target
        if _state["pick"]:
            pick_end(context)
        if self.grab is not None:
            self.grab = None
            _state["float"] = None
        if self.stroke and obj is not None:
            self._end_stroke(context, obj)
        _drawing = False
        _state.update(hover=None, rect=None)
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)
        tag_redraw_all(context)
        return {'FINISHED'}

    def cancel(self, context):
        if self._timer is not None:
            self._finish(context)

    # -- modal
    def modal(self, context, event):
        if not _drawing:  # Confirm pressed
            return self._finish(context)
        vs = context.scene.voxel_settings
        obj = vs.target
        if obj is None:
            return self._finish(context)

        editing = obj.mode == 'EDIT'
        if editing != self.editing:  # Tab between Edit / Object = resume / pause
            if _state["pick"]:
                pick_end(context)
            if self.grab is not None:
                self._grab_end(context, obj, False)
            if self.stroke:
                self._end_stroke(context, obj)
            self.sel_press = None
            _state["rect"] = None
            self._set_editing(context, obj, editing)
        if not editing or event.type == 'TIMER':
            return {'PASS_THROUGH'}

        if self.grab is not None:
            return self._grab_modal(context, event, obj)
        if _state["pick"]:
            return self._pick_modal(context, event, obj, vs)

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            if self.sel_press:
                self._select_end(context, event, obj)
                return {'RUNNING_MODAL'}
            if self.stroke:
                if self.drag and _state["hover"]:
                    self._apply(context, obj, vs, _state["hover"], self._mode(vs))
                self._end_stroke(context, obj)
                return {'RUNNING_MODAL'}

        # Checked before the region test: Blender's own undo (e.g. with the mouse
        # over the sidebar) would desync the mesh from the cells.
        if (event.type == 'Z' and event.value == 'PRESS'
                and (event.ctrl or event.oskey) and not event.alt):
            if not self.stroke:
                self._history(context, obj, redo=event.shift)
            return {'RUNNING_MODAL'}

        region = region_under_mouse(context.area, event)
        if region is None:  # over sidebar/toolbar/header/other editors: let them work
            if event.type == 'MOUSEMOVE' and _state["hover"] is not None:
                _state["hover"] = None
                context.area.tag_redraw()
            return {'PASS_THROUGH'}

        if (event.value == 'PRESS' and not event.type.endswith('MOUSE')
                and self._key(context, region, event, obj, vs)):
            tag_redraw_all(context)
            return {'RUNNING_MODAL'}

        brush_key = False
        if event.type == 'RIGHTMOUSE' and event.ctrl:
            if event.value == 'PRESS':
                vs.shape = _cycle(SHAPE_ITEMS, vs.shape)
            brush_key = True
        elif event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'} and event.ctrl:
            step = 1 if event.type == 'WHEELUPMOUSE' else -1
            vs.brush_size = max(1, min(256, vs.brush_size + step))
            brush_key = True
        elif (event.type == 'LEFTMOUSE' and event.shift and event.value == 'PRESS'
              and not self.stroke and vs.mode not in ('SELECT', 'MOVE')):
            vs.rotation = (vs.rotation + 1) % 4
            brush_key = True
        if brush_key:
            if not self.stroke:
                self._step(context, region, event, obj, vs, False)  # refresh the preview
            tag_redraw_all(context)
            return {'RUNNING_MODAL'}

        if event.type == 'RIGHTMOUSE':  # pick colour / Shift: replace that colour everywhere
            if event.value == 'PRESS' and not self.stroke:
                self._sync(obj)
                hit = self._hit(*self._ray(region, event, obj))
                if hit:
                    src, dst = self.cells[hit[0]], vs.color_index
                    if not event.shift:
                        vs.color_index = src
                    elif src != dst:
                        self._commit(context, obj, {c: [src, dst] for c, k in self.cells.items()
                                                    if k == src})
                tag_redraw_all(context)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.alt:  # Alt+LMB = emulated MMB navigation
                return {'PASS_THROUGH'}
            if event.value == 'PRESS':
                self._press(context, region, event, obj, vs)
            return {'RUNNING_MODAL'}

        if event.type == 'MOUSEMOVE':
            if self.sel_press:
                x0, y0, _op, r = self.sel_press
                _state["rect"] = (x0, y0, event.mouse_x - r.x, event.mouse_y - r.y)
                context.area.tag_redraw()
                return {'PASS_THROUGH'}
            self._sync(obj)
            if not self.stroke:
                self.size = self._size_for(obj, vs)
            deleting = self.deleting if self.stroke else event.ctrl
            self._step(context, region, event, obj, vs, deleting)
        return {'PASS_THROUGH'}


class _SessionOp:
    @classmethod
    def poll(cls, context):
        return context.scene.voxel_settings.target is not None


class VOXELDRAW_OT_clear(_SessionOp, bpy.types.Operator):
    bl_idname = "voxeldraw.clear"
    bl_label = "Clear"
    bl_description = "Delete all voxels of the current session (Ctrl+Z while drawing restores them)"

    def execute(self, context):
        vs = context.scene.voxel_settings
        obj = vs.target
        cells = load_cells(obj)
        commit(context.scene, obj, cells, {c: [k, 0] for c, k in cells.items()}, session_size(obj, vs))
        _sel.clear()
        tag_redraw_all(context)
        return {'FINISHED'}


class VOXELDRAW_OT_selection(_SessionOp, bpy.types.Operator):
    bl_idname = "voxeldraw.selection"
    bl_label = "Selection"
    bl_description = "Apply a command to the selected voxels"
    action: bpy.props.EnumProperty(items=[
        ("ALL", "Select All", ""), ("NONE", "Select None", ""), ("DELETE", "Delete", ""),
        ("PAINT", "Paint", "Recolour the selection with the active colour"),
        ("COPY", "Copy", ""), ("ROTATE", "Rotate 90°", ""), ("FLIP", "Flip", ""),
        ("MIRROR", "Mirror", "Mirrored copy across the plane of the first voxel")])
    axis: bpy.props.IntProperty(min=0, max=2)

    def execute(self, context):
        vs = context.scene.voxel_settings
        obj = vs.target
        cells = load_cells(obj)
        changes = selection_action(vs, cells, self.action, self.axis, ccw=True)
        commit(context.scene, obj, cells, changes, session_size(obj, vs))
        tag_redraw_all(context)
        return {'FINISHED'}


class VOXELDRAW_OT_replace_color(_SessionOp, bpy.types.Operator):
    bl_idname = "voxeldraw.replace_color"
    bl_label = "Replace"
    bl_description = "Recolour every voxel of the 'From' index with the active colour (Shift+RMB)"

    def execute(self, context):
        vs = context.scene.voxel_settings
        obj = vs.target
        src, dst = vs.replace_from, vs.color_index
        cells = load_cells(obj)
        if src != dst:
            commit(context.scene, obj, cells, {c: [src, dst] for c, k in cells.items() if k == src},
                   session_size(obj, vs))
        tag_redraw_all(context)
        return {'FINISHED'}


class VOXELDRAW_OT_eyedropper(bpy.types.Operator):
    bl_idname = "voxeldraw.eyedropper"
    bl_label = "Pick Colour"
    bl_description = "Click a voxel to make its colour the active one (also RMB while drawing)"

    @classmethod
    def poll(cls, context):
        return _drawing

    def execute(self, context):
        _state["pick"] = True
        context.window.cursor_modal_set('EYEDROPPER')
        context.workspace.status_text_set("Click a voxel to pick its colour    RMB / Esc: cancel")
        return {'FINISHED'}


class VOXELDRAW_OT_palette_add(bpy.types.Operator):
    bl_idname = "voxeldraw.palette_add"
    bl_label = "Add Colour"
    bl_description = "Add a colour at the end of the palette (copy of the active one)"

    @classmethod
    def poll(cls, context):
        return 1 < len(context.scene.voxel_settings.palette) < 256

    def execute(self, context):
        vs = context.scene.voxel_settings
        rgb = tuple(vs.palette[min(vs.color_index, len(vs.palette) - 1)].color)
        vs.palette.add()["color"] = rgb
        vs.color_index = len(vs.palette) - 1
        return {'FINISHED'}


class VOXELDRAW_OT_palette_remove(bpy.types.Operator):
    bl_idname = "voxeldraw.palette_remove"
    bl_label = "Remove Colour"
    bl_description = "Remove this colour from the palette (the following ones shift up)"
    index: bpy.props.IntProperty(min=1)

    def execute(self, context):
        vs = context.scene.voxel_settings
        k = self.index
        if not 0 < k < len(vs.palette) or len(vs.palette) <= 2:
            self.report({'ERROR'}, "The palette needs at least one colour")
            return {'CANCELLED'}
        obj = vs.target
        cells = load_cells(obj) if obj is not None else {}
        used = sum(1 for v in cells.values() if v == k)
        if used:
            self.report({'ERROR'}, f"Colour {k} is used by {used} voxels: recolour them first "
                                   "(Replace, Paint or the Fill brush)")
            return {'CANCELLED'}
        vs.palette.remove(k)
        if cells:  # indices above k shift down; old undo steps would point to the wrong colours
            rev = save_cells(obj, remap_removed_color(cells, k), session_size(obj, vs))
            _hist.update(undo=[], redo=[], rev=rev, obj=obj.name)
            refresh_session(None, context)
        new_clip = remap_removed_color(_clip, k)
        _clip.clear()
        _clip.update(new_clip)
        vs.color_index = max(1, min(vs.color_index - (vs.color_index > k), len(vs.palette) - 1))
        if vs.replace_from >= len(vs.palette):
            vs.replace_from = len(vs.palette) - 1
        tag_redraw_all(context)
        return {'FINISHED'}


class VOXELDRAW_OT_reset_palette(bpy.types.Operator):
    bl_idname = "voxeldraw.reset_palette"
    bl_label = "Default Palette"
    bl_description = "Restore the MagicaVoxel default palette"

    def execute(self, context):
        set_palette(context.scene.voxel_settings, default_palette())
        refresh_session(None, context)
        return {'FINISHED'}


class VOXELDRAW_OT_load_palette(bpy.types.Operator, ImportHelper):
    bl_idname = "voxeldraw.load_palette"
    bl_label = "Load Palette"
    bl_description = "Load the palette of a .vox file, or of an image (256 x 1 MagicaVoxel palette PNG)"
    filter_glob: bpy.props.StringProperty(default="*.vox;*.png;*.bmp;*.tga;*.jpg", options={'HIDDEN'})

    def execute(self, context):
        try:
            if self.filepath.lower().endswith(".vox"):
                with open(self.filepath, "rb") as f:
                    pal = read_vox(f.read())[1]
                if pal is None:
                    raise ValueError("The file has no palette")
            else:
                img = bpy.data.images.load(self.filepath, check_existing=False)
                px = np.empty(len(img.pixels), np.float32)
                img.pixels.foreach_get(px)
                row = px.reshape(img.size[1], img.size[0], 4)[-1, :255, :3]  # top row
                bpy.data.images.remove(img)
                pal = [(0.0, 0.0, 0.0)] + [tuple(c) for c in row.tolist()]
                pal += default_palette()[len(pal):]
        except (OSError, ValueError, RuntimeError) as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        vs = context.scene.voxel_settings
        ensure_palette(vs)
        set_palette(vs, pal)
        refresh_session(None, context)
        return {'FINISHED'}


class VOXELDRAW_OT_import_vox(bpy.types.Operator, ImportHelper):
    bl_idname = "voxeldraw.import_vox"
    bl_label = "Import .vox"
    bl_description = "Load a MagicaVoxel .vox (voxels + palette) into the session, replacing it"
    filter_glob: bpy.props.StringProperty(default="*.vox", options={'HIDDEN'})

    def execute(self, context):
        try:
            with open(self.filepath, "rb") as f:
                new, pal = read_vox(f.read())
        except (OSError, ValueError) as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        if not new:  # would silently wipe the session
            self.report({'ERROR'}, "The file contains no voxels")
            return {'CANCELLED'}
        scene = context.scene
        vs = scene.voxel_settings
        ensure_palette(vs)
        if pal is not None:
            set_palette(vs, pal)
        if vs.target is None and context.view_layer.objects.active is not None \
                and context.view_layer.objects.active.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        obj = ensure_session_object(context)
        ensure_material(obj)
        old = load_cells(obj)
        changes = {c: [old.get(c, 0), k] for c, k in new.items() if old.get(c, 0) != k}
        changes.update({c: [k, 0] for c, k in old.items() if c not in new})
        refresh_session(None, context)  # new palette on the current voxels
        commit(scene, obj, old, changes, session_size(obj, vs))
        _sel.clear()
        tag_redraw_all(context)
        self.report({'INFO'}, f"Imported {len(new)} voxels")
        return {'FINISHED'}


class VOXELDRAW_OT_export_vox(_SessionOp, bpy.types.Operator, ExportHelper):
    bl_idname = "voxeldraw.export_vox"
    bl_label = "Export .vox"
    bl_description = "Save the session as a MagicaVoxel .vox file (max 256 per axis)"
    filename_ext = ".vox"
    filter_glob: bpy.props.StringProperty(default="*.vox", options={'HIDDEN'})

    def execute(self, context):
        cells = load_cells(context.scene.voxel_settings.target)
        if not cells:
            self.report({'ERROR'}, "Nothing to export")
            return {'CANCELLED'}
        try:
            data = vox_bytes(cells, palette_array(context.scene))
            with open(self.filepath, "wb") as f:
                f.write(data)
        except (OSError, ValueError) as e:
            self.report({'ERROR'}, str(e))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Exported {len(cells)} voxels")
        return {'FINISHED'}


class VOXELDRAW_OT_confirm(_SessionOp, bpy.types.Operator):
    bl_idname = "voxeldraw.confirm"
    bl_label = "Confirm"
    bl_description = "Finish the session: bake the voxels into a single mesh"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global _drawing
        vs = context.scene.voxel_settings
        obj = vs.target
        if obj.mode != 'OBJECT' and context.view_layer.objects.active != obj:
            self.report({'ERROR'}, "Leave Edit Mode first")
            return {'CANCELLED'}  # before touching anything: the session goes on
        _drawing = False  # the modal operator notices and stops
        _hist.update(undo=[], redo=[], rev=None, obj=None)
        _sel.clear()
        if obj.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        tmp = bpy.data.meshes.get(TMP_MESH)
        if tmp is not None:
            bpy.data.meshes.remove(tmp)

        cells = load_cells(obj)
        me = obj.data
        if not cells:  # nothing drawn: discard the empty session object
            bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.meshes.remove(me)
            tag_redraw_all(context)
            self.report({'INFO'}, "Nothing drawn, session discarded")
            return {'FINISHED'}

        size = float(obj.get("_vd_size", vs.voxel_size))
        rebuild_mesh(obj, cells, size, palette_array(context.scene), greedy=vs.greedy)

        for key in ("_vd_cells", "_vd_cols", "_vd_size", "_vd_rev"):
            obj.pop(key, None)
        vs.target = None
        tag_redraw_all(context)
        self.report({'INFO'}, f"Voxel mesh confirmed ({len(me.polygons)} faces)")
        return {'FINISHED'}


# ---------------------------------------------------------------- UI
class VOXELDRAW_UL_palette(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # number | name (double-click to rename) | colour | X remove
        row = layout.row(align=True)
        num = row.row()
        num.ui_units_x = 1.5
        num.label(text=str(index))
        row.prop(item, "name", text="", emboss=False)
        swatch = row.row()
        swatch.ui_units_x = 3
        swatch.prop(item, "color", text="")
        row.operator("voxeldraw.palette_remove", text="", icon='X', emboss=False).index = index

    def filter_items(self, context, data, propname):
        flags = [self.bitflag_filter_item] * len(getattr(data, propname))
        if flags:
            flags[0] = 0  # index 0 = empty voxel
        return flags, []


class _Panel:
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "VoxelDraw"


class VOXELDRAW_PT_panel(_Panel, bpy.types.Panel):
    bl_label = "VoxelDraw"
    bl_idname = "VOXELDRAW_PT_panel"

    def draw(self, context):
        layout = self.layout
        vs = context.scene.voxel_settings
        obj = vs.target
        has_cells = obj is not None and "_vd_cells" in obj

        col = layout.column(align=True)
        row = col.row(align=True)
        row.scale_y = 1.4
        row.enabled = not _drawing
        row.operator("voxeldraw.start", text="Start Voxel", icon='PLAY')
        col.operator("voxeldraw.clear", text="Clear", icon='TRASH')
        col.operator("voxeldraw.confirm", text="Confirm", icon='CHECKMARK')
        col.prop(vs, "greedy")
        if obj is not None:
            if _drawing and obj.mode == 'EDIT':
                layout.label(text="Drawing  (Tab = pause)", icon='INFO')
            elif _drawing:
                layout.label(text="Paused  (Edit Mode = resume)", icon='INFO')
            else:
                layout.label(text="Session open: Start Voxel to resume", icon='INFO')

        col = layout.column(align=True)
        sub = col.column()
        sub.enabled = not has_cells
        sub.prop(vs, "voxel_size")
        col.prop(vs, "axis")
        col.prop(vs, "z")
        col.prop(vs, "show_floor")
        col.prop(vs, "show_bounds")
        col.prop(vs, "show_legend")
        col.prop(vs, "tentacle")


class VOXELDRAW_PT_brush(_Panel, bpy.types.Panel):
    bl_label = "Brush"
    bl_parent_id = "VOXELDRAW_PT_panel"

    def draw(self, context):
        layout = self.layout
        vs = context.scene.voxel_settings
        layout.row().prop(vs, "mode", expand=True)
        if vs.mode == 'MOVE':
            layout.row().prop(vs, "move_axis", expand=True)
            if not _sel:
                layout.label(text="Select voxels first (Select mode)", icon='INFO')
        layout.row().prop(vs, "brush", expand=True)
        col = layout.column(align=True)
        col.enabled = vs.brush == 'SHAPE'
        col.prop(vs, "shape")
        col.prop(vs, "brush_size")
        col.prop(vs, "rotation")
        layout.prop(vs, "fill")
        row = layout.row(align=True)
        row.prop(vs, "mirror_live", text="Live Mirror", toggle=True, icon='MOD_MIRROR')
        axes = row.row(align=True)
        axes.active = vs.mirror_live
        axes.prop(vs, "mirror", text="", toggle=True)
        layout.prop(vs, "use_limit")
        col = layout.column(align=True)
        col.enabled = vs.use_limit
        col.prop(vs, "model_size", text="")


class VOXELDRAW_PT_selection(_Panel, bpy.types.Panel):
    bl_label = "Selection"
    bl_parent_id = "VOXELDRAW_PT_panel"

    def draw(self, context):
        layout = self.layout
        layout.label(text=f"{len(_sel)} selected    clipboard: {len(_clip)}")
        row = layout.row(align=True)
        row.operator("voxeldraw.selection", text="All").action = 'ALL'
        row.operator("voxeldraw.selection", text="None").action = 'NONE'
        for action, label in (("MIRROR", "Mirror"), ("ROTATE", "Rotate"), ("FLIP", "Flip")):
            row = layout.row(align=True)
            row.label(text=label)
            for axis, name in enumerate("XYZ"):
                op = row.operator("voxeldraw.selection", text=name)
                op.action, op.axis = action, axis
        row = layout.row(align=True)
        row.operator("voxeldraw.selection", text="Delete", icon='TRASH').action = 'DELETE'
        row.operator("voxeldraw.selection", text="Paint", icon='BRUSH_DATA').action = 'PAINT'
        row.operator("voxeldraw.selection", text="Copy", icon='COPYDOWN').action = 'COPY'


class VOXELDRAW_PT_palette(_Panel, bpy.types.Panel):
    bl_label = "Palette"
    bl_parent_id = "VOXELDRAW_PT_panel"

    def draw(self, context):
        layout = self.layout
        vs = context.scene.voxel_settings
        if len(vs.palette) < 2:
            layout.label(text="Created by Start Voxel", icon='COLOR')
            layout.operator("voxeldraw.load_palette", icon='FILEBROWSER')
            return
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.operator("voxeldraw.eyedropper", text="", icon='EYEDROPPER')
        if vs.color_index < len(vs.palette):
            item = vs.palette[vs.color_index]
            row.label(text=f"{vs.color_index}  {item.name}")
            row.prop(item, "color", text="")
        row.operator("voxeldraw.palette_add", text="", icon='ADD')
        layout.template_list("VOXELDRAW_UL_palette", "", vs, "palette", vs, "color_index", rows=8)
        row = layout.row(align=True)
        row.prop(vs, "replace_from")
        row.operator("voxeldraw.replace_color", text="→ active")
        row = layout.row(align=True)
        row.operator("voxeldraw.load_palette", text="Load", icon='FILEBROWSER')
        row.operator("voxeldraw.reset_palette", text="Default", icon='LOOP_BACK')


class VOXELDRAW_PT_file(_Panel, bpy.types.Panel):
    bl_label = "MagicaVoxel"
    bl_parent_id = "VOXELDRAW_PT_panel"

    def draw(self, context):
        row = self.layout.row(align=True)
        row.operator("voxeldraw.import_vox", icon='IMPORT')
        row.operator("voxeldraw.export_vox", icon='EXPORT')


# ---------------------------------------------------------------- viewport preview
def _lines(shader, pts, color, idx=None):
    batch = batch_for_shader(shader, 'LINES', {"pos": pts}, indices=idx)
    shader.uniform_float("color", color)
    batch.draw(shader)


def _draw_cubes(shader, cells, size, color):
    p = np.array(list(cells), np.float32).reshape(-1, 3)
    if len(p):
        pts = ((p[:, None, :] + CUBE_CORNERS) * size).reshape(-1, 3)
        idx = (np.arange(len(p), dtype=np.int32)[:, None, None] * 8 + CUBE_LINES).reshape(-1, 2)
        _lines(shader, pts, color, idx)


def _toward_viewer(rv3d, obj, pts, amount):
    """Move object-space points `amount` toward the viewer: a depth bias, so ghost
    edges lying on voxel edges are not half hidden by the faces next to them."""
    inv = obj.matrix_world.inverted_safe()
    if rv3d.is_perspective:
        d = np.array(inv @ rv3d.view_matrix.inverted().translation, np.float32) - pts
    else:
        d = np.broadcast_to(np.array(inv.to_3x3() @ (rv3d.view_rotation @ Vector((0, 0, 1))),
                                     np.float32), pts.shape)
    n = np.linalg.norm(d, axis=1)[:, None]
    return pts + d / np.maximum(n, 1e-9) * amount


def _draw_ghost(rv3d, obj, cells, size, color, xray=False):
    """Voxels about to be placed / erased / painted / moved: translucent fill,
    see-through hint, thick outline with a contrasting halo."""
    tris, segs = ghost_geometry(cells)
    if not len(segs):
        return
    segs = _toward_viewer(rv3d, obj, segs * size, 0.02 * size)
    flat = gpu.shader.from_builtin('UNIFORM_COLOR')
    flat.bind()
    gpu.state.depth_test_set('NONE')
    _lines(flat, segs, (*color[:3], 0.25))  # parts hidden behind voxels
    if not xray:
        gpu.state.depth_test_set('LESS_EQUAL')
    batch = batch_for_shader(flat, 'TRIS', {"pos": tris * size})
    flat.uniform_float("color", (*color[:3], 0.18))
    batch.draw(flat)

    s = bpy.context.preferences.system.ui_scale
    dark = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2] < 0.3
    wide = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
    wide.bind()
    wide.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    batch = batch_for_shader(wide, 'LINES', {"pos": segs})
    for width, col in ((5.0, (1, 1, 1, 0.6) if dark else (0, 0, 0, 0.6)), (2.5, (*color[:3], 1))):
        wide.uniform_float("lineWidth", width * s)
        wide.uniform_float("color", col)
        batch.draw(wide)
    gpu.state.depth_test_set('LESS_EQUAL')


def _draw_box(shader, lo, hi, size, color):
    lo = np.array(lo, np.float32)
    _lines(shader, (lo + CUBE_CORNERS * (np.array(hi, np.float32) - lo)) * size, color, CUBE_LINES)


def _floor_grid(vs, size):
    """Grid lines + quad on the floor plane: over the model limit, else 64 x 64 voxels."""
    a = AXIS_INDEX[vs.axis]
    b, c = [i for i in range(3) if i != a]
    if vs.use_limit:
        (u0, u1), (v0, v1) = (0, vs.model_size[b]), (0, vs.model_size[c])
    else:
        (u0, u1), (v0, v1) = (-32, 32), (-32, 32)

    def pt(u, v):
        p = [0.0, 0.0, 0.0]
        p[a], p[b], p[c] = vs.z, u * size, v * size
        return p
    lines = [pt(u, v) for u in range(u0, u1 + 1) for v in (v0, v1)]
    lines += [pt(u, v) for v in range(v0, v1 + 1) for u in (u0, u1)]
    return lines, [pt(u0, v0), pt(u1, v0), pt(u1, v1), pt(u0, v1)]


def draw_preview():
    if not _drawing:
        return
    context = bpy.context
    area = context.area
    if area is None or area.as_pointer() != _state["area"]:
        return
    vs = context.scene.voxel_settings
    obj = vs.target
    if obj is None or obj.mode != 'EDIT':
        return
    size = _state["size"]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')

    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.depth_mask_set(False)
    gpu.matrix.push()
    gpu.matrix.multiply_matrix(obj.matrix_world)
    shader.bind()

    if vs.show_floor:
        lines, quad = _floor_grid(vs, size)
        batch = batch_for_shader(shader, 'TRIS', {"pos": quad}, indices=((0, 1, 2), (2, 3, 0)))
        shader.uniform_float("color", (1, 1, 0, 0.04))
        batch.draw(shader)
        _lines(shader, lines, (1, 1, 1, 0.12))
    if vs.show_bounds:
        if vs.use_limit:
            _draw_box(shader, (0, 0, 0), tuple(vs.model_size), size, (1, 1, 1, 0.35))
        elif _state["bounds"]:
            lo, hi = _state["bounds"]
            _draw_box(shader, lo, [h + 1 for h in hi], size, (1, 1, 1, 0.2))

    for a, corners in mirror_planes(vs, _state["bounds"]):
        quad = np.array(corners, np.float32) * size
        batch = batch_for_shader(shader, 'TRIS', {"pos": quad}, indices=((0, 1, 2), (2, 3, 0)))
        shader.uniform_float("color", (*AXIS_COLORS[a][:3], 0.08))
        batch.draw(shader)
        _lines(shader, quad, (*AXIS_COLORS[a][:3], 0.6), ((0, 1), (1, 2), (2, 3), (3, 0)))

    if _sel:  # placed voxels: thin outline
        _draw_cubes(shader, _sel, size, (1, 0.55, 0, 1))
    rv3d = context.region_data
    cells = _state["hover"]
    if cells:
        if _state["mode"] == 'ERASE':
            color = (1, 0.1, 0.1, 1)
        elif _state["mode"] == 'SELECT' or vs.color_index >= len(vs.palette):
            color = (1, 1, 1, 1)
        else:
            color = (*vs.palette[vs.color_index].color, 1)
        _draw_ghost(rv3d, obj, cells, size, color)
    if _state["float"]:  # moving voxels stay visible through the model
        _draw_ghost(rv3d, obj, _state["float"], size, (0.2, 0.9, 1, 1), xray=True)

    gpu.matrix.pop()
    gpu.state.depth_mask_set(True)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def legend_lines(vs):
    shape = f"{vs.shape.title()} {vs.brush_size}" if vs.brush == 'SHAPE' else vs.brush.title()
    mirror = "".join("XYZ"[a] for a in live_mirror_axes(vs)) or "off"
    mode = f"Move {vs.move_axis.title()}" if vs.mode == 'MOVE' else vs.mode.title()
    head = (f"{mode}  |  {shape}{'  fill' if vs.fill else ''}  |  Mirror {mirror}  |  "
            f"Colour {vs.color_index}  |  Selected {len(_sel)}")
    return (head,) + LEGEND


def _draw_gizmo(region, rv3d, obj, vs):
    """Move-mode gizmo: one arrow per axis + a centre square, in region pixels."""
    cells = _state["float"] or _sel
    giz = gizmo_screen(region, rv3d, obj, cells, _state["size"])
    if giz is None:
        return
    c2, ends = giz
    s = bpy.context.preferences.system.ui_scale
    hot = _state["line"] if _state["float"] else _state["gizmo_hover"]
    locked = None if vs.move_axis == 'FREE' else 'XYZ'.index(vs.move_axis)
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    shader.bind()

    def tris(pts, color):
        batch = batch_for_shader(shader, 'TRIS', {"pos": [(x, y, 0) for x, y in pts]})
        shader.uniform_float("color", color)
        batch.draw(shader)
    for a, e in ends.items():
        col = AXIS_COLORS[a]
        if a == hot or a == locked:
            col = tuple(min(1.0, v + 0.35) for v in col[:3]) + (1,)
        d = (e - c2).normalized()
        n = Vector((-d.y, d.x)) * (2.5 if a == hot else 1.5) * s
        p, q = c2 + d * 14 * s, e
        tris([p - n, q - n, q + n, p - n, q + n, p + n], col)
        w = Vector((-d.y, d.x)) * 7 * s
        tris([e + d * 14 * s, e - w, e + w], col)
    h = (7 if hot == 'C' else 5) * s
    c = (1, 1, 1, 1) if hot == 'C' else (0.85, 0.85, 0.85, 0.9)
    tris([(c2.x - h, c2.y - h), (c2.x + h, c2.y - h), (c2.x + h, c2.y + h),
          (c2.x - h, c2.y - h), (c2.x + h, c2.y + h), (c2.x - h, c2.y + h)], c)


def draw_overlay():
    if not _drawing:
        return
    context = bpy.context
    area = context.area
    if area is None or area.as_pointer() != _state["area"]:
        return
    vs = context.scene.voxel_settings
    obj = vs.target
    if obj is None or obj.mode != 'EDIT':
        return
    rect = _state["rect"]
    if rect:
        x0, y0, x1, y1 = rect
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": [
            (x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0), (x0, y0, 0)]})
        shader.bind()
        shader.uniform_float("color", (1, 1, 1, 0.9))
        batch.draw(shader)
    if vs.mode == 'MOVE':
        gpu.state.blend_set('ALPHA')
        _draw_gizmo(context.region, context.region_data, obj, vs)
        gpu.state.blend_set('NONE')
    if not vs.show_legend:
        return
    scale = context.preferences.system.ui_scale
    x = 12 * scale + sum(r.width for r in area.regions if r.type == 'TOOLS')
    blf.size(0, 11 * scale)
    blf.enable(0, blf.SHADOW)
    blf.shadow(0, 3, 0, 0, 0, 1)
    blf.shadow_offset(0, 1, -1)
    lines = legend_lines(vs)
    for i, text in enumerate(reversed(lines)):
        a = 1.0 if i == len(lines) - 1 else 0.75  # status line brighter
        blf.color(0, 1, 1, 1, a)
        blf.position(0, x, (12 + i * 15) * scale, 0)
        blf.draw(0, text)
    blf.disable(0, blf.SHADOW)


@persistent
def _on_load(_dummy):
    global _drawing
    _drawing = False  # modal operators do not survive a file load
    _sel.clear()
    _hist.update(undo=[], redo=[], rev=None, obj=None)  # it belonged to the previous file


classes = (
    VoxelColor,
    VoxelSettings,
    VOXELDRAW_OT_start,
    VOXELDRAW_OT_clear,
    VOXELDRAW_OT_selection,
    VOXELDRAW_OT_replace_color,
    VOXELDRAW_OT_eyedropper,
    VOXELDRAW_OT_palette_add,
    VOXELDRAW_OT_palette_remove,
    VOXELDRAW_OT_reset_palette,
    VOXELDRAW_OT_load_palette,
    VOXELDRAW_OT_import_vox,
    VOXELDRAW_OT_export_vox,
    VOXELDRAW_OT_confirm,
    VOXELDRAW_UL_palette,
    VOXELDRAW_PT_panel,
    VOXELDRAW_PT_brush,
    VOXELDRAW_PT_selection,
    VOXELDRAW_PT_palette,
    VOXELDRAW_PT_file,
)
_handles = []


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.voxel_settings = bpy.props.PointerProperty(type=VoxelSettings)
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(draw_preview, (), 'WINDOW', 'POST_VIEW'))
    _handles.append(bpy.types.SpaceView3D.draw_handler_add(draw_overlay, (), 'WINDOW', 'POST_PIXEL'))
    bpy.app.handlers.load_post.append(_on_load)


def unregister():
    global _drawing
    _drawing = False
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    for h in _handles:
        bpy.types.SpaceView3D.draw_handler_remove(h, 'WINDOW')
    _handles.clear()
    del bpy.types.Scene.voxel_settings
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
