# Run: blender -b --factory-startup --python test_vd_ui.py
# Viewport panel framework + the add-on's panels, driven by simulated events
# (no GPU needed: a fake painter measures text).
import importlib.util
import json
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "voxel_draw", os.path.join(HERE, "__init__.py"), submodule_search_locations=[HERE])
vd = importlib.util.module_from_spec(spec)
sys.modules["voxel_draw"] = vd
spec.loader.exec_module(vd)
U = vd.vd_ui


class FakePainter:
    def text_size(self, s, size, font=None):
        return len(s) * size * 0.55, size * 0.75

    def __getattr__(self, name):  # rect, frame, text, icon, ... draw nothing
        return lambda *a, **k: None


class Ev:
    def __init__(self, type, value='PRESS', shift=False, ctrl=False):
        self.type, self.value, self.shift, self.ctrl, self.alt = type, value, shift, ctrl, False


def centre(w):
    x, y, ww, hh = w.rect
    return x + ww / 2, y + hh / 2


def click(ui, w, dx=0.0):
    x, y = centre(w)
    ui.handle(Ev('MOUSEMOVE', 'NOTHING'), x + dx, y)
    r1 = ui.handle(Ev('LEFTMOUSE', 'PRESS'), x + dx, y)
    r2 = ui.handle(Ev('LEFTMOUSE', 'RELEASE'), x + dx, y)
    return r1, r2


P = FakePainter()
BOUNDS = (40, 0, 1600, 900)

# --- geometry helpers
tris = U.rounded_rect(0, 0, 100, 40, 6)
assert len(tris) % 3 == 0 and all(-1e-6 <= x <= 100 + 1e-6 and -1e-6 <= y <= 40 + 1e-6 for x, y in tris)
assert U.clip_rect((0, 0, 10, 10), (5, 5, 10, 10)) == (5, 5, 5, 5) and U.clip_rect((0, 0, 1, 1), (5, 5, 1, 1)) is None
assert len(U.rounded_rect(0, 0, 10, 10, 0)) == 6
dat = U.load_dat("ops.generic.select_box")  # Blender's own toolbar icon
assert dat and len(dat[0]) == len(dat[1]) and len(dat[0]) % 3 == 0
assert all(0 <= u <= 1 and 0 <= v <= 1 for u, v in dat[0])
assert U.load_dat("no.such.icon") is None

# --- widgets in a panel
state = {"n": 0, "v": 10, "on": False, "sel": 1}
btn = U.Button("Go", on_click=lambda env: state.__setitem__("n", state["n"] + 1), tooltip="Run it")
off = U.Button("Off", enabled=False, on_click=lambda env: state.__setitem__("n", 99))
tog = U.Button("Tog", active=lambda: state["on"], on_click=lambda env: state.__setitem__("on", not state["on"]))
sld = U.Slider("Size", lambda: state["v"], lambda v: state.__setitem__("v", v), 1, 256, min_w=200)
grid = U.ColorGrid(lambda: 256, lambda i: (i / 255, 0, 0), lambda: state["sel"],
                   lambda i: state.__setitem__("sel", i), cell=20)
body = U.Column(U.Row(btn, off, tog, equal=True), sld, grid)
panel = U.Panel("p", "Panel", body, anchor=(0, 0), offset=(8, 8), resizable=True, size=(220, 300))
other = U.Panel("q", "Other", U.Column(U.Label("hi")), anchor=(1, 1), offset=(8, 8))
saved = []
ui = U.UI([panel, other], on_layout=saved.append)
ui.layout(P, BOUNDS, 1.0)
x, y, w, h = panel.rect
assert (x, y + h) == (48, 892) and (w, h) == (220, 300)  # top-left, 8 px from the free area
ox, oy, ow, oh = other.rect
assert ox + ow == 1592 and oy == 8  # bottom-right anchor
assert ui.handle(Ev('MOUSEMOVE', 'NOTHING'), 800, 450) is None  # empty viewport: not ours
assert ui.handle(Ev('LEFTMOUSE'), 800, 450) is None

assert click(ui, btn) == ('UI', 'UI') and state["n"] == 1
click(ui, off)
assert state["n"] == 1  # disabled
click(ui, tog)
assert state["on"] is True
# press, drag away, release outside: no click
bx, by = centre(btn)
ui.handle(Ev('LEFTMOUSE'), bx, by)
ui.handle(Ev('MOUSEMOVE', 'NOTHING'), 900, 300)
ui.handle(Ev('LEFTMOUSE', 'RELEASE'), 900, 300)
assert state["n"] == 1 and ui.capture is None
# tooltip after the delay
ui.handle(Ev('MOUSEMOVE', 'NOTHING'), bx, by)
assert ui.hot is btn and not ui.tick(ui.tip_since + 0.1) and ui.tick(ui.tip_since + 0.6)

# slider: drag, fine drag, click on the ends, wheel, Esc restores
sx, sy = centre(sld)
ui.handle(Ev('MOUSEMOVE', 'NOTHING'), sx, sy)
ui.handle(Ev('LEFTMOUSE'), sx, sy)
ui.handle(Ev('MOUSEMOVE', 'NOTHING'), sx + 50, sy)
assert state["v"] == 10 + round(50 / sld.rect[2] * 255), state["v"]
ui.handle(Ev('LEFTMOUSE', 'RELEASE'), sx + 50, sy)
v = state["v"]
ui.handle(Ev('LEFTMOUSE'), sx, sy)
ui.handle(Ev('MOUSEMOVE', 'NOTHING', shift=True), sx + 50, sy)
assert state["v"] == v + round(50 / sld.rect[2] * 255 * 0.1)
ui.handle(Ev('ESC'), sx + 50, sy)
assert state["v"] == v and ui.capture is None  # cancelled
v = state["v"]
click(ui, sld, dx=sld.rect[2] * 0.45)
assert state["v"] == v + 1
click(ui, sld, dx=-sld.rect[2] * 0.45)
assert state["v"] == v
ui.handle(Ev('WHEELUPMOUSE'), sx, sy)
assert state["v"] == v + 1
state["v"] = 256
ui.handle(Ev('WHEELUPMOUSE'), sx, sy)
assert state["v"] == 256  # clamped

# colour grid: click picks, wheel scrolls (panel too small for 255 cells)
gx, gy, gw, gh = grid.rect
cols = int(gw // 20)
ui.handle(Ev('MOUSEMOVE', 'NOTHING'), gx + 20 * 2 + 5, gy + gh - 20 - 5)
ui.handle(Ev('LEFTMOUSE'), gx + 20 * 2 + 5, gy + gh - 20 - 5)
ui.handle(Ev('LEFTMOUSE', 'RELEASE'), gx + 20 * 2 + 5, gy + gh - 20 - 5)
assert state["sel"] == 1 + cols + 2 and grid.tip() == f"{1 + cols + 2}"
ui.handle(Ev('WHEELDOWNMOUSE'), gx + 5, gy + 5)
assert grid.scroll == 20
ui.handle(Ev('WHEELUPMOUSE'), gx + 5, gy + 5)
ui.handle(Ev('WHEELUPMOUSE'), gx + 5, gy + 5)
assert grid.scroll == 0  # clamped at the top
# right click and wheel over a panel are eaten (no colour pick / zoom behind it)
assert ui.handle(Ev('RIGHTMOUSE'), gx + 5, gy + 5) == 'UI'
assert ui.handle(Ev('WHEELUPMOUSE'), *centre(other)) == 'UI'
# keys are never taken
assert ui.handle(Ev('A'), sx, sy) is None

# move the panel by its header: anchor follows the nearest third, layout is saved
hx, hy = x + 60, y + h - 10
ui.handle(Ev('MOUSEMOVE', 'NOTHING'), hx, hy)
assert ui.hot_part == (panel, "header")
ui.handle(Ev('LEFTMOUSE'), hx, hy)
ui.handle(Ev('MOUSEMOVE', 'NOTHING'), hx + 1300, hy - 300)
ui.handle(Ev('LEFTMOUSE', 'RELEASE'), hx + 1300, hy - 300)
nx, ny, nw, nh = panel.rect
assert (nx, ny + nh) == (x + 1300, y + h - 300) and panel.fx == 1 and panel.fy == 0.5
assert saved and saved[-1]["p"]["fx"] == 1
# the window gets smaller: still right-anchored, kept inside
ui.layout(P, (40, 0, 1200, 700), 1.0)
assert panel.rect[0] + panel.rect[2] == 1200 - (1600 - (x + 1300 + nw))  # same gap to the right edge
assert all(40 <= q.rect[0] and q.rect[0] + q.rect[2] <= 1200 for q in ui.panels)
ui.layout(P, BOUNDS, 1.0)
# resize from the grip: grows, keeps its top-left corner, respects the minimum
ui.set_state(None)  # back to the top-left corner, room to grow
ui.layout(P, BOUNDS, 1.0)
x, y, w, h = panel.rect
gx, gy, _, _ = panel.grip_rect(ui)
ui.handle(Ev('MOUSEMOVE', 'NOTHING'), gx + 3, gy + 3)
assert ui.hot_part == (panel, "grip")
ui.handle(Ev('LEFTMOUSE'), gx + 3, gy + 3)
ui.handle(Ev('MOUSEMOVE', 'NOTHING'), gx + 63, gy - 97)
ui.handle(Ev('LEFTMOUSE', 'RELEASE'), gx + 63, gy - 97)
assert (panel.w, panel.h) == (280, 400) and (panel.rect[0], panel.rect[1] + panel.rect[3]) == (x, y + h)
ui.handle(Ev('LEFTMOUSE'), panel.grip_rect(ui)[0] + 3, panel.grip_rect(ui)[1] + 3)
ui.handle(Ev('MOUSEMOVE', 'NOTHING'), 0, 2000)
ui.handle(Ev('LEFTMOUSE', 'RELEASE'), 0, 2000)
assert (panel.w, panel.h) == panel.min_size
# collapse with the chevron, state round trip, reset
cx, cy, cw, ch = panel.chevron_rect()
ui.handle(Ev('MOUSEMOVE', 'NOTHING'), cx + cw / 2, cy + ch / 2)
ui.handle(Ev('LEFTMOUSE'), cx + cw / 2, cy + ch / 2)
assert panel.collapsed and panel.rect[3] == U.HEADER
st = json.loads(json.dumps(ui.get_state()))
ui.set_state(None)
assert not panel.collapsed and panel.w == 220
ui.set_state(st)
assert panel.collapsed and panel.w == 150
ui.set_state({"p": "garbage"})  # bad saved data falls back to defaults
assert panel.fx == 0
# scale 2: everything doubles
ui.set_state(None)
ui.layout(P, BOUNDS, 2.0)
assert panel.rect[2] == 440 and panel.rect[0] == 40 + 16

# --- the add-on's panels on real settings
vd.register()
ctx = bpy.context
vs = ctx.scene.voxel_settings
vd.ensure_palette(vs)
obj = vd.ensure_session_object(ctx)
vd.save_cells(obj, {(0, 0, 0): 1, (1, 0, 0): 2}, 1.0)
vd.start_viewport_ui(vs)
ui = vd._ui
ui.layout(P, BOUNDS, 1.0)
rects = [q.rect for q in ui.panels if q.visible]
assert all(BOUNDS[0] <= r[0] and r[0] + r[2] <= BOUNDS[2] and r[1] >= 0 and r[1] + r[3] <= 900 for r in rects)
assert not [(a, b) for i, a in enumerate(rects) for b in rects[i + 1:] if U.clip_rect(a, b)]  # no overlap
names = {q.name for q in ui.panels}
assert names == {"tools", "brush", "session", "symmetry", "palette", "selection", "scene", "shortcuts"}


def find(pred):
    for q in ui.panels:
        for w in U._walk(q.body):
            if pred(w):
                return w
    raise AssertionError("widget not found")


reports = []
ui.env = type("Env", (), {"op": type("Op", (), {"report": lambda self, t, m: reports.append(m)})()})()
click(ui, find(lambda w: isinstance(w, U.Button) and w.text == "Erase"))
assert vs.mode == 'ERASE'
click(ui, find(lambda w: isinstance(w, U.Button) and w.text == "Line"))
assert vs.brush == 'LINE'
size = find(lambda w: isinstance(w, U.Slider) and w.text == "Size")
assert not size.enabled  # only for the Shape brush
vs.brush = 'SHAPE'
ui.handle(Ev('WHEELUPMOUSE'), *centre(size))
assert vs.brush_size == 2
live = find(lambda w: isinstance(w, U.Button) and w.text == "Live Mirror")
click(ui, live)
assert vs.mirror_live
mz = find(lambda w: isinstance(w, U.Button) and w.text == "Z" and w.on_click.__name__ == "flip")
click(ui, mz)
assert tuple(vs.mirror) == (True, False, True)
pal = find(lambda w: isinstance(w, U.ColorGrid))
gx, gy, gw, gh = pal.rect
click_at = (gx + 20 * 3 + 5, gy + gh - 5)
ui.handle(Ev('MOUSEMOVE', 'NOTHING'), *click_at)
ui.handle(Ev('LEFTMOUSE'), *click_at)
ui.handle(Ev('LEFTMOUSE', 'RELEASE'), *click_at)
assert vs.color_index == 4
hue = find(lambda w: isinstance(w, U.Slider) and w.text == "Value")
before = tuple(vs.palette[4].color)
hue.set(0.5)
assert tuple(vs.palette[4].color) != before
vsize = find(lambda w: isinstance(w, U.Slider) and w.text == "Voxel Size")
assert not vsize.enabled  # the session has voxels
assert not find(lambda w: isinstance(w, U.Button) and w.text == "Delete").enabled  # nothing selected
vd._sel.add((0, 0, 0))
assert find(lambda w: isinstance(w, U.Button) and w.text == "Delete").enabled
# a failing operator becomes a warning, the UI keeps going
vs.color_index = 1
click(ui, find(lambda w: isinstance(w, U.Button) and w.tooltip.startswith("Remove the active colour")))
assert reports and "used by" in reports[-1], reports
# layout persists in the scene and is restored on the next start
tools = next(q for q in ui.panels if q.name == "tools")
tools.collapsed = True
ui.layout_changed()
assert json.loads(vs.ui_layout)["tools"]["collapsed"] is True
vd.stop_viewport_ui()
vd.start_viewport_ui(vs)
assert next(q for q in vd._ui.panels if q.name == "tools").collapsed
assert bpy.ops.voxeldraw.reset_ui() == {'FINISHED'} and vs.ui_layout == ""
assert not next(q for q in vd._ui.panels if q.name == "tools").collapsed
vs.ui_layout = "{not json"
vd.start_viewport_ui(vs)  # broken saved layout: defaults
assert vd._ui is not None
# draw handlers: added per session, removed safely (twice is fine)
vd._add_draw_handlers()
assert len(vd._handles) == 2
vd._remove_draw_handlers()
vd._remove_draw_handlers()
assert vd._handles == []
# proportional editing: the panel row drives Blender's own tool settings
ts = ctx.scene.tool_settings
vd.start_viewport_ui(vs)
ui = vd._ui
ui.layout(P, BOUNDS, 1.0)
ui.env = type("Env", (), {"op": type("Op", (), {"report": lambda self, t, m: reports.append(m)})()})()
ts.use_proportional_edit = False
prop_btn = find(lambda w: isinstance(w, U.Button) and w.text == "Proportional")
sharp = find(lambda w: isinstance(w, U.Button) and w.tooltip == "Falloff: Sharp")
assert not sharp.enabled
click(ui, prop_btn)
assert ts.use_proportional_edit and sharp.enabled
click(ui, sharp)
assert ts.proportional_edit_falloff == 'SHARP'
psize = find(lambda w: isinstance(w, U.Slider) and w.text == "Size (voxels)")
vd.set_prop_distance(ts, 3.0)
assert abs(psize.get() - 3.0) < 1e-6  # session voxel size 1
psize.set(5.0)
assert abs(vd.prop_distance(ts) - 5.0) < 1e-6
proj = find(lambda w: isinstance(w, U.Button) and w.text == "Projected")
ts.use_proportional_connected = True
assert not proj.enabled  # like Blender: Projected is off while Connected is on
ts.use_proportional_connected = False
assert len(vd.FALLOFF_ITEMS) == 8 and all(i.curve for i in vd.FALLOFF_ICONS.values())

# a real grab (G) with proportional editing: rays faked, everything else is the operator's
import types  # noqa: E402
from mathutils import Quaternion  # noqa: E402
slab = {(x, y, z): 1 for x in range(-4, 5) for y in range(-4, 5) for z in range(-2, 1)}
vd.save_cells(obj, slab, 1.0)
G = type("G", (), {k: v for k, v in vd.VOXELDRAW_OT_start.__dict__.items() if callable(v)})
g = G()
g.cells, g.bounds, g.size, g.rev, g.face_key = dict(slab), vd.bounds_of(slab), 1.0, int(obj["_vd_rev"]), None
vd._hist.update(undo=[], redo=[], rev=g.rev, obj=obj.name)
region = types.SimpleNamespace(data=types.SimpleNamespace(view_rotation=Quaternion()))
lift = [0.0]
g._ray = lambda region, event, obj: ((100.0, 0.5, 0.5 + lift[0]), (-1.0, 0.0, 0.0))  # ray at height 0.5 + lift
fctx = types.SimpleNamespace(scene=ctx.scene, workspace=types.SimpleNamespace(status_text_set=lambda t: None),
                             window_manager=ctx.window_manager, area=types.SimpleNamespace(tag_redraw=lambda: None),
                             view_layer=ctx.view_layer)
ts.use_proportional_edit, ts.proportional_edit_falloff, ts.use_proportional_projected = True, 'LINEAR', False
vd.set_prop_distance(ts, 3.0)
vs.prop_stretch = True
vd._sel.clear()
vd._sel.add((0, 0, 0))
g._grab_start(fctx, region, None, obj, {(0, 0, 0): 1}, {(0, 0, 0)}, line=2)
assert g.grab["prop"] is not None and abs(g.grab["prop"]["radius"] - 3) < 1e-6
lift[0] = 3.0
g._grab_update(region, None, obj)
assert g.grab["delta"] == (0, 0, 3) and (0, 0, 3) in vd._state["float"]
ev = types.SimpleNamespace(type='WHEELUPMOUSE', value='PRESS', ctrl=False, shift=False, alt=False,
                           mouse_x=0, mouse_y=0)
real_rum, vd.region_under_mouse = vd.region_under_mouse, lambda area, event: region
g._grab_modal(fctx, ev, obj)  # wheel: proportional size x 1.1, weights recomputed
vd.region_under_mouse = real_rum
assert abs(vd.prop_distance(ts) - 3.3) < 1e-6 and abs(g.grab["prop"]["radius"] - 3.3) < 1e-6
g._grab_end(fctx, obj, True)
hill = vd.load_cells(obj)
assert (0, 0, 3) in hill and vd._sel == {(0, 0, 3)} and len(vd._hist["undo"]) == 1
for x in range(-4, 5):
    for y in range(-4, 5):
        col = sorted(z for (a, b, z) in hill if (a, b) == (x, y))
        assert col == list(range(col[0], col[-1] + 1))  # solid, no gaps
assert vd._state["prop"] is None and vd._state["float_del"] is None
ts.use_proportional_edit = False

# sessions in the Session panel: one button per open session (current lit), New
second = vd.new_session_object(ctx)  # the current one now; obj is paused
vd.save_cells(second, {(0, 0, 0): 3}, 1.0)
ui.layout(P, BOUNDS, 1.0)
names = {obj.name, second.name}
slots = [w for q in ui.panels for w in U._walk(q.body)
         if isinstance(w, U.Button) and w.visible and U._val(w.text) in names]
assert sorted(U._val(w.text) for w in slots) == sorted(names)
assert [U._val(w.text) for w in slots if U._val(w.active)] == [second.name]
vd._drawing = True  # the modal does the switch: the buttons only ask for it
ui.env = type("Env", (), {"context": bpy.context,
                          "op": type("Op", (), {"report": lambda self, t, m: reports.append(m)})()})()
click(ui, next(w for w in slots if U._val(w.text) == obj.name))
assert vd._state["switch"] == obj.name
click(ui, find(lambda w: isinstance(w, U.Button) and "start a new one" in U._val(w.tooltip)))
assert vd._state["switch"] == ""
vd._state["switch"] = None
vd._drawing = False
vs.target = obj
for extra in range(vd.SESSION_SLOTS):  # more than fit: the rest are in the sidebar
    vd.new_session_object(ctx)
vd.forget_sessions()
ui.layout(P, BOUNDS, 1.0)
more = find(lambda w: isinstance(w, U.Label) and "sidebar" in str(U._val(w.text)))
assert more.visible and U._val(more.text) == "+2 (sidebar)"
rects = [q.rect for q in ui.panels if q.visible]
assert not [(a, b) for i, a in enumerate(rects) for b in rects[i + 1:] if U.clip_rect(a, b)]  # still no overlap
vs.target = obj

# the tool never leaves handlers / panels behind: errors, Blender cancelling it, closed area
import types  # noqa: E402
F = type("F", (), {k: v for k, v in vd.VOXELDRAW_OT_start.__dict__.items() if callable(v)})
op = F()
op._timer, op.grab, op.spin, op.stroke = None, None, None, False
op.report = lambda t, m: reports.append(m)
move = types.SimpleNamespace(type='MOUSEMOVE', value='NOTHING')


def arm():
    vd._drawing = True
    vd.start_viewport_ui(vs)
    vd._add_draw_handlers()


def clean():
    return not vd._drawing and vd._handles == [] and vd._ui is None


arm()
op._modal = lambda c, e: 1 / 0
assert op.modal(ctx, move) == {'CANCELLED'} and clean() and "internal error" in reports[-1]
arm()
op.stroke = True
op._end_stroke = lambda c, o: (_ for _ in ()).throw(RuntimeError("disk full"))
assert op.modal(ctx, move) == {'CANCELLED'} and clean()  # even if saving the stroke fails
del op._modal, op._end_stroke
op.stroke = False
arm()
op.cancel(ctx)
assert clean()
arm()
vd._state["area"] = 12345  # our viewport...
fake = types.SimpleNamespace(scene=ctx.scene, area=None, window_manager=ctx.window_manager, workspace=None)
real_exists = vd.area_exists
vd.area_exists = lambda ptr: True  # ...is in another workspace: wait
assert vd.VOXELDRAW_OT_start._modal(op, fake, move) == {'PASS_THROUGH'} and vd._drawing
vd.area_exists = lambda ptr: False  # ...was closed: stop
assert vd.VOXELDRAW_OT_start._modal(op, fake, move) == {'FINISHED'} and clean()
vd.area_exists = real_exists
assert not vd.area_exists(12345)

vd._add_draw_handlers()
vd.unregister()  # disabling the add-on mid-session cleans up too
assert vd._handles == [] and vd._ui is None
print("UI TESTS PASSED")
