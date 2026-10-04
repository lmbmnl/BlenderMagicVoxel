# SPDX-License-Identifier: MIT
# BlenderMagicVoxel by lmbmnl - https://github.com/lmbmnl/BlenderMagicVoxel
"""Small widget kit drawn inside a 3D viewport region (RetopoFlow style).

No add-on logic lives here: widgets read and write through callables and call
`on_click(env)` with whatever environment the host puts in `UI.env`. Drawing
goes through a painter object (GPUPainter: gpu + blf), so layout and hit
testing can run, and be tested, without a GPU.

Coordinates are region pixels, origin bottom-left (like Blender). Sizes in this
module are unscaled pixels; everything is multiplied by `UI.s` (the user's UI
scale) when laid out.
"""

import os
import time
from math import cos, pi, sin

ROW, PAD, GAP, RADIUS, FONT, ICON, HEADER = 22, 6, 3, 4, 11, 16, 24
SYMBOLS = "NotoSansSymbols2-Regular.woff2"  # fonts shipped in Blender's datafiles/fonts
MONO = "DejaVuSansMono.woff2"
EMOJI = "NotoEmoji-VariableFont_wght.woff2"

THEME = {
    "panel": (0.11, 0.11, 0.12, 0.93), "header": (0.17, 0.17, 0.19, 0.97),
    "title": (0.86, 0.86, 0.9, 1), "widget": (0.235, 0.235, 0.255, 1),
    "hover": (0.31, 0.31, 0.335, 1), "press": (0.19, 0.19, 0.21, 1),
    "on": (0.26, 0.44, 0.76, 1), "on_hover": (0.32, 0.51, 0.85, 1),
    "text": (0.93, 0.93, 0.95, 1), "dim": (0.6, 0.6, 0.64, 1),
    "outline": (0, 0, 0, 0.55), "fill": (0.29, 0.47, 0.8, 0.9),
    "accent": (1.0, 0.62, 0.15, 1), "tip": (0.05, 0.05, 0.06, 0.97),
    "sep": (1, 1, 1, 0.08),
}


def _val(v):
    return v() if callable(v) else v


def _mix(c, a):
    return (c[0], c[1], c[2], c[3] * a)


# ---------------------------------------------------------------- geometry (pure)
def rounded_rect(x, y, w, h, r, seg=4):
    """Triangles (flat list of (x, y), 3 per triangle) of a rounded rectangle."""
    r = max(0.0, min(r, w / 2, h / 2))
    if r <= 0.5:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y), (x + w, y + h), (x, y + h)]
    ring = rounded_outline(x, y, w, h, r, seg)
    c = (x + w / 2, y + h / 2)
    tris = []
    for i in range(len(ring)):
        tris += [c, ring[i], ring[(i + 1) % len(ring)]]
    return tris


def rounded_outline(x, y, w, h, r, seg=4):
    """Closed outline (counter-clockwise points, last != first) of a rounded rect."""
    r = max(0.0, min(r, w / 2, h / 2))
    if r <= 0.5:
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    pts = []
    for cx, cy, a0 in ((x + w - r, y + r, -pi / 2), (x + w - r, y + h - r, 0.0),
                       (x + r, y + h - r, pi / 2), (x + r, y + r, pi)):
        for i in range(seg + 1):
            a = a0 + pi / 2 * i / seg
            pts.append((cx + r * cos(a), cy + r * sin(a)))
    return pts


def clip_rect(r, clip):
    """Intersection of two (x, y, w, h) rects, or None."""
    x0, y0 = max(r[0], clip[0]), max(r[1], clip[1])
    x1, y1 = min(r[0] + r[2], clip[0] + clip[2]), min(r[1] + r[3], clip[1] + clip[3])
    return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None


# ---------------------------------------------------------------- icons
class Icon:
    """One of Blender's toolbar icons (datafiles/icons/<name>.dat: coloured vector
    triangles), or a glyph of a font shipped with Blender. Blender's named UI icons
    ('TRASH', ...) can't be drawn from Python outside UILayout, hence these two."""
    __slots__ = ("dat", "glyph", "font")

    def __init__(self, dat=None, glyph=None, font=SYMBOLS):
        self.dat, self.glyph, self.font = dat, glyph, font


def DAT(name):
    return Icon(dat=name)


def GLYPH(char, font=SYMBOLS):
    return Icon(glyph=char, font=font)


_dat_cache = {}


def datafiles_dir(sub):
    try:
        import bpy
        return bpy.utils.system_resource('DATAFILES', path=sub)
    except Exception:  # outside Blender
        return ""


def load_dat(name, folder=None):
    """(coords [(x, y) 0..1] 3 per triangle, colours [(r, g, b, a) 0..1] per vertex),
    or None if the icon is missing or unreadable."""
    key = (name, folder)
    if key not in _dat_cache:
        _dat_cache[key] = None
        try:
            with open(os.path.join(folder or datafiles_dir("icons"), name + ".dat"), "rb") as f:
                d = f.read()
            if d[:4] == b"VCO\x00" and (len(d) - 8) % 18 == 0:
                n = (len(d) - 8) // 18 * 3  # vertices
                co = d[8:8 + n * 2]
                cl = d[8 + n * 2:]
                coords = [(co[2 * i] / 255, co[2 * i + 1] / 255) for i in range(n)]
                cols = [tuple(cl[4 * i + k] / 255 for k in range(4)) for i in range(n)]
                _dat_cache[key] = (coords, cols)
        except OSError:
            pass
    return _dat_cache[key]


# ---------------------------------------------------------------- GPU painter
def builtin_shader(name):
    """Built-in shader by its 4.x / 5.x name, falling back to the 3.x '2D_' / '3D_'
    names. None if neither exists."""
    import gpu
    for n in (name, "2D_" + name, "3D_" + name):
        try:
            return gpu.shader.from_builtin(n)
        except (ValueError, KeyError):
            continue
    return None


_font_ids = {}


def font_id(name):
    """blf id of a font from Blender's datafiles/fonts (0 = default UI font).
    Never unloaded: Blender itself may use the same font as a fallback."""
    if not name:
        return 0
    if name not in _font_ids:
        import blf
        fid = blf.load(os.path.join(datafiles_dir("fonts"), name))
        _font_ids[name] = fid if fid >= 0 else 0
    return _font_ids[name]


class GPUPainter:
    """Draws with gpu + blf in a POST_PIXEL handler (no bgl)."""

    def __init__(self):
        from gpu_extras.batch import batch_for_shader
        import gpu
        import blf
        self._batch, self._gpu, self._blf = batch_for_shader, gpu, blf
        self.flat = builtin_shader('UNIFORM_COLOR')
        self.smooth = builtin_shader('SMOOTH_COLOR')
        self.poly = builtin_shader('POLYLINE_UNIFORM_COLOR')

    def tris(self, pts, color):
        if not pts:
            return
        b = self._batch(self.flat, 'TRIS', {"pos": pts})
        self.flat.bind()
        self.flat.uniform_float("color", color)
        b.draw(self.flat)

    def rect(self, x, y, w, h, color, r=0):
        self.tris(rounded_rect(x, y, w, h, r), color)

    def frame(self, x, y, w, h, color, r=0, width=1.0):
        pts = rounded_outline(x, y, w, h, r)
        self.polyline(pts + pts[:1], color, width)

    def polyline(self, pts, color, width=1.0):
        if self.poly is None:  # very old Blender: thin lines
            b = self._batch(self.flat, 'LINE_STRIP', {"pos": pts})
            self.flat.bind()
            self.flat.uniform_float("color", color)
            b.draw(self.flat)
            return
        b = self._batch(self.poly, 'LINE_STRIP', {"pos": [(x, y, 0) for x, y in pts]})
        self.poly.bind()
        self.poly.uniform_float("viewportSize", self._gpu.state.viewport_get()[2:])
        self.poly.uniform_float("lineWidth", width)
        self.poly.uniform_float("color", color)
        b.draw(self.poly)

    def color_rects(self, rects):
        """Many solid rects in one batch: [(x, y, w, h, (r, g, b, a))]."""
        if not rects:
            return
        pos, col = [], []
        for x, y, w, h, c in rects:
            pos += [(x, y), (x + w, y), (x + w, y + h), (x, y), (x + w, y + h), (x, y + h)]
            col += [c] * 6
        b = self._batch(self.smooth, 'TRIS', {"pos": pos, "color": col})
        self.smooth.bind()
        b.draw(self.smooth)

    def text_size(self, s, size, font=None):
        fid = font_id(font)
        self._blf.size(fid, size)
        return self._blf.dimensions(fid, s)

    def text(self, s, x, y, size, color, font=None):
        """`y` = baseline."""
        fid = font_id(font)
        self._blf.size(fid, size)
        self._blf.color(fid, *color)
        self._blf.position(fid, x, y, 0)
        self._blf.draw(fid, s)

    def icon(self, icon, x, y, size, alpha=1.0, color=None):
        """Icon in the square (x, y, size, size)."""
        if icon.dat:
            data = load_dat(icon.dat)
            if data is None:
                return
            coords, cols = data
            pos = [(x + u * size, y + v * size) for u, v in coords]
            col = [(c[0], c[1], c[2], c[3] * alpha) for c in cols]
            b = self._batch(self.smooth, 'TRIS', {"pos": pos, "color": col})
            self.smooth.bind()
            b.draw(self.smooth)
        elif icon.glyph:
            gs = size * 0.9
            w, h = self.text_size(icon.glyph, gs, icon.font)
            self.text(icon.glyph, x + (size - w) / 2, y + (size - gs * 0.72) / 2,
                      gs, _mix(color or THEME["text"], alpha), icon.font)


# ---------------------------------------------------------------- widgets
class Widget:
    interactive = False

    def __init__(self, *, tooltip="", enabled=True, visible=True, flex=False):
        self.tooltip, self._enabled, self._visible, self.flex = tooltip, enabled, visible, flex
        self.rect = (0, 0, 0, 0)
        self.pressed = False

    @property
    def enabled(self):
        return bool(_val(self._enabled))

    @property
    def visible(self):
        return bool(_val(self._visible))

    def tip(self):
        return _val(self.tooltip)

    def measure(self, ui):
        return 0, 0

    def place(self, ui, x, y, w, h):
        self.rect = (x, y, w, h)

    def contains(self, mx, my):
        x, y, w, h = self.rect
        return x <= mx < x + w and y <= my < y + h

    def find(self, mx, my):
        """Deepest interactive widget under the mouse."""
        return self if self.interactive and self.visible and self.contains(mx, my) else None

    def draw(self, ui, p):
        pass

    # interaction: press returns True to keep the mouse until the release
    def press(self, ui, mx, my, ev):
        return False

    def drag(self, ui, mx, my, ev):
        pass

    def release(self, ui, mx, my, ev):
        pass

    def cancel(self, ui):
        pass

    def wheel(self, ui, step, ev):
        return False

    def motion(self, ui, mx, my):
        """Mouse moved over the widget (hover only). True = redraw."""
        return False


class Box(Widget):
    """Stack children vertically or horizontally. equal=True gives every child of
    a row the same width; flex children share the space left over."""

    def __init__(self, children, *, vertical=True, spacing=GAP, padding=0, equal=False, **kw):
        super().__init__(**kw)
        self.children, self.vertical = list(children), vertical
        self.spacing, self.padding, self.equal = spacing, padding, equal

    def shown(self):
        return [c for c in self.children if c.visible]

    def measure(self, ui):
        kids = self.shown()
        if not kids:
            return 0, 0
        sizes = [c.measure(ui) for c in kids]
        sp, pad = self.spacing * ui.s, self.padding * ui.s
        if self.vertical:
            return (max(w for w, _ in sizes) + 2 * pad,
                    sum(h for _, h in sizes) + sp * (len(kids) - 1) + 2 * pad)
        ws = [w for w, _ in sizes]
        total = max(ws) * len(ws) if self.equal else sum(ws)
        return total + sp * (len(kids) - 1) + 2 * pad, max(h for _, h in sizes) + 2 * pad

    def place(self, ui, x, y, w, h):
        super().place(ui, x, y, w, h)
        kids = self.shown()
        if not kids:
            return
        sp, pad = self.spacing * ui.s, self.padding * ui.s
        x, y, w, h = x + pad, y + pad, w - 2 * pad, h - 2 * pad
        sizes = [c.measure(ui) for c in kids]
        if self.vertical:
            extra = h - sum(s[1] for s in sizes) - sp * (len(kids) - 1)
            flex = [c for c in kids if c.flex]
            top = y + h
            for c, (_cw, ch) in zip(kids, sizes):
                if flex and c.flex:
                    ch += max(0.0, extra) / len(flex)
                c.place(ui, x, top - ch, w, ch)
                top -= ch + sp
        else:
            n = len(kids)
            if self.equal:
                widths = [(w - sp * (n - 1)) / n] * n
            else:
                widths = [s[0] for s in sizes]
                flex = [i for i, c in enumerate(kids) if c.flex]
                extra = w - sum(widths) - sp * (n - 1)
                for i in flex:
                    widths[i] += max(0.0, extra) / len(flex)
            left = x
            for c, cw in zip(kids, widths):
                c.place(ui, left, y, cw, h)
                left += cw + sp

    def find(self, mx, my):
        if not self.contains(mx, my):
            return None
        for c in reversed(self.shown()):
            hit = c.find(mx, my)
            if hit is not None:
                return hit
        return None

    def draw(self, ui, p):
        for c in self.shown():
            c.draw(ui, p)


def Row(*children, **kw):
    return Box(children, vertical=False, **kw)


def Column(*children, **kw):
    return Box(children, vertical=True, **kw)


class Separator(Widget):
    def measure(self, ui):
        return 2 * ui.s, 5 * ui.s

    def draw(self, ui, p):
        x, y, w, h = self.rect
        p.rect(x, y + h / 2 - 0.5 * ui.s, w, 1 * ui.s, THEME["sep"])


def _icon_text_size(ui, p, icon, text, size):
    s = ui.s
    tw = p.text_size(text, size)[0] if text else 0
    iw = ICON * s if icon else 0
    return iw, tw, iw + tw + (4 * s if icon and text else 0)


def _fit(p, text, size, width):
    """Text cut with an ellipsis to fit `width`."""
    if not text or p.text_size(text, size)[0] <= width:
        return text
    while text and p.text_size(text + "…", size)[0] > width:
        text = text[:-1]
    return text + "…" if text else ""


def _draw_icon_text(ui, p, x, y, w, h, icon, text, color, alpha, align="CENTER", size=None):
    s = ui.s
    size = size or FONT * s
    iw, tw, total = _icon_text_size(ui, p, icon, text, size)
    room = w - 2 * PAD * s
    if total > room and text:  # shorten the text, keep the icon
        text = _fit(p, text, size, room - iw - (4 * s if icon else 0))
        iw, tw, total = _icon_text_size(ui, p, icon, text, size)
    left = x + (w - total) / 2 if align == "CENTER" else x + PAD * s
    if icon:
        p.icon(icon, left, y + (h - ICON * s) / 2, ICON * s, alpha, color)
        left += iw + 4 * s
    if text:
        p.text(text, left, y + h / 2 - size * 0.36, size, _mix(color, alpha))


class Label(Widget):
    def __init__(self, text, *, icon=None, small=False, dim=False, align="LEFT", **kw):
        super().__init__(**kw)
        self.text, self.icon, self.small, self.dim, self.align = text, icon, small, dim, align

    def size(self, ui):
        return (FONT - (1.5 if self.small else 0)) * ui.s

    def measure(self, ui):
        _, _, total = _icon_text_size(ui, ui.painter, self.icon, _val(self.text), self.size(ui))
        return total + 2 * PAD * ui.s, (ROW - (5 if self.small else 0)) * ui.s

    def draw(self, ui, p):
        x, y, w, h = self.rect
        _draw_icon_text(ui, p, x, y, w, h, self.icon, _val(self.text),
                        THEME["dim"] if self.dim else THEME["text"], 1.0, self.align, self.size(ui))


class Button(Widget):
    """Click button. `active` (bool or callable) highlights it: toggles and radio
    buttons are Buttons whose `active` reads the bound value."""
    interactive = True

    def __init__(self, text="", *, icon=None, on_click=None, active=False, min_w=0, align="CENTER", **kw):
        super().__init__(**kw)
        self.text, self.icon, self.on_click, self.active, self.min_w = text, icon, on_click, active, min_w
        self.align = align

    def measure(self, ui):
        s = ui.s
        _, _, total = _icon_text_size(ui, ui.painter, self.icon, _val(self.text), FONT * s)
        w = ROW * s if not _val(self.text) else total + 2 * PAD * s
        return max(w, self.min_w * s), ROW * s

    def draw(self, ui, p):
        x, y, w, h = self.rect
        on, en = bool(_val(self.active)), self.enabled
        hover = en and ui.hot is self
        if self.pressed and hover:
            bg = THEME["press"]
        elif on:
            bg = THEME["on_hover"] if hover else THEME["on"]
        else:
            bg = THEME["hover"] if hover else THEME["widget"]
        p.rect(x, y, w, h, bg if en else _mix(bg, 0.5), RADIUS * ui.s)
        _draw_icon_text(ui, p, x, y, w, h, self.icon, _val(self.text), THEME["text"],
                        1.0 if en else 0.35, self.align)

    def press(self, ui, mx, my, ev):
        if not self.enabled:
            return False
        self.pressed = True
        return True

    def release(self, ui, mx, my, ev):
        self.pressed = False
        if self.enabled and self.contains(mx, my) and self.on_click:
            self.on_click(ui.env)
            ui.changed_values = True

    def cancel(self, ui):
        self.pressed = False


class Slider(Widget):
    """Drag horizontally to change, Shift = fine; click the left / right end or use
    the wheel to step; Esc while dragging restores the value."""
    interactive = True

    def __init__(self, text, get, set, lo, hi, *, step=1, digits=0, log=False, min_w=90, **kw):
        super().__init__(**kw)
        self.text, self.get, self.set, self.lo, self.hi = text, get, set, lo, hi
        self.step, self.digits, self.log, self.min_w = step, digits, log, min_w
        self.start = None

    def frac(self, v):
        lo, hi = _val(self.lo), _val(self.hi)
        if self.log and lo > 0:
            from math import log
            return (log(max(v, lo)) - log(lo)) / (log(hi) - log(lo))
        return (v - lo) / (hi - lo) if hi > lo else 0.0

    def value_at(self, f):
        lo, hi = _val(self.lo), _val(self.hi)
        f = max(0.0, min(1.0, f))
        if self.log and lo > 0:
            v = lo * (hi / lo) ** f
        else:
            v = lo + (hi - lo) * f
        return self.clean(v)

    def clean(self, v):
        lo, hi = _val(self.lo), _val(self.hi)
        v = max(lo, min(hi, v))
        return int(round(v)) if self.digits == 0 else round(v, self.digits)

    def label(self):
        v = self.get()
        num = f"{v:d}" if self.digits == 0 else f"{v:.{self.digits}f}"
        t = _val(self.text)
        return f"{t}  {num}" if t else num

    def measure(self, ui):
        tw = ui.painter.text_size(self.label(), FONT * ui.s)[0]
        return max(self.min_w * ui.s, tw + 4 * PAD * ui.s), ROW * ui.s

    def draw(self, ui, p):
        x, y, w, h = self.rect
        s, en = ui.s, self.enabled
        hover = en and (ui.hot is self or self.start is not None)
        bg = THEME["hover"] if hover else THEME["widget"]
        p.rect(x, y, w, h, bg if en else _mix(bg, 0.5), RADIUS * s)
        f = max(0.0, min(1.0, self.frac(self.get())))
        if f > 0:
            p.rect(x, y, max(w * f, 2 * RADIUS * s), h, _mix(THEME["fill"], 1 if en else 0.4), RADIUS * s)
        a = 1.0 if en else 0.35
        if hover and self.start is None:
            for ch, cx in (("‹", x + 4 * s), ("›", x + w - 10 * s)):
                p.text(ch, cx, y + h / 2 - FONT * s * 0.36, FONT * s, _mix(THEME["dim"], a))
        text = _fit(p, self.label(), FONT * s, w - 4 * PAD * s)
        tw = p.text_size(text, FONT * s)[0]
        p.text(text, x + (w - tw) / 2, y + h / 2 - FONT * s * 0.36, FONT * s, _mix(THEME["text"], a))

    def press(self, ui, mx, my, ev):
        if not self.enabled:
            return False
        self.start = (mx, self.get(), False)
        return True

    def drag(self, ui, mx, my, ev):
        x0, v0, moved = self.start
        dx = mx - x0
        if not moved and abs(dx) < 3 * ui.s:
            return
        self.start = (x0, v0, True)
        fine = 0.1 if getattr(ev, "shift", False) else 1.0
        v = self.value_at(self.frac(v0) + dx / max(self.rect[2], 1) * fine)
        if v != self.get():
            self.set(v)
            ui.changed_values = True

    def release(self, ui, mx, my, ev):
        x0, v0, moved = self.start
        self.start = None
        if not moved and self.contains(mx, my):  # click on an end = one step
            x, _y, w, _h = self.rect
            side = -1 if mx < x + w * 0.3 else 1 if mx > x + w * 0.7 else 0
            if side:
                self.set(self.clean(self.get() + side * self.step))
                ui.changed_values = True

    def cancel(self, ui):
        if self.start is not None:
            self.set(self.start[1])
            self.start = None

    def wheel(self, ui, step, ev):
        if not self.enabled:
            return False
        self.set(self.clean(self.get() + step * self.step))
        ui.changed_values = True
        return True


class Swatch(Widget):
    """Colour patch (optionally clickable)."""

    def __init__(self, color, *, size=(ROW, ROW), on_click=None, **kw):
        super().__init__(**kw)
        self.color, self.size, self.on_click = color, size, on_click
        self.interactive = on_click is not None

    def measure(self, ui):
        return self.size[0] * ui.s, self.size[1] * ui.s

    def draw(self, ui, p):
        x, y, w, h = self.rect
        c = _val(self.color)
        if c is None:
            return
        p.rect(x, y, w, h, (*c[:3], 1), RADIUS * ui.s)
        p.frame(x, y, w, h, THEME["outline"], RADIUS * ui.s, 1.5 * ui.s)

    def press(self, ui, mx, my, ev):
        return self.enabled

    def release(self, ui, mx, my, ev):
        if self.enabled and self.contains(mx, my) and self.on_click:
            self.on_click(ui.env)
            ui.changed_values = True


class ColorGrid(Widget):
    """Scrollable grid of colour cells: indices first..count()-1. Columns follow
    the available width, so it reflows when its panel is resized."""
    interactive = True

    def __init__(self, count, color, active, select, *, first=1, cell=20, name=None, **kw):
        kw.setdefault("flex", True)
        super().__init__(**kw)
        self.count, self.color, self.active, self.select = count, color, active, select
        self.first, self.cell, self.name = first, cell, name
        self.scroll, self.mouse_index = 0.0, None

    def measure(self, ui):
        return self.cell * 6 * ui.s, self.cell * 2 * ui.s

    def grid(self, ui):
        """(cell px, columns, content height)."""
        c = self.cell * ui.s
        cols = max(1, int(self.rect[2] // c))
        n = max(0, self.count() - self.first)
        return c, cols, -(-n // cols) * c

    def place(self, ui, x, y, w, h):
        super().place(ui, x, y, w, h)
        _c, _cols, content = self.grid(ui)
        self.scroll = max(0.0, min(self.scroll, content - h))

    def index_at(self, ui, mx, my):
        c, cols, _ = self.grid(ui)
        x, y, w, h = self.rect
        col, row = int((mx - x) // c), int((y + h + self.scroll - my) // c)
        if not (0 <= col < cols and row >= 0):
            return None
        i = self.first + row * cols + col
        return i if i < self.count() else None

    def cell_rect(self, ui, i):
        c, cols, _ = self.grid(ui)
        x, y, w, h = self.rect
        row, col = divmod(i - self.first, cols)
        return x + col * c, y + h + self.scroll - (row + 1) * c, c, c

    def draw(self, ui, p):
        c, cols, content = self.grid(ui)
        x, y, w, h = self.rect
        s = ui.s
        p.rect(x, y, w, h, (0.07, 0.07, 0.08, 1), RADIUS * s)
        rows_from = int(self.scroll // c)
        rows_to = int((self.scroll + h) // c) + 1
        rects, marks = [], []
        for i in range(self.first + rows_from * cols, min(self.count(), self.first + rows_to * cols)):
            r = clip_rect(self.cell_rect(ui, i), self.rect)
            if r is None:
                continue
            g = 1 * s  # gap between cells
            col = self.color(i)
            rects.append((r[0] + g, r[1] + g, max(0, r[2] - 2 * g), max(0, r[3] - 2 * g), (*col[:3], 1)))
            if i == self.active() or (ui.hot is self and i == self.mouse_index):
                marks.append((i, r))
        p.color_rects(rects)
        for i, r in marks:
            color = THEME["accent"] if i == self.active() else (1, 1, 1, 0.9)
            p.frame(r[0] + s, r[1] + s, r[2] - 2 * s, r[3] - 2 * s, color, 0, 2 * s)
        if content > h:  # scrollbar
            bar = max(16 * s, h * h / content)
            top = y + h - (h - bar) * self.scroll / (content - h)
            p.rect(x + w - 4 * s, top - bar, 3 * s, bar, (1, 1, 1, 0.35), 1.5 * s)

    def motion(self, ui, mx, my):
        i = self.index_at(ui, mx, my)
        if i != self.mouse_index:
            self.mouse_index = i
            return True
        return False

    def tip(self):
        if self.mouse_index is None:
            return ""
        name = self.name(self.mouse_index) if self.name else ""
        return f"{self.mouse_index}  {name}".strip()

    def press(self, ui, mx, my, ev):
        i = self.index_at(ui, mx, my)
        if i is not None and self.enabled:
            self.select(i)
            ui.changed_values = True
        return False

    def wheel(self, ui, step, ev):
        c, _cols, content = self.grid(ui)
        if content <= self.rect[3]:
            return False
        self.scroll = max(0.0, min(content - self.rect[3], self.scroll - step * c))
        return True


# ---------------------------------------------------------------- panels
class Panel:
    """Floating panel: header (drag to move, chevron to collapse), body, optional
    resize grip (bottom-right). Its place is an anchor (fx, fy: 0 / 0.5 / 1 of the
    free width / height, fy measured from the top) plus an offset, so it sticks to
    the nearest edge when the viewport is resized."""

    def __init__(self, name, title, body, *, icon=None, anchor=(0, 0), offset=(8, 8),
                 resizable=False, size=(220, 300), min_size=(150, 110), collapsed=False,
                 visible=True):
        self.name, self.title, self.body, self.icon = name, title, body, icon
        self.defaults = dict(fx=anchor[0], fy=anchor[1], dx=offset[0], dy=offset[1],
                             w=size[0], h=size[1], collapsed=collapsed)
        self.resizable, self.min_size, self._visible = resizable, min_size, visible
        self.rect = (0, 0, 0, 0)
        self._hh = HEADER
        self.reset()

    def reset(self):
        for k, v in self.defaults.items():
            setattr(self, k, v)

    @property
    def visible(self):
        return bool(_val(self._visible))

    def state(self):
        return {k: getattr(self, k) for k in self.defaults}

    def load(self, d):
        for k in self.defaults:
            if k in d:
                setattr(self, k, type(self.defaults[k])(d[k]))

    # -- geometry
    def header_h(self, ui):
        return HEADER * ui.s

    def chevron_rect(self):
        x, y, w, h = self.rect
        hh = self._hh
        return x + w - hh, y + h - hh, hh, hh

    def grip_rect(self, ui):
        x, y, w, h = self.rect
        g = 14 * ui.s
        return x + w - g, y, g, g

    def natural_size(self, ui):
        s, p = ui.s, ui.painter
        title_w = (p.text_size(self.title, FONT * s)[0] + (ICON + 6) * s * bool(self.icon)
                   + HEADER * s + 2 * PAD * s)
        if self.collapsed:
            return max(title_w, (self.w if self.resizable else 0) * s), self.header_h(ui)
        if self.resizable:
            return (max(self.w, self.min_size[0]) * s, max(self.h, self.min_size[1]) * s)
        bw, bh = self.body.measure(ui)
        return max(title_w, bw + 2 * PAD * s), self.header_h(ui) + bh + 2 * PAD * s

    def layout(self, ui, bounds):
        L, B, R, T = bounds
        s = ui.s
        self._hh = self.header_h(ui)
        w, h = self.natural_size(ui)
        w, h = min(w, max(R - L, 40 * s)), min(h, max(T - B, self._hh))
        x = L + self.fx * (R - L - w) + self.dx * s * (1 - 2 * (self.fx == 1))
        top = T - self.fy * (T - B - h) - self.dy * s * (1 - 2 * (self.fy == 1))
        x = max(L, min(x, R - w))
        top = min(T, max(top, B + h))
        self.rect = (x, top - h, w, h)
        if not self.collapsed:
            pad = PAD * s
            self.body.place(ui, x + pad, top - h + pad, w - 2 * pad, h - self._hh - 2 * pad)

    def move_to(self, ui, bounds, x, top):
        """Store an absolute position as anchor + offset (nearest third of the view)."""
        L, B, R, T = bounds
        s = ui.s
        _, _, w, h = self.rect
        cx, cy = x + w / 2, top - h / 2
        self.fx = 0 if cx < L + (R - L) / 3 else 1 if cx > L + 2 * (R - L) / 3 else 0.5
        self.fy = 0 if cy > T - (T - B) / 3 else 1 if cy < T - 2 * (T - B) / 3 else 0.5
        base_x = L + self.fx * (R - L - w)
        base_top = T - self.fy * (T - B - h)
        self.dx = (x - base_x) / s * (1 - 2 * (self.fx == 1))
        self.dy = (base_top - top) / s * (1 - 2 * (self.fy == 1))

    def contains(self, mx, my):
        x, y, w, h = self.rect
        return self.visible and x <= mx < x + w and y <= my < y + h

    def in_header(self, mx, my):
        x, y, w, h = self.rect
        return y + h - self._hh <= my < y + h

    # -- drawing
    def draw(self, ui, p):
        x, y, w, h = self.rect
        s = ui.s
        r = RADIUS * 1.5 * s
        p.rect(x, y, w, h, THEME["panel"], r)
        hh = self._hh
        p.rect(x, y + h - hh, w, hh, THEME["header"], r)
        if not self.collapsed:  # square off the header's bottom corners
            p.rect(x, y + h - hh, w, r, THEME["header"])
        left = x + PAD * s
        if self.icon:
            p.icon(self.icon, left, y + h - hh + (hh - ICON * s) / 2, ICON * s)
            left += (ICON + 6) * s
        title = _fit(p, self.title, FONT * s, x + w - hh - left)
        p.text(title, left, y + h - hh / 2 - FONT * s * 0.36, FONT * s, THEME["title"])
        cx, cy, cw, ch = self.chevron_rect()
        hot = ui.hot_part == (self, "chevron")
        p.icon(GLYPH("▸" if self.collapsed else "▾"), cx + (cw - ICON * s) / 2,
               cy + (ch - ICON * s) / 2, ICON * s, 1.0 if hot else 0.6)
        if not self.collapsed:
            self.body.draw(ui, p)
            if self.resizable:
                gx, gy, gw, gh = self.grip_rect(ui)
                c = (1, 1, 1, 0.55 if ui.hot_part == (self, "grip") else 0.25)
                for k in (1, 2, 3):
                    d = k * 4 * s
                    p.polyline([(gx + gw - d, gy + 2 * s), (gx + gw - 2 * s, gy + d)], c, 1.2 * s)
        p.frame(x, y, w, h, THEME["outline"], r, 1.0 * s)


class _MoveDrag:
    def __init__(self, ui, panel, mx, my):
        self.panel, self.ui = panel, ui
        x, y, w, h = panel.rect
        self.grab = (mx - x, my - (y + h))

    def drag(self, ui, mx, my, ev):
        x, top = mx - self.grab[0], my - self.grab[1]
        self.panel.move_to(ui, ui.bounds, x, top)
        self.panel.layout(ui, ui.bounds)

    def release(self, ui, mx, my, ev):
        ui.layout_changed()

    def cancel(self, ui):
        ui.layout_changed()


class _ResizeDrag:
    def __init__(self, ui, panel, mx, my):
        self.panel = panel
        x, y, w, h = panel.rect
        self.start = (mx, my, panel.w, panel.h, x, y + h)

    def drag(self, ui, mx, my, ev):
        mx0, my0, w0, h0, x, top = self.start
        p = self.panel
        p.w = max(p.min_size[0], w0 + (mx - mx0) / ui.s)
        p.h = max(p.min_size[1], h0 + (my0 - my) / ui.s)
        p.layout(ui, ui.bounds)
        p.move_to(ui, ui.bounds, x, top)  # keep the top-left corner where it is
        p.layout(ui, ui.bounds)

    def release(self, ui, mx, my, ev):
        ui.layout_changed()

    def cancel(self, ui):
        ui.layout_changed()


class _WidgetDrag:
    def __init__(self, widget):
        self.widget = widget

    def drag(self, ui, mx, my, ev):
        self.widget.drag(ui, mx, my, ev)

    def release(self, ui, mx, my, ev):
        self.widget.release(ui, mx, my, ev)

    def cancel(self, ui):
        self.widget.cancel(ui)


# ---------------------------------------------------------------- manager
class UI:
    """Owns the panels, routes events, draws. handle() returns 'UI' (event used),
    'OVER' (mouse move over a panel: hide the host's hover feedback) or None."""
    TIP_DELAY = 0.5

    def __init__(self, panels, *, on_layout=None):
        self.panels = list(panels)
        self.on_layout = on_layout
        self.s, self.bounds, self.painter, self.env = 1.0, (0, 0, 100, 100), None, None
        self.capture = None
        self.hot = self.hot_part = None
        self.mouse = (0, 0)
        self.tip_since, self.tip_shown = 0.0, False
        self.redraw = False
        self.changed_values = False

    # -- persistence
    def get_state(self):
        return {p.name: p.state() for p in self.panels}

    def set_state(self, state):
        for p in self.panels:
            p.reset()
            if state and isinstance(state.get(p.name), dict):
                p.load(state[p.name])

    def layout_changed(self):
        if self.on_layout:
            self.on_layout(self.get_state())

    # -- layout / draw
    def layout(self, painter, bounds, scale):
        self.painter, self.bounds, self.s = painter, bounds, max(scale, 0.1)
        for p in self.panels:
            if p.visible:
                p.layout(self, bounds)

    def draw(self, painter):
        for p in self.panels:
            if p.visible:
                p.draw(self, painter)
        if self.tip_shown and self.hot is not None and self.capture is None:
            self._draw_tip(painter, self.hot.tip())

    def _draw_tip(self, p, text):
        if not text:
            return
        s = self.s
        lines = text.split("\n")
        size = (FONT - 0.5) * s
        w = max(p.text_size(t, size)[0] for t in lines) + 2 * PAD * s
        lh = size * 1.45
        h = lh * len(lines) + PAD * s
        mx, my = self.mouse
        L, B, R, T = self.bounds
        x = max(L, min(mx + 14 * s, R - w))
        y = my - 22 * s - h
        if y < B:
            y = my + 22 * s
        p.rect(x, y, w, h, THEME["tip"], RADIUS * s)
        for i, t in enumerate(lines):
            p.text(t, x + PAD * s, y + h - PAD * s / 2 - lh * (i + 1) + size * 0.4, size, THEME["text"])

    # -- events
    def panel_at(self, mx, my):
        for p in reversed(self.panels):
            if p.contains(mx, my):
                return p
        return None

    def contains(self, mx, my):
        return self.panel_at(mx, my) is not None

    def _hover(self, mx, my):
        self.mouse = (mx, my)
        p = self.panel_at(mx, my)
        hot, part = None, None
        if p is not None:
            if p.in_header(mx, my):
                cx, cy, cw, ch = p.chevron_rect()
                part = (p, "chevron" if cx <= mx < cx + cw else "header")
            elif p.resizable and not p.collapsed and _inside(p.grip_rect(self), mx, my):
                part = (p, "grip")
            elif not p.collapsed:
                hot = p.body.find(mx, my)
        changed = hot is not self.hot or part != self.hot_part
        if hot is not self.hot:
            self.tip_since, self.tip_shown = time.monotonic(), False
        self.hot, self.hot_part = hot, part
        if hot is not None and hot.motion(self, mx, my):
            changed = True
            self.tip_since, self.tip_shown = time.monotonic(), False
        return p, changed

    def leave(self):
        """Mouse left the region: drop hover state. True = redraw."""
        had = self.hot is not None or self.hot_part is not None or self.tip_shown
        self.hot = self.hot_part = None
        self.tip_shown = False
        return had

    def handle(self, ev, mx, my):
        t, v = ev.type, ev.value
        self.changed_values = False
        if self.capture is not None:
            if t in ('MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'):
                self.mouse = (mx, my)
                self.capture.drag(self, mx, my, ev)
                self.redraw = True
            elif t == 'LEFTMOUSE' and v == 'RELEASE':
                cap, self.capture = self.capture, None
                cap.release(self, mx, my, ev)
                self._hover(mx, my)
                self.redraw = True
            elif t in ('ESC', 'RIGHTMOUSE') and v == 'PRESS':
                cap, self.capture = self.capture, None
                cap.cancel(self)
                self.redraw = True
            elif not t.startswith(('MOUSE', 'TIMER', 'WINDOW', 'NDOF', 'TRACKPAD')) and t not in (
                    'LEFTMOUSE', 'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'):
                return None  # keyboard keeps working while dragging
            return 'UI'

        if t in ('MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'):
            p, changed = self._hover(mx, my)
            if changed:
                self.redraw = True
            return 'OVER' if p is not None else None

        p = self.panel_at(mx, my)
        if p is None:
            return None
        if t == 'LEFTMOUSE' and v in ('PRESS', 'DOUBLE_CLICK'):
            self.panels.remove(p)  # bring to front
            self.panels.append(p)
            self._hover(mx, my)
            self.tip_shown = False
            part = self.hot_part[1] if self.hot_part else None
            if part == "chevron":
                p.collapsed = not p.collapsed
                p.layout(self, self.bounds)
                self.layout_changed()
            elif part == "header":
                self.capture = _MoveDrag(self, p, mx, my)
            elif part == "grip":
                self.capture = _ResizeDrag(self, p, mx, my)
            elif self.hot is not None and self.hot.press(self, mx, my, ev):
                self.capture = _WidgetDrag(self.hot)
            self.redraw = True
            return 'UI'
        if t == 'LEFTMOUSE':
            return 'UI'  # release over a panel without a press on it
        if t in ('WHEELUPMOUSE', 'WHEELDOWNMOUSE'):
            step = 1 if t == 'WHEELUPMOUSE' else -1
            self._hover(mx, my)
            used = self.hot is not None and self.hot.wheel(self, step, ev)
            if not used and not p.collapsed:  # scroll a grid from anywhere in its panel
                for w in _walk(p.body):
                    if isinstance(w, ColorGrid) and w.visible and w.wheel(self, step, ev):
                        break
            self.redraw = True
            return 'UI'  # the view doesn't zoom under a panel
        if t == 'RIGHTMOUSE':
            return 'UI'  # no colour picking through a panel
        return None

    def tick(self, now=None):
        """Call on timer events: True when a tooltip has to appear (redraw)."""
        now = time.monotonic() if now is None else now
        if (self.hot is not None and not self.tip_shown and self.capture is None
                and now - self.tip_since >= self.TIP_DELAY and self.hot.tip()):
            self.tip_shown = True
            return True
        return False

    def cancel(self):
        if self.capture is not None:
            cap, self.capture = self.capture, None
            cap.cancel(self)


def _inside(r, mx, my):
    return r[0] <= mx < r[0] + r[2] and r[1] <= my < r[1] + r[3]


def _walk(w):
    yield w
    for c in getattr(w, "children", ()):
        yield from _walk(c)
