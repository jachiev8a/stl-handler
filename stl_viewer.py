"""Drag an .stl onto the window: bounding-box dimensions + GPU-rendered preview.

Left-drag orbits, scroll zooms, right-drag pans. 'w' wireframe, 'a' axes, 'z' reset view.
The button (or 'd') toggles bounding-box dimensions drawn on the model.
"""
import ctypes
import itertools
import math
import os
import sys

import numpy as np
import pyglet
import trimesh
from pyglet import gl
from trimesh.viewer import SceneViewer

HINT = "drop an .stl here"
VIEW = (math.radians(60), 0, math.radians(45))  # 3/4 view; trimesh's default is face-on
RES = (900, 700)
# trimesh's look_at fits the axis-aligned bounds, so a rotated view lands too close.
# Distance as a multiple of the bounding-box diagonal instead; turn this knob to taste.
FIT = 1.4
LIGHT, AMBIENT = (0.85, 0.85, 0.85), (0.30, 0.30, 0.30)  # brightness knobs
BACKGROUND = (38, 40, 46, 255)  # trimesh wants 0-255 RGBA
TEXT = (215, 215, 220, 255)
DIM = (255, 196, 92, 255)       # dimension outline
DIM_TEXT = (120, 226, 240, 255)  # the numbers: amber's complement, so they read
                                 # against both the outline and the grey surface
BUTTON = (12, 12, 176, 30)      # x, y, w, h from the bottom-left corner
RESET = (196, 12, 96, 30)
EDIT_FORMAT = "[{typed}_] was {original:.2f}"   # how the height reads while being typed
MAX_GROW = 3.0                  # ceiling on a typed height, times the ORIGINAL height:
                                # measured against the original so repeated edits can't
                                # creep (3x then 3x again would otherwise reach 9x)
BBOX = "bbox"                   # scene node holding the dimension box
EDIT = (255, 140, 205, 255)     # the height while being typed: distinct from both
                                # the amber outline and the cyan resting numbers
MARGIN = 0.05                   # mm kept clear of each cut band's edges


def describe(mesh, path):
    dims = " x ".join(f"{d:.2f}" for d in mesh.extents)
    return f"{os.path.basename(path)}   {dims} mm   {len(mesh.faces)} triangles"


def box_edges(bounds):
    """The 12 edges of an axis-aligned box as a Path3D. bounding_box.outline() on a
    Trimesh comes back with zero entities, so build the segments directly."""
    corners = np.array(list(itertools.product(*zip(*bounds))))
    segments = [(a, b) for i, a in enumerate(corners) for b in corners[i + 1:]
                if np.count_nonzero(a != b) == 1]  # edges differ in exactly one axis
    path = trimesh.load_path(np.array(segments))
    path.colors = np.tile(DIM, (len(path.entities), 1))
    return path


def cut_bands(mesh):
    """-> [(lo, hi)] z-ranges where every triangle is a vertical wall.

    A horizontal slab taken from inside such a band can be removed by sliding the
    vertices above it straight down: the wall triangles just get shorter, so the
    topology, triangle count and watertightness are all preserved, and no feature
    (floor, rim, fillet) is touched. Anything non-vertical is forbidden."""
    zmin, zmax = mesh.triangles[:, :, 2].min(axis=1), mesh.triangles[:, :, 2].max(axis=1)
    sloped = np.abs(mesh.face_normals[:, 2]) >= 1e-3
    blocked = sorted(zip(zmin[sloped], zmax[sloped]))
    bands, edge = [], mesh.bounds[0][2]
    for lo, hi in blocked:  # walk the blocked spans; the gaps between them are free
        if lo - edge > 2 * MARGIN:
            bands.append((edge + MARGIN, lo - MARGIN))
        edge = max(edge, hi)
    if mesh.bounds[1][2] - edge > 2 * MARGIN:
        bands.append((edge + MARGIN, mesh.bounds[1][2] - MARGIN))
    return bands


def cut_gaps(mesh):
    """-> [(lo, hi)] slices that can actually be removed.

    A cut must land inside a vertical band AND contain no vertex: swallowing a vertex
    row would drag it out of position instead of shortening the wall around it. So the
    usable slices are the gaps between consecutive vertex heights within each band."""
    heights = np.unique(mesh.vertices[:, 2])
    gaps = []
    for lo, hi in cut_bands(mesh):
        edges = np.concatenate(([lo], heights[(heights > lo) & (heights < hi)], [hi]))
        gaps.extend((a + MARGIN, b - MARGIN) for a, b in zip(edges[:-1], edges[1:])
                    if b - a > 2 * MARGIN)
    return gaps


def shorten(mesh, target):
    """-> a copy of mesh with its Z extent set to target, by cutting plain wall only.

    Raises ValueError when the part has too little featureless wall to give up."""
    delta = mesh.extents[2] - target
    if abs(delta) < 1e-9:
        raise ValueError(f"already {target:.2f} mm tall")
    gaps = cut_gaps(mesh)
    if not gaps:
        raise ValueError("no plain vertical wall to cut")
    room = sum(b - a for a, b in gaps)
    if delta > room:
        raise ValueError(f"needs {delta:.2f} mm of cut, part only has {room:.2f} mm of "
                         f"plain wall (min height {mesh.extents[2] - room:.2f} mm)")
    vertices = mesh.vertices.copy()
    if delta < 0:  # growing: stretch one wall segment, which needs no spare room
        _, top = max(gaps, key=lambda g: g[1] - g[0])
        vertices[vertices[:, 2] >= top, 2] -= delta
    else:
        left = delta
        for lo, hi in sorted(gaps, reverse=True):  # top down: lower gaps keep their z
            take = min(left, hi - lo)
            vertices[vertices[:, 2] >= lo + take, 2] -= take
            left -= take
            if left <= 1e-12:
                break
    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces.copy(), process=False)


def check_target(target, original_height):
    """Raise if a typed height is nonsense, before any geometry work happens."""
    if not np.isfinite(target) or target <= 0:
        raise ValueError("height must be a positive number")
    if target > original_height * MAX_GROW:
        raise ValueError(f"{target:.2f} mm is over {MAX_GROW:g}x the original "
                         f"{original_height:.2f} mm (max {original_height * MAX_GROW:.2f})")


def mesh_of(scene):
    """The part itself, as opposed to the dimension box."""
    return next((g for n, g in scene.geometry.items() if n != BBOX), None)


def build_scene(mesh, camera=None):
    scene = trimesh.Scene(mesh)
    scene.add_geometry(box_edges(mesh.bounds), geom_name=BBOX, node_name=BBOX)
    if camera is None:
        scene.set_camera(angles=VIEW, distance=scene.scale * FIT, resolution=RES)
    else:
        scene.camera_transform = camera  # an edit keeps the view you were looking from
    return scene


def scene_for(path):
    """-> (scene, caption). Empty scene when path is None."""
    if path is None:
        return trimesh.Scene(), HINT
    mesh = trimesh.load(path, force="mesh")
    if not len(mesh.faces):
        raise ValueError("no triangles in file")
    return build_scene(mesh), describe(mesh, path)


class Viewer(SceneViewer):
    # trimesh builds the pyglet window itself and never passes file_drops=True, so the
    # window never advertises XdndAware and X11 skips it. Force it on; the False that
    # Window.__init__ assigns lands in the no-op setter.
    _file_drops = property(lambda self: True, lambda self, value: None)

    def __init__(self, path=None):
        scene, caption = scene_for(path)
        super().__init__(scene, caption=caption, resolution=RES, start_loop=False,
                         background=BACKGROUND)
        self.text = pyglet.text.Label(caption, font_name="monospace", font_size=12,
                                      color=TEXT, x=12, y=self.height - 24)
        self.button = pyglet.text.Label("", font_name="monospace", font_size=11,
                                        color=TEXT, x=BUTTON[0] + 12, y=BUTTON[1] + 10)
        self.reset_label = pyglet.text.Label("reset (r)", font_name="monospace", font_size=11,
                                             color=TEXT, x=RESET[0] + 12, y=RESET[1] + 10)
        self.dim_labels = [pyglet.text.Label("", font_name="monospace", font_size=14.4,
                                             color=DIM_TEXT, anchor_x="center", anchor_y="center")
                           for _ in range(3)]
        self.status = pyglet.text.Label("", font_name="monospace", font_size=11,
                                        color=DIM_TEXT, x=12, y=self.height - 44)
        self.dims_on = False
        self.path, self.mesh = path, mesh_of(scene)
        self.original = self.mesh
        self.editing = None      # the digits typed so far, or None when not editing
        self._label_at = [None, None, None]   # last screen pos of each dimension label
        self._apply_dims()

    def _apply_dims(self):
        """The box lives in the scene permanently; the toggle just hides its node."""
        if self.dims_on:
            self.unhide_geometry(BBOX)
        else:
            self.hide_geometry(BBOX)
        self.button.text = f"dimensions: {'on' if self.dims_on else 'off'}  (d)"

    def toggle_dims(self):
        self.dims_on = not self.dims_on
        self._apply_dims()

    def set_caption(self, text):
        super().set_caption(text)
        if getattr(self, "text", None):  # base __init__ captions before the label exists
            self.text.text = text

    def on_file_drop(self, x, y, paths):
        try:
            scene, caption = scene_for(paths[0])
        except Exception as exc:
            self.set_caption(f"{os.path.basename(paths[0])}: {exc}")
            return
        self._swap_scene(scene)
        self.path, self.mesh = paths[0], mesh_of(scene)
        self.original = self.mesh
        self.editing, self.status.text = None, ""
        self.set_caption(caption)

    def on_mouse_press(self, x, y, buttons, modifiers):
        bx, by, bw, bh = BUTTON
        if bx <= x <= bx + bw and by <= y <= by + bh:
            self.toggle_dims()  # swallow the click so it doesn't also start a camera drag
            return
        rx, ry, rw, rh = RESET
        if rx <= x <= rx + rw and ry <= y <= ry + rh:
            self._reset()
            return
        at = self._label_at[2]  # only the height is editable; X/Y have no cut bands
        if self.dims_on and self.mesh is not None and at and math.dist(at, (x, y)) < 34:
            self._start_edit()
            return
        super().on_mouse_press(x, y, buttons, modifiers)

    def _start_edit(self):
        was = self.original.extents[2]
        self.editing = ""
        self.status.text = (f"height in mm - was {was:.2f}, max {was * MAX_GROW:.2f} - "
                            "Enter applies, Esc cancels")

    def on_text(self, text):
        if self.editing is not None and (text.isdigit() or (text == "." and "." not in self.editing)):
            self.editing += text

    def _apply_edit(self):
        try:
            target = float(self.editing)
            check_target(target, self.original.extents[2])
            mesh = shorten(self.mesh, target)
        except (ValueError, ZeroDivisionError) as exc:
            self.status.text = str(exc)
            self.editing = None
            return
        self.mesh = mesh
        self._swap_scene(build_scene(mesh, camera=self.scene.camera_transform.copy()))
        self.editing = None
        self.set_caption(describe(mesh, self.path))
        self.status.text = "edited - 's' saves a copy, original untouched"

    def _reset(self):
        """Back to the mesh as loaded. The view is left alone - 'z' already resets that."""
        if self.mesh is None or self.mesh is self.original:
            return
        self.mesh = self.original
        self._swap_scene(build_scene(self.mesh, camera=self.scene.camera_transform.copy()))
        self.editing = None
        self.set_caption(describe(self.mesh, self.path))
        self.status.text = "reset to original"

    def _save(self):
        if self.mesh is None:
            return
        stem, _ = os.path.splitext(self.path)
        out = f"{stem}_h{self.mesh.extents[2]:.2f}mm.stl"
        self.mesh.export(out)
        self.status.text = f"wrote {os.path.basename(out)}"

    def on_key_press(self, symbol, modifiers):
        key = pyglet.window.key
        if self.editing is not None:  # swallow everything while typing a number
            if symbol == key.BACKSPACE:
                self.editing = self.editing[:-1]
            elif symbol in (key.ENTER, key.RETURN, key.NUM_ENTER) and self.editing:
                self._apply_edit()
            elif symbol == key.ESCAPE:
                self.editing, self.status.text = None, ""
            return
        if symbol == key.D:
            self.toggle_dims()
            return
        if symbol == key.S:
            self._save()
            return
        if symbol == key.R:
            self._reset()
            return
        super().on_key_press(symbol, modifiers)

    def _dim_anchors(self):
        """-> [(world point, length)] per axis: the midpoint of whichever of the four
        parallel box edges sits nearest the camera, so numbers land on the near side."""
        if self.scene.is_empty:
            return []
        lo, hi = self.scene.bounds
        eye = self.scene.camera_transform[:3, 3]
        out = []
        for axis in range(3):
            others = [i for i in range(3) if i != axis]
            points = []
            for pick in itertools.product((lo, hi), repeat=2):
                p = np.empty(3)
                p[axis] = (lo[axis] + hi[axis]) / 2
                for i, corner in zip(others, pick):
                    p[i] = corner[i]
                points.append(p)
            out.append((min(points, key=lambda p: np.linalg.norm(p - eye)), hi[axis] - lo[axis]))
        return out

    @staticmethod
    def _to_screen(point, mv, proj, view):
        """gluProject one world point to window coords. -> (x, y) or None if behind."""
        wx, wy, wz = gl.GLdouble(), gl.GLdouble(), gl.GLdouble()
        gl.gluProject(*(float(c) for c in point), mv, proj, view,
                      ctypes.byref(wx), ctypes.byref(wy), ctypes.byref(wz))
        return None if not 0.0 <= wz.value <= 1.0 else (wx.value, wy.value)

    def _draw_buttons(self):
        edited = self.mesh is not None and self.mesh is not self.original
        for (bx, by, bw, bh), label, live in ((BUTTON, self.button, True),
                                              (RESET, self.reset_label, edited)):
            shade = (58, 62, 72, 235) if live else (46, 48, 54, 200)  # dim when inert
            pyglet.graphics.draw(4, gl.GL_QUADS,
                                 ("v2f", (bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh)),
                                 ("c4B", shade * 4))
            label.color = TEXT if live else (120, 122, 130, 255)
            label.draw()

    def _swap_scene(self, scene):
        """Point the viewer at a new scene. The camera lands on whatever transform the
        scene carries, so a drop re-fits and an edit keeps the view you had."""
        self.scene = self._scene = scene
        scene._redraw = self._redraw
        self._initial_camera_transform = scene.camera_transform.copy()
        # drop the previous mesh's GPU buffers, then let the base class re-upload
        self.batch = pyglet.graphics.Batch()
        for d in (self.vertex_list, self.vertex_list_hash, self.vertex_list_mode, self.textures):
            d.clear()
        self._update_vertex_list()
        self.reset_view()
        self.update_flags()
        self._apply_dims()  # a fresh scene starts with nothing hidden

    def _headlight(self):
        """trimesh's autolight puts two dim (0.235) point lights at the bounding-box
        corners, which renders a grey mesh near-black. One bright light in eye space
        instead, so it follows the camera. w=0 in POSITION means directional."""
        gl.glPushMatrix()
        gl.glLoadIdentity()  # POSITION is multiplied by the modelview: identity = eye space
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(-0.3, 0.4, 1.0, 0.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (gl.GLfloat * 4)(*LIGHT, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (gl.GLfloat * 4)(*AMBIENT, 1.0))
        gl.glPopMatrix()
        gl.glEnable(gl.GL_LIGHT0)
        gl.glDisable(gl.GL_LIGHT1)

    def on_draw(self):
        self._headlight()
        super().on_draw()
        # trimesh leaves a lit 3D context; drop to 2D for the overlay, then restore.
        # PushAttrib puts lighting and depth-test back (without it every later frame is
        # unlit and un-occluded), and CURRENT_BIT puts the colour back: trimesh enables
        # GL_COLOR_MATERIAL, so the label's dark text colour would become the mesh's.
        # capture the 3D matrices before switching to 2D: the labels are placed by
        # projecting world points through them, so they track the model as it orbits
        mv, proj = (gl.GLdouble * 16)(), (gl.GLdouble * 16)()
        view = (gl.GLint * 4)()
        gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX, mv)
        gl.glGetDoublev(gl.GL_PROJECTION_MATRIX, proj)
        gl.glGetIntegerv(gl.GL_VIEWPORT, view)
        gl.glPushAttrib(gl.GL_ENABLE_BIT | gl.GL_CURRENT_BIT | gl.GL_TEXTURE_BIT)
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glOrtho(0, max(self.width, 1), 0, max(self.height, 1), -1, 1)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_LIGHTING)
        self.text.y = self.height - 24
        self.text.draw()
        self.status.y = self.height - 44
        self.status.draw()
        self._draw_buttons()
        anchors = self._dim_anchors() if self.dims_on else []  # empty scene -> no anchors
        self._label_at = [None, None, None]  # stale positions must not stay clickable
        if anchors:
            middle = self._to_screen(self.scene.centroid, mv, proj, view) or (0, 0)
            for axis, (label, (point, length)) in enumerate(zip(self.dim_labels, anchors)):
                at = self._to_screen(point, mv, proj, view)
                if at:
                    # push the number off its edge, away from the model's centre,
                    # so the box line doesn't strike through the digits
                    away = np.array(at) - middle
                    norm = np.linalg.norm(away)
                    label.text = f"{length:.2f}"
                    label.x, label.y = np.array(at) + (away / norm * 22 if norm > 1 else 0)
                    self._label_at[axis] = (label.x, label.y)
                    if axis == 2 and self.editing is not None:
                        # keep the value being replaced visible next to the new one
                        label.text = EDIT_FORMAT.format(typed=self.editing,
                                                        original=self.original.extents[2])
                        label.color = EDIT
                    else:
                        label.color = DIM_TEXT
                    label.draw()
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPopMatrix()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glPopAttrib()


def selftest():
    """A 2x3x4 box, through a real .stl file, comes back with the right extents."""
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), "box.stl")
    trimesh.creation.box((2, 3, 4)).export(path)
    _, caption = scene_for(path)
    assert "2.00 x 3.00 x 4.00 mm" in caption, caption
    assert "12 triangles" in caption, caption
    assert scene_for(None)[1] == HINT

    # shortening cuts plain wall only: exact height, other axes and topology untouched
    mesh = trimesh.load(path, force="mesh")
    short = shorten(mesh, 3.0)
    assert abs(short.extents[2] - 3.0) < 1e-9, short.extents
    assert np.allclose(short.extents[:2], (2, 3)), short.extents
    assert short.is_watertight and len(short.faces) == len(mesh.faces)
    assert (short.area_faces > 1e-9).all(), "cut collapsed a face"
    assert abs(shorten(mesh, 6.0).extents[2] - 6.0) < 1e-9, "growing should work too"
    try:
        shorten(mesh, 0.01)  # more cut than the part has plain wall
        raise AssertionError("should have refused")
    except ValueError:
        pass

    # the typed-height guard, measured against the original so edits can't creep
    check_target(4 * MAX_GROW, 4.0)          # exactly at the ceiling is allowed
    for bad in (0.0, -5.0, float("nan"), 4 * MAX_GROW + 0.01):
        try:
            check_target(bad, 4.0)
            raise AssertionError(f"should have refused {bad}")
        except ValueError:
            pass
    print("ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        Viewer(*sys.argv[1:2])
        pyglet.app.run()
