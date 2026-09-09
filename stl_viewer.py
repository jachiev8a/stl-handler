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
BBOX = "bbox"                   # scene node holding the dimension box


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


def scene_for(path):
    """-> (scene, caption). Empty scene when path is None."""
    if path is None:
        return trimesh.Scene(), HINT
    mesh = trimesh.load(path, force="mesh")
    if not len(mesh.faces):
        raise ValueError("no triangles in file")
    scene = trimesh.Scene(mesh)
    scene.add_geometry(box_edges(mesh.bounds), geom_name=BBOX, node_name=BBOX)
    scene.set_camera(angles=VIEW, distance=scene.scale * FIT, resolution=RES)
    return scene, describe(mesh, path)


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
        self.dim_labels = [pyglet.text.Label("", font_name="monospace", font_size=14.4,
                                             color=DIM_TEXT, anchor_x="center", anchor_y="center")
                           for _ in range(3)]
        self.dims_on = False
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
        self.scene = self._scene = scene
        self.scene._redraw = self._redraw
        self._initial_camera_transform = scene.camera_transform.copy()
        # drop the previous mesh's GPU buffers, then let the base class re-upload
        self.batch = pyglet.graphics.Batch()
        for d in (self.vertex_list, self.vertex_list_hash, self.vertex_list_mode, self.textures):
            d.clear()
        self._update_vertex_list()
        self.reset_view()
        self.update_flags()
        self._apply_dims()  # a fresh scene starts with nothing hidden
        self.set_caption(caption)

    def on_mouse_press(self, x, y, buttons, modifiers):
        bx, by, bw, bh = BUTTON
        if bx <= x <= bx + bw and by <= y <= by + bh:
            self.toggle_dims()  # swallow the click so it doesn't also start a camera drag
            return
        super().on_mouse_press(x, y, buttons, modifiers)

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.D:
            self.toggle_dims()
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

    def _draw_button(self):
        bx, by, bw, bh = BUTTON
        pyglet.graphics.draw(4, gl.GL_QUADS,
                             ("v2f", (bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh)),
                             ("c4B", (58, 62, 72, 235) * 4))
        self.button.draw()

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
        self._draw_button()
        anchors = self._dim_anchors() if self.dims_on else []  # empty scene -> no anchors
        if anchors:
            middle = self._to_screen(self.scene.centroid, mv, proj, view) or (0, 0)
            for label, (point, length) in zip(self.dim_labels, anchors):
                at = self._to_screen(point, mv, proj, view)
                if at:
                    # push the number off its edge, away from the model's centre,
                    # so the box line doesn't strike through the digits
                    away = np.array(at) - middle
                    norm = np.linalg.norm(away)
                    label.text = f"{length:.2f}"
                    label.x, label.y = np.array(at) + (away / norm * 22 if norm > 1 else 0)
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
    print("ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        Viewer(*sys.argv[1:2])
        pyglet.app.run()
