"""Drag an .stl onto the window: bounding-box dimensions + GPU-rendered preview.

Left-drag orbits, scroll zooms, right-drag pans. 'w' wireframe, 'a' axes, 'z' reset view.
"""
import math
import os
import sys

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


def describe(mesh, path):
    dims = " x ".join(f"{d:.2f}" for d in mesh.extents)
    return f"{os.path.basename(path)}   {dims} mm   {len(mesh.faces)} triangles"


def scene_for(path):
    """-> (scene, caption). Empty scene when path is None."""
    if path is None:
        return trimesh.Scene(), HINT
    mesh = trimesh.load(path, force="mesh")
    if not len(mesh.faces):
        raise ValueError("no triangles in file")
    scene = trimesh.Scene(mesh)
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
        self.set_caption(caption)

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
