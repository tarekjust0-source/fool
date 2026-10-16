"""Multi-view camera list operators and intermediate-result inspection."""

import os

import bpy
import numpy as np
from bpy.props import EnumProperty, IntProperty

from ...core import camera as camlib
from .. import services, storage
from .common import active_mesh, run_safely, show_image

VIEW_SETS = {
    "STANDARD": [("FRONT", True), ("RIGHT", True), ("LEFT", True), ("BACK", True), ("TOP", True),
                 ("BOTTOM", True)],
    "CHARACTER": [("FRONT", True), ("FRONT_RIGHT", True), ("FRONT_LEFT", True), ("BACK", True),
                  ("BACK_RIGHT", True), ("BACK_LEFT", True), ("TOP", True), ("BOTTOM", False)],
    "MINIMAL": [("FRONT", True), ("BACK", True), ("TOP", False)],
    "FULL": services.DEFAULT_VIEWS,
}


class ATS_OT_views_preset(bpy.types.Operator):
    """Replace the view list with a preset camera set"""

    bl_idname = "ats.views_preset"
    bl_label = "View Preset"
    preset: EnumProperty(items=[
        ("STANDARD", "6 Axis Views", "Front, sides, back, top, bottom"),
        ("CHARACTER", "Character (3/4 views)", "Overlapping 3/4 views: smoother seams on organic shapes"),
        ("MINIMAL", "Front + Back", "Fewest requests (flat objects like shields, signs)"),
        ("FULL", "All 10 (axis on, 3/4 off)", "Every preset view"),
    ])

    def execute(self, context):
        sc = context.scene.ats
        sc.views.clear()
        for key, on in VIEW_SETS[self.preset]:
            v = sc.views.add()
            v.preset = key
            v.enabled = on
            v.label = camlib.VIEW_LABELS[key]
            services.view_uid(v)
        sc.active_view = 0
        return {"FINISHED"}


class ATS_OT_view_add_viewport(bpy.types.Operator):
    """Add the current 3D viewport angle as a custom view (great for hidden or important areas)"""

    bl_idname = "ats.view_add_viewport"
    bl_label = "Add Viewport View"

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D"

    def execute(self, context):
        sc = context.scene.ats
        services.ensure_default_views(sc)
        cam = run_safely(self, services.viewport_camera, context, int(sc.view_size))
        if cam is None:
            return {"CANCELLED"}
        v = sc.views.add()
        v.preset = "CUSTOM"
        v.enabled = True
        v.label = f"Viewport {sum(1 for x in sc.views if x.preset == 'CUSTOM')}"
        v.custom_view = [float(x) for x in np.asarray(cam.view).reshape(-1)]
        v.custom_proj = [float(x) for x in np.asarray(cam.proj).reshape(-1)]
        v.custom_ortho = cam.ortho
        services.view_uid(v)
        sc.active_view = len(sc.views) - 1
        return {"FINISHED"}


class ATS_OT_view_remove(bpy.types.Operator):
    """Remove the selected view"""

    bl_idname = "ats.view_remove"
    bl_label = "Remove View"

    def execute(self, context):
        sc = context.scene.ats
        if 0 <= sc.active_view < len(sc.views):
            sc.views.remove(sc.active_view)
            sc.active_view = max(0, sc.active_view - 1)
        return {"FINISHED"}


class ATS_OT_view_move(bpy.types.Operator):
    """Change the order views are generated in (earlier views define the design)"""

    bl_idname = "ats.view_move"
    bl_label = "Move View"
    direction: IntProperty(default=-1)

    def execute(self, context):
        sc = context.scene.ats
        i = sc.active_view
        j = i + self.direction
        if 0 <= i < len(sc.views) and 0 <= j < len(sc.views):
            sc.views.move(i, j)
            sc.active_view = j
        return {"FINISHED"}


class ATS_OT_view_show(bpy.types.Operator):
    """Show an intermediate image of the last run for the selected view"""

    bl_idname = "ats.view_show"
    bl_label = "Show View Image"
    kind: EnumProperty(items=[("render", "Render", "What was sent to the AI"),
                              ("mask", "Mask", "Region the AI was asked to paint"),
                              ("generated", "AI Output", "Raw image returned by the provider"),
                              ("aligned", "Aligned", "After auto-alignment / colour matching"),
                              ("depth", "Depth", "Depth map (used by ComfyUI ControlNet)")])

    def execute(self, context):
        sc = context.scene.ats
        if not (0 <= sc.active_view < len(sc.views)):
            return {"CANCELLED"}
        v = sc.views[sc.active_view]
        if not v.debug_dir:
            self.report({"WARNING"}, "No intermediate images for this view yet")
            return {"CANCELLED"}
        path = os.path.join(storage.resolve(v.debug_dir), f"{self.kind}.png")
        img = storage.load_debug_image(path, f"ATS View {v.label} {self.kind}")
        if img is None:
            self.report({"WARNING"}, "Image file not found (intermediate images may have been disabled)")
            return {"CANCELLED"}
        if not show_image(context, img):
            self.report({"INFO"}, f"Loaded '{img.name}' – open it in an Image Editor")
        return {"FINISHED"}


class ATS_OT_view_preview_camera(bpy.types.Operator):
    """Align the 3D viewport to the selected view's camera"""

    bl_idname = "ats.view_preview_camera"
    bl_label = "Look Through View"

    @classmethod
    def poll(cls, context):
        return context.area is not None and context.area.type == "VIEW_3D" and active_mesh(context) is not None

    def execute(self, context):
        import mathutils
        from .. import meshio
        sc = context.scene.ats
        services.ensure_default_views(sc)
        if not (0 <= sc.active_view < len(sc.views)):
            return {"CANCELLED"}
        obj = active_mesh(context)
        mesh = run_safely(self, meshio.extract, obj, context, uv_name=obj.ats.uv_map or None,
                          use_modifiers=sc.use_modifiers)
        if mesh is None:
            return {"CANCELLED"}
        cam = services.view_camera(sc, sc.views[sc.active_view], mesh, 256)
        rv3d = context.space_data.region_3d
        inv = np.linalg.inv(cam.view)
        m = mathutils.Matrix([list(r) for r in inv])
        rv3d.view_perspective = "ORTHO" if cam.ortho else "PERSP"
        lo, hi = mesh.bounds()
        center = mathutils.Vector(((lo + hi) / 2).tolist())
        rv3d.view_location = center
        rv3d.view_rotation = m.to_quaternion()
        rv3d.view_distance = (m.translation - center).length
        return {"FINISHED"}


CLASSES = (ATS_OT_views_preset, ATS_OT_view_add_viewport, ATS_OT_view_remove, ATS_OT_view_move,
           ATS_OT_view_show, ATS_OT_view_preview_camera)
