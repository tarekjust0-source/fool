"""Local refinement: mask painting, face masks, correction, accept/reject."""

import os

import bpy
import numpy as np

from .. import compat, layers, services, storage
from .common import active_mesh, run_safely


def _mask_image(obj, size: int) -> bpy.types.Image:
    st = obj.ats
    img = st.mask_image
    if img is None or img.name not in bpy.data.images or img.size[0] == 0:
        img = storage.get_or_create_image(f"ATS_{obj.name}_Mask", size, non_color=True)
        compat.array_to_image(img, np.zeros((size, size, 3), np.float32))
        st.mask_image = img
    elif tuple(img.size) != (size, size):
        img.scale(size, size)
    img.use_fake_user = True
    return img


def _save_mask(obj):
    img = obj.ats.mask_image
    if img is not None:
        storage.save_image_datablock(img, os.path.join(storage.object_dir(obj), "refine_mask.png"))


class ATS_OT_mask_paint(bpy.types.Operator):
    """Paint the region to correct directly on the model (white = edit). Uses Blender's Texture Paint"""

    bl_idname = "ats.mask_paint"
    bl_label = "Paint Mask"

    @classmethod
    def poll(cls, context):
        return active_mesh(context) is not None

    def execute(self, context):
        obj = active_mesh(context)
        sc = context.scene.ats
        if not obj.data.uv_layers:
            self.report({"ERROR"}, "The object needs a UV map")
            return {"CANCELLED"}
        size = layers.texture_size(obj, sc)
        img = _mask_image(obj, size)
        uv = obj.ats.uv_map or obj.data.uv_layers.active.name
        if uv in obj.data.uv_layers:
            obj.data.uv_layers.active = obj.data.uv_layers[uv]
        if context.mode != "PAINT_TEXTURE":
            bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
        ts = context.tool_settings
        ip = ts.image_paint
        ip.mode = "IMAGE"
        ip.canvas = img
        try:
            ups = ts.unified_paint_settings
            ups.use_unified_color = True
            ups.color = (1.0, 1.0, 1.0)
        except AttributeError:
            pass
        try:
            if ip.brush is not None:
                ip.brush.color = (1.0, 1.0, 1.0)
        except AttributeError:
            pass
        self.report({"INFO"}, "Paint white where the texture should change (Ctrl: subtract). Press 'Done' when "
                              "finished")
        return {"FINISHED"}


class ATS_OT_mask_done(bpy.types.Operator):
    """Leave mask painting and return to Object Mode"""

    bl_idname = "ats.mask_done"
    bl_label = "Done Painting"

    def execute(self, context):
        obj = active_mesh(context)
        if context.mode == "PAINT_TEXTURE":
            bpy.ops.object.mode_set(mode="OBJECT")
        if obj is not None:
            _save_mask(obj)
        return {"FINISHED"}


class ATS_OT_mask_from_faces(bpy.types.Operator):
    """Add the faces selected in Edit Mode to the painted mask"""

    bl_idname = "ats.mask_from_faces"
    bl_label = "Mask From Selected Faces"

    @classmethod
    def poll(cls, context):
        return active_mesh(context) is not None

    def execute(self, context):
        obj = active_mesh(context)
        size = layers.texture_size(obj, context.scene.ats)
        m = run_safely(self, services.face_mask, obj, size)
        if m is None:
            return {"CANCELLED"}
        if m.sum() == 0:
            self.report({"WARNING"}, "No faces selected (select faces in Edit Mode)")
            return {"CANCELLED"}
        img = _mask_image(obj, size)
        cur = compat.image_to_array(img)[..., 0]
        new = np.maximum(cur, m)
        compat.array_to_image(img, np.repeat(new[..., None], 3, axis=2))
        _save_mask(obj)
        return {"FINISHED"}


class ATS_OT_mask_clear(bpy.types.Operator):
    """Clear the refinement mask"""

    bl_idname = "ats.mask_clear"
    bl_label = "Clear Mask"
    invert: bpy.props.BoolProperty(default=False, name="Invert instead")

    def execute(self, context):
        obj = active_mesh(context)
        if obj is None or obj.ats.mask_image is None:
            return {"CANCELLED"}
        img = obj.ats.mask_image
        size = img.size[0]
        if self.invert:
            cur = compat.image_to_array(img)[..., :3]
            compat.array_to_image(img, 1.0 - cur)
        else:
            compat.array_to_image(img, np.zeros((size, size, 3), np.float32))
        _save_mask(obj)
        return {"FINISHED"}


class ATS_OT_refine(bpy.types.Operator):
    """Generate a correction for the masked area. The result is a pending layer you can accept or reject"""

    bl_idname = "ats.refine"
    bl_label = "Generate Correction"

    @classmethod
    def poll(cls, context):
        from .. import runtime
        return runtime.active_job() is None and active_mesh(context) is not None

    def invoke(self, context, event):
        need, notice = services.needs_consent(context)
        if need:
            self._notice = notice
            return context.window_manager.invoke_props_dialog(self, width=520)
        return self.execute(context)

    def draw(self, context):
        from ..prefs import _wrap
        col = self.layout.column(align=True)
        col.label(text="Before the first request, please confirm:", icon="INFO")
        for line in _wrap(getattr(self, "_notice", ""), 80):
            col.label(text=line)

    def execute(self, context):
        from .. import prefs
        obj = active_mesh(context)
        if context.mode == "PAINT_TEXTURE":
            bpy.ops.object.mode_set(mode="OBJECT")
            _save_mask(obj)
        need, _ = services.needs_consent(context)
        if need:
            p = prefs.get_prefs(context)
            p.settings_for(p.provider).consent = True
        job = run_safely(self, services.start_refine, context, obj)
        if job is None:
            return {"CANCELLED"}
        if job.status == "FAILED":
            self.report({"ERROR"}, job.error)
            return {"CANCELLED"}
        return {"FINISHED"}


class ATS_OT_refine_resolve(bpy.types.Operator):
    """Accept or reject the pending correction"""

    bl_idname = "ats.refine_resolve"
    bl_label = "Resolve Correction"
    accept: bpy.props.BoolProperty(default=True)

    def execute(self, context):
        obj = active_mesh(context)
        if obj is None or not services.resolve_pending(obj, self.accept):
            self.report({"WARNING"}, "No pending correction")
            return {"CANCELLED"}
        services.apply_material(obj, context.scene.ats)
        return {"FINISHED"}


CLASSES = (ATS_OT_mask_paint, ATS_OT_mask_done, ATS_OT_mask_from_faces, ATS_OT_mask_clear, ATS_OT_refine,
           ATS_OT_refine_resolve)
