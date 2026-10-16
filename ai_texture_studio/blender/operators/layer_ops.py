"""Layer stack and history operators."""

import os

import bpy
import numpy as np
from bpy.props import EnumProperty, IntProperty

from ...core import imageio
from .. import compat, layers, services, storage
from .common import active_mesh, run_safely, show_image


def _obj(context):
    obj = active_mesh(context)
    return obj if obj is not None and obj.ats.layers is not None else None


class _LayerOp:
    @classmethod
    def poll(cls, context):
        obj = active_mesh(context)
        return obj is not None and len(obj.ats.layers) > 0


class ATS_OT_layer_op(_LayerOp, bpy.types.Operator):
    """Layer stack operation"""

    bl_idname = "ats.layer_op"
    bl_label = "Layer Operation"
    action: EnumProperty(items=[("DUPLICATE", "Duplicate", "Duplicate the active layer"),
                                ("DELETE", "Delete", "Delete the active layer (undoable; file is kept)"),
                                ("UP", "Move Up", "Move the active layer up"),
                                ("DOWN", "Move Down", "Move the active layer down"),
                                ("MERGE_DOWN", "Merge Down", "Merge the active layer into the one below"),
                                ("FLATTEN", "Flatten", "Merge all visible layers into one")])

    def execute(self, context):
        obj = active_mesh(context)
        st = obj.ats
        i = st.active_layer
        if not (0 <= i < len(st.layers)) and self.action != "FLATTEN":
            return {"CANCELLED"}
        if self.action == "DUPLICATE":
            run_safely(self, layers.duplicate_layer, obj, i)
        elif self.action == "DELETE":
            layers.remove_layer(obj, i)
        elif self.action == "UP":
            layers.move_layer(obj, i, 1)
        elif self.action == "DOWN":
            layers.move_layer(obj, i, -1)
        elif self.action == "MERGE_DOWN":
            if i == 0:
                self.report({"WARNING"}, "Nothing below to merge into")
                return {"CANCELLED"}
            run_safely(self, layers.merge_down, obj, i)
        elif self.action == "FLATTEN":
            run_safely(self, layers.flatten, obj)
        layers.recomposite(obj, save=True)
        services.apply_material(obj, context.scene.ats)
        return {"FINISHED"}


class ATS_OT_layers_undo(bpy.types.Operator):
    """Undo the last layer-stack change (AI Texture Studio's own history; layer files are never deleted)"""

    bl_idname = "ats.layers_undo"
    bl_label = "Undo Layer Change"
    redo: bpy.props.BoolProperty(default=False)

    @classmethod
    def poll(cls, context):
        return active_mesh(context) is not None

    def execute(self, context):
        obj = active_mesh(context)
        label = layers.redo(obj) if self.redo else layers.undo(obj)
        if label is None:
            self.report({"INFO"}, "Nothing to redo" if self.redo else "Nothing to undo")
            return {"CANCELLED"}
        if obj.ats.layers:
            services.apply_material(obj, context.scene.ats)
        self.report({"INFO"}, f"{'Redo' if self.redo else 'Undo'}: {label}")
        return {"FINISHED"}


class ATS_OT_layer_import_existing(bpy.types.Operator):
    """Import the object's current base-colour image as a layer (keeps existing look as starting point)"""

    bl_idname = "ats.layer_import_existing"
    bl_label = "Import Existing Texture"
    image_name: bpy.props.StringProperty(name="Image")

    @classmethod
    def poll(cls, context):
        return active_mesh(context) is not None

    def invoke(self, context, event):
        from .. import meshio
        obj = active_mesh(context)
        tex = meshio.existing_textures(obj)
        if not tex:
            self.report({"WARNING"}, "No image textures found in this object's materials")
            return {"CANCELLED"}
        if not self.image_name:
            self.image_name = tex[0][1]
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        self.layout.prop_search(self, "image_name", bpy.data, "images")

    def execute(self, context):
        obj = active_mesh(context)
        img = bpy.data.images.get(self.image_name)
        if img is None or img.size[0] == 0:
            self.report({"ERROR"}, "Image not found or empty")
            return {"CANCELLED"}
        storage.ensure_uid(obj)
        sc = context.scene.ats
        if not obj.ats.uv_map:
            from .. import meshio
            obj.ats.uv_map = meshio.default_uv_name(obj)
        if obj.ats.texture_size == 0:
            obj.ats.texture_size = int(sc.texture_size)
        size = obj.ats.texture_size
        arr = imageio.to_uint8(compat.image_to_array(img))
        if arr.shape[:2] != (size, size):
            arr = imageio.resize(arr, size, size)
        arr[..., 3] = 255
        layers.add_layer(obj, arr, name=f"Base: {img.name}", kind="BASE", index=0)
        layers.recomposite(obj, save=True)
        services.apply_material(obj, sc)
        return {"FINISHED"}


class ATS_OT_layer_add_paint(bpy.types.Operator):
    """Add an empty transparent layer for hand painting"""

    bl_idname = "ats.layer_add_paint"
    bl_label = "Add Paint Layer"

    @classmethod
    def poll(cls, context):
        obj = active_mesh(context)
        return obj is not None and bool(obj.data.uv_layers)

    def execute(self, context):
        obj = active_mesh(context)
        storage.ensure_uid(obj)
        sc = context.scene.ats
        if obj.ats.texture_size == 0:
            obj.ats.texture_size = int(sc.texture_size)
        size = obj.ats.texture_size
        layers.add_layer(obj, np.zeros((size, size, 4), np.uint8), name="Paint", kind="PAINT")
        return {"FINISHED"}


class ATS_OT_layer_edit(_LayerOp, bpy.types.Operator):
    """Open the active layer as an image to paint/edit it (Texture Paint or Image Editor)"""

    bl_idname = "ats.layer_edit"
    bl_label = "Edit Layer Pixels"
    paint_on_model: bpy.props.BoolProperty(name="Paint on Model", default=True)

    def execute(self, context):
        obj = active_mesh(context)
        ly = obj.ats.layers[obj.ats.active_layer]
        path = storage.resolve(ly.filepath)
        name = f"ATS Edit {obj.name} {ly.uid}"
        img = bpy.data.images.get(name)
        if img is None:
            img = bpy.data.images.load(path, check_existing=False)
            img.name = name
        else:
            img.reload()
        img["ats_layer_uid"] = ly.uid
        img["ats_object"] = obj.name
        if self.paint_on_model:
            uv = obj.ats.uv_map
            if uv in obj.data.uv_layers:
                obj.data.uv_layers.active = obj.data.uv_layers[uv]
            if context.mode != "PAINT_TEXTURE":
                bpy.ops.object.mode_set(mode="TEXTURE_PAINT")
            ip = context.tool_settings.image_paint
            ip.mode = "IMAGE"
            ip.canvas = img
        else:
            show_image(context, img)
        self.report({"INFO"}, "Edit the layer, then press 'Apply Layer Edits'")
        return {"FINISHED"}


class ATS_OT_layer_apply_edits(bpy.types.Operator):
    """Store edits made to a layer image as a new version of that layer (undoable)"""

    bl_idname = "ats.layer_apply_edits"
    bl_label = "Apply Layer Edits"

    @classmethod
    def poll(cls, context):
        return active_mesh(context) is not None

    def execute(self, context):
        obj = active_mesh(context)
        if context.mode == "PAINT_TEXTURE":
            bpy.ops.object.mode_set(mode="OBJECT")
        applied = 0
        for img in list(bpy.data.images):
            uid = img.get("ats_layer_uid")
            if not uid or img.get("ats_object") != obj.name:
                continue
            i, ly = layers.find_layer(obj, uid)
            if ly is None:
                continue
            if applied == 0:
                layers.push_undo(obj, "Edit layer pixels")
            arr = compat.image_to_array(img)
            new_path = os.path.join(storage.object_dir(obj), f"layer_{uid}_edit{np.random.randint(1 << 30)}.png")
            storage.write_png(new_path, arr)
            ly.filepath = storage.to_stored(new_path)
            bpy.data.images.remove(img)
            applied += 1
        if not applied:
            self.report({"WARNING"}, "No layer is being edited")
            return {"CANCELLED"}
        layers.recomposite(obj, save=True)
        return {"FINISHED"}


class ATS_OT_recomposite(bpy.types.Operator):
    """Rebuild and save the base colour texture from the visible layers"""

    bl_idname = "ats.recomposite"
    bl_label = "Update Texture"

    @classmethod
    def poll(cls, context):
        obj = active_mesh(context)
        return obj is not None and len(obj.ats.layers) > 0

    def execute(self, context):
        obj = active_mesh(context)
        run_safely(self, layers.recomposite, obj, save=True)
        services.apply_material(obj, context.scene.ats)
        return {"FINISHED"}


class ATS_OT_history_restore(bpy.types.Operator):
    """Add the layer produced by this history entry back to the stack"""

    bl_idname = "ats.history_restore"
    bl_label = "Restore From History"
    index: IntProperty()

    def execute(self, context):
        obj = active_mesh(context)
        if obj is None or not (0 <= self.index < len(obj.ats.history)):
            return {"CANCELLED"}
        h = obj.ats.history[self.index]
        if not h.filepath or not os.path.exists(storage.resolve(h.filepath)):
            self.report({"ERROR"}, "This entry has no stored result (failed, or files were removed)")
            return {"CANCELLED"}
        layers.add_layer(obj, None, name=f"Restored: {h.layer_name or h.kind}", kind="VIEW", info=h.prompt,
                         filepath=storage.resolve(h.filepath), history_uid=h.uid)
        layers.recomposite(obj, save=True)
        services.apply_material(obj, context.scene.ats)
        return {"FINISHED"}


class ATS_OT_history_clear(bpy.types.Operator):
    """Remove all history entries of the active object (files stay on disk)"""

    bl_idname = "ats.history_clear"
    bl_label = "Clear History"

    def execute(self, context):
        obj = active_mesh(context)
        if obj is not None:
            obj.ats.history.clear()
        return {"FINISHED"}


CLASSES = (ATS_OT_layer_op, ATS_OT_layers_undo, ATS_OT_layer_import_existing, ATS_OT_layer_add_paint,
           ATS_OT_layer_edit, ATS_OT_layer_apply_edits, ATS_OT_recomposite, ATS_OT_history_restore,
           ATS_OT_history_clear)
