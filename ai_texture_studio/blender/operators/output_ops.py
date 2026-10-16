"""PBR maps, AO bake, saving, packing, export, material restore."""

import os

import bpy

from .. import bake, export, layers, materials, services, storage
from .common import active_mesh, run_safely


class _NeedsLayers:
    @classmethod
    def poll(cls, context):
        obj = active_mesh(context)
        return obj is not None and len(obj.ats.layers) > 0


class ATS_OT_pbr_generate(_NeedsLayers, bpy.types.Operator):
    """Create the enabled PBR maps from the base colour (labelled approximations) and wire them up"""

    bl_idname = "ats.pbr_generate"
    bl_label = "Generate Maps"

    def execute(self, context):
        obj = active_mesh(context)
        wm = context.window_manager
        wm.progress_begin(0, 1)
        try:
            made = run_safely(self, services.generate_pbr, context, obj)
            if made is None:
                return {"CANCELLED"}
            if context.scene.ats.pbr_ao:
                if run_safely(self, self._bake, context, obj) is None:
                    return {"CANCELLED"}
                made.append("AO (baked)")
        finally:
            wm.progress_end()
        self.report({"INFO"}, "Created: " + ", ".join(made) if made else "No maps enabled")
        return {"FINISHED"}

    @staticmethod
    def _bake(context, obj):
        sc = context.scene.ats
        img = bake.bake_ao(context, obj, layers.texture_size(obj, sc), sc.ao_samples, obj.ats.uv_map)
        services.register_baked_ao(obj, img)
        services.apply_material(obj, sc)
        return img


class ATS_OT_bake_ao(_NeedsLayers, bpy.types.Operator):
    """Bake ambient occlusion from the mesh geometry with Cycles (blocks Blender while baking)"""

    bl_idname = "ats.bake_ao"
    bl_label = "Bake AO"

    def execute(self, context):
        obj = active_mesh(context)
        if not materials.is_applied(obj):
            services.apply_material(obj, context.scene.ats)
        img = run_safely(self, ATS_OT_pbr_generate._bake, context, obj)
        if img is None:
            return {"CANCELLED"}
        self.report({"INFO"}, f"Baked {img.name}")
        return {"FINISHED"}


class ATS_OT_save_textures(_NeedsLayers, bpy.types.Operator):
    """Write all maps as PNG files with clear names to the export folder"""

    bl_idname = "ats.save_textures"
    bl_label = "Save Textures"

    def execute(self, context):
        obj = active_mesh(context)
        sc = context.scene.ats
        layers.recomposite(obj, save=True)
        files = run_safely(self, export.save_textures, obj, services.current_maps(obj), sc.export_dir,
                           orm=sc.export_orm, dx_normal=sc.export_dx_normal)
        if files is None:
            return {"CANCELLED"}
        self.report({"INFO"}, f"Saved {len(files)} textures to {bpy.path.abspath(sc.export_dir)}")
        return {"FINISHED"}


class ATS_OT_pack_textures(_NeedsLayers, bpy.types.Operator):
    """Pack the generated maps into the .blend file"""

    bl_idname = "ats.pack_textures"
    bl_label = "Pack Into .blend"

    def execute(self, context):
        obj = active_mesh(context)
        layers.recomposite(obj, save=True)
        n = export.pack_images(services.current_maps(obj))
        self.report({"INFO"}, f"Packed {n} image(s)")
        return {"FINISHED"}


class ATS_OT_export_model(bpy.types.Operator):
    """Export the selected object with its textured material (game-ready glTF/FBX)"""

    bl_idname = "ats.export_model"
    bl_label = "Export Model"

    @classmethod
    def poll(cls, context):
        return active_mesh(context) is not None

    def execute(self, context):
        obj = active_mesh(context)
        sc = context.scene.ats
        if obj.ats.layers:
            layers.recomposite(obj, save=True)
            services.apply_material(obj, sc)
            export.save_textures(obj, services.current_maps(obj), sc.export_dir, orm=sc.export_orm,
                                 dx_normal=sc.export_dx_normal)
        path = os.path.join(sc.export_dir, storage.safe_name(obj.name))
        out = run_safely(self, export.export_model, context, obj, path, sc.export_format)
        if out is None:
            return {"CANCELLED"}
        self.report({"INFO"}, f"Exported {out}")
        return {"FINISHED"}


class ATS_OT_restore_materials(bpy.types.Operator):
    """Put the object's original materials back (generated materials and textures are kept)"""

    bl_idname = "ats.restore_materials"
    bl_label = "Restore Original Materials"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        obj = active_mesh(context)
        return obj is not None and materials.is_applied(obj)

    def execute(self, context):
        n = materials.restore_originals(active_mesh(context))
        self.report({"INFO"}, f"Restored {n} slot(s)")
        return {"FINISHED"}


class ATS_OT_apply_material(_NeedsLayers, bpy.types.Operator):
    """(Re)apply the generated textures to the object's material slots"""

    bl_idname = "ats.apply_material"
    bl_label = "Apply Material"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        obj = active_mesh(context)
        layers.recomposite(obj, save=True)
        run_safely(self, services.apply_material, obj, context.scene.ats)
        return {"FINISHED"}


class ATS_OT_transfer_uv(_NeedsLayers, bpy.types.Operator):
    """Re-bake the generated maps onto another UV map of this mesh (e.g. the original game UVs)"""

    bl_idname = "ats.transfer_uv"
    bl_label = "Transfer to UV Map"

    def execute(self, context):
        obj = active_mesh(context)
        sc = context.scene.ats
        names = run_safely(self, services.transfer_to_uv, context, obj, sc.transfer_uv)
        if names is None:
            return {"CANCELLED"}
        self.report({"INFO"}, f"Created {len(names)} image(s) for UV map '{sc.transfer_uv}': {', '.join(names)}")
        return {"FINISHED"}


CLASSES = (ATS_OT_pbr_generate, ATS_OT_bake_ao, ATS_OT_save_textures, ATS_OT_pack_textures, ATS_OT_export_model,
           ATS_OT_restore_materials, ATS_OT_apply_material, ATS_OT_transfer_uv)
