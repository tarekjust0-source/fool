"""3D Viewport sidebar UI: View3D > Sidebar > "AI Texture" tab."""

import json

import bpy

from ... import providers
from ...core.pbr import MAP_PROVENANCE
from .. import compat, credentials, layers, materials, prefs, runtime

CATEGORY = "AI Texture"


def _obj(context):
    o = context.active_object
    return o if o is not None and o.type == "MESH" else None


class ATS_UL_views(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.label(text=item.label or item.preset, icon="VIEW_CAMERA" if item.preset != "CUSTOM" else "VIEW3D")
        if item.status:
            row.label(text=item.status)
        op = row.operator("ats.generate", text="", icon="PLAY", emboss=False)
        op.only_view = index


class ATS_UL_layers(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "visible", text="", icon="HIDE_OFF" if item.visible else "HIDE_ON", emboss=False)
        ico = {"VIEW": "VIEW_CAMERA", "FILL": "MOD_SMOOTH", "CORRECTION": "BRUSH_DATA", "BASE": "IMAGE_DATA",
               "PAINT": "TPAINT_HLT", "TILE": "TEXTURE", "MERGED": "SELECT_EXTEND"}.get(item.kind, "IMAGE_DATA")
        row.prop(item, "name", text="", emboss=False, icon=ico)
        if item.pending:
            row.label(text="pending", icon="TIME")
        sub = row.row()
        sub.scale_x = 0.55
        sub.prop(item, "opacity", text="", slider=True)

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        # top layer first, like painting applications
        order = list(reversed(range(len(items))))
        ordering = [0] * len(items)
        for pos, idx in enumerate(order):
            ordering[idx] = pos
        return [self.bitflag_filter_item] * len(items), ordering


class ATS_UL_history(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        ico = {"DONE": "CHECKMARK", "ACCEPTED": "CHECKMARK", "FAILED": "ERROR", "CANCELLED": "CANCEL",
               "REJECTED": "X", "PENDING": "TIME", "RUNNING": "SORTTIME"}.get(item.status, "DOT")
        row.label(text=f"{item.timestamp[5:16]} {item.kind}: {item.prompt[:40]}", icon=ico)
        if item.filepath:
            row.operator("ats.history_restore", text="", icon="RECOVER_LAST", emboss=False).index = index

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        n = len(items)
        return [self.bitflag_filter_item] * n, [n - 1 - i for i in range(n)]


class ATS_UL_prompts(bpy.types.UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "prompt", text="", emboss=False)
        row.operator("ats.prompt_use", text="", icon="IMPORT", emboss=False).index = index
        row.operator("ats.prompt_remove", text="", icon="X", emboss=False).index = index


class _Base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY


class ATS_PT_main(_Base, bpy.types.Panel):
    bl_label = "AI Texture Studio"
    bl_idname = "ATS_PT_main"

    def draw(self, context):
        layout = self.layout
        obj = _obj(context)
        job = runtime.active_job()
        last = runtime.last_job()
        if job is not None:
            box = layout.box()
            compat.progress(box, job.progress, job.message or job.title)
            box.operator("ats.cancel", icon="CANCEL")
        elif last is not None and last.status in ("FAILED", "CANCELLED"):
            box = layout.box()
            box.alert = last.status == "FAILED"
            box.label(text=f"Last request {last.status.lower()}", icon="ERROR" if last.status == "FAILED" else "CANCEL")
            col = box.column(align=True)
            for line in prefs._wrap(last.error or last.message, 42):
                col.label(text=line)
            if last.status == "FAILED" and not last.retryable:
                col.label(text="(fix the cause above, then retry)")
            box.operator("ats.retry", icon="FILE_REFRESH")

        box = layout.box()
        if obj is None:
            box.label(text="Select a mesh object", icon="INFO")
            return
        row = box.row()
        row.label(text=obj.name, icon="MESH_DATA")
        row.label(text=f"{len(obj.data.polygons):,} faces")
        if obj.data.uv_layers:
            row = box.row(align=True)
            row.prop_search(obj.ats, "uv_map", obj.data, "uv_layers", text="UV")
        else:
            box.label(text="No UV map", icon="ERROR")
        row = box.row(align=True)
        row.operator("ats.create_uv", icon="UV")
        row.operator("ats.analyze", icon="VIEWZOOM")
        if obj.ats.analysis:
            try:
                info = json.loads(obj.ats.analysis)
            except ValueError:
                info = None
            if info and info.get("object") == obj.name:
                col = box.column(align=True)
                d = info.get("dimensions", [0, 0, 0])
                col.label(text=f"Size {d[0]:.2f} x {d[1]:.2f} x {d[2]:.2f} m, {info.get('tri_count', 0):,} tris")
                if info.get("uv_name"):
                    col.label(text=f"UV '{info['uv_name']}': {info.get('uv_islands', 0)} islands, "
                                   f"{info.get('uv_coverage', 0):.0%} used, overlap {info.get('uv_overlap', 0):.0%}")
                mats = info.get("materials", [])
                col.label(text=f"{len(mats)} material slot(s): {', '.join(mats)[:60]}")
                for t in info.get("textures", [])[:3]:
                    col.label(text=t[:70], icon="IMAGE_DATA")
                for lvl, msg in info.get("messages", []):
                    ico = {"ERROR": "ERROR", "WARNING": "ERROR", "INFO": "INFO"}[lvl]
                    first = True
                    for line in prefs._wrap(msg, 46):
                        sub = col.row()
                        sub.alert = lvl == "ERROR"
                        sub.label(text=line, icon=ico if first else "BLANK1")
                        first = False

        try:
            p = prefs.get_prefs(context)
            prov = providers.get(p.provider)
            s = p.settings_for(prov.id)
            row = layout.row(align=True)
            label = f"{prov.name}: {s.model or prov.default_model or 'custom'}"
            row.label(text=label, icon="DESKTOP" if prov.is_local else "WORLD")
            row.operator("ats.open_prefs", text="", icon="PREFERENCES")
            if prov.requires_key and not credentials.get_key(prov.id, s):
                layout.label(text="API key missing – open settings", icon="ERROR")
            ok, msg = prefs.network_allowed(prov, providers.ProviderConfig(base_url=s.base_url))
            if not ok:
                layout.label(text="Online access disabled in Blender", icon="ERROR")
        except Exception as exc:  # noqa: BLE001
            layout.label(text=str(exc)[:60], icon="ERROR")


class ATS_PT_prompt(_Base, bpy.types.Panel):
    bl_label = "Prompt & Reference"
    bl_parent_id = "ATS_PT_main"

    def draw(self, context):
        sc = context.scene.ats
        layout = self.layout
        col = layout.column(align=True)
        col.prop(sc, "prompt", text="", icon="TEXT")
        col.prop(sc, "negative", text="", icon="CANCEL")
        layout.prop(sc, "art_direction", icon="WORLD")
        box = layout.box()
        row = box.row(align=True)
        row.prop(sc, "use_reference", text="")
        row.template_ID(sc, "reference", open="image.open")
        row.operator("ats.load_reference", text="", icon="FILEBROWSER")
        if sc.reference is not None and sc.use_reference:
            try:
                sc.reference.preview_ensure()
                box.template_icon(icon_value=sc.reference.preview.icon_id, scale=5.0)
            except Exception:  # noqa: BLE001
                pass
        row = layout.row(align=True)
        row.operator("ats.preview_prompt", icon="HIDE_OFF")
        if len(sc.prompt_history):
            layout.label(text="Recent prompts")
            layout.template_list("ATS_UL_prompts", "", sc, "prompt_history", sc, "active_prompt", rows=3)


class ATS_PT_style(_Base, bpy.types.Panel):
    bl_label = "Style & Resolution"
    bl_parent_id = "ATS_PT_main"

    def draw(self, context):
        sc = context.scene.ats
        layout = self.layout
        col = layout.column()
        col.prop(sc, "style")
        col.prop(sc, "material")
        col.prop(sc, "detail", slider=True)
        col.prop(sc, "wear", slider=True)
        col.prop(sc, "strength", slider=True)
        col.prop(sc, "delight")
        box = layout.box()
        row = box.row()
        row.prop(sc, "use_palette")
        if sc.use_palette:
            row.prop(sc, "palette_count", text="")
            r = box.row(align=True)
            for i in range(sc.palette_count):
                r.prop(sc, f"color_{i + 1}", text="")
        obj = _obj(context)
        if obj is not None:
            row = box.row(align=True)
            row.prop(obj.ats, "palette_lock", slider=True)
            if sc.style == "PIXEL_ART" or obj.ats.pixelate:
                row = layout.row(align=True)
                row.prop(obj.ats, "pixelate")
                sub = row.row(align=True)
                sub.active = obj.ats.pixelate
                sub.prop(obj.ats, "pixel_size", text="Px")
                sub.prop(obj.ats, "pixel_colors", text="Lv")
        row = layout.row(align=True)
        row.prop(sc, "texture_size")
        row.prop(sc, "view_size")
        if obj is not None and obj.ats.texture_size and obj.ats.layers and \
                str(obj.ats.texture_size) != sc.texture_size:
            layout.label(text=f"Existing layers are {obj.ats.texture_size}px (kept for this object)", icon="INFO")
        row = layout.row(align=True)
        row.prop(sc, "use_seed")
        sub = row.row()
        sub.active = sc.use_seed
        sub.prop(sc, "seed", text="")


class ATS_PT_generate(_Base, bpy.types.Panel):
    bl_label = "Generate Texture"
    bl_parent_id = "ATS_PT_main"

    def draw(self, context):
        sc = context.scene.ats
        layout = self.layout
        layout.prop(sc, "mode", expand=True)
        if sc.mode == "TILE":
            layout.prop(sc, "tile_scale")
        layout.prop(sc, "material_mode")
        col = layout.column(align=True)
        col.prop(sc, "regenerate")
        col.prop(sc, "use_modifiers")
        n_views = sum(1 for v in sc.views if v.enabled) if sc.mode == "MULTIVIEW" else 1
        if sc.mode == "MULTIVIEW" and len(sc.views) == 0:
            n_views = 6
        col = layout.column()
        col.scale_y = 1.6
        busy = runtime.active_job() is not None
        col.enabled = not busy
        col.operator("ats.generate", text=f"Generate ({n_views} request{'s' if n_views != 1 else ''})",
                     icon="SHADERFX").all_selected = False
        n_sel = len([o for o in context.selected_objects if o.type == "MESH"])
        if n_sel > 1:
            row = layout.row(align=True)
            row.operator("ats.generate", text=f"Generate All {n_sel} Selected",
                         icon="OUTLINER_OB_GROUP_INSTANCE").all_selected = True
            row.prop(sc, "chain_reference", text="", icon="LINKED")
        layout.label(text="Requests to hosted providers are billed by them.", icon="INFO")


class ATS_PT_multiview(_Base, bpy.types.Panel):
    bl_label = "Multi-View"
    bl_parent_id = "ATS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        sc = context.scene.ats
        layout = self.layout
        if len(sc.views) == 0:
            layout.label(text="Default views are created on first use")
        row = layout.row()
        row.template_list("ATS_UL_views", "", sc, "views", sc, "active_view", rows=5)
        col = row.column(align=True)
        col.operator("ats.view_add_viewport", text="", icon="ADD")
        col.operator("ats.view_remove", text="", icon="REMOVE")
        col.separator()
        col.operator("ats.view_move", text="", icon="TRIA_UP").direction = -1
        col.operator("ats.view_move", text="", icon="TRIA_DOWN").direction = 1
        col.separator()
        col.operator_menu_enum("ats.views_preset", "preset", text="", icon="PRESET")
        if 0 <= sc.active_view < len(sc.views):
            v = sc.views[sc.active_view]
            box = layout.box()
            row = box.row(align=True)
            row.prop(v, "label", text="")
            row.operator("ats.view_preview_camera", text="", icon="HIDE_OFF")
            if v.debug_dir:
                box.label(text="Inspect last run:")
                grid = box.grid_flow(columns=3, align=True)
                for k, lbl in (("render", "Render"), ("mask", "Mask"), ("generated", "AI Output"),
                               ("aligned", "Aligned"), ("depth", "Depth")):
                    grid.operator("ats.view_show", text=lbl).kind = k
        col = layout.column(align=True)
        row = col.row(align=True)
        row.prop(sc, "ortho")
        sub = row.row()
        sub.active = not sc.ortho
        sub.prop(sc, "fov")
        col.prop(sc, "align_to_object")
        col = layout.column(align=True)
        col.prop(sc, "min_facing", slider=True)
        col.prop(sc, "band_px")
        row = layout.row(align=True)
        row.prop(sc, "auto_align", toggle=True)
        row.prop(sc, "color_match", toggle=True)
        row.prop(sc, "fill_unseen", toggle=True)


class ATS_PT_refine(_Base, bpy.types.Panel):
    bl_label = "Local Refinement"
    bl_parent_id = "ATS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        sc = context.scene.ats
        obj = _obj(context)
        layout = self.layout
        if obj is None:
            return
        pending = [ly for ly in obj.ats.layers if ly.pending]
        if pending:
            box = layout.box()
            box.label(text=f"Pending: {pending[0].name}", icon="TIME")
            row = box.row(align=True)
            row.prop(pending[0], "visible", text="Compare (toggle)", toggle=True, icon="HIDE_OFF")
            row = box.row(align=True)
            row.scale_y = 1.3
            row.operator("ats.refine_resolve", text="Accept", icon="CHECKMARK").accept = True
            row.operator("ats.refine_resolve", text="Reject", icon="X").accept = False
            return
        layout.prop(sc, "refine_prompt", text="", icon="BRUSH_DATA")
        layout.prop(sc, "refine_source")
        box = layout.box()
        row = box.row(align=True)
        if context.mode == "PAINT_TEXTURE":
            row.operator("ats.mask_done", icon="CHECKMARK")
        else:
            row.operator("ats.mask_paint", icon="BRUSH_DATA")
        row.operator("ats.mask_from_faces", text="", icon="FACESEL")
        row.operator("ats.mask_clear", text="", icon="TRASH").invert = False
        row.operator("ats.mask_clear", text="", icon="ARROW_LEFTRIGHT").invert = True
        row = layout.row(align=True)
        row.prop(sc, "refine_space", expand=True)
        row = layout.row(align=True)
        row.prop(sc, "refine_grow")
        row.prop(sc, "refine_feather")
        col = layout.column()
        col.scale_y = 1.4
        col.operator("ats.refine", icon="SHADERFX")
        if sc.refine_space == "VIEW":
            layout.label(text="Uses the current viewport angle", icon="VIEW3D")


class ATS_PT_layers(_Base, bpy.types.Panel):
    bl_label = "Texture Layers"
    bl_parent_id = "ATS_PT_main"

    def draw(self, context):
        obj = _obj(context)
        layout = self.layout
        if obj is None:
            return
        st = obj.ats
        row = layout.row()
        row.template_list("ATS_UL_layers", "", st, "layers", st, "active_layer", rows=5)
        col = row.column(align=True)
        col.operator("ats.layer_add_paint", text="", icon="ADD")
        col.operator("ats.layer_op", text="", icon="REMOVE").action = "DELETE"
        col.separator()
        col.operator("ats.layer_op", text="", icon="TRIA_UP").action = "UP"
        col.operator("ats.layer_op", text="", icon="TRIA_DOWN").action = "DOWN"
        col.separator()
        col.operator("ats.layer_op", text="", icon="DUPLICATE").action = "DUPLICATE"
        col.operator("ats.layer_op", text="", icon="TRIA_DOWN_BAR").action = "MERGE_DOWN"
        if 0 <= st.active_layer < len(st.layers):
            ly = st.layers[st.active_layer]
            box = layout.box()
            row = box.row(align=True)
            row.prop(ly, "blend", text="")
            row.prop(ly, "opacity", slider=True)
            if ly.info:
                for line in prefs._wrap(ly.info, 48):
                    box.label(text=line)
            row = box.row(align=True)
            row.operator("ats.layer_edit", icon="TPAINT_HLT")
            row.operator("ats.layer_apply_edits", text="Apply Edits")
        row = layout.row(align=True)
        sub = row.row(align=True)
        sub.enabled = layers.can_undo(obj)
        sub.operator("ats.layers_undo", text="Undo", icon="LOOP_BACK").redo = False
        sub = row.row(align=True)
        sub.enabled = layers.can_redo(obj)
        sub.operator("ats.layers_undo", text="Redo", icon="LOOP_FORWARDS").redo = True
        row = layout.row(align=True)
        row.operator("ats.layer_import_existing", icon="IMPORT")
        row.operator("ats.layer_op", text="Flatten").action = "FLATTEN"
        row = layout.row(align=True)
        row.operator("ats.recomposite", icon="FILE_REFRESH")
        row.prop(st, "gutter")
        row = layout.row(align=True)
        row.operator("ats.layers_resize", icon="FULLSCREEN_ENTER")
        row.operator("ats.cleanup_files", icon="TRASH")


class ATS_PT_history(_Base, bpy.types.Panel):
    bl_label = "History"
    bl_parent_id = "ATS_PT_layers"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        obj = _obj(context)
        if obj is None:
            return
        st = obj.ats
        layout = self.layout
        layout.template_list("ATS_UL_history", "", st, "history", st, "active_history", rows=4)
        if 0 <= st.active_history < len(st.history):
            h = st.history[st.active_history]
            col = layout.column(align=True)
            col.label(text=f"{h.status} – {h.provider} {h.model}")
            for line in prefs._wrap(h.prompt, 48):
                col.label(text=line)
            for line in prefs._wrap(h.detail, 48):
                col.label(text=line)
        layout.operator("ats.history_clear", icon="TRASH")


class ATS_PT_pbr(_Base, bpy.types.Panel):
    bl_label = "PBR Maps"
    bl_parent_id = "ATS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        sc = context.scene.ats
        layout = self.layout
        col = layout.column(align=True)
        col.label(text="Base Color: " + MAP_PROVENANCE["BASE_COLOR"], icon="CHECKMARK")
        for prop, key in (("pbr_roughness", "ROUGHNESS"), ("pbr_metallic", "METALLIC"), ("pbr_normal", "NORMAL"),
                          ("pbr_height", "HEIGHT"), ("pbr_ao", "AO")):
            row = col.row()
            row.prop(sc, prop)
            row.label(text=MAP_PROVENANCE[key].split(" (")[0])
        box = layout.box()
        if sc.pbr_roughness:
            row = box.row(align=True)
            row.prop(sc, "rough_override", text="")
            sub = row.row()
            sub.active = sc.rough_override
            sub.prop(sc, "rough_base")
            box.prop(sc, "rough_variation", slider=True)
        if sc.pbr_metallic:
            row = box.row(align=True)
            row.prop(sc, "metal_value")
            row.prop(sc, "metal_from_mask", toggle=True)
        if sc.pbr_normal:
            box.prop(sc, "normal_strength")
        if sc.pbr_normal or sc.pbr_height:
            box.prop(sc, "height_detail")
        if sc.pbr_ao:
            row = box.row(align=True)
            row.prop(sc, "ao_samples")
            row.prop(sc, "ao_multiply", toggle=True)
        layout.label(text="Approximations are not measured material data.", icon="INFO")
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.operator("ats.pbr_generate", icon="NODE_MATERIAL")
        row.operator("ats.bake_ao", text="", icon="RENDER_STILL")


class ATS_PT_export(_Base, bpy.types.Panel):
    bl_label = "Export"
    bl_parent_id = "ATS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        sc = context.scene.ats
        obj = _obj(context)
        layout = self.layout
        layout.prop(sc, "export_dir")
        row = layout.row(align=True)
        row.prop(sc, "export_orm", toggle=True)
        row.prop(sc, "export_dx_normal", toggle=True)
        row = layout.row(align=True)
        row.operator("ats.save_textures", icon="FILE_IMAGE")
        row.operator("ats.pack_textures", icon="PACKAGE")
        layout.prop(sc, "export_format", text="")
        col = layout.column()
        col.scale_y = 1.3
        col.operator("ats.export_model", icon="EXPORT")
        if obj is not None and obj.data.uv_layers and len(obj.data.uv_layers) > 1:
            box = layout.box()
            box.label(text="Bake result onto another UV map")
            row = box.row(align=True)
            row.prop_search(sc, "transfer_uv", obj.data, "uv_layers", text="")
            row.operator("ats.transfer_uv", text="Transfer")
        if obj is not None and materials.is_applied(obj):
            row = layout.row(align=True)
            row.operator("ats.apply_material", icon="MATERIAL")
            row.operator("ats.restore_materials", icon="LOOP_BACK")


class ATS_PT_provider(_Base, bpy.types.Panel):
    bl_label = "AI Provider"
    bl_parent_id = "ATS_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        p = prefs.get_prefs(context)
        layout.prop(p, "provider", text="")
        try:
            prov = providers.get(p.provider)
        except providers.ProviderError as exc:
            layout.label(text=str(exc), icon="ERROR")
            return
        s = p.settings_for(prov.id)
        if prov.id != "MOCK":
            layout.prop(s, "model")
        if prov.requires_key or prov.id == "COMFYUI":
            row = layout.row(align=True)
            row.label(text=credentials.key_status(prov.id, s)[:48], icon="LOCKED")
            if s.key_mode in ("SESSION", "KEYRING"):
                row.operator("ats.set_api_key", text="", icon="KEY_HLT").provider = prov.id
        caps = prov.capabilities(prefs.provider_config(context, prov.id)[1]) if prov.id == "COMFYUI" else \
            prov.capabilities(providers.ProviderConfig())
        col = layout.column(align=True)
        b = providers.base
        col.label(text="Supports: " + ", ".join(n for flag, n in (
            (b.MASK, "true inpainting"), (b.SOFT_MASK, "guided edits"), (b.REFERENCE, "reference images"),
            (b.NEGATIVE_PROMPT, "negative prompt"), (b.SEED, "seed"), (b.DEPTH_CONTROL, "depth control"))
            if flag in caps))
        for line in prefs._wrap(prov.data_notice, 46):
            col.label(text=line)
        row = layout.row(align=True)
        row.operator("ats.test_connection", icon="URL").provider = prov.id
        row.operator("ats.open_prefs", text="All Settings", icon="PREFERENCES")


CLASSES = (ATS_UL_views, ATS_UL_layers, ATS_UL_history, ATS_UL_prompts, ATS_PT_main, ATS_PT_prompt, ATS_PT_style,
           ATS_PT_generate, ATS_PT_multiview, ATS_PT_refine, ATS_PT_layers, ATS_PT_history, ATS_PT_pbr,
           ATS_PT_export, ATS_PT_provider)
