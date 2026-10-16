"""Geometric ambient-occlusion bake with Cycles (main thread, blocking).

Everything the bake touches (render engine, samples, selection, active
object, temporary image nodes) is restored afterwards.
"""

from __future__ import annotations

import os

import bpy

from . import compat, storage

TEMP_NODE = "ATS Bake Target"


def bake_ao(context, obj, size: int, samples: int = 32, uv_name: str = "") -> bpy.types.Image:
    scene = context.scene
    if not obj.material_slots or any(s.material is None for s in obj.material_slots):
        raise RuntimeError("Every material slot needs a material before baking (generate a texture first).")
    name = f"ATS_{obj.name}_Ao"
    img = storage.get_or_create_image(name, size, non_color=True)
    prev = {
        "engine": scene.render.engine,
        "samples": getattr(scene.cycles, "samples", None) if hasattr(scene, "cycles") else None,
        "active": context.view_layer.objects.active,
        "selected": [o for o in context.view_layer.objects if o.select_get()],
        "mode": obj.mode,
        "uv_active": obj.data.uv_layers.active_index,
    }
    temp_nodes = []
    try:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        scene.render.engine = "CYCLES"
        scene.cycles.samples = samples
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        if uv_name and uv_name in obj.data.uv_layers:
            obj.data.uv_layers.active = obj.data.uv_layers[uv_name]
        for mat in {s.material for s in obj.material_slots}:
            compat.ensure_nodes(mat)
            nt = mat.node_tree
            n = nt.nodes.new("ShaderNodeTexImage")
            n.name = TEMP_NODE
            n.image = img
            for other in nt.nodes:
                other.select = False
            n.select = True
            nt.nodes.active = n
            temp_nodes.append((mat, n.name))
        bpy.ops.object.bake(type="AO", margin=8, use_clear=True, target="IMAGE_TEXTURES",
                            uv_layer=uv_name if uv_name else "")
    finally:
        for mat, nname in temp_nodes:
            node = mat.node_tree.nodes.get(nname)
            if node is not None:
                mat.node_tree.nodes.remove(node)
        scene.render.engine = prev["engine"]
        if prev["samples"] is not None:
            scene.cycles.samples = prev["samples"]
        for o in context.view_layer.objects:
            o.select_set(o in prev["selected"])
        context.view_layer.objects.active = prev["active"]
        if 0 <= prev["uv_active"] < len(obj.data.uv_layers):
            obj.data.uv_layers.active_index = prev["uv_active"]
    path = os.path.join(storage.object_dir(obj), f"{storage.safe_name(obj.name)}_AO.png")
    storage.save_image_datablock(img, path)
    compat.set_colorspace(img, True)
    return img
