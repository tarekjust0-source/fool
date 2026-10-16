"""Extract a world-space triangle snapshot from a Blender object (main thread)."""

from __future__ import annotations

import bpy
import numpy as np

from ..core.meshdata import MeshData, compute_face_normals
from . import compat


def default_uv_name(obj) -> str:
    me = obj.data
    if not me.uv_layers:
        return ""
    st = obj.ats
    if st.uv_map and st.uv_map in me.uv_layers:
        return st.uv_map
    for uv in me.uv_layers:
        if uv.active_render:
            return uv.name
    return me.uv_layers.active.name if me.uv_layers.active else me.uv_layers[0].name


def extract(obj, context=None, *, uv_name: str | None = None, use_modifiers: bool = True,
            extra_uvs=()) -> MeshData:
    ctx = context or bpy.context
    if obj.type != "MESH":
        raise ValueError("Active object is not a mesh")
    if obj.mode == "EDIT":
        obj.update_from_editmode()
    uv_name = default_uv_name(obj) if uv_name is None else uv_name
    if use_modifiers:
        dg = ctx.evaluated_depsgraph_get()
        src = obj.evaluated_get(dg)
        me = src.to_mesh()
    else:
        src = obj
        me = obj.to_mesh()
    try:
        me.calc_loop_triangles()
        T = len(me.loop_triangles)
        V = len(me.vertices)
        co = np.empty(V * 3, np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        M = np.array(obj.matrix_world, np.float64)
        pos = co @ M[:3, :3].T + M[:3, 3]
        tv = np.empty(T * 3, np.int32)
        me.loop_triangles.foreach_get("vertices", tv)
        tris = tv.reshape(-1, 3)
        tl = np.empty(T * 3, np.int32)
        me.loop_triangles.foreach_get("loops", tl)
        loops = tl.reshape(-1, 3)
        mat = np.empty(T, np.int32)
        me.loop_triangles.foreach_get("material_index", mat)
        poly = np.empty(T, np.int32)
        me.loop_triangles.foreach_get("polygon_index", poly)
        psel = np.zeros(len(me.polygons), bool)
        if len(me.polygons):
            me.polygons.foreach_get("select", psel)
        if uv_name and uv_name in me.uv_layers:
            uv = compat.uv_vectors(me.uv_layers[uv_name])
            tri_uvs = uv[loops]
        else:
            uv_name = ""
            tri_uvs = np.zeros((T, 3, 2), np.float32)
        extra = {}
        for name in extra_uvs:
            if name in me.uv_layers:
                extra[name] = compat.uv_vectors(me.uv_layers[name])[loops]
        cn_local = compat.corner_normals(me)[loops]  # (T,3,3)
        nm = np.linalg.inv(M[:3, :3]).T
        cn = cn_local.reshape(-1, 3) @ nm.T
        cn /= np.maximum(np.linalg.norm(cn, axis=1, keepdims=True), 1e-12)
        fn = compute_face_normals(pos, tris)
        if np.linalg.det(M[:3, :3]) < 0:  # mirrored object: winding flips
            fn = -fn
        R = M[:3, :3] / np.maximum(np.linalg.norm(M[:3, :3], axis=0, keepdims=True), 1e-12)
        return MeshData(name=obj.name, positions=pos, tris=tris, tri_uvs=tri_uvs.astype(np.float32),
                        tri_normals=fn.astype(np.float32), corner_normals=cn.reshape(-1, 3, 3).astype(np.float32),
                        tri_material=mat, tri_selected=psel[poly] if len(psel) else np.zeros(T, bool),
                        basis=R, uv_name=uv_name, extra_uvs=extra)
    finally:
        src.to_mesh_clear()


def existing_textures(obj) -> list[tuple[str, str, int, int]]:
    """(slot material, image name, width, height) for image nodes in slot materials."""
    out = []
    for slot in obj.material_slots:
        mat = slot.material
        if not mat or not mat.node_tree:
            continue
        for node in mat.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image is not None:
                out.append((mat.name, node.image.name, node.image.size[0], node.image.size[1]))
    return out


def shared_mesh_users(obj) -> int:
    return max(0, obj.data.users - 1)
