"""Material setup that never destroys the user's materials.

PER_SLOT: each slot's material is *copied* to ``<name>_ATS`` and the copy is
wired with the generated maps; the original gets a fake user so it is kept
and can be restored with "Restore Original Materials".
SINGLE: one new ``ATS_<object>`` material on every slot.

Node layout: UV Map -> Image Texture(s) -> Principled BSDF, with correct
colour spaces (Base Color sRGB; data maps Non-Color), a tangent-space
Normal Map node bound to the same UV map, and a "glTF Material Output"
group for AO so the glTF exporter writes the occlusion texture.
"""

from __future__ import annotations

import json

import bpy

from . import compat

PREFIX = "ATS "
MAP_SOCKETS = {
    "BASE_COLOR": "Base Color",
    "ROUGHNESS": "Roughness",
    "METALLIC": "Metallic",
    "NORMAL": "Normal",
}
NON_COLOR = {"ROUGHNESS", "METALLIC", "NORMAL", "HEIGHT", "AO"}
GLTF_GROUP = "glTF Material Output"


def _state(obj) -> dict:
    try:
        return json.loads(obj.ats.material_state) if obj.ats.material_state else {}
    except ValueError:
        return {}


def _save_state(obj, st: dict):
    obj.ats.material_state = json.dumps(st)


def is_applied(obj) -> bool:
    return bool(_state(obj).get("slots"))


def _new_material(name: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    compat.ensure_nodes(mat)
    return mat


def ensure_materials(obj, mode: str = "PER_SLOT") -> list[bpy.types.Material]:
    """Make sure every slot uses an ATS material. Returns the ATS materials."""
    st = _state(obj)
    if st.get("slots") and st.get("mode") == mode:
        mats = []
        ok = True
        for entry, slot in zip(st["slots"], obj.material_slots):
            mat = bpy.data.materials.get(entry.get("ats") or "")
            if mat is None:
                ok = False
                break
            if slot.material != mat:
                slot.material = mat
            mats.append(mat)
        if ok and len(st["slots"]) == len(obj.material_slots):
            return mats
    if st.get("slots"):
        restore_originals(obj)
        st = {}
    st = {"mode": mode, "slots": [], "added_slot": False}
    if len(obj.material_slots) == 0:
        obj.data.materials.append(None)
        st["added_slot"] = True
    single = None
    if mode == "SINGLE":
        single = _new_material(f"ATS_{obj.name}")
        single["ats_owner"] = obj.ats.uid
    mats = []
    for slot in obj.material_slots:
        orig = slot.material
        entry = {"original": orig.name if orig else None,
                 "original_fake_user": bool(orig.use_fake_user) if orig else False, "link": slot.link}
        if single is not None:
            mat = single
        elif orig is not None:
            mat = orig.copy()
            mat.name = f"{orig.name}_ATS"
            mat["ats_owner"] = obj.ats.uid
        else:
            mat = _new_material(f"ATS_{obj.name}")
            mat["ats_owner"] = obj.ats.uid
        if orig is not None:
            orig.use_fake_user = True
        compat.ensure_nodes(mat)
        slot.material = mat
        entry["ats"] = mat.name
        st["slots"].append(entry)
        if mat not in mats:
            mats.append(mat)
    _save_state(obj, st)
    return mats


def restore_originals(obj) -> int:
    st = _state(obj)
    slots = st.get("slots") or []
    n = 0
    for entry, slot in zip(slots, obj.material_slots):
        orig = bpy.data.materials.get(entry["original"]) if entry.get("original") else None
        slot.material = orig
        if orig is not None:
            orig.use_fake_user = entry.get("original_fake_user", False)
        n += 1
    if st.get("added_slot") and len(obj.data.materials) == 1 and obj.data.materials[0] is None:
        obj.data.materials.pop(index=0)
    obj.ats.material_state = ""
    return n


def _find_output(nt):
    outs = [n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"]
    for n in outs:
        if n.is_active_output:
            return n
    if outs:
        return outs[0]
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    return out


def _find_principled(nt, out):
    surf = out.inputs.get("Surface")
    if surf is not None and surf.is_linked:
        node = surf.links[0].from_node
        if node.type == "BSDF_PRINCIPLED":
            return node
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED":
            nt.links.new(n.outputs[0], out.inputs["Surface"])
            return n
    n = nt.nodes.new("ShaderNodeBsdfPrincipled")
    n.location = (out.location.x - 300, out.location.y)
    nt.links.new(n.outputs[0], out.inputs["Surface"])
    return n


def _node(nt, name, type_, location, label=None):
    n = nt.nodes.get(name)
    if n is None or n.bl_idname != type_:
        if n is not None:
            nt.nodes.remove(n)
        n = nt.nodes.new(type_)
        n.name = name
    n.label = label or name.replace(PREFIX, "")
    n.location = location
    return n


def _gltf_group():
    g = bpy.data.node_groups.get(GLTF_GROUP)
    if g is None:
        g = bpy.data.node_groups.new(GLTF_GROUP, "ShaderNodeTree")
        compat.new_group_input(g, "Occlusion", "NodeSocketFloat")
        g.nodes.new("NodeGroupInput")
    return g


def wire(mat: bpy.types.Material, maps: dict, uv_name: str, *, ao_multiply: bool = False,
         pixel_art: bool = False):
    """Create/update ATS nodes in ``mat``. ``maps``: kind -> bpy Image."""
    compat.ensure_nodes(mat)
    nt = mat.node_tree
    out = _find_output(nt)
    bsdf = _find_principled(nt, out)
    x0, y0 = bsdf.location.x - 900, bsdf.location.y
    uvn = _node(nt, PREFIX + "UV", "ShaderNodeUVMap", (x0 - 250, y0 - 200), "ATS UV Map")
    uvn.uv_map = uv_name
    links = nt.links
    row = 0
    tex_nodes = {}
    for kind in ("BASE_COLOR", "ROUGHNESS", "METALLIC", "NORMAL", "HEIGHT", "AO"):
        img = maps.get(kind)
        name = PREFIX + kind.title().replace("_", " ")
        if img is None:
            continue
        compat.set_colorspace(img, kind in NON_COLOR)
        t = _node(nt, name, "ShaderNodeTexImage", (x0, y0 - row * 280), f"ATS {kind.replace('_', ' ').title()}")
        t.image = img
        t.interpolation = "Closest" if (pixel_art and kind == "BASE_COLOR") else "Linear"
        links.new(uvn.outputs["UV"], t.inputs["Vector"])
        tex_nodes[kind] = t
        row += 1

    def link_to(socket_name, from_socket):
        sock = bsdf.inputs.get(socket_name)
        if sock is None:
            return
        for lk in list(sock.links):
            links.remove(lk)
        links.new(from_socket, sock)

    if "BASE_COLOR" in tex_nodes:
        col = tex_nodes["BASE_COLOR"].outputs["Color"]
        if ao_multiply and "AO" in tex_nodes:
            mix = _node(nt, PREFIX + "AO Multiply", "ShaderNodeMix", (x0 + 350, y0), "ATS AO Multiply")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs[0].default_value = 1.0
            links.new(col, mix.inputs[6])
            links.new(tex_nodes["AO"].outputs["Color"], mix.inputs[7])
            col = mix.outputs[2]
        else:
            m = nt.nodes.get(PREFIX + "AO Multiply")
            if m is not None:
                nt.nodes.remove(m)
        link_to("Base Color", col)
    if "ROUGHNESS" in tex_nodes:
        link_to("Roughness", tex_nodes["ROUGHNESS"].outputs["Color"])
    if "METALLIC" in tex_nodes:
        link_to("Metallic", tex_nodes["METALLIC"].outputs["Color"])
    if "NORMAL" in tex_nodes:
        nm = _node(nt, PREFIX + "Normal Map", "ShaderNodeNormalMap", (x0 + 350, y0 - 3 * 280), "ATS Normal Map")
        nm.space = "TANGENT"
        nm.uv_map = uv_name
        links.new(tex_nodes["NORMAL"].outputs["Color"], nm.inputs["Color"])
        link_to("Normal", nm.outputs["Normal"])
    if "AO" in tex_nodes:
        grp = _node(nt, PREFIX + "glTF Output", "ShaderNodeGroup", (out.location.x, out.location.y - 250),
                    "glTF Material Output")
        grp.node_tree = _gltf_group()
        occ = grp.inputs.get("Occlusion")
        if occ is not None:
            for lk in list(occ.links):
                links.remove(lk)
            links.new(tex_nodes["AO"].outputs["Color"], occ)
    # Frame for readability
    frame = _node(nt, PREFIX + "Frame", "NodeFrame", (x0 - 300, y0 + 100), "AI Texture Studio")
    for n in list(tex_nodes.values()) + [uvn]:
        n.parent = frame
    return bsdf


def apply(obj, maps: dict, uv_name: str, mode: str, **kw) -> list[bpy.types.Material]:
    mats = ensure_materials(obj, mode)
    for mat in mats:
        wire(mat, maps, uv_name, **kw)
    return mats
