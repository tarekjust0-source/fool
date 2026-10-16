"""Validation through real Cycles renders of the textured object.

These tests judge results the way a user would: by rendering the mesh with
Blender's own renderer through the material the add-on built, from cameras
that were *not* used for generation (where seams would be visible).
They use synthetic-but-deterministic providers (no network, no AI model).
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: E402

import bpy  # noqa: E402
import mathutils  # noqa: E402

MOD = bootstrap.install_and_enable()
prov_mod = __import__(f"{MOD}.providers", fromlist=["base"])
pbase = prov_mod.base
imageio = __import__(f"{MOD}.core.imageio", fromlist=["x"])
camlib = __import__(f"{MOD}.core.camera", fromlist=["x"])
render_mod = __import__(f"{MOD}.core.render", fromlist=["x"])
runtime = __import__(f"{MOD}.blender.runtime", fromlist=["x"])
layers_mod = __import__(f"{MOD}.blender.layers", fromlist=["x"])
services = __import__(f"{MOD}.blender.services", fromlist=["x"])
meshio = __import__(f"{MOD}.blender.meshio", fromlist=["x"])

RES = 192


def field(P):
    return np.clip(np.stack([0.5 + 0.4 * np.sin(P[:, 0] * 2.3 + P[:, 2] * 0.9),
                             0.5 + 0.4 * np.sin(P[:, 1] * 1.9 - P[:, 0] * 0.6),
                             0.5 + 0.4 * np.cos(P[:, 2] * 2.1 + P[:, 1] * 0.4)], axis=1), 0, 1).astype(np.float32)


class OracleProvider(pbase.Provider):
    """Paints a smooth 3D colour field consistently from every camera (an ideal multi-view model)."""
    id = "ORACLE"
    name = "Oracle"
    is_local = True
    requires_key = False
    default_model = "oracle"

    def __init__(self):
        self.mesh = None
        self.cams = []
        self.calls = 0
        self.solid = None          # when set: paint the masked region this colour (refinement checks)

    def capabilities(self, cfg):
        return {pbase.IMAGE_EDIT, pbase.MASK}

    def _generate(self, req, cfg, cancel=None):
        self.calls += 1
        h, w = req.height, req.width
        if self.solid is not None:
            img = req.image.astype(np.float32) / 255.0
            img[req.mask] = self.solid
            return pbase.ImageResult(imageio.to_uint8(img), self.id, "oracle")
        cam = camlib.ViewCamera.from_dict(req.camera)     # the camera this request was rendered from
        geo = render_mod.render_geometry(self.mesh, render_mod.resized_camera(cam, w))
        img = np.full((h, w, 3), 0.5, np.float32)
        m = geo.mask
        P = np.einsum("nk,nkj->nj", geo.bary[m].astype(np.float64), self.mesh.positions[self.mesh.tris[geo.tri[m]]])
        img[m] = field(P)
        return pbase.ImageResult(imageio.to_uint8(img), self.id, "oracle")


ORACLE = OracleProvider()
prov_mod.register(ORACLE)


@pytest.fixture(autouse=True)
def scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.cameras):
        for d in list(coll):
            try:
                coll.remove(d)
            except Exception:
                pass
    sc = bpy.context.scene.ats
    for prop in sc.bl_rna.properties:
        if prop.identifier != "rna_type" and prop.type != "COLLECTION" and not prop.is_readonly:
            sc.property_unset(prop.identifier)
    sc.views.clear()
    sc.prompt = "test"
    sc.texture_size = "1024"
    sc.view_size = "512"
    p = bpy.context.preferences.addons[MOD].preferences
    p.provider = "ORACLE"
    ORACLE.solid = None
    runtime._state["last"] = None
    yield


def make_active(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def prepare_oracle(obj):
    sc = bpy.context.scene.ats
    services.ensure_default_views(sc)
    mesh = meshio.extract(obj, bpy.context, uv_name=obj.ats.uv_map or None)
    ORACLE.mesh = mesh
    ORACLE.cams = [c for _, c in services.enabled_cameras(sc, mesh, int(sc.view_size))]
    return mesh


def generate(obj, **kw):
    make_active(obj)
    assert bpy.ops.ats.generate(**kw) == {"FINISHED"}
    job = runtime.last_job()
    assert job.status == "DONE", job.error
    return job


def cycles_render(obj, cam: "camlib.ViewCamera"):
    """Unlit render of the add-on's base colour through Blender's own material/UV path."""
    scene = bpy.context.scene
    cdata = bpy.data.cameras.new("ValCam")
    cobj = bpy.data.objects.new("ValCam", cdata)
    scene.collection.objects.link(cobj)
    cobj.matrix_world = mathutils.Matrix(np.linalg.inv(cam.view).tolist())
    cdata.type = "PERSP"
    cdata.sensor_fit = "VERTICAL"
    cdata.angle_y = 2 * np.arctan(1.0 / cam.proj[1, 1])
    cdata.clip_start = 0.01
    cdata.clip_end = 1000
    scene.camera = cobj
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.render.film_transparent = True
    saved = []
    for slot in obj.material_slots:
        nt = slot.material.node_tree
        out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
        prev = out.inputs["Surface"].links[0].from_socket if out.inputs["Surface"].is_linked else None
        em = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(nt.nodes["ATS Base Color"].outputs["Color"], em.inputs["Color"])
        nt.links.new(em.outputs[0], out.inputs["Surface"])
        saved.append((nt, out, prev, em))
    path = os.path.join(tempfile.mkdtemp(), "r.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    for nt, out, prev, em in saved:
        nt.nodes.remove(em)
        if prev is not None:
            nt.links.new(prev, out.inputs["Surface"])
    bpy.data.objects.remove(cobj)
    bpy.context.view_layer.update()
    with open(path, "rb") as fh:
        img = imageio.rgba(imageio.decode_image(fh.read()))
    return img


def symmetric_cam(name, direction, mesh, fov_deg: float = 40.0):
    """Plain perspective camera looking at the bounding-box centre (exactly reproducible in Blender)."""
    lo, hi = mesh.bounds()
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo)) / 2
    d = np.asarray(direction, np.float64)
    d /= np.linalg.norm(d)
    dist = radius / np.sin(np.radians(fov_deg) / 2) * 1.08
    up = (0, 1, 0) if abs(d[2]) > 0.99 else (0, 0, 1)
    view = camlib.look_at(center + d * dist, center, up)
    proj = camlib.perspective(np.radians(fov_deg), 1.0, dist * 0.05, dist * 4)
    return camlib.ViewCamera(name, view, proj, RES, RES, False, {})


def expected_image(mesh, cam):
    geo = render_mod.render_geometry(mesh, cam)
    img = np.zeros((cam.height, cam.width, 3), np.float32)
    m = geo.mask
    P = np.einsum("nk,nkj->nj", geo.bary[m].astype(np.float64), mesh.positions[mesh.tris[geo.tri[m]]])
    img[m] = field(P)
    return img, m


def _erode(m, n):
    masks = __import__(f"{MOD}.core.masks", fromlist=["x"])
    return masks.erode(m, n)


def render_error(obj, mesh, cam):
    rendered = cycles_render(obj, cam)
    exp, m = expected_image(mesh, cam)
    inside = _erode(m & (rendered[..., 3] > 250), 2)        # ignore silhouette pixels
    err = np.abs(rendered[..., :3].astype(np.float32) / 255.0 - exp).max(axis=2)
    return err[inside], inside.sum()


def test_seams_between_views_in_cycles_render():
    """Render from 3/4 angles that no generation camera used; seams would show as error spikes."""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=1.0)
    obj = bpy.context.active_object
    bpy.ops.object.shade_smooth()
    mesh = prepare_oracle(obj)
    generate(obj)
    for key in ("FRONT_RIGHT", "BACK_LEFT"):
        cam = symmetric_cam(key, camlib.VIEW_PRESETS[key], mesh)
        err, n = render_error(obj, mesh, cam)
        assert n > 5000
        print(f"{key}: mean {err.mean() * 255:.2f}/255, p99 {np.percentile(err, 99) * 255:.1f}/255")
        assert err.mean() < 6.0 / 255.0
        assert np.percentile(err, 99) < 40.0 / 255.0


def test_single_view_regeneration_in_cycles_render():
    bpy.ops.mesh.primitive_cube_add(size=2)
    obj = bpy.context.active_object
    mesh = prepare_oracle(obj)
    generate(obj)
    front = symmetric_cam("FRONT", camlib.VIEW_PRESETS["FRONT"], mesh)
    right = symmetric_cam("RIGHT", camlib.VIEW_PRESETS["RIGHT"], mesh)
    before_front = cycles_render(obj, front)
    before_right = cycles_render(obj, right)
    ORACLE.solid = np.array([0.9, 0.1, 0.1], np.float32)       # "regenerated" front is solid red
    generate(obj, only_view=0)
    after_front = cycles_render(obj, front)
    after_right = cycles_render(obj, right)
    fa = after_front[..., 3] > 250
    inner = _erode(fa, 12)
    red = (after_front[..., 0] > 200) & (after_front[..., 1] < 60)
    assert red[inner].mean() > 0.95, "front face must show the regenerated view"
    assert not np.array_equal(before_front, after_front)
    ra = _erode(after_right[..., 3] > 250, 12)
    diff = np.abs(after_right[..., :3].astype(int) - before_right[..., :3].astype(int)).max(axis=2)
    assert (diff[ra] <= 2).mean() > 0.97, "the right side must be unchanged"


def test_refinement_mask_maps_to_selected_face_in_render():
    bpy.ops.mesh.primitive_cube_add(size=2)
    obj = bpy.context.active_object
    prepare_oracle(obj)
    generate(obj)
    me = obj.data
    # select the face pointing to -Y (front) only
    for p in me.polygons:
        p.select = p.normal.y < -0.9
    sc = bpy.context.scene.ats
    sc.refine_source = "FACES"
    sc.refine_space = "UV"
    sc.refine_prompt = "solid blue"
    sc.refine_feather = 0
    sc.refine_grow = 0
    ORACLE.solid = np.array([0.1, 0.2, 0.95], np.float32)
    make_active(obj)
    mesh = meshio.extract(obj, bpy.context, uv_name=obj.ats.uv_map)
    front = symmetric_cam("FRONT_RIGHT", camlib.VIEW_PRESETS["FRONT_RIGHT"], mesh)
    before = cycles_render(obj, front)
    assert bpy.ops.ats.refine() == {"FINISHED"}
    assert bpy.ops.ats.refine_resolve(accept=True) == {"FINISHED"}
    after = cycles_render(obj, front)
    # project the 3D face classification of each pixel
    geo = render_mod.render_geometry(mesh, front)
    face_normal = mesh.tri_normals[np.maximum(geo.tri, 0)]
    is_front = geo.mask & (face_normal[..., 1] < -0.9)
    is_other = geo.mask & (face_normal[..., 1] >= -0.9)
    alpha = after[..., 3] > 250
    fi = _erode(is_front & alpha, 3)
    oi = _erode(is_other & alpha, 3)
    blue = (after[..., 2] > 200) & (after[..., 0] < 60)
    assert blue[fi].mean() > 0.95, "the selected face must receive the correction"
    diff = np.abs(after[..., :3].astype(int) - before[..., :3].astype(int)).max(axis=2)
    assert (diff[oi] <= 2).mean() > 0.98, "unselected faces must stay unchanged"


def test_exported_glb_works_in_clean_session():
    bpy.ops.mesh.primitive_cube_add(size=2)
    obj = bpy.context.active_object
    obj.name = "Crate"
    prepare_oracle(obj)
    generate(obj)
    sc = bpy.context.scene.ats
    sc.pbr_roughness = sc.pbr_metallic = sc.pbr_normal = True
    sc.pbr_ao = True
    sc.ao_samples = 4
    make_active(obj)
    assert bpy.ops.ats.pbr_generate() == {"FINISHED"}
    out = tempfile.mkdtemp()
    sc.export_dir = out + os.sep
    assert bpy.ops.ats.export_model() == {"FINISHED"}
    glb = os.path.join(out, "Crate.glb")
    original = layers_mod.current_texture(obj, obj.ats.texture_size)[0]
    ref = os.path.join(out, "expected.npy")
    np.save(ref, original)
    result = run_clean_blender(CLEAN_IMPORT_CHECK.format(glb=glb, ref=ref))
    assert result.get("ok"), result
    assert result["base_size"] == 1024
    assert result["base_error"] < 2.0 / 255.0, result
    assert result["roughness"] and result["metallic"] and result["normal"], result


CLEAN_IMPORT_CHECK = r"""
import json, sys
import bpy, numpy as np
out = {{"ok": False}}
try:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath={glb!r})
    objs = [o for o in bpy.data.objects if o.type == "MESH"]
    mat = objs[0].material_slots[0].material
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    def source(sock):
        s = bsdf.inputs[sock]
        if not s.is_linked:
            return None
        node = s.links[0].from_node
        for _ in range(6):
            if node.type == "TEX_IMAGE":
                return node.image
            ins = [i for i in node.inputs if i.is_linked]
            if not ins:
                return None
            node = ins[0].links[0].from_node
        return None
    base = source("Base Color")
    px = np.empty(base.size[0] * base.size[1] * 4, np.float32)
    base.pixels.foreach_get(px)
    px = px.reshape(base.size[1], base.size[0], 4)[::-1, :, :3]
    ref = np.load({ref!r})
    covered = np.abs(ref).sum(axis=2) > 0
    out.update(ok=True, meshes=len(objs), base_size=base.size[0],
               base_error=float(np.abs(px[covered] - ref[covered]).mean()),
               roughness=source("Roughness") is not None, metallic=source("Metallic") is not None,
               normal=source("Normal") is not None)
except Exception as exc:
    out["error"] = repr(exc)
print("ATS_RESULT " + json.dumps(out))
"""


def run_clean_blender(code: str) -> dict:
    """Run ``code`` in a brand-new Blender process (no add-on, factory settings)."""
    import json
    import subprocess
    binary = bpy.app.binary_path
    if binary and os.path.basename(binary).lower().startswith("blender"):
        cmd = [binary, "-b", "--factory-startup", "--python-expr", code]
    else:                                     # bpy as a Python module
        cmd = [sys.executable, "-c", code]
    env = {k: v for k, v in os.environ.items() if k != "BLENDER_USER_RESOURCES"}
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith("ATS_RESULT "):
            return json.loads(line[len("ATS_RESULT "):])
    return {"ok": False, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}
