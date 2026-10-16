"""Blender integration tests (run inside Blender's Python).

Run with the ``bpy`` module:   python -m pytest tests/blender
or real Blender:               blender -b --python tests/blender/run_in_blender.py

The extension ZIP is built, installed through Blender's extension system into
an isolated user profile and enabled; tests then drive the real operators.
All generation uses the offline *test* provider (synthetic images).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa: E402

import bpy  # noqa: E402

MOD = bootstrap.install_and_enable()
PKG = bootstrap.package()
prov_mod = __import__(f"{MOD}.providers", fromlist=["base"])
pbase = prov_mod.base
imageio = __import__(f"{MOD}.core.imageio", fromlist=["x"])
camlib = __import__(f"{MOD}.core.camera", fromlist=["x"])
runtime = __import__(f"{MOD}.blender.runtime", fromlist=["x"])
layers_mod = __import__(f"{MOD}.blender.layers", fromlist=["x"])
services = __import__(f"{MOD}.blender.services", fromlist=["x"])
meshio = __import__(f"{MOD}.blender.meshio", fromlist=["x"])


LAST_ERROR = []


def prefs():
    return bpy.context.preferences.addons[MOD].preferences


@pytest.fixture(autouse=True)
def clean_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.cameras, bpy.data.armatures):
        for d in list(coll):
            try:
                coll.remove(d)
            except Exception:
                pass
    p = prefs()
    p.provider = "MOCK"
    p.mock.mock_fail_mode = ""
    sc = bpy.context.scene.ats
    sc.prompt = "weathered dark iron armor with scratches"
    sc.negative = ""
    sc.texture_size = "512"
    sc.view_size = "512"
    sc.regenerate = False
    sc.mode = "MULTIVIEW"
    sc.views.clear()
    sc.material_mode = "PER_SLOT"
    sc.reference = None
    sc.ortho = False
    runtime._state["last"] = None
    LAST_ERROR.clear()
    yield
    runtime.shutdown()


def snapshot_mesh(obj):
    me = obj.data
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    return {
        "co": co.copy(), "nverts": len(me.vertices), "npolys": len(me.polygons), "nedges": len(me.edges),
        "uvs": [u.name for u in me.uv_layers], "matrix": [list(r) for r in obj.matrix_world],
        "mods": [(m.name, m.type) for m in obj.modifiers],
        "shape_keys": [k.name for k in me.shape_keys.key_blocks] if me.shape_keys else [],
        "vgroups": [g.name for g in obj.vertex_groups],
        "mat_index": [p.material_index for p in me.polygons],
    }


def assert_preserved(before, obj, *, allow_new_uv=None):
    after = snapshot_mesh(obj)
    assert np.allclose(before["co"], after["co"]), "vertex positions changed"
    for k in ("nverts", "npolys", "nedges", "mods", "shape_keys", "vgroups", "mat_index"):
        assert before[k] == after[k], f"{k} changed"
    assert np.allclose(np.array(before["matrix"]), np.array(after["matrix"])), "transform changed"
    uvs = list(after["uvs"])
    if allow_new_uv and allow_new_uv in uvs:
        uvs.remove(allow_new_uv)
    assert uvs == before["uvs"], "UV maps changed"


def ctx_override(obj):
    win = bpy.context.window_manager.windows[0] if bpy.context.window_manager.windows else None
    kw = {"active_object": obj, "object": obj, "selected_objects": [obj], "selected_editable_objects": [obj]}
    if win is not None:
        kw["window"] = win
    return bpy.context.temp_override(**kw)


def make_active(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def add_cube(name="Box", uv=True):
    bpy.ops.mesh.primitive_cube_add(size=2)
    obj = bpy.context.active_object
    obj.name = name
    if not uv:
        while obj.data.uv_layers:
            obj.data.uv_layers.remove(obj.data.uv_layers[0])
    return obj


def call(op, **kw):
    """Run an operator; ERROR reports raise RuntimeError in Python -> CANCELLED."""
    try:
        return op(**kw)
    except RuntimeError as exc:
        LAST_ERROR.append(str(exc))
        return {"CANCELLED"}


def generate(obj, **kw):
    make_active(obj)
    res = call(bpy.ops.ats.generate, **kw)
    return res, runtime.last_job()


# ---------------------------------------------------------------------------

def test_mesh_without_uv_is_rejected_then_unwrapped():
    obj = add_cube("NoUV", uv=False)
    before = snapshot_mesh(obj)
    res, job = generate(obj)
    assert res == {"CANCELLED"}
    assert "no UV map" in LAST_ERROR[-1]
    assert len(obj.ats.layers) == 0
    make_active(obj)
    assert bpy.ops.ats.create_uv(method="SMART") == {"FINISHED"}
    assert obj.ats.uv_map == "ATS_UV"
    assert_preserved(before, obj, allow_new_uv="ATS_UV")
    res, job = generate(obj)
    assert res == {"FINISHED"} and job.status == "DONE", job.error
    assert sum(1 for ly in obj.ats.layers if ly.kind == "VIEW") >= 4
    assert obj.ats.base_image is not None and obj.ats.base_image.size[0] == 512


def test_create_uv_preserves_existing_uvs_and_selection():
    obj = add_cube("HasUV")
    me = obj.data
    orig = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers[0].uv.foreach_get("vector", orig)
    me.polygons[0].select = True
    for p in me.polygons[1:]:
        p.select = False
    sel_before = [p.select for p in me.polygons]
    make_active(obj)
    assert bpy.ops.ats.create_uv(method="LIGHTMAP") == {"FINISHED"}
    after = np.empty(len(me.loops) * 2, np.float32)
    me.uv_layers["UVMap"].uv.foreach_get("vector", after)
    assert np.allclose(orig, after)
    assert [p.select for p in me.polygons] == sel_before
    assert "ATS_UV" in me.uv_layers


def test_existing_materials_multi_slot_are_preserved_and_restorable():
    obj = add_cube("Slots")
    m1 = bpy.data.materials.new("Painted")
    if bpy.app.version < (5, 0, 0):
        m1.use_nodes = True
    m1.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.123
    tex_img = bpy.data.images.new("old_tex", 64, 64)
    tn = m1.node_tree.nodes.new("ShaderNodeTexImage")
    tn.image = tex_img
    m2 = bpy.data.materials.new("Trim")
    obj.data.materials.append(m1)
    obj.data.materials.append(m2)
    for i, p in enumerate(obj.data.polygons):
        p.material_index = i % 2
    n_nodes_m1 = len(m1.node_tree.nodes)
    before = snapshot_mesh(obj)
    res, job = generate(obj)
    assert job.status == "DONE", job.error
    names = [s.material.name for s in obj.material_slots]
    assert names == ["Painted_ATS", "Trim_ATS"]
    # originals untouched and protected from garbage collection
    assert len(m1.node_tree.nodes) == n_nodes_m1
    assert abs(m1.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value - 0.123) < 1e-6
    assert m1.use_fake_user and m2.use_fake_user
    # copy keeps slot-specific settings
    cp = bpy.data.materials["Painted_ATS"]
    assert abs(cp.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value - 0.123) < 1e-6
    assert_preserved(before, obj)
    info = json.loads(services.analyze(bpy.context, obj) and obj.ats.analysis)
    assert any("old_tex" in t for t in info["textures"])
    make_active(obj)
    assert bpy.ops.ats.restore_materials() == {"FINISHED"}
    assert [s.material.name for s in obj.material_slots] == ["Painted", "Trim"]
    assert not m1.use_fake_user


def _shader_input_source(mat, socket):
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    s = bsdf.inputs[socket]
    return s.links[0].from_node if s.is_linked else None


def test_material_nodes_and_pbr_maps():
    obj = add_cube("PBR")
    sc = bpy.context.scene.ats
    sc.material = "METAL"
    sc.pbr_roughness = sc.pbr_metallic = sc.pbr_normal = True
    sc.pbr_height = True
    sc.pbr_ao = True
    sc.ao_samples = 4
    res, job = generate(obj)
    assert job.status == "DONE", job.error
    make_active(obj)
    assert bpy.ops.ats.pbr_generate() == {"FINISHED"}
    mat = obj.material_slots[0].material
    base = _shader_input_source(mat, "Base Color")
    assert base.type == "TEX_IMAGE" and base.image.colorspace_settings.name == "sRGB"
    for sock in ("Roughness", "Metallic"):
        n = _shader_input_source(mat, sock)
        assert n.type == "TEX_IMAGE"
        assert n.image.colorspace_settings.is_data or n.image.colorspace_settings.name == "Non-Color"
    nm = _shader_input_source(mat, "Normal")
    assert nm.type == "NORMAL_MAP" and nm.space == "TANGENT" and nm.uv_map == obj.ats.uv_map
    assert nm.inputs["Color"].links[0].from_node.image.colorspace_settings.name == "Non-Color"
    uvn = mat.node_tree.nodes["ATS UV"]
    assert uvn.uv_map == obj.ats.uv_map
    assert base.inputs["Vector"].links[0].from_node == uvn
    grp = mat.node_tree.nodes.get("ATS glTF Output")
    assert grp is not None and grp.inputs["Occlusion"].is_linked
    # metallic preset METAL -> 1.0 everywhere
    met = _shader_input_source(mat, "Metallic").image
    px = np.empty(met.size[0] * met.size[1] * 4, np.float32)
    met.pixels.foreach_get(px)
    assert px.reshape(-1, 4)[:, 0].mean() > 0.99
    ao = mat.node_tree.nodes["ATS Ao"].image
    aopx = np.empty(ao.size[0] * ao.size[1] * 4, np.float32)
    ao.pixels.foreach_get(aopx)
    assert 0.2 < aopx.reshape(-1, 4)[:, 0].mean() <= 1.0  # a real bake, not empty


def test_projection_alignment_matches_blender_render():
    """A screen-space pattern returned for one view must re-appear in a Cycles
    render from a Blender camera matching that view (validates camera model,
    rasteriser, UV/image orientation and material wiring together)."""

    class PatternProvider(pbase.Provider):
        id = "PATTERN"
        name = "Pattern"
        is_local = True
        requires_key = False
        default_model = "pattern"

        def capabilities(self, cfg):
            return {pbase.IMAGE_EDIT, pbase.MASK}

        def generate(self, req, cfg, cancel=None):
            h, w = req.height, req.width
            yy, xx = np.mgrid[0:h, 0:w]
            img = np.zeros((h, w, 3), np.uint8)
            img[..., 0] = np.where((xx // 32) % 2 == 0, 230, 20)
            img[..., 1] = np.where((yy // 32) % 2 == 0, 230, 20)
            img[..., 2] = 120
            return pbase.ImageResult(img, self.id, "pattern")

    prov_mod.register(PatternProvider())
    p = prefs()
    p.provider = "PATTERN"
    p.settings_for("PATTERN")  # mock settings group is fine
    obj = add_cube("Align")
    obj.rotation_euler = (0.3, 0.2, 0.5)
    bpy.ops.object.transform_apply(rotation=True)
    sc = bpy.context.scene.ats
    sc.ortho = True
    sc.align_to_object = False
    sc.views.clear()
    v = sc.views.add()
    v.preset = "FRONT_RIGHT"
    v.label = "FR"
    sc.min_facing = 0.0
    sc.fill_unseen = False
    sc.texture_size = "1024"
    sc.auto_align = False
    sc.color_match = False
    res, job = generate(obj)
    assert job.status == "DONE", job.error
    # Build matching Blender camera
    mesh = meshio.extract(obj, bpy.context, uv_name=obj.ats.uv_map)
    cam = services.view_camera(sc, sc.views[0], mesh, 256)
    cdata = bpy.data.cameras.new("C")
    cdata.type = "ORTHO"
    half_w = 1.0 / cam.proj[0, 0]
    cdata.ortho_scale = 2 * half_w
    cdata.shift_x = -cam.proj[0, 3] / 2
    cdata.shift_y = -cam.proj[1, 3] / 2
    import mathutils
    cobj = bpy.data.objects.new("C", cdata)
    bpy.context.scene.collection.objects.link(cobj)
    cobj.matrix_world = mathutils.Matrix(np.linalg.inv(cam.view).tolist())
    scene = bpy.context.scene
    scene.camera = cobj
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.render.resolution_x = scene.render.resolution_y = 256
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.render.film_transparent = True
    # unlit: base colour texture -> emission
    mat = obj.material_slots[0].material
    nt = mat.node_tree
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(nt.nodes["ATS Base Color"].outputs["Color"], em.inputs["Color"])
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    path = os.path.join(tempfile.mkdtemp(), "r.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    with open(path, "rb") as fh:
        rendered = imageio.decode_image(fh.read())
    rendered = imageio.rgba(rendered)
    alpha = rendered[..., 3] > 250
    yy, xx = np.mgrid[0:256, 0:256]
    exp_r = np.where(((xx * 2) // 32) % 2 == 0, 230, 20)
    exp_g = np.where(((yy * 2) // 32) % 2 == 0, 230, 20)
    # ignore pixels near pattern edges (bilinear) and silhouettes
    stable = alpha.copy()
    stable &= ((xx * 2) % 32 > 3) & ((xx * 2) % 32 < 29) & ((yy * 2) % 32 > 3) & ((yy * 2) % 32 < 29)
    assert stable.sum() > 2000
    r_ok = (np.abs(rendered[..., 0].astype(int) - exp_r) < 60)[stable].mean()
    g_ok = (np.abs(rendered[..., 1].astype(int) - exp_g) < 60)[stable].mean()
    assert r_ok > 0.97 and g_ok > 0.97, (r_ok, g_ok)


def _low_poly_character():
    bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=0.4, depth=1.2, location=(0, 0, 0.6))
    body = bpy.context.active_object
    bpy.ops.mesh.primitive_uv_sphere_add(segments=10, ring_count=6, radius=0.35, location=(0, 0, 1.5))
    bpy.ops.mesh.primitive_cube_add(size=0.25, location=(0.6, 0, 0.9))
    bpy.ops.mesh.primitive_cube_add(size=0.25, location=(-0.6, 0, 0.9))
    for o in bpy.context.view_layer.objects:
        o.select_set(o.type == "MESH")
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    char = bpy.context.active_object
    char.name = "Orc"
    make_active(char)
    bpy.ops.ats.create_uv(method="SMART", set_render=False)
    char.shape_key_add(name="Basis")
    k = char.shape_key_add(name="Smile")
    k.data[0].co.z += 0.1
    vg = char.vertex_groups.new(name="Spine")
    vg.add([0, 1, 2], 1.0, "REPLACE")
    arm = bpy.data.armatures.new("Rig")
    rig = bpy.data.objects.new("Rig", arm)
    bpy.context.scene.collection.objects.link(rig)
    mod = char.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    char.modifiers.new("Bevel", "BEVEL").width = 0.01
    return char


def test_low_poly_character_full_workflow_preserves_scene_data():
    char = _low_poly_character()
    before = snapshot_mesh(char)
    other = add_cube("Untouched")
    other.location = (5, 0, 0)
    bpy.context.view_layer.update()
    other_before = snapshot_mesh(other)
    sc = bpy.context.scene.ats
    sc.style = "HAND_PAINTED"
    sc.prompt = "stylized low-poly orc skin, muted green tones, painted shadows"
    make_active(char)
    bpy.ops.ats.views_preset(preset="CHARACTER")
    res, job = generate(char)
    assert job.status == "DONE", job.error
    # local refinement in UV space from selected faces
    me = char.data
    for i, p in enumerate(me.polygons):
        p.select = i < 6
    sc.refine_prompt = "replace with rusty iron plate"
    sc.refine_source = "FACES"
    sc.refine_space = "UV"
    make_active(char)
    assert bpy.ops.ats.refine() == {"FINISHED"}
    assert runtime.last_job().status == "DONE", runtime.last_job().error
    assert any(ly.pending for ly in char.ats.layers)
    assert bpy.ops.ats.refine_resolve(accept=True) == {"FINISHED"}
    assert not any(ly.pending for ly in char.ats.layers)
    assert bpy.ops.ats.pbr_generate() == {"FINISHED"}
    out = tempfile.mkdtemp()
    sc.export_dir = out + os.sep
    assert bpy.ops.ats.export_model() == {"FINISHED"}
    assert os.path.getsize(os.path.join(out, "Orc.glb")) > 10_000
    assert os.path.exists(os.path.join(out, "Orc_BaseColor.png"))
    assert os.path.exists(os.path.join(out, "Orc_ORM.png"))
    sc.export_format = "FBX"
    assert bpy.ops.ats.export_model() == {"FINISHED"}
    assert os.path.getsize(os.path.join(out, "Orc.fbx")) > 1000
    assert_preserved(before, char)
    assert_preserved(other_before, other)
    assert len(other.material_slots) == 0, "unrelated object was modified"
    assert char.modifiers["Armature"].object.name == "Rig"


def test_occluded_surfaces_are_reported_and_filled():
    # A small cube hidden inside a larger hollow-ish arrangement: inner faces are never visible.
    bpy.ops.mesh.primitive_cube_add(size=2)
    outer = bpy.context.active_object
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0, 0))
    inner = bpy.context.active_object
    for o in (outer, inner):
        o.select_set(True)
    bpy.context.view_layer.objects.active = outer
    bpy.ops.object.join()
    obj = bpy.context.active_object
    make_active(obj)
    bpy.ops.ats.create_uv(method="LIGHTMAP")
    info = services.analyze(bpy.context, obj)
    assert info["view_coverage"] < 0.95
    assert any("see" in m for _, m in info["messages"])
    res, job = generate(obj)
    assert job.status == "DONE", job.error
    fill = [ly for ly in obj.ats.layers if ly.kind == "FILL"]
    assert fill and "not AI-generated" in fill[0].info
    res = job.result[obj.name]
    assert res["unseen_fraction"] > 0.05


@pytest.mark.parametrize("mode,kind", [("network", "network"), ("timeout", "timeout"),
                                       ("rate_limit", "rate_limit"), ("auth", "auth"),
                                       ("content_policy", "content_policy"),
                                       ("invalid_image", "invalid_response"), ("blank_image", "invalid_response")])
def test_failed_requests_never_create_layers(mode, kind):
    obj = add_cube("Fail")
    prefs().mock.mock_fail_mode = mode
    res, job = generate(obj)
    assert job.status == "FAILED"
    assert job.error_kind == kind
    assert len(obj.ats.layers) == 0
    assert obj.ats.history[-1].status == "FAILED"
    assert len(obj.material_slots) == 0, "material must not change when nothing was generated"


def test_partial_failure_keeps_finished_views_and_retry_completes():
    obj = add_cube("Partial")
    mock = prov_mod.get("MOCK")
    mock.calls.clear()
    prefs().mock.mock_fail_mode = "fail_after:2"
    res, job = generate(obj)
    assert job.status == "FAILED" and job.retryable
    views = [ly for ly in obj.ats.layers if ly.kind == "VIEW"]
    assert len(views) == 2
    prefs().mock.mock_fail_mode = ""
    n_before = len(mock.calls)
    make_active(obj)
    assert bpy.ops.ats.retry() == {"FINISHED"}
    assert runtime.last_job().status == "DONE"
    # front/right were already painted -> those views are skipped (no paid requests)
    new_calls = len(mock.calls) - n_before
    assert new_calls <= 4
    assert any(v.status.startswith("skipped") for v in bpy.context.scene.ats.views)


def test_missing_credentials_blocks_before_network():
    obj = add_cube("Keys")
    p = prefs()
    p.provider = "OPENAI"
    p.openai.key_mode = "ENV"
    p.openai.env_var = "ATS_TEST_DEFINITELY_UNSET"
    p.openai.consent = True
    os.environ.pop("ATS_TEST_DEFINITELY_UNSET", None)
    res, job = generate(obj)
    assert res == {"CANCELLED"}
    assert "API key" in LAST_ERROR[-1]
    assert len(obj.ats.layers) == 0
    with pytest.raises(services.UserError, match="API key"):
        services.resolve_provider(bpy.context)


def test_api_key_is_redacted_from_errors():
    import http.server
    import threading
    secret = "sk-test-SECRET-1234567890abcdef"

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({"error": {"message": f"Incorrect API key provided: {secret}"}}).encode()
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        obj = add_cube("Redact")
        p = prefs()
        p.provider = "OPENAI"
        p.openai.key_mode = "PREFS"
        p.openai.api_key = secret
        p.openai.consent = True
        p.openai.base_url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
        res, job = generate(obj)
        assert job.status == "FAILED" and job.error_kind == "auth"
        assert secret not in job.error
        assert "[REDACTED]" in job.error
        for h in obj.ats.history:
            assert secret not in h.detail
    finally:
        srv.shutdown()
        p.openai.api_key = ""
        p.openai.base_url = ""


def test_save_reload_and_layer_files():
    obj = add_cube("Persist")
    res, job = generate(obj)
    assert job.status == "DONE"
    n_layers = len(obj.ats.layers)
    d = tempfile.mkdtemp()
    path = os.path.join(d, "scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=path)
    obj = bpy.data.objects["Persist"]
    for ly in obj.ats.layers:
        assert ly.filepath.startswith("//ATS_textures"), ly.filepath
        assert os.path.exists(bpy.path.abspath(ly.filepath))
    base = obj.ats.base_image
    assert bpy.path.abspath(base.filepath_raw).startswith(d)
    bpy.ops.wm.open_mainfile(filepath=path)
    obj = bpy.data.objects["Persist"]
    assert len(obj.ats.layers) == n_layers
    img = obj.material_slots[0].material.node_tree.nodes["ATS Base Color"].image
    assert img.size[0] == 512 and img.has_data or img.size[0] == 512
    rgb, a = layers_mod.current_texture(obj, 512)
    assert (a > 0.5).mean() > 0.3
    make_active(obj)
    assert bpy.ops.ats.recomposite() == {"FINISHED"}


def test_repeated_generation_correction_undo_and_history():
    obj = add_cube("Repeat")
    sc = bpy.context.scene.ats
    mock = prov_mod.get("MOCK")
    res, job = generate(obj)
    n1 = len(obj.ats.layers)
    calls = len(mock.calls)
    # second run: everything already textured -> all views skipped, no requests
    res, job = generate(obj)
    assert job.status == "DONE"
    assert len(mock.calls) == calls
    assert len(obj.ats.layers) == n1
    # explicit regenerate paints new layers on top
    sc.regenerate = True
    sc.prompt = "old wooden shield with chipped paint"
    res, job = generate(obj)
    assert job.status == "DONE"
    n2 = len(obj.ats.layers)
    assert n2 > n1
    # correction via painted mask, rejected then accepted
    size = obj.ats.texture_size
    mimg = bpy.data.images.new("m", size, size)
    # Default cube UVs form a cross; paint inside its middle column (top-down rows/cols).
    top_down = np.zeros((size, size, 4), np.float32)
    top_down[int(size * 0.30):int(size * 0.45), int(size * 0.40):int(size * 0.60)] = 1.0
    mimg.pixels.foreach_set(top_down[::-1].reshape(-1))
    obj.ats.mask_image = mimg
    sc.refine_source = "PAINT"
    sc.refine_space = "UV"
    sc.refine_prompt = "rusty iron"
    make_active(obj)
    assert bpy.ops.ats.refine() == {"FINISHED"}
    pending = [ly for ly in obj.ats.layers if ly.pending]
    assert len(pending) == 1
    px = layers_mod.layer_pixels(obj, pending[0], size)
    m = services.painted_mask(obj, size) > 0.5   # top-down, as the add-on reads it
    masks_mod = __import__(f"{MOD}.core.masks", fromlist=["x"])
    near = masks_mod.dilate(m, 2 * (sc.refine_grow + sc.refine_feather) + 2)  # box feather vs 4-neighbour dilation
    assert px[..., 3][~near].max() == 0, "correction must stay inside the mask"
    assert (px[..., 3][m] > 0).mean() > 0.3, "correction must cover the (UV-covered part of the) mask"
    rgb_before, _ = layers_mod.current_texture(obj, size, include_pending=False)
    assert bpy.ops.ats.refine_resolve(accept=False) == {"FINISHED"}
    assert len(obj.ats.layers) == n2
    rgb_after, _ = layers_mod.current_texture(obj, size)
    assert np.allclose(rgb_before, rgb_after, atol=1e-6), "reject must restore the exact texture"
    assert bpy.ops.ats.refine() == {"FINISHED"}
    assert bpy.ops.ats.refine_resolve(accept=True) == {"FINISHED"}
    statuses = [h.status for h in obj.ats.history]
    assert "REJECTED" in statuses and "ACCEPTED" in statuses
    # layer ops + undo/redo
    n3 = len(obj.ats.layers)
    obj.ats.active_layer = n3 - 1
    assert bpy.ops.ats.layer_op(action="DUPLICATE") == {"FINISHED"}
    assert len(obj.ats.layers) == n3 + 1
    assert bpy.ops.ats.layer_op(action="MERGE_DOWN") == {"FINISHED"}
    assert len(obj.ats.layers) == n3
    assert bpy.ops.ats.layer_op(action="DELETE") == {"FINISHED"}
    assert len(obj.ats.layers) == n3 - 1
    assert bpy.ops.ats.layers_undo() == {"FINISHED"}
    assert len(obj.ats.layers) == n3
    assert bpy.ops.ats.layers_undo(redo=True) == {"FINISHED"}
    assert len(obj.ats.layers) == n3 - 1
    # restore from history
    hidx = next(i for i, h in enumerate(obj.ats.history) if h.filepath)
    assert bpy.ops.ats.history_restore(index=hidx) == {"FINISHED"}
    assert obj.ats.layers[-1].name.startswith("Restored")
    # prompt history recorded both prompts
    prompts = [p.prompt for p in sc.prompt_history]
    assert "old wooden shield with chipped paint" in prompts and "rusty iron" in prompts


def test_mask_outside_surface_is_rejected():
    obj = add_cube("EmptyMask")
    generate(obj)
    size = obj.ats.texture_size
    mimg = bpy.data.images.new("m2", size, size)
    arr = np.zeros((size, size, 4), np.float32)
    arr[:40, :40] = 1.0   # bottom-left corner: no cube UVs there
    mimg.pixels.foreach_set(arr.reshape(-1))
    obj.ats.mask_image = mimg
    sc = bpy.context.scene.ats
    sc.refine_source = "PAINT"
    sc.refine_space = "UV"
    sc.refine_prompt = "anything"
    mock = prov_mod.get("MOCK")
    c = len(mock.calls)
    make_active(obj)
    assert call(bpy.ops.ats.refine) == {"CANCELLED"}
    assert "does not touch the surface" in LAST_ERROR[-1]
    assert len(mock.calls) == c, "no request may be sent for an unusable mask"


def test_layer_visibility_and_opacity_affect_output():
    obj = add_cube("Vis")
    generate(obj)
    size = obj.ats.texture_size
    full, _ = layers_mod.current_texture(obj, size)
    top = obj.ats.layers[-1]
    top.visible = False
    hidden, _ = layers_mod.current_texture(obj, size)
    assert not np.allclose(full, hidden)
    top.visible = True
    top.opacity = 0.0
    zero, _ = layers_mod.current_texture(obj, size)
    assert np.allclose(zero, hidden, atol=1e-6)


def test_tileable_mode():
    obj = add_cube("Tile")
    sc = bpy.context.scene.ats
    sc.mode = "TILE"
    sc.prompt = "mossy cobblestone"
    res, job = generate(obj)
    assert job.status == "DONE", job.error
    assert obj.ats.layers[-1].kind == "TILE"


def test_multi_object_consistent_generation():
    a = add_cube("A")
    b = add_cube("B")
    b.location = (4, 0, 0)
    for o in (a, b):
        o.select_set(True)
    bpy.context.view_layer.objects.active = a
    mock = prov_mod.get("MOCK")
    mock.calls.clear()
    sc = bpy.context.scene.ats
    sc.art_direction = "muted medieval palette"
    sc.chain_reference = True
    assert bpy.ops.ats.generate(all_selected=True) == {"FINISHED"}
    assert runtime.last_job().status == "DONE"
    assert a.ats.layers and b.ats.layers
    # every request carries the shared art direction; B's requests carry A's result as reference
    assert all("muted medieval palette" in r.prompt for r in mock.calls)
    n_a = sum(1 for ly in a.ats.layers if ly.kind == "VIEW")
    assert all(len(r.references) == 1 for r in mock.calls[n_a:])


def test_viewport_refine_requires_viewport_and_reports():
    obj = add_cube("VP")
    generate(obj)
    sc = bpy.context.scene.ats
    sc.refine_space = "VIEW"
    sc.refine_source = "FACES"
    sc.refine_prompt = "gold trim"
    obj.data.polygons[0].select = True
    make_active(obj)
    assert call(bpy.ops.ats.refine) == {"CANCELLED"}  # no 3D viewport in background mode
    assert "3D Viewport" in LAST_ERROR[-1]


def test_import_existing_texture_as_base_layer():
    obj = add_cube("Existing")
    img = bpy.data.images.new("painted", 256, 256)
    arr = np.zeros((256, 256, 4), np.float32)
    arr[..., 0] = 1.0
    arr[..., 3] = 1.0
    img.pixels.foreach_set(arr.reshape(-1))
    mat = bpy.data.materials.new("Old")
    if bpy.app.version < (5, 0, 0):
        mat.use_nodes = True
    n = mat.node_tree.nodes.new("ShaderNodeTexImage")
    n.image = img
    obj.data.materials.append(mat)
    make_active(obj)
    assert bpy.ops.ats.layer_import_existing(image_name="painted") == {"FINISHED"}
    assert obj.ats.layers[0].kind == "BASE"
    rgb, a = layers_mod.current_texture(obj, obj.ats.texture_size)
    assert rgb[..., 0].mean() > 0.99
    # generation now only paints areas not covered: everything is covered -> nothing requested
    mock = prov_mod.get("MOCK")
    c = len(mock.calls)
    generate(obj)
    assert len(mock.calls) == c


def test_preview_prompt_reflects_style_controls():
    sc = bpy.context.scene.ats
    sc.style = "PIXEL_ART"
    sc.wear = 0.9
    sc.detail = 0.1
    sc.use_palette = True
    sc.palette_count = 2
    sc.color_1 = (1, 0, 0)
    text = services.preview_prompt(sc)
    assert "pixel art" in text.lower()
    assert "heavily weathered" in text
    assert "minimal detail" in text
    assert "#FF0000" in text
    assert "STYLE PRESET: pixel-art" in text


def test_uv_transfer_to_other_map():
    obj = add_cube("Transfer")
    generate(obj)
    make_active(obj)
    bpy.ops.ats.create_uv(method="LIGHTMAP", uv_name="Game")
    obj.ats.uv_map = "UVMap"
    sc = bpy.context.scene.ats
    sc.transfer_uv = "Game"
    assert bpy.ops.ats.transfer_uv() == {"FINISHED"}
    assert bpy.data.images.get("ATS_Transfer_BaseColor_Game") is not None
