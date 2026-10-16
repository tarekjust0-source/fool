"""End-to-end test of AI Texture Studio against a REAL AI provider.

Run inside your own Blender (4.2+), with network access and an API key:

    # macOS/Linux
    export OPENAI_API_KEY=sk-...            # or STABILITY_API_KEY / GEMINI_API_KEY
    blender -b --factory-startup --online-mode --python tools/real_provider_test.py -- \\
        --provider OPENAI --yes

    # Windows (PowerShell)
    $env:OPENAI_API_KEY="sk-..."
    & "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe" -b --factory-startup --online-mode `
        --python tools\\real_provider_test.py -- --provider OPENAI --yes

    # ComfyUI (local; no key):
    blender -b --factory-startup --python tools/real_provider_test.py -- \\
        --provider COMFYUI --model sd_xl_base_1.0.safetensors --controlnet <depth-controlnet-file> --yes

Options: --model ID, --base-url URL, --views N (default 3), --skip-character, --out DIR, --zip PATH.
Without --yes the script only prints what it would send and how many requests that is.

What it does (each step is PASS/FAIL in out/report.md and out/report.json):
  A. simple cube: generate -> validate image/layers/material/UV -> Cycles render -> save .blend ->
     reopen -> verify the texture persisted (pixel hash) -> render again and compare.
  B. low-poly character with rig + shape keys: generate (3/4 views) -> local correction on selected
     faces -> PBR maps -> GLB export -> re-import in a separate factory-clean Blender process.
Renders and the raw provider outputs are saved for you to LOOK AT: automated checks cannot judge
whether the texture is beautiful, only whether the pipeline worked.

The extension is installed into a temporary extension repository; with --factory-startup nothing
in your Blender profile is changed. The offline test provider is refused (a real test must call a
real model); --allow-test-provider exists only to check this script itself and stamps every report
"NOT A REAL PROVIDER TEST".
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_ENV = {"OPENAI": "OPENAI_API_KEY", "STABILITY": "STABILITY_API_KEY", "GEMINI": "GEMINI_API_KEY"}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser(description="Real-provider end-to-end test")
    ap.add_argument("--provider", required=True, choices=["OPENAI", "STABILITY", "GEMINI", "COMFYUI", "MOCK"])
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--controlnet", default="", help="ComfyUI depth ControlNet file name")
    ap.add_argument("--views", type=int, default=3, help="views per object (cost control)")
    ap.add_argument("--skip-character", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "ats_real_provider_test"))
    ap.add_argument("--zip", default="")
    ap.add_argument("--yes", action="store_true", help="actually send requests (they may cost money)")
    ap.add_argument("--allow-test-provider", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


class Report:
    def __init__(self, out, args, real):
        self.out, self.args, self.real = out, args, real
        self.steps = []
        self.t0 = time.time()

    def step(self, name, ok, detail=""):
        self.steps.append({"step": name, "ok": bool(ok), "detail": str(detail)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)
        return ok

    def write(self):
        ok = all(s["ok"] for s in self.steps) and bool(self.steps)
        data = {"real_provider_test": self.real, "provider": self.args.provider, "model": self.args.model,
                "blender": bpy.app.version_string, "all_passed": ok, "seconds": round(time.time() - self.t0, 1),
                "steps": self.steps}
        with open(os.path.join(self.out, "report.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        lines = ["# AI Texture Studio real-provider test", ""]
        if not self.real:
            lines += ["**NOT A REAL PROVIDER TEST - the offline test provider was used. "
                      "These results say nothing about real AI generation.**", ""]
        lines += [f"- Provider: {self.args.provider} {self.args.model}", f"- Blender: {bpy.app.version_string}",
                  f"- Result: {'ALL PASSED' if ok else 'FAILURES'}", "", "| Step | Result | Detail |",
                  "|---|---|---|"]
        lines += [f"| {s['step']} | {'PASS' if s['ok'] else 'FAIL'} | {s['detail'].replace('|', '/')} |"
                  for s in self.steps]
        lines += ["", "Look at the images in `renders/` and `provider_outputs/`: automated checks verify the "
                      "pipeline, not the artistic quality."]
        with open(os.path.join(self.out, "report.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        return ok


# ---------------------------------------------------------------------------
# set-up
# ---------------------------------------------------------------------------

def build_zip(args):
    if args.zip:
        return os.path.abspath(args.zip)
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import build
    return build.build(tempfile.mkdtemp(prefix="ats_zip_"))


def install(zip_path):
    repo_dir = tempfile.mkdtemp(prefix="ats_ext_repo_")
    repos = bpy.context.preferences.extensions.repos
    repo = repos.new(name="ATS real-provider test", module="ats_realtest", custom_directory=repo_dir, source="USER")
    res = bpy.ops.extensions.package_install_files(filepath=zip_path, repo=repo.module, enable_on_install=True)
    if "FINISHED" not in res:
        raise RuntimeError(f"install failed: {res}")
    mod = f"bl_ext.{repo.module}.ai_texture_studio"
    if mod not in bpy.context.preferences.addons:
        import addon_utils
        addon_utils.enable(mod, default_set=True)
    return mod


def configure(mod, args):
    p = bpy.context.preferences.addons[mod].preferences
    p.provider = args.provider
    s = p.settings_for(args.provider)
    if args.model:
        s.model = args.model
    if args.base_url:
        s.base_url = args.base_url
    if args.provider in KEY_ENV:
        s.key_mode = "ENV"
        s.env_var = KEY_ENV[args.provider]
    if args.provider == "COMFYUI":
        s.key_mode = "ENV"
        s.env_var = "COMFYUI_TOKEN"
        if args.controlnet:
            s.controlnet = args.controlnet
    s.consent = True          # the person running this script with --yes is the informed action
    p.timeout = 300.0
    return p, s


def scene_settings(views, prompt, style):
    sc = bpy.context.scene.ats
    sc.prompt = prompt
    sc.style = style
    sc.texture_size = "1024"
    sc.view_size = "1024"
    sc.views.clear()
    for key in views:
        v = sc.views.add()
        v.preset = key
        v.enabled = True
        v.label = key.replace("_", " ").title()
    return sc


def make_active(obj):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def pixel_hash(img):
    import numpy as np
    buf = np.empty(img.size[0] * img.size[1] * 4, np.float32)
    img.pixels.foreach_get(buf)
    return hashlib.sha256(np.round(buf * 255).astype("uint8").tobytes()).hexdigest()[:16], buf


def lit_render(path, obj, direction=(1.0, -1.2, 0.8), res=640):
    import mathutils
    scene = bpy.context.scene
    lo = mathutils.Vector((min(v[0] for v in obj.bound_box), min(v[1] for v in obj.bound_box),
                           min(v[2] for v in obj.bound_box)))
    hi = mathutils.Vector((max(v[0] for v in obj.bound_box), max(v[1] for v in obj.bound_box),
                           max(v[2] for v in obj.bound_box)))
    center = obj.matrix_world @ ((lo + hi) / 2)
    radius = (obj.matrix_world.to_3x3() @ (hi - lo)).length / 2
    cam = bpy.data.objects.get("ATS_TestCam") or bpy.data.objects.new("ATS_TestCam", bpy.data.cameras.new("ATS_TestCam"))
    if cam.name not in scene.collection.objects:
        scene.collection.objects.link(cam)
    d = mathutils.Vector(direction).normalized()
    cam.location = center + d * radius * 3.2
    cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 50
    sun = bpy.data.objects.get("ATS_TestSun") or bpy.data.objects.new("ATS_TestSun", bpy.data.lights.new("ATS_TestSun", "SUN"))
    if sun.name not in scene.collection.objects:
        scene.collection.objects.link(sun)
    sun.data.energy = 3.0
    sun.rotation_euler = (0.8, 0.2, 0.6)
    if scene.world is None:
        scene.world = bpy.data.worlds.new("ATS_TestWorld")
    scene.world.color = (0.25, 0.25, 0.25)
    scene.camera = cam
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 24
    scene.cycles.device = "CPU"
    scene.render.resolution_x = scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def copy_provider_outputs(obj, dest, svc):
    os.makedirs(dest, exist_ok=True)
    n = 0
    sc = bpy.context.scene.ats
    for v in sc.views:
        if v.debug_dir and v.debug_object == obj.name:
            src = bpy.path.abspath(v.debug_dir)
            for kind in ("render", "mask", "generated", "aligned"):
                f = os.path.join(src, f"{kind}.png")
                if os.path.exists(f):
                    shutil.copy2(f, os.path.join(dest, f"{obj.name}_{v.label}_{kind}.png".replace(" ", "_")))
                    n += 1
    return n


def check_generation(rep, name, obj, job, mod, args, expect_calls_max):
    layers = __import__(f"{mod}.blender.layers", fromlist=["x"])
    if not rep.step(f"{name}: job finished", job is not None and job.status == "DONE",
                    "" if job is None else (job.error or job.status)):
        return False
    res = (job.result or {}).get(obj.name, {})
    calls = res.get("calls", 0)
    rep.step(f"{name}: real requests sent", 0 < calls <= expect_calls_max, f"{calls} request(s)")
    views = [ly for ly in obj.ats.layers if ly.kind == "VIEW"]
    rep.step(f"{name}: one AI layer per generated view", len(views) == calls, f"{len(views)} view layers")
    real = args.provider != "MOCK"
    rep.step(f"{name}: layers come from {args.provider}",
             all(args.provider in ly.info for ly in views)
             and (not real or "MOCK" not in " ".join(ly.info for ly in views)),
             "; ".join(ly.info[:60] for ly in views[:2]))
    _, frac = layers.provenance(obj, obj.ats.texture_size)
    rep.step(f"{name}: AI-generated share of surface", frac["AI"] > 0.3,
             f"AI {frac['AI']:.0%}, low-angle {frac['AI_LOW']:.0%}, colour-extended {frac['EXTEND']:.0%}")
    img = obj.ats.base_image
    import numpy as np
    _, buf = pixel_hash(img)
    rgb = buf.reshape(-1, 4)[:, :3]
    rep.step(f"{name}: base colour image is a real texture", img.size[0] == obj.ats.texture_size and rgb.std() > 0.02,
             f"{img.size[0]}px, std {rgb.std():.3f}")
    mat = obj.material_slots[0].material if obj.material_slots else None
    ok = False
    detail = "no material"
    if mat is not None and mat.node_tree:
        bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        tex = mat.node_tree.nodes.get("ATS Base Color")
        uvn = mat.node_tree.nodes.get("ATS UV")
        linked = bsdf is not None and bsdf.inputs["Base Color"].is_linked and \
            bsdf.inputs["Base Color"].links[0].from_node in (tex, mat.node_tree.nodes.get("ATS AO Multiply"))
        ok = bool(linked and tex is not None and tex.image == img and uvn is not None and uvn.uv_map == obj.ats.uv_map)
        detail = f"material {mat.name}, UV map '{uvn.uv_map if uvn else '?'}'"
    rep.step(f"{name}: texture wired to Principled BSDF on the working UV map", ok, detail)
    return True


# ---------------------------------------------------------------------------

def test_cube(rep, mod, args, out):
    svc = __import__(f"{mod}.blender.services", fromlist=["x"])
    runtime = __import__(f"{mod}.blender.runtime", fromlist=["x"])
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    bpy.ops.mesh.primitive_cube_add(size=2)
    obj = bpy.context.active_object
    obj.name = "Crate"
    # Two opposite 3/4 views + bottom see all six faces of a box with only 3 requests.
    views = ["FRONT_RIGHT", "BACK_LEFT", "BOTTOM", "FRONT", "BACK", "TOP"][:max(1, args.views)]
    scene_settings(views, "Old wooden crate with metal corner brackets, visible wood grain, game asset",
                   "STYLIZED")
    make_active(obj)
    try:
        bpy.ops.ats.generate()
    except RuntimeError as exc:
        rep.step("A cube: generate", False, exc)
        return
    job = runtime.last_job()
    if not check_generation(rep, "A cube", obj, job, mod, args, len(views)):
        return
    rep.step("A cube: provider outputs saved", copy_provider_outputs(obj, os.path.join(out, "provider_outputs"), svc) > 0)
    r1 = lit_render(os.path.join(out, "renders", "A_cube.png"), obj)
    rep.step("A cube: Cycles render", os.path.getsize(r1) > 10_000, r1)
    h1, _ = pixel_hash(obj.ats.base_image)
    n_layers = len(obj.ats.layers)
    blend = os.path.join(out, "A_cube.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    bpy.ops.wm.open_mainfile(filepath=blend)
    obj = bpy.data.objects["Crate"]
    rep.step("A cube: layers persisted after reopen", len(obj.ats.layers) == n_layers, f"{len(obj.ats.layers)} layers")
    img = obj.material_slots[0].material.node_tree.nodes["ATS Base Color"].image
    img.reload()
    h2, _ = pixel_hash(img)
    rep.step("A cube: texture pixels identical after reopen", h1 == h2, f"{h1} vs {h2}")
    rep.step("A cube: texture files stored next to the .blend",
             all(ly.filepath.startswith("//") and os.path.exists(bpy.path.abspath(ly.filepath)) for ly in obj.ats.layers))
    r2 = lit_render(os.path.join(out, "renders", "A_cube_reopened.png"), obj)
    import numpy as np
    a = np.array(bpy.data.images.load(r1).pixels[:])
    b = np.array(bpy.data.images.load(r2).pixels[:])
    rep.step("A cube: render identical after reopen", a.shape == b.shape and np.abs(a - b).mean() < 0.01,
             f"mean diff {np.abs(a - b).mean():.4f}" if a.shape == b.shape else "size differs")


def build_character():
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
    bpy.ops.ats.create_uv(method="SMART")
    char.shape_key_add(name="Basis")
    char.shape_key_add(name="Smile")
    arm = bpy.data.objects.new("Rig", bpy.data.armatures.new("Rig"))
    bpy.context.scene.collection.objects.link(arm)
    char.modifiers.new("Armature", "ARMATURE").object = arm
    return char


def test_character(rep, mod, args, out):
    runtime = __import__(f"{mod}.blender.runtime", fromlist=["x"])
    svc = __import__(f"{mod}.blender.services", fromlist=["x"])
    layers = __import__(f"{mod}.blender.layers", fromlist=["x"])
    bpy.ops.wm.read_homefile(use_empty=True)
    char = build_character()
    import numpy as np
    co_before = np.array([v.co[:] for v in char.data.vertices])
    views = ["FRONT_RIGHT", "BACK_LEFT", "FRONT_LEFT", "BACK_RIGHT", "TOP"][:max(1, args.views)]
    sc = scene_settings(views, "Stylized low-poly orc, muted green skin, leather straps, painted shadows",
                        "HAND_PAINTED")
    make_active(char)
    try:
        bpy.ops.ats.generate()
    except RuntimeError as exc:
        rep.step("B character: generate", False, exc)
        return
    if not check_generation(rep, "B character", char, runtime.last_job(), mod, args, len(views)):
        return
    copy_provider_outputs(char, os.path.join(out, "provider_outputs"), svc)
    lit_render(os.path.join(out, "renders", "B_character.png"), char, direction=(0.6, -1.4, 0.5))
    size = char.ats.texture_size
    before, _ = layers.current_texture(char, size)
    mw = char.matrix_world
    for p in char.data.polygons:
        p.select = (mw @ p.center).z > 1.25   # the head (world space: the joined origin is offset)
    sc.refine_source = "FACES"
    sc.refine_space = "UV"
    sc.refine_prompt = "green skin with a red war-paint stripe"
    make_active(char)
    try:
        bpy.ops.ats.refine()
        job = runtime.last_job()
        ok = job.status == "DONE"
        rep.step("B character: local correction request", ok, job.error or job.status)
    except RuntimeError as exc:
        rep.step("B character: local correction request", False, exc)
        ok = False
    if ok:
        mask = svc.refine_mask(char, sc, size) > 0.01
        pending, _ = layers.current_texture(char, size)
        changed = np.abs(pending - before).max(axis=2) > 0.02
        outside = changed & ~mask
        rep.step("B character: correction stays inside the selected faces", outside.mean() < 0.001,
                 f"{changed[mask].mean():.0%} of mask changed, {outside.mean():.3%} outside")
        bpy.ops.ats.refine_resolve(accept=True)
        lit_render(os.path.join(out, "renders", "B_character_corrected.png"), char, direction=(0.6, -1.4, 0.5))
    sc.pbr_roughness = sc.pbr_metallic = sc.pbr_normal = True
    sc.pbr_ao = True
    sc.ao_samples = 16
    make_active(char)
    bpy.ops.ats.pbr_generate()
    mat = char.material_slots[0].material
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    rep.step("B character: PBR maps connected",
             all(bsdf.inputs[k].is_linked for k in ("Base Color", "Roughness", "Metallic", "Normal")))
    co_after = np.array([v.co[:] for v in char.data.vertices])
    rep.step("B character: mesh/rig/shape keys preserved",
             np.allclose(co_before, co_after) and len(char.data.shape_keys.key_blocks) == 2
             and char.modifiers["Armature"].object is not None)
    sc.export_dir = os.path.join(out, "export") + os.sep
    bpy.ops.ats.export_model()
    glb = os.path.join(out, "export", "Orc.glb")
    rep.step("B character: GLB exported", os.path.exists(glb) and os.path.getsize(glb) > 10_000, glb)
    code = ("import bpy, json; bpy.ops.wm.read_factory_settings(use_empty=True); "
            f"bpy.ops.import_scene.gltf(filepath={glb!r}); "
            "m=[o for o in bpy.data.objects if o.type=='MESH'][0].material_slots[0].material; "
            "b=next(n for n in m.node_tree.nodes if n.type=='BSDF_PRINCIPLED'); "
            "print('ATS_RESULT '+json.dumps({k: b.inputs[k].is_linked for k in "
            "('Base Color','Roughness','Metallic','Normal')}))")
    binary = bpy.app.binary_path
    if binary and os.path.basename(binary).lower().startswith("blender"):
        cmd = [binary, "-b", "--factory-startup", "--python-expr", code]
    else:                                   # running through the "bpy" Python module
        cmd = [sys.executable, "-c", code]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("ATS_RESULT ")), "")
    res = json.loads(line[11:]) if line else {}
    rep.step("B character: GLB re-imports in a clean Blender with textures connected",
             bool(res) and all(res.values()), res or proc.stderr[-300:])


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out, "renders"), exist_ok=True)
    real = args.provider != "MOCK"
    if not real and not args.allow_test_provider:
        print("Refusing to run with the offline test provider: a real-provider test must call a real model.")
        sys.exit(2)
    if not real:
        os.environ["ATS_ENABLE_TEST_PROVIDER"] = "1"
    elif os.environ.get("ATS_ENABLE_TEST_PROVIDER") == "1":
        print("Unset ATS_ENABLE_TEST_PROVIDER for a real test.")
        sys.exit(2)
    if args.provider in KEY_ENV and not os.environ.get(KEY_ENV[args.provider]):
        print(f"Set the environment variable {KEY_ENV[args.provider]} first.")
        sys.exit(2)
    if args.provider not in ("COMFYUI", "MOCK") and not bpy.app.online_access:
        print("Blender started without online access: add --online-mode to the command line.")
        sys.exit(2)
    n_req = max(1, args.views) + (0 if args.skip_character else max(1, args.views) + 1)
    print(f"This test sends about {n_req} image request(s) to {args.provider}"
          f"{' (billed by the provider)' if args.provider not in ('COMFYUI', 'MOCK') else ''}: renders of a cube"
          f"{'' if args.skip_character else ' and a simple character'}, masks and prompts. Nothing else is sent.")
    if not args.yes:
        print("Dry run only. Add --yes to send the requests.")
        sys.exit(0)
    os.environ["ATS_SYNC_JOBS"] = "1"
    rep = Report(args.out, args, real)
    try:
        mod = install(build_zip(args))
        rep.step("extension installed and enabled", True, mod)
        p, s = configure(mod, args)
        args.model = s.model
        prov_mod = __import__(f"{mod}.providers", fromlist=["x"])
        prov, cfg = __import__(f"{mod}.blender.prefs", fromlist=["x"]).provider_config(bpy.context)
        try:
            rep.step("provider connection test", True, prov.test_connection(cfg))
        except prov_mod.ProviderError as exc:
            rep.step("provider connection test", False, exc.user_message())
            raise SystemExit(1)
        test_cube(rep, mod, args, args.out)
        if not args.skip_character:
            test_character(rep, mod, args, args.out)
    except SystemExit:
        pass
    except Exception as exc:                      # noqa: BLE001
        rep.step("unexpected error", False, f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    ok = rep.write()
    print(f"\nReport: {os.path.join(args.out, 'report.md')}  ->  {'ALL PASSED' if ok else 'FAILURES'}"
          + ("" if real else "   (NOT A REAL PROVIDER TEST)"), flush=True)
    code = 0 if ok else 1
    binary = bpy.app.binary_path
    if not (binary and os.path.basename(binary).lower().startswith("blender")):
        # The PyPI "bpy" module can abort during interpreter teardown after a Cycles render
        # (independent of this add-on); exit directly so the exit code reflects the report.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    sys.exit(code)


if __name__ == "__main__":
    main()
