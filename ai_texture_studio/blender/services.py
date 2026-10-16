"""High-level actions used by operators (main thread).

These functions gather everything a background job needs from Blender
(mesh snapshot, current texture, settings, reference pixels, provider
config), start the job, and apply its results back to the scene.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import bpy
import numpy as np

from ..core import camera as camlib
from ..core import imageio, masks, meshdata, pbr, pipeline, raster
from ..core.projection import ProjectionSettings, TextureState
from ..core.prompting import StyleSettings, compose
from ..core.secrets import redact
from .. import providers
from . import compat, layers, materials, meshio, prefs, runtime, storage
from .props import palette


class UserError(Exception):
    """Problem the user can fix; shown as an operator error report."""


DEFAULT_VIEWS = [("FRONT", True), ("RIGHT", True), ("LEFT", True), ("BACK", True), ("TOP", True),
                 ("BOTTOM", True), ("FRONT_RIGHT", False), ("FRONT_LEFT", False), ("BACK_RIGHT", False),
                 ("BACK_LEFT", False)]


# ---------------------------------------------------------------------------
# settings -> plain data
# ---------------------------------------------------------------------------

def style_settings(sc) -> StyleSettings:
    return StyleSettings(style=sc.style, material=sc.material, detail=sc.detail, wear=sc.wear,
                         strength=sc.strength, use_palette=sc.use_palette, palette=palette(sc),
                         delight=sc.delight, art_direction=sc.art_direction)


def reference_arrays(sc, max_side: int = 1024) -> list:
    if not (sc.use_reference and sc.reference is not None):
        return []
    img = sc.reference
    if img.size[0] == 0 or img.size[1] == 0:
        try:
            img.reload()
        except Exception:  # noqa: BLE001
            pass
    if img.size[0] == 0:
        raise UserError(f"Reference image '{img.name}' could not be loaded.")
    arr = imageio.to_uint8(compat.image_to_array(img))[..., :3]
    h, w = arr.shape[:2]
    s = min(1.0, max_side / float(max(h, w)))
    if s < 1.0:
        arr = imageio.resize(arr, max(16, int(w * s)), max(16, int(h * s)))
    return [arr]


def make_spec(sc, prompt: str | None = None) -> pipeline.GenerationSpec:
    text = sc.prompt if prompt is None else prompt
    return pipeline.GenerationSpec(prompt=text, negative=sc.negative, style=style_settings(sc),
                                   seed=sc.seed if sc.use_seed else None, references=reference_arrays(sc),
                                   request_size=int(sc.view_size))


def preview_prompt(sc, purpose="first_view") -> str:
    cp = compose(sc.prompt, sc.negative, style_settings(sc), purpose=purpose, view_name="front",
                 has_reference=bool(sc.use_reference and sc.reference))
    return f"PROMPT: {cp.prompt}\n\nNEGATIVE: {cp.negative}\n\nSTYLE PRESET: {cp.style_preset}  " \
           f"STRENGTH: {cp.strength:.2f}"


def ensure_default_views(sc):
    if len(sc.views) == 0:
        for key, on in DEFAULT_VIEWS:
            v = sc.views.add()
            v.preset = key
            v.enabled = on
            v.label = camlib.VIEW_LABELS[key]
    for v in sc.views:
        view_uid(v)


def view_uid(v) -> str:
    """Stable key linking a view to the layers it produced."""
    if not v.uid:
        v.uid = uuid.uuid4().hex[:8]
    return v.uid


def view_camera(sc, v, mesh: meshdata.MeshData, size: int) -> camlib.ViewCamera:
    if v.preset == "CUSTOM":
        view = np.array(v.custom_view, np.float64).reshape(4, 4)
        proj = np.array(v.custom_proj, np.float64).reshape(4, 4)
        cam = camlib.ViewCamera("CUSTOM", view, proj, size, size, v.custom_ortho,
                                {"label": v.label or "Custom", "key": view_uid(v)})
        return cam
    basis = mesh.basis if sc.align_to_object else None
    cam = camlib.preset_camera(v.preset, camlib.VIEW_PRESETS[v.preset], mesh.positions, size, basis=basis,
                               ortho=sc.ortho, fov_deg=sc.fov)
    cam.meta["label"] = v.label or camlib.VIEW_LABELS[v.preset]
    cam.meta["key"] = view_uid(v)
    return cam


def enabled_cameras(sc, mesh, size, only_index=None):
    ensure_default_views(sc)
    cams = []
    for i, v in enumerate(sc.views):
        if only_index is not None and i != only_index:
            continue
        if only_index is None and not v.enabled:
            continue
        cams.append((i, view_camera(sc, v, mesh, size)))
    return cams


def projection_settings(sc, size) -> ProjectionSettings:
    return ProjectionSettings(texture_size=size, view_size=int(sc.view_size), min_facing=sc.min_facing,
                              band_px=sc.band_px, auto_align=sc.auto_align, color_match=sc.color_match,
                              fill_unseen=sc.fill_unseen)


def viewport_camera(context, size: int, label: str = "Viewport") -> camlib.ViewCamera:
    space = context.space_data
    rv3d = getattr(space, "region_3d", None) if space and space.type == "VIEW_3D" else None
    if rv3d is None:
        rv3d = getattr(context, "region_data", None)
    if rv3d is None:
        raise UserError("Run this from a 3D Viewport.")
    region = None
    if context.area is not None:
        for r in context.area.regions:
            if r.type == "WINDOW":
                region = r
    w, h = (region.width, region.height) if region else (1, 1)
    persp = rv3d.view_perspective != "ORTHO" and rv3d.is_perspective
    cam = camlib.viewport_camera("CUSTOM", rv3d.view_matrix, rv3d.window_matrix, w, h, size, persp)
    cam.meta["label"] = label
    return cam


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def validate_object(obj):
    if obj is None:
        raise UserError("Select a mesh object.")
    if obj.type != "MESH":
        raise UserError(f"'{obj.name}' is not a mesh.")
    if len(obj.data.polygons) == 0:
        raise UserError(f"'{obj.name}' has no faces.")
    if not obj.data.uv_layers:
        raise UserError(f"'{obj.name}' has no UV map. Use 'Create UV Map' in the Selected Object panel "
                        "(adds a new map; nothing existing is changed).")


def resolve_provider(context):
    prov, cfg = prefs.provider_config(context)
    problems = prov.validate(cfg)
    if problems:
        raise UserError(" ".join(problems) + " (Preferences > Add-ons > AI Texture Studio)")
    ok, msg = prefs.network_allowed(prov, cfg)
    if not ok:
        raise UserError(msg)
    return prov, cfg


def needs_consent(context) -> tuple[bool, str]:
    p = prefs.get_prefs(context)
    prov = providers.get(p.provider)
    s = p.settings_for(prov.id)
    if prov.id == "MOCK" or s.consent:
        return False, ""
    return True, prov.data_notice


# ---------------------------------------------------------------------------
# history
# ---------------------------------------------------------------------------

def add_history(obj, kind: str, prompt: str, negative: str = "", provider: str = "", model: str = "",
                detail: str = "", status: str = "RUNNING") -> str:
    h = obj.ats.history.add()
    h.uid = uuid.uuid4().hex[:10]
    h.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    h.kind = kind
    h.status = status
    h.prompt = prompt[:1000]
    h.negative = negative[:500]
    h.provider = provider
    h.model = model
    h.detail = redact(detail)[:1000]
    obj.ats.active_history = len(obj.ats.history) - 1
    while len(obj.ats.history) > 200:
        obj.ats.history.remove(0)
    return h.uid


def update_history(obj, uid: str, **fields):
    for h in obj.ats.history:
        if h.uid == uid:
            for k, v in fields.items():
                setattr(h, k, redact(v)[:1000] if isinstance(v, str) else v)
            return h
    return None


def add_prompt_history(sc, prompt: str, negative: str):
    if not prompt.strip():
        return
    for i, it in enumerate(sc.prompt_history):
        if it.prompt == prompt and it.negative == negative:
            sc.prompt_history.remove(i)
            break
    it = sc.prompt_history.add()
    it.prompt = prompt
    it.negative = negative
    it.style = sc.style
    it.timestamp = time.strftime("%Y-%m-%d %H:%M")
    sc.prompt_history.move(len(sc.prompt_history) - 1, 0)
    while len(sc.prompt_history) > 30:
        sc.prompt_history.remove(len(sc.prompt_history) - 1)
    sc.active_prompt = 0


def object_by_uid(uid: str):
    for o in bpy.data.objects:
        st = getattr(o, "ats", None)
        if st is not None and st.uid == uid:
            return o
    return None


# ---------------------------------------------------------------------------
# materials
# ---------------------------------------------------------------------------

def current_maps(obj) -> dict:
    maps = {}
    if obj.ats.base_image is not None:
        maps["BASE_COLOR"] = obj.ats.base_image
    try:
        names = json.loads(obj.ats.maps or "{}")
    except ValueError:
        names = {}
    for kind, name in names.items():
        img = bpy.data.images.get(name)
        if img is not None:
            maps[kind] = img
    return maps


def apply_material(obj, sc):
    uv = obj.ats.uv_map or meshio.default_uv_name(obj)
    return materials.apply(obj, current_maps(obj), uv, sc.material_mode, ao_multiply=sc.ao_multiply,
                           pixel_art=obj.ats.pixelate)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _prepare_object(context, obj, sc, regenerate: bool, exclude_view_key: str = ""):
    validate_object(obj)
    storage.ensure_uid(obj)
    st = obj.ats
    if not st.uv_map or st.uv_map not in obj.data.uv_layers:
        st.uv_map = meshio.default_uv_name(obj)
    if st.texture_size == 0 or not st.layers:
        st.texture_size = int(sc.texture_size)
    size = st.texture_size
    missing = layers.missing_layer_files(obj)
    if missing:
        raise UserError(f"Layer files missing on disk: {', '.join(missing[:3])}. Delete those layers first.")
    mesh = meshio.extract(obj, context, uv_name=st.uv_map, use_modifiers=sc.use_modifiers)
    if not st.layers:
        return mesh, size, None, None, None, None
    # "Start From Scratch" still respects protected layers (imported base, accepted fixes, paint).
    return (mesh, size) + layers.planning_texture(obj, size, exclude_view_key=exclude_view_key,
                                                  protected_only=regenerate)


def start_generation(context, objects, *, only_view: int | None = None):
    sc = context.scene.ats
    prov, cfg = resolve_provider(context)
    p = prefs.get_prefs(context)
    if sc.mode == "MULTIVIEW" and not sc.prompt.strip() and not (sc.use_reference and sc.reference):
        raise UserError("Enter a prompt or set a reference image.")
    if sc.mode == "TILE" and not sc.prompt.strip():
        raise UserError("Enter a prompt describing the tileable material.")
    spec = make_spec(sc)
    items = []
    job = runtime.BridgeJob("generate", "Generate")
    job.context["out_dirs"] = {}
    job.context["objects"] = {}
    job.context["history"] = {}
    job.context["keep_debug"] = p.keep_debug_views
    job.context["view_index"] = {}
    job.context["replace_key"] = ""
    job.context["created"] = set()
    total_calls = 0
    ensure_default_views(sc)
    replace_key = ""
    if only_view is not None and sc.mode == "MULTIVIEW" and 0 <= only_view < len(sc.views):
        # Regenerate one view: its previous layers are ignored while planning,
        # so it repaints exactly what it painted before, with the other views as context.
        replace_key = view_uid(sc.views[only_view])
        job.context["replace_key"] = replace_key
    for obj in objects:
        mesh, size, rgb, alpha, protected, grazing = _prepare_object(context, obj, sc, sc.regenerate, replace_key)
        state = TextureState(mesh, size, rgb, alpha, protected, grazing)
        if state.uv.index.size == 0:
            raise UserError(f"'{obj.name}': UV map '{mesh.uv_name}' covers no area inside 0-1.")
        cams = enabled_cameras(sc, mesh, int(sc.view_size), only_view) if sc.mode == "MULTIVIEW" else []
        if sc.mode == "MULTIVIEW" and not cams:
            raise UserError("Enable at least one view in the Multi-View panel.")
        key = obj.name
        job.context["out_dirs"][key] = storage.object_dir(obj)
        job.context["objects"][key] = obj.ats.uid
        job.context["view_index"][key] = {cam.meta["key"]: i for i, cam in cams}
        detail = f"{sc.mode.lower()} | {len(cams)} views | {size}px | style {sc.style}"
        job.context["history"][key] = add_history(obj, "Generate" if sc.mode == "MULTIVIEW" else "Tile",
                                                  sc.prompt, sc.negative, prov.id, cfg.model, detail)
        items.append((key, mesh, [c for _, c in cams], state))
        total_calls += max(1, len(cams))
    for _, v in enumerate(sc.views):
        v.status = ""
    add_prompt_history(sc, sc.prompt, sc.negative)
    ps_size = {k: s.size for k, _, _, s in items}
    ps_map = {k: projection_settings(sc, ps_size[k]) for k in ps_size}
    mode = sc.mode
    tile_scale = sc.tile_scale
    chain = sc.chain_reference and len(items) > 1
    job.title = f"Generating {len(items)} object(s), up to {total_calls} {prov.name} request(s)"

    def worker(job):
        results = {}
        n = len(items)
        for k, (key, mesh, cams, state) in enumerate(items):
            job.check_cancel()
            if mode == "TILE":
                job.set_progress(k / n, f"{key}: generating tileable material")
                results[key] = pipeline.run_tile(job, mesh, state, prov, cfg, spec, scale=tile_scale, object_name=key)
            else:
                results[key] = pipeline.run_multiview(job, mesh, cams, state, prov, cfg, spec, ps_map[key],
                                                      object_name=key, progress_range=(k / n, (k + 1) / n))
                first = results[key].pop("first_generated", None)
                if chain and k == 0 and first is not None:
                    spec.references = list(spec.references) + [first]
        return results

    job.context["first_layer_done"] = set()
    runtime.start(job, worker, _on_generation_event, _on_generation_finished)
    return job


def _obj_for(job, payload):
    uid = job.context["objects"].get(payload.get("object", ""))
    obj = object_by_uid(uid) if uid else None
    if obj is None:
        raise RuntimeError("Target object no longer exists")
    return obj


def _on_generation_event(job, kind, payload):
    sc = bpy.context.scene.ats
    if kind == "view_skipped":
        idx = job.context["view_index"].get(payload["object"], {}).get(payload.get("view_key"))
        if idx is not None and idx < len(sc.views):
            sc.views[idx].status = f"skipped ({payload['untextured_fraction']:.1%} new)"
        return
    if kind not in ("view_done", "grazing_done", "fill_done", "tile_done"):
        return
    obj = _obj_for(job, payload)
    key = payload["object"]
    if key not in job.context["first_layer_done"]:
        layers.push_undo(obj, "Generate")
        job.context["first_layer_done"].add(key)
    hist = job.context["history"].get(key, "")
    view_key = payload.get("view_key", "") if kind == "view_done" else ""
    st = payload.get("stats", {})
    if kind == "view_done":
        name = f"{payload['label']}"
        kind_l = "VIEW"
        info = (f"AI-generated ({payload['provider']} {payload.get('model', '')}); "
                f"claimed {st.get('claimed_fraction', 0):.1%} of the surface")
        if view_key and view_key == job.context.get("replace_key"):
            older = [ly for ly in obj.ats.layers if ly.view_key == view_key and ly.kind == "VIEW"]
            for ly in older:          # keep for comparison / undo, but hide
                ly.visible = False
                if not ly.name.endswith("(previous)"):
                    ly.name = f"{ly.name} (previous)"
            name = f"{payload['label']} v{len(older) + 1}"
    elif kind == "grazing_done":
        name = "Low-angle views"
        kind_l = "GRAZING"
        info = (f"AI-generated but seen only at flat angles ({st.get('fallback_fraction', 0):.1%} of the surface); "
                "a better view will repaint these texels")
    elif kind == "fill_done":
        name = "Hidden-area fill"
        kind_l = "FILL"
        info = (f"NOT AI-generated: colour extended from the nearest painted surface into "
                f"{st.get('unseen_fraction', 0):.1%} of the surface that no camera saw")
    else:
        name = "Tileable material"
        kind_l = "TILE"
        info = f"AI-generated swatch ({payload['provider']} {payload.get('model', '')})"
    index = 0 if kind in ("grazing_done", "fill_done") else None
    ly = layers.add_layer(obj, None, name=name, kind=kind_l, info=info, filepath=payload["layer_path"],
                          history_uid=hist, index=index, push=False, view_key=view_key)
    job.context["created"].add(ly.uid)
    if index is not None:
        # keep the user's active selection on the newest generated layer
        obj.ats.active_layer = len(obj.ats.layers) - 1
    if hist and kind not in ("grazing_done", "fill_done"):
        h = update_history(obj, hist)
        if h is not None:
            h.layer_uid = ly.uid
            h.filepath = ly.filepath
            h.layer_name = ly.name
    if kind == "view_done":
        idx = job.context["view_index"].get(key, {}).get(view_key)
        if idx is not None and idx < len(sc.views):
            v = sc.views[idx]
            v.status = f"done ({payload['stats'].get('claimed_fraction', 0):.0%} of surface)"
            v.debug_dir = storage.to_stored(payload.get("debug_dir", "")) if payload.get("debug_dir") else ""
            v.debug_object = obj.name
    preview = payload.get("preview")
    if preview is not None and preview.shape[0] == obj.ats.texture_size:
        layers.write_preview(obj, preview)   # cheap: final recomposite runs when the job ends
    else:
        layers.recomposite(obj, save=False)
    wired = job.context.setdefault("materials_done", set())
    if key not in wired:   # wire materials once per object; later events only update pixels
        apply_material(obj, sc)
        wired.add(key)


def _drop_superseded_fills(obj, created: set):
    """After a successful multi-view run, older low-angle / fill layers are obsolete.

    The run planned without them and produced its own for every texel that is
    still unpainted, so keeping them would only stack stale colour extensions.
    (The removal is part of the run's undo step.)
    """
    st = obj.ats
    for i in reversed(range(len(st.layers))):
        ly = st.layers[i]
        if ly.kind in ("GRAZING", "FILL") and ly.uid not in created:
            st.layers.remove(i)
    st.active_layer = min(st.active_layer, max(0, len(st.layers) - 1))


def _on_generation_finished(job):
    sc = bpy.context.scene.ats
    for key, hist_uid in job.context.get("history", {}).items():
        obj = object_by_uid(job.context["objects"].get(key, ""))
        if obj is None:
            continue
        res = (job.result or {}).get(key) if isinstance(job.result, dict) else None
        if job.status == "DONE":
            detail = ""
            if isinstance(res, dict) and "views" in res:
                hidden = res.get("unseen_fraction")
                detail = (f"views: {', '.join(res['views']) or 'none'}; skipped: {', '.join(res['skipped']) or 'none'}"
                          + (f"; colour-extended (not AI) {hidden:.1%}" if hidden is not None else ""))
            update_history(obj, hist_uid, status="DONE", detail=detail)
            if isinstance(res, dict) and res.get("calls"):
                _drop_superseded_fills(obj, job.context["created"])
        else:
            update_history(obj, hist_uid, status=job.status, detail=job.error or job.message)
        if obj.ats.layers:
            layers.recomposite(obj, save=True)
            apply_material(obj, sc)
    for d in job.context.get("out_dirs", {}).values():
        storage.prune_debug_runs(d)
    if job.context.get("first_layer_done"):
        runtime.push_undo_step("AI Texture Studio: Generate")


# ---------------------------------------------------------------------------
# refinement
# ---------------------------------------------------------------------------

def face_mask(obj, size: int) -> np.ndarray:
    """UV-space mask of faces selected in Edit Mode (original mesh, no modifiers)."""
    mesh = meshio.extract(obj, uv_name=obj.ats.uv_map or None, use_modifiers=False)
    sel = np.flatnonzero(mesh.tri_selected)
    m = np.zeros((size, size), np.float32)
    if sel.size == 0:
        return m
    uvs = raster.rasterize_uv(mesh.tri_uvs, size, tri_subset=sel)
    m.reshape(-1)[uvs.index] = 1.0
    return m


def painted_mask(obj, size: int) -> np.ndarray:
    img = obj.ats.mask_image
    if img is None or img.size[0] == 0:
        return np.zeros((size, size), np.float32)
    arr = compat.image_to_array(img)[..., 0]
    if arr.shape != (size, size):
        arr = imageio.resize(arr.astype(np.float32), size, size)
    return np.clip(arr, 0, 1).astype(np.float32)


def refine_mask(obj, sc, size: int) -> np.ndarray:
    m = np.zeros((size, size), np.float32)
    if sc.refine_source in ("PAINT", "BOTH"):
        m = np.maximum(m, painted_mask(obj, size))
    if sc.refine_source in ("FACES", "BOTH"):
        m = np.maximum(m, face_mask(obj, size))
    hard = m > 0.5
    if sc.refine_grow:
        hard = masks.dilate(hard, sc.refine_grow)
    soft = masks.feather(hard, sc.refine_feather) if sc.refine_feather else hard.astype(np.float32)
    # never lose the core of the selection to feathering
    return np.maximum(soft, (masks.erode(hard, max(1, sc.refine_feather // 2))).astype(np.float32))


def start_refine(context, obj):
    sc = context.scene.ats
    validate_object(obj)
    if not obj.ats.layers:
        raise UserError("Generate or import a texture first (no layers yet).")
    if any(ly.pending for ly in obj.ats.layers):
        raise UserError("Accept or reject the pending correction first.")
    if not sc.refine_prompt.strip():
        raise UserError("Describe the correction (e.g. 'replace with rusty iron').")
    prov, cfg = resolve_provider(context)
    p = prefs.get_prefs(context)
    size = layers.texture_size(obj, sc)
    uv_mask = refine_mask(obj, sc, size)
    if (uv_mask > 0.5).sum() < 4:
        raise UserError("The mask is empty. Paint a mask or select faces in Edit Mode.")
    mesh = meshio.extract(obj, context, uv_name=obj.ats.uv_map, use_modifiers=sc.use_modifiers)
    rgb, alpha = layers.current_texture(obj, size)
    state = TextureState(mesh, size, rgb, alpha)
    if (uv_mask.reshape(-1)[state.uv.index] > 0.5).sum() < 4:
        raise UserError("The mask does not touch the surface (it only covers empty UV space). "
                        "Paint on the model or select faces.")
    spec = make_spec(sc, prompt=sc.refine_prompt)
    spec.design_context = sc.prompt if sc.prompt.strip() != sc.refine_prompt.strip() else ""
    cam = viewport_camera(context, int(sc.view_size), "Viewport") if sc.refine_space == "VIEW" else None
    job = runtime.BridgeJob("refine", "Local refinement")
    job.context["out_dirs"] = {obj.name: storage.object_dir(obj)}
    job.context["objects"] = {obj.name: obj.ats.uid}
    job.context["keep_debug"] = p.keep_debug_views
    job.context["history"] = {obj.name: add_history(obj, "Correction", sc.refine_prompt, sc.negative, prov.id,
                                                    cfg.model, f"{sc.refine_space} | mask {sc.refine_source}")}
    add_prompt_history(sc, sc.refine_prompt, sc.negative)
    name = obj.name

    def worker(job):
        if cam is not None:
            return pipeline.run_refine_view(job, mesh, cam, state, uv_mask, prov, cfg, spec, object_name=name)
        return pipeline.run_refine_uv(job, state, uv_mask, prov, cfg, spec, object_name=name)

    runtime.start(job, worker, _on_refine_event, _on_refine_finished)
    return job


def _on_refine_event(job, kind, payload):
    if kind != "refine_done":
        return
    obj = _obj_for(job, payload)
    hist = job.context["history"].get(payload["object"], "")
    stats = payload.get("stats", {})
    info = f"{payload['provider']} {payload.get('model', '')}"
    if "visible_fraction" in stats:
        info += f" | {stats['visible_fraction']:.0%} of mask visible"
    layers.push_undo(obj, "Correction")
    ly = layers.add_layer(obj, None, name=f"Fix: {bpy.context.scene.ats.refine_prompt[:40]}", kind="CORRECTION",
                          info=info, filepath=payload["layer_path"], pending=True, history_uid=hist, push=False)
    job.context["pending_uid"] = ly.uid
    job.context["visible_fraction"] = stats.get("visible_fraction")
    h = update_history(obj, hist, status="PENDING")
    if h is not None:
        h.layer_uid = ly.uid
        h.filepath = ly.filepath
        h.layer_name = ly.name
    layers.recomposite(obj, save=False)
    apply_material(obj, bpy.context.scene.ats)


def _on_refine_finished(job):
    for key, hist_uid in job.context.get("history", {}).items():
        obj = object_by_uid(job.context["objects"].get(key, ""))
        if obj is None:
            continue
        if job.status != "DONE":
            update_history(obj, hist_uid, status=job.status, detail=job.error or job.message)
    if job.context.get("pending_uid"):
        runtime.push_undo_step("AI Texture Studio: Correction")


def resolve_pending(obj, accept: bool) -> bool:
    st = obj.ats
    for i, ly in enumerate(st.layers):
        if ly.pending:
            if accept:
                ly.pending = False
                for h in st.history:
                    if h.layer_uid == ly.uid:
                        h.status = "ACCEPTED"
            else:
                for h in st.history:
                    if h.layer_uid == ly.uid:
                        h.status = "REJECTED"
                layers.remove_layer(obj, i, push=True)
            layers.recomposite(obj, save=True)
            return True
    return False


# ---------------------------------------------------------------------------
# PBR maps
# ---------------------------------------------------------------------------

def _save_map(obj, kind: str, arr: np.ndarray, size: int) -> bpy.types.Image:
    name = f"ATS_{obj.name}_{kind.title().replace('_', '')}"
    img = storage.get_or_create_image(name, size, non_color=kind != "BASE_COLOR")
    img["ats_map"] = kind
    compat.array_to_image(img, arr)
    path = os.path.join(storage.object_dir(obj), f"{storage.safe_name(obj.name)}_{kind.title().replace('_', '')}.png")
    storage.save_image_datablock(img, path)
    compat.set_colorspace(img, kind != "BASE_COLOR")
    maps = json.loads(obj.ats.maps or "{}")
    maps[kind] = img.name
    obj.ats.maps = json.dumps(maps)
    return img


def generate_pbr(context, obj) -> list[str]:
    sc = context.scene.ats
    if not obj.ats.layers:
        raise UserError("Generate a base colour texture first.")
    size = layers.texture_size(obj, sc)
    base = layers.recomposite(obj, save=True)
    rgb = compat.image_to_array(base)[..., :3]
    _, alpha = layers.current_texture(obj, size)
    valid = alpha > 0.02
    made = []
    preset = sc.material
    if sc.pbr_height or sc.pbr_normal:
        h = pbr.height_from_color(rgb, valid, detail=sc.height_detail)
        if sc.pbr_height:
            _save_map(obj, "HEIGHT", h, size)
            made.append("Height (approx.)")
        if sc.pbr_normal:
            n = pbr.normal_from_height(h, sc.normal_strength)
            _save_map(obj, "NORMAL", n, size)
            made.append("Normal (approx.)")
    if sc.pbr_roughness:
        r = pbr.roughness_map(rgb, preset, base=sc.rough_base if sc.rough_override else None,
                              variation=sc.rough_variation, wear=sc.wear)
        _save_map(obj, "ROUGHNESS", r, size)
        made.append("Roughness (approx.)")
    if sc.pbr_metallic:
        val = sc.metal_value if sc.metal_value >= 0 else pbr.MATERIAL_PRESETS.get(preset, (0, 0, 0))[1]
        mk = painted_mask(obj, size) if sc.metal_from_mask else None
        _save_map(obj, "METALLIC", pbr.metallic_map(size, val, mk), size)
        made.append(f"Metallic (value {val:.2f})")
    # Remove maps the user switched off so the material reflects the panel.
    maps = json.loads(obj.ats.maps or "{}")
    for kind, flag in (("HEIGHT", sc.pbr_height), ("NORMAL", sc.pbr_normal), ("ROUGHNESS", sc.pbr_roughness),
                       ("METALLIC", sc.pbr_metallic)):
        if not flag:
            maps.pop(kind, None)
    obj.ats.maps = json.dumps(maps)
    _strip_unused_nodes(obj, maps)
    apply_material(obj, sc)
    return made


def _strip_unused_nodes(obj, maps):
    for slot in obj.material_slots:
        mat = slot.material
        if mat is None or not mat.node_tree or "ats_owner" not in mat:
            continue
        for kind in ("ROUGHNESS", "METALLIC", "NORMAL", "HEIGHT", "AO"):
            if kind not in maps:
                node = mat.node_tree.nodes.get(materials.PREFIX + kind.title())
                if node is not None:
                    mat.node_tree.nodes.remove(node)
                if kind == "NORMAL":
                    nm = mat.node_tree.nodes.get(materials.PREFIX + "Normal Map")
                    if nm is not None:
                        mat.node_tree.nodes.remove(nm)


def register_baked_ao(obj, img: bpy.types.Image):
    img["ats_map"] = "AO"
    maps = json.loads(obj.ats.maps or "{}")
    maps["AO"] = img.name
    obj.ats.maps = json.dumps(maps)


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def analyze(context, obj) -> dict:
    sc = context.scene.ats
    validate_basic = obj is not None and obj.type == "MESH"
    if not validate_basic:
        raise UserError("Select a mesh object.")
    info = {"object": obj.name}
    has_uv = bool(obj.data.uv_layers)
    mesh = meshio.extract(obj, context, uv_name=(obj.ats.uv_map or None) if has_uv else "",
                          use_modifiers=sc.use_modifiers)
    rep = meshdata.analyze(mesh)
    M = np.array(obj.matrix_world)
    scale = np.linalg.norm(M[:3, :3], axis=0)
    if np.linalg.det(M[:3, :3]) < 0:
        rep.add("INFO", "Object has negative scale (mirrored); normals are handled.")
    if scale.max() > 0 and scale.max() / max(scale.min(), 1e-9) > 1.01:
        rep.add("INFO", "Non-uniform object scale; texture density follows world size.")
    mods = [f"{m.name} ({m.type})" for m in obj.modifiers]
    if mods and sc.use_modifiers:
        rep.add("INFO", f"Projecting onto the evaluated mesh with modifiers: {', '.join(mods[:4])}.")
    if obj.data.shape_keys:
        rep.add("INFO", f"{len(obj.data.shape_keys.key_blocks)} shape keys (preserved, current mix is used).")
    if any(m.type == "ARMATURE" for m in obj.modifiers):
        rep.add("INFO", "Rigged mesh: texture is projected in the current pose; use rest pose for best results.")
    users = meshio.shared_mesh_users(obj)
    if users:
        rep.add("WARNING", f"Mesh data is shared with {users} other object(s); slot materials linked to DATA "
                           "will change for them too.")
    tex = meshio.existing_textures(obj)
    info["textures"] = [f"{m}: {i} ({w}x{h})" for m, i, w, h in tex][:8]
    info["materials"] = [s.material.name if s.material else "(empty)" for s in obj.material_slots]
    info["uv_maps"] = [u.name for u in obj.data.uv_layers]
    if rep.suitable and has_uv:
        size = int(sc.view_size) // 4
        cams = [c for _, c in enabled_cameras(sc, mesh, max(128, size))]
        if cams:
            rep.view_coverage = meshdata.estimate_view_coverage(mesh, cams)
            if rep.view_coverage < 0.9:
                rep.add("WARNING", f"Enabled views see {rep.view_coverage:.0%} of the surface. Hidden areas get "
                                   "colour-extended; add views (e.g. 3/4 or viewport views) or refine them later.")
            else:
                rep.add("INFO", f"Enabled views see {rep.view_coverage:.0%} of the surface.")
    info.update(rep.to_dict())
    obj.ats.analysis = json.dumps(info)
    return info


# ---------------------------------------------------------------------------
# UV creation (explicit user action, adds a NEW UV map)
# ---------------------------------------------------------------------------

def create_uv_map(context, obj, method: str, angle: float, margin: float, set_render: bool,
                  name: str = "ATS_UV") -> str:
    me = obj.data
    if len(me.uv_layers) >= 8:
        raise UserError("Mesh already has the maximum of 8 UV maps.")
    prev_mode = obj.mode
    prev_active = context.view_layer.objects.active
    prev_sel = {o.name: o.select_get() for o in context.view_layer.objects}
    sel_state = None
    for o in context.view_layer.objects:
        o.select_set(o == obj)
    context.view_layer.objects.active = obj
    try:
        if prev_mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        sel_state = (np.array([p.select for p in me.polygons]), np.array([v.select for v in me.vertices]),
                     np.array([e.select for e in me.edges]))
        uv = me.uv_layers.new(name=name)
        if uv is None:
            raise UserError("Could not add a UV map.")
        # Keep only the name: entering Edit Mode reallocates mesh layers, which
        # invalidates Python references to them.
        uv_name = uv.name
        me.uv_layers.active = uv
        del uv
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        if method == "SMART":
            bpy.ops.uv.smart_project(angle_limit=angle, island_margin=margin, area_weight=0.0,
                                     correct_aspect=True, scale_to_bounds=False)
        elif method == "LIGHTMAP":
            bpy.ops.uv.lightmap_pack(PREF_CONTEXT="ALL_FACES", PREF_PACK_IN_ONE=True, PREF_MARGIN_DIV=max(0.05, margin * 10))
        else:
            bpy.ops.uv.cube_project(cube_size=1.0, correct_aspect=True, scale_to_bounds=True)
            bpy.ops.uv.pack_islands(margin=margin)
        bpy.ops.object.mode_set(mode="OBJECT")
        if set_render:
            me.uv_layers[uv_name].active_render = True
        obj.ats.uv_map = uv_name
        return uv_name
    finally:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        if sel_state is not None:
            ps, vs, es = sel_state
            me.polygons.foreach_set("select", ps)
            me.vertices.foreach_set("select", vs)
            me.edges.foreach_set("select", es)
            me.update()
        for o in context.view_layer.objects:
            if o.name in prev_sel:
                o.select_set(prev_sel[o.name])
        context.view_layer.objects.active = prev_active
        if prev_mode != "OBJECT" and prev_active == obj:
            try:
                bpy.ops.object.mode_set(mode=prev_mode)
            except RuntimeError:
                pass


def transfer_to_uv(context, obj, target_uv: str) -> list[str]:
    """Re-bake the ATS maps from the working UV map onto another UV map."""
    src_uv = obj.ats.uv_map
    if target_uv == src_uv or target_uv not in obj.data.uv_layers:
        raise UserError("Choose a different, existing UV map.")
    mesh = meshio.extract(obj, context, uv_name=src_uv, use_modifiers=False, extra_uvs=(target_uv,))
    dst = mesh.extra_uvs[target_uv]
    out = []
    for kind, img in current_maps(obj).items():
        size = img.size[0]
        src = compat.image_to_array(img)
        uvs = raster.rasterize_uv(dst, size)
        suv = mesh.uv_points(uvs.tri, uvs.bary)
        col = imageio.sample_bilinear(src, suv[:, 0] * src.shape[1], (1 - suv[:, 1]) * src.shape[0])
        res = np.zeros((size * size, 4), np.float32)
        res[uvs.index] = col
        res = res.reshape(size, size, 4)
        valid = np.zeros(size * size, bool)
        valid[uvs.index] = True
        rgb, _ = masks.fill_gutter(res[..., :3], valid.reshape(size, size), 8)
        name = f"{img.name}_{storage.safe_name(target_uv)}"
        new = storage.get_or_create_image(name, size, non_color=kind != "BASE_COLOR")
        compat.array_to_image(new, rgb)
        path = os.path.join(storage.object_dir(obj), f"{name}.png")
        storage.save_image_datablock(new, path)
        new["ats_map"] = kind
        out.append(new.name)
    return out
