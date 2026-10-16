# Architecture and technical plan

## Research summary

**Modddif (functional reference).** Its texture documentation (docs.modddif.com/project/texture) could not be fetched directly from the development sandbox, because the host was not reachable. The summary below therefore comes from search-indexed descriptions of that page and modddif.com:

- Multi-view generation uses preset camera angles whose images are projected onto the object, with one layer per angle.
- It only generates the parts of the object that are not yet textured.
- A reference image can be selected or deselected alongside an optional prompt.
- A single-view mode lets you validate each image, change the prompt or reference, and choose the angle.
- Inpainting fixes local problems; disabling an existing texture layer regenerates from scratch.

AI Texture Studio reproduces this *workflow* with its own implementation and UI. It adds explicit occlusion handling, seam bands, alignment and colour matching, labelled hidden-area fill, refinement with accept/reject, its own layer undo, and PBR provenance labels.

**Blender.**

- Packaging: 4.2+ extension format (`blender_manifest.toml`), relative imports, and `bpy.app.online_access` must be respected.
- Threading: `bpy` is not thread-safe, so all scene changes happen on the main thread via `bpy.app.timers`.
- Version changes handled: `FILE_PATH` properties need `PATH_SUPPORTS_BLEND_RELATIVE` on 5.0; `Material.use_nodes` is deprecated in 5.0; mesh layer references are invalidated by mode switches.

**Image models.** None of the accessible hosted APIs is natively multi-view consistent. Practical consistency comes from the literature's *sequential inpainting* approach (TEXTure, Text2Tex): render the partially textured mesh and inpaint only the new region with existing texture as context. Structure is conveyed by the render itself, or by a depth ControlNet for ComfyUI.

## Option comparison (summary)

| Option | Quality | Consistency tools | Privacy | Cost | Integration |
|---|---|---|---|---|---|
| OpenAI GPT Image | very good prompt following | multi-image refs, soft mask | cloud | per image | simple REST |
| Stability AI | good | true inpaint, structure control, seed | cloud | credits | simple REST |
| Gemini image | very good | multi-image input | cloud | per image | simple REST |
| ComfyUI (SDXL/Flux + ControlNet) | depends on model | true inpaint + **depth** | **local** | free | needs a GPU and setup |
| 3D texturing SaaS (Meshy, Tripo) | good | native 3D | cloud | credits | replaces UVs/mesh ⇒ rejected |

## Package layout

```
ai_texture_studio/
  blender_manifest.toml
  __init__.py              register()/unregister() – imports bpy lazily
  core/                    pure Python + numpy, thread-safe, unit-tested
    raster.py              vectorised triangle rasteriser (UV + screen, z-buffer, persp.-correct)
    camera.py              preset / viewport cameras (OpenGL conventions)
    render.py              mesh renders (clay + texture, depth, normals), visibility tests
    projection.py          multi-view projection / blending / alignment / fill
    pipeline.py            multi-view, refinement (view + UV), tile pipelines
    compositing.py         layer blend modes
    pbr.py                 map derivation (labelled approximations)
    prompting.py           style controls → prompt text + parameters
    imageio.py masks.py    PNG codec (all filter types), resampling, morphology, padding
    http.py secrets.py     urllib client (multipart, retries), key redaction
    jobs.py meshdata.py    background jobs; mesh snapshot + analysis
  providers/               provider interface + OpenAI, Stability, Gemini, ComfyUI, test mock
  blender/                 everything that imports bpy
    props.py prefs.py      data model (Scene.ats, Object.ats), preferences, credentials
    runtime.py             job ↔ main-thread bridge (timer), main-thread call queue
    services.py            gather inputs, start jobs, apply results, analysis, UV creation
    layers.py storage.py   layer stack, compositing to image, undo; PNG storage, save handlers
    materials.py bake.py export.py
    operators/  ui/
```

## Threading model

1. **Operator (main thread).** Validate, take a mesh snapshot (evaluated mesh, world space), read the current composite and reference pixels, and resolve the provider config and key.
2. **Worker thread.** Render views, call the provider, align, project and write layer PNGs. A preview composite is computed in the worker too.
3. **Timer (main thread).** Drain events: add a layer and copy preview pixels (about 0.1 s), and wire materials once.
4. **Finish.** Full recomposite and save, history update, undo push.

Cancellation is checked between steps and during retry waits. A late HTTP result after cancelling is discarded.

## Multi-view algorithm (per view, in order)

1. Rasterise the mesh from the camera (z-buffer, perspective-correct barycentrics). Render the current texture (gutter-padded) and shaded clay for untextured surface.
2. AI mask = visible untextured pixels, dilated by *Seam Blend* pixels. The first view paints the whole object.
3. Provider call: render, mask, depth, references and the composed prompt.
4. Post-process: resize; phase-correlation **auto-alignment** for providers that may drift; **colour transfer** using a ring of existing texture around the mask.
5. UV-space projection. For every texel (UV-rasterised once), compute its world point and normal, then run the visibility test: frustum, depth buffer with slope-scaled tolerance, back-face culling.
   - **Claim** untextured texels with facing ≥ *Min Facing* that are away from silhouettes or depth discontinuities.
   - Texels already textured inside the band get a **cross-fade** alpha.
   - Grazing texels become **fallback** candidates (best facing wins).
6. The view becomes an RGBA layer; the in-memory state is updated so the next view sees it.
7. After all views: fill layer = fallbacks, plus push-pull colour extension for surfaces no camera saw (reported as a fraction).

Globally inverted normals are detected (signed volume) and re-oriented in the snapshot. Thin two-sided objects keep separate front and back paint.

Measured accuracy with a perfectly consistent "oracle" provider (`tests/unit/test_projection_quality.py`):

- Sphere: mean error 0.2/255 at 100% coverage.
- Occluding scene: 1.6/255.
- With 9 px provider drift: 0.5/255 with auto-align versus 4.1/255 without.

## Layers and undo

Layers are immutable PNG files. Editing produces a new file, and the stack keeps references and settings. Layer operations push JSON snapshots of the stack, which gives reliable undo/redo independent of Blender's memfile undo (that undo does not track image pixels). Base colour = composite of visible layers, plus output filters (pixelate, palette lock), plus edge padding.

## Materials

Materials are never modified in place: slot materials are copied, or a single new material is used. Original names and fake-user states are recorded in `Object.ats.material_state` for one-click restore. Nodes are named `ATS …` so re-applying updates them instead of duplicating.

## Roadmap

1. Live verification against each provider; tune prompts per model.
2. Flux Fill / Kontext provider; ComfyUI IP-Adapter template.
3. AI material-estimation maps (roughness, metal, normal) via ComfyUI workflows; clearly labelled as AI-estimated.
4. View upscaling; GPU (`gpu` module) rasterisation for very dense meshes; UDIM tiles.
5. Automatic "problem area" detection (low-confidence texels) suggesting extra views.
