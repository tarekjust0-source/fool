# Testing

There are three levels of testing. Only the first runs automatically. **None of the automated tests contacts a real AI service or runs a real AI model.**

| Level | What | Who runs it |
|---|---|---|
| 1. Automated | 124 tests: unit, Blender integration, Cycles-render validation, UI draw, harness self-check | `pytest` (CI or you) |
| 2. Real provider | `tools/real_provider_test.py`: the full workflow against OpenAI / Stability / Gemini / ComfyUI | you, with credentials |
| 3. Interactive | checklist below for things that need a Blender window | you |

## 1. Automated tests

Last run in this repository: **124 passed on Blender 4.2.23 LTS and 124 passed on Blender 5.0.1** (PyPI `bpy` modules, Python 3.11, Linux, CPU only). Both runs used the freshly built `dist/ai_texture_studio-1.0.0.zip`, installed through Blender's extension installer.

| File | Tests | Covers |
|---|---|---|
| `tests/unit/test_core.py` | 48 | PNG codec (all 5 filter types, corruption), resampling, rasteriser (coverage, z-buffer, huge triangles, UV overlap), cameras, multi-view pipeline (coverage, skipping, occlusion, min-facing deferral, flipped normals, thin two-sided shield), non-square provider output cropping, alignment, colour match, 3D hidden-area fill, local refinement (UV and view), refinement design context, tiles, cancellation, compositing, padding, normal-map orientation, PBR ranges, prompt composition, key redaction, mesh analysis |
| `tests/unit/test_projection_quality.py` | 3 | reconstruction error against ground truth with a perfectly consistent oracle provider (sphere 0.2/255 mean), occluding scene, drift + auto-align |
| `tests/unit/test_providers.py` | 30 | exact wire format for OpenAI / Stability / Gemini / ComfyUI against **local fake servers**: multipart fields, mask conventions, headers, JSON, `output_format`, `input_fidelity` rules for gpt-image-2, retry-without-rejected-parameter, model listing (OpenAI, paginated Gemini, both ComfyUI `/object_info` formats), error classification, `Retry-After`, timeouts, network errors, invalid bodies, cancellation |
| `tests/blender/test_integration.py` | 34 | real operators in Blender: see list below |
| `tests/blender/test_render_validation.py` | 4 | **Cycles renders** through the add-on's own material: seams from unused 3/4 angles (0.8/255 mean, 1.6/255 p99), single-view regeneration (front changes, right side pixel-identical), face-mask refinement (only the selected face changes), GLB re-import in a **separate factory-clean Blender process** |
| `tests/blender/test_ui.py` | 2 | every panel, UI list and preferences page drawn in all states; every property and operator must exist |
| `tests/blender/test_real_provider_harness.py` | 3 | the real-provider script refuses the test provider and missing keys, dry run sends nothing, and the self-check run is labelled *NOT A REAL PROVIDER TEST* |

Integration scenarios (`test_integration.py`):

- **UV maps:** a mesh with no UV map is rejected, then Create UV Map; existing UVs and selection are preserved.
- **Materials:** multiple material slots get copies; originals are untouched and restorable. Node connections and colour spaces are correct, and a real AO bake runs.
- **Projection alignment:** a screen-space pattern lands correctly in a Cycles render from a matching camera.
- **Low-poly character with rig, shape keys and modifiers:**
  - full workflow: generate, refine, accept, PBR, GLB + FBX export
  - all scene data preserved
  - an unrelated object stays untouched
- **Occlusion:** occluded geometry is reported and colour-extended, with a "not AI-generated" label.
- **Failures:**
  - 7 failure modes create no layers and leave materials unchanged
  - a partial failure plus Retry pays only for the missing views
  - a failed correction changes nothing
  - missing credentials are blocked before any network access
  - the API key is redacted from errors and history
- **Persistence:** save, reload and Save As elsewhere work; opening another file during a job is safe.
- **Repeated work:**
  - repeated generation skips painted areas
  - correction reject/accept with exact pixel restore
  - layer operations, undo/redo, history restore and prompt history
- **Audit regressions:**
  - single-view regeneration sends exactly one request, repaints only that view and keeps the old version hidden
  - a new view paints areas that were only colour-extended
  - custom viewport views record status and intermediate images
  - protected base layers are never repainted
- **Other:** visibility/opacity, tileable mode, multi-object shared look, viewport-refinement guard, importing an existing texture, prompt preview, UV transfer, resize and cleanup.

### Running

```bash
python3.11 -m venv venv
venv/bin/pip install "bpy==4.2.23" "numpy==1.26.4" pytest    # or bpy==5.0.1 with numpy 1.26
venv/bin/python -m pytest tests                  # everything (~3.5 min on 4 CPU cores)
venv/bin/python -m pytest tests/unit             # pure Python, no Blender needed beyond numpy
ATS_ZIP=dist/ai_texture_studio-1.0.0.zip venv/bin/python -m pytest tests   # test a specific build
```

Inside a real Blender installation (pytest is installed into `tests/.blender_site` on first run):

```bash
blender --background --factory-startup --python tests/blender/run_in_blender.py
```

The tests install the add-on into an isolated user profile. Generation uses synthetic providers:

- the offline **test provider** (registered only with `ATS_ENABLE_TEST_PROVIDER=1`; obviously synthetic patterns);
- an "oracle" that paints a known 3D colour field, so projection can be compared with ground truth.

Neither is a model. Results from them show that the *pipeline* is correct, not that a real model produces good textures.

> Note for the PyPI `bpy` module: it can abort during interpreter shutdown after a Cycles render, independent of this add-on. The test results are reported before that happens. Real Blender exits normally.

## 2. Real-provider test

Requires a provider key (or a running ComfyUI) and network access. It sends about **7 image requests** (3 for a cube, 3 for a character, 1 correction). `--views N` changes the per-object count, and `--skip-character` limits it to the cube.

```bash
# OpenAI (Stability: STABILITY_API_KEY, --provider STABILITY; Gemini: GEMINI_API_KEY, --provider GEMINI)
export OPENAI_API_KEY=sk-...
blender -b --factory-startup --online-mode --python tools/real_provider_test.py -- --provider OPENAI --yes
```

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY="sk-..."
& "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe" -b --factory-startup --online-mode `
  --python tools\real_provider_test.py -- --provider OPENAI --yes
```

```bash
# ComfyUI (local; no key, no online mode needed for 127.0.0.1)
blender -b --factory-startup --python tools/real_provider_test.py -- --provider COMFYUI \
  --model sd_xl_base_1.0.safetensors --controlnet <your depth ControlNet file> --yes
```

Useful options: `--model <id>`, `--base-url <url>`, `--out <dir>`, `--zip <path>`. Omit `--yes` to see only what would be sent.

Checks (PASS/FAIL in `report.md` / `report.json`):

**A. Simple cube**
- connection test
- job finishes
- requests sent are 1..N, with one AI layer per request, each from the chosen provider
- AI share of the surface
- the base colour image is a real (non-blank) texture
- the texture is wired to the Principled BSDF on the working UV map
- provider outputs saved; Cycles render
- after save and reopen: same layers, **pixel-identical texture**, files next to the `.blend`, identical render

**B. Low-poly character with rig and shape keys**
- the same generation checks
- the local correction stays inside the selected faces
- PBR maps connected
- mesh, rig and shape keys preserved
- GLB exported and re-imported in a clean Blender process with all textures connected

**Then look at the pictures.** `renders/` holds lit Cycles renders and `provider_outputs/` the raw images each view received and returned. The script cannot judge artistic quality, seams in real model output, or whether the model respected the silhouette. If something looks wrong, the *render* / *generated* / *aligned* images show where it happened. Please keep `report.md` when reporting problems; it contains no key.

## 3. Interactive checklist

These need a Blender window and cannot be automated headless:

1. **Progress.** Generate with a hosted provider: the consent dialog appears once, progress updates per view, and you can orbit while it runs.
2. **Cancel** midway: finished views stay, the rest stop. **Retry** requests only the missing views.
3. **Paint Mask.** Texture Paint opens with the mask image and paints white; *Done Painting* returns to Object Mode.
4. **From Viewport** refinement: orbit so the area faces you, Generate Correction, *Compare* toggle, then Accept / Reject.
5. **Regenerate one view.** ▶ on a view: only that region changes; the old layer is hidden as "(previous)".
6. **Inspect views.** Multi-View › *Render / Mask / AI Output / Aligned* open in an Image Editor.
7. **Provenance map.** **Show Provenance Map** opens the provenance image with percentages.
8. **Ctrl+Z** after a generation removes it; Ctrl+Shift+Z brings it back. If the texture looks blank after redo, press *Update Texture*.
9. **List Models** shows models for your key; clicking one sets it.
