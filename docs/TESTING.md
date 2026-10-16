# Testing

## Automated tests

| Suite | Runs in | What it covers |
|---|---|---|
| `tests/unit/test_core.py` (45 tests) | any Python 3.11 with numpy | PNG codec (all 5 filter types, corruption), resampling, rasteriser (coverage, z-buffer, huge triangles, UV overlap), cameras, multi-view pipeline (coverage, skipping, occlusion, min-facing deferral, flipped normals, thin two-sided shield), alignment, colour match, local refinement (UV and view), tiles, cancellation, compositing modes, padding/fill, normal-map orientation, PBR ranges, prompt composition, key redaction, mesh analysis |
| `tests/unit/test_projection_quality.py` (3 tests) | same | quantitative reconstruction error with a perfectly consistent oracle provider, including occlusion and drift + auto-align |
| `tests/unit/test_providers.py` (25 tests) | same | exact wire format for OpenAI / Stability / Gemini / ComfyUI against local fake servers (multipart fields, mask conventions, headers, JSON), error classification (401/402/429/400-policy/5xx), Retry-After, timeouts, network errors, invalid image bodies, cancellation during back-off, missing keys |
| `tests/blender/test_integration.py` (28 tests) | Blender (bpy module or real Blender) | the extension zip is **built, installed via Blender's extension installer and enabled**, then real operators are driven (see list below) |
| `tests/blender/test_ui.py` (2 tests) | Blender | every panel, UI list and preferences page drawn in all states with a validating layout (every property and operator must exist) |

Integration scenarios:

- mesh with no UV map (rejected, then Create UV Map)
- UV creation preserving existing UVs and selection
- existing UVs and materials with multiple slots: copies made, originals untouched and restorable
- material node connections and colour spaces, plus a real AO bake
- **projection alignment against a Cycles render** from a matching Blender camera
- low-poly character with armature, shape keys, vertex groups and modifiers through the full workflow (generate, refine, accept, PBR, GLB + FBX export), with preservation of all scene data and of an unrelated object
- occluded inner geometry reported and filled
- failed requests (network, timeout, rate limit, auth, content policy, invalid image, blank image) create no layers and leave materials unchanged
- partial failure keeps finished views; Retry pays only for the rest
- missing credentials blocked before any network access
- API key redacted from errors and history
- save, reload, *Save As* elsewhere (relative path remap)
- repeated generation (skips already textured areas), regenerate, correction reject/accept with exact pixel restore, layer operations, undo/redo, history restore, prompt history
- visibility and opacity effects, tileable mode, multi-object shared look, viewport refinement guard
- import of an existing texture, prompt preview reflecting style controls, UV transfer, resize and cleanup

Results in the development environment (bpy 4.2.23 LTS and 5.0.1 from PyPI, Python 3.11, Linux): **all 103 tests passed on both versions**, run against the built ZIP.

### Running

```bash
python3.11 -m venv venv
venv/bin/pip install "bpy==4.2.23" "numpy==1.26.4" pytest   # or bpy==5.0.1
venv/bin/python -m pytest tests            # everything
venv/bin/python -m pytest tests/unit       # pure Python only (no Blender needed beyond numpy)
```

Inside a real Blender installation (pytest is installed into `tests/.blender_site` on first run):

```bash
blender --background --factory-startup --python tests/blender/run_in_blender.py
blender -b --factory-startup --python tests/blender/run_in_blender.py -- -k refine
```

**Not covered by automation** (headless Blender has no interactive UI, so these need the manual checklist below):

- the *From Viewport* refinement operator (it needs a 3D viewport; the underlying pipeline *is* unit-tested)
- interactive mask painting in Texture Paint mode
- showing images in an Image Editor window
- Blender's global Ctrl+Z step pushed after a job (a no-op in background mode)
- Test Connection against real services

The tests use an isolated Blender user profile, and the offline **test provider**, which is registered only when `ATS_ENABLE_TEST_PROVIDER=1`. It produces obviously synthetic patterns. It is not available to normal users and is not a substitute for generation.

## Manual end-to-end checklist (real provider)

Outbound access to AI APIs was blocked where this add-on was developed, so please run this once with your own key (about 5–10 requests).

1. Install the zip, enable *Allow Online Access*, set the provider key, then press **Test Connection**. Expect "Connected…".
2. Default cube or Suzanne, prompt *"Weathered dark iron armor with hand-painted scratches and subtle edge wear"*, style Hand-painted, texture 1024, view 1024.
   **Generate**:
   - The consent dialog appears once.
   - Progress shows per view and Blender stays responsive (orbit while it runs).
   - Each view appears on the model as it finishes.
3. Multi-View › select *Front* › *Render / AI Output / Aligned*: the AI output should match the render's silhouette.
4. **Cancel** a second run midway: finished views stay, nothing else changes. **Retry**: only missing views are requested.
5. Paint a mask on the side, set the correction to *"Replace this area with rusty iron"*, method *From Viewport*, **Generate Correction** → *Compare* → **Reject** (texture returns exactly) → generate again → **Accept**.
6. Set a reference image and regenerate with *Start From Scratch*; the colours should follow the reference.
7. **PBR Maps › Generate Maps** (with AO) → check the Shading workspace: nodes connected, data maps Non-Color.
8. **Export Model** (GLB) → import into a glTF viewer or game engine and check the base colour, normal, ORM and AO.
9. Save the .blend, restart Blender, reopen it: layers, history and material are intact.
10. Wrong key → clear error, no layers added, and the key is not visible in the message.
