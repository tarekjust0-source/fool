# AI Texture Studio for Blender

AI-assisted, **UV-aware** texturing inside Blender. It renders your own mesh from several cameras, has an AI image model paint each view and projects the result back onto the mesh's **existing UV map**. Seams are blended and occlusion is respected. You can then fix individual areas with local AI corrections and keep everything in a non-destructive layer stack. The result is a Principled BSDF material ready for glTF/FBX export.

Inspired by the texturing workflow of Modddif (multi-view generation, reference images, layers, inpainting), but written from scratch with an original UI and codebase.

- **Blender:** 4.2 LTS or newer (extension format). Automated tests run on **4.2.23 LTS** and **5.0.1**.
- **Dependencies:** none beyond Blender's bundled Python and numpy. `keyring` is optional, for OS credential storage.
- **AI providers:** OpenAI GPT Image, Stability AI, Google Gemini image models, or **ComfyUI** (local or self-hosted).
- **License:** GPL-3.0-or-later.

> **Status in one sentence:** the add-on, its Blender integration and its projection pipeline are tested automatically on two Blender versions, including real Cycles renders. **No request has yet been made to a real AI service:** the development environment had neither credentials nor network access to any provider or model host. Run [`tools/real_provider_test.py`](#test-it-with-a-real-provider-10-minutes) once to verify your provider.

---

## Install

1. Download `dist/ai_texture_studio-1.0.0.zip` (or build it: `python tools/build.py`).
2. In Blender: **Edit › Preferences › Get Extensions › ▾ (top right) › Install from Disk…** and pick the zip.
3. For hosted providers, enable **Allow Online Access** (Preferences › System › Network). The add-on respects this switch; local ComfyUI on `127.0.0.1` works without it.
4. Open **Preferences › Add-ons › AI Texture Studio**, choose a provider and set up its key (see below). **List Models** shows the model ids your account can use.
5. In the 3D Viewport press **N** and open the **AI Texture** tab.

## Configure a provider (details in [docs/PROVIDERS.md](docs/PROVIDERS.md))

| Provider | What you need | Default model | Key environment variable |
|---|---|---|---|
| OpenAI GPT Image | API key with image access | `gpt-image-1.5` (presets: 1, 1-mini, 2) | `OPENAI_API_KEY` |
| Stability AI | API key (credits) | fixed v2beta endpoints | `STABILITY_API_KEY` |
| Google Gemini | Gemini API key | `gemini-3.1-flash-image` | `GEMINI_API_KEY` |
| ComfyUI | Running server plus checkpoint; a depth ControlNet is recommended | your checkpoint | none (optional token) |

**Key storage**, set per provider:

- **Environment variable** (recommended).
- **Session only**: memory only.
- **System keyring**: needs the `keyring` package.
- **Blender preferences**: **plain text** in `userpref.blend`.

Keys are redacted from every message, history entry and console print. Before the first request to a hosted provider, a dialog shows what will be sent; those requests are billed by the provider. Requests per run = enabled views that still have something to paint; finished views are skipped. A correction is one request.

## Test it with a real provider (≈10 minutes)

```bash
export OPENAI_API_KEY=sk-...                       # or STABILITY_API_KEY / GEMINI_API_KEY
blender -b --factory-startup --online-mode --python tools/real_provider_test.py -- --provider OPENAI --yes
```

About 7 image requests are sent: 3 for a cube, 3 for a character, and 1 correction. The script reports PASS/FAIL for each step:

- generate
- image validation
- material and UV wiring
- Cycles render
- save, reopen and pixel-identical texture
- local-correction locality
- PBR wiring
- GLB re-import in a clean Blender

It writes `ats_real_provider_test/report.md` plus renders and the raw provider outputs **for you to look at**. Without `--yes` it only prints what it would send. Full instructions, including Windows and ComfyUI, are in [docs/TESTING.md](docs/TESTING.md#2-real-provider-test).

## Quick start

1. Select a mesh and click **Analyze Mesh**. It reports UVs, overlaps, materials, existing textures, flipped normals, and how much surface the enabled cameras see. No UVs? **Create UV Map** adds a *new* map.
2. Write a prompt; optionally pick a reference image.
3. Choose a style, material, detail, wear, prompt strength and an optional palette. Texture size is 512–4096.
4. **Generate (N requests)**: each finished view appears on the model as its own layer.
5. **A view looks wrong?** Click ▶ on that view in *Multi-View* to regenerate only it. The other views stay as they are and act as context; the old version is kept, hidden, for comparison.
6. **Local Refinement:** paint a mask or select faces, describe the fix, then **Generate Correction** → *Compare* → **Accept / Reject**.
7. **PBR Maps:** roughness, normal and height approximations (labelled as such), metallic from a value, and AO baked from the geometry.
8. **Export:** **Save Textures** and **Export Model** (`.glb` / `.fbx`).

Example prompts:

- *Weathered dark iron armor with hand-painted scratches and subtle edge wear* (Hand-painted, Metal, Wear 0.6).
- *Stylized low-poly orc skin, muted green tones, painted shadows, and subtle color variation* (Low-poly, palette on).
- *Old wooden shield with chipped paint, visible wood grain, and a worn medieval appearance* (views: Front + Back).
- *Realistic leather straps with stitching, creases, and darkened edges* (Realistic, Leather).

More in [docs/USAGE.md](docs/USAGE.md).

## What is verified, and how

### Verified automatically: 124 tests, run in this repository on Blender 4.2.23 LTS and 5.0.1

No real AI model or service is involved in any of these; generation uses deterministic synthetic providers.

- **Installation and UI:**
  - the extension ZIP installs through Blender's own extension installer and enables
  - every panel, list and preferences page draws in all states
- **Scene preservation:**
  - mesh, topology, transforms, modifiers, armature, shape keys and vertex groups unchanged
  - original materials kept and restorable; unrelated objects untouched
- **Projection correctness in real Cycles renders:**
  - a pattern painted in a view reappears at the right place in a render from a matching Blender camera
  - with a perfectly consistent provider, renders from **3/4 angles no generation camera used** match ground truth within **0.8/255 mean, 1.6/255 p99**, so the projection adds no visible seams
- **Single-view regeneration in Cycles renders:** the regenerated view's region changes, while the other sides stay pixel-identical.
- **Local refinement in Cycles renders:**
  - a face-selection mask changes exactly that face; untouched faces stay unchanged
  - Reject restores the texture exactly
  - a failed correction changes nothing
- **Failures:** network, timeout, rate limit, auth, content policy, invalid and blank images all produce no layers and no material change. A partial failure keeps the finished views, and Retry pays only for the rest.
- **Persistence:** save, reopen and *Save As* elsewhere keep relative texture paths and pixel-identical textures.
- **Export:** GLB re-imported in a **separate factory-clean Blender process** keeps base colour (pixel-matched), roughness, metallic and normal connections.
- **Providers (against local fake servers, not the real services):**
  - endpoints, headers, multipart fields, mask conventions
  - parameter fallbacks, error classification, `Retry-After`, timeouts, cancellation, model listing

### Implemented and tested only offline: needs your credentials to confirm

- Real generation quality and API acceptance with **OpenAI**, **Stability AI**, **Gemini** and **ComfyUI**. Request formats follow current public documentation (checked October 2026), but providers change models and parameters often.
- Whether a specific model honours masks, reference images and the "keep the silhouette" instruction well enough for good projection. Auto-align and colour matching compensate for small drifts; their effect on real model outputs is unmeasured.
- **Test Connection** and **List Models** against the live APIs.

→ Run `tools/real_provider_test.py` (above) and look at the renders.

### Needs interactive testing in a Blender window

Headless automation cannot exercise these; the [manual checklist](docs/TESTING.md#3-interactive-checklist) covers them:

- **From Viewport** refinement (needs a 3D viewport; the underlying pipeline is tested)
- Mask painting in Texture Paint mode
- Image Editor display
- Progress updates while generating in the background
- Cancel during a running request
- Ctrl+Z after a generation
- The consent dialog

### Known limitations

- **Cross-view consistency** depends on the model. Image models are not 3D-aware; context inpainting, seam bands and colour matching reduce mismatches but cannot guarantee identical details on both sides of a seam. Regenerate single views or use Local Refinement.
- **Hidden surfaces** (seen by no camera) are **colour-extended from the nearest painted surface in 3D, not AI-painted**. They sit on their own *Hidden-area fill* layer, shown red in the **Provenance Map**. Pixels seen only at flat angles sit on a *Low-angle views* layer. A view added later will repaint both.
- **PBR maps:** only base colour is AI-generated. Roughness, normal and height are **image-derived approximations**, not physically measured or AI-generated maps; metallic is a user value; AO is a real geometric bake.
- **Overlapping or mirrored UVs** share texture. UDIM and UVs outside 0–1 are unsupported.
- **Cancel** cannot abort a request the provider is already processing, so it may still be billed.
- **Model ids change.** For example, `gemini-2.5-flash-image` is scheduled for retirement. Use **List Models** when a model is rejected.

## Tests

```bash
python3.11 -m venv venv && venv/bin/pip install bpy==4.2.23 numpy==1.26.4 pytest
venv/bin/python -m pytest tests                     # or: blender -b --factory-startup --python tests/blender/run_in_blender.py
```

See [docs/TESTING.md](docs/TESTING.md).

## Documentation

- [docs/USAGE.md](docs/USAGE.md): workflow, single-view regeneration, protection, provenance, tips for stylized assets.
- [docs/PROVIDERS.md](docs/PROVIDERS.md): setup, capabilities, costs, data handling, model churn.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): design and multi-view algorithm.
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): common problems.
- [docs/TESTING.md](docs/TESTING.md): automated tests, real-provider test, interactive checklist.
