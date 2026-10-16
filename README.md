# AI Texture Studio for Blender

AI-assisted, **UV-aware** texturing inside Blender. It renders your own mesh from several cameras, has an AI image model paint each view and projects the result back onto the mesh's **existing UV map**. Seams are blended and occlusion is respected. You can then fix individual areas with local AI corrections and keep everything in a non-destructive layer stack. The result is a Principled BSDF material ready for glTF/FBX export.

Inspired by the texturing workflow of Modddif (multi-view generation, reference images, layers, inpainting), but written from scratch with an original UI and codebase.

- **Blender:** 4.2 LTS or newer (extension format). Automated tests run on **4.2.23 LTS** and **5.0.1**.
- **Dependencies:** none beyond Blender's bundled Python and numpy. `keyring` is optional, for OS credential storage.
- **AI providers:** OpenAI GPT Image, Stability AI, Google Gemini image models, or **ComfyUI** (local or self-hosted; nothing leaves your machine).
- **License:** GPL-3.0-or-later.

> **Honest status:** every feature below is implemented and covered by automated tests that use an offline *test* provider and fake HTTP servers. Request formats follow each provider's public API documentation. They were **not** verified against the live services in this development environment, because outbound access to AI APIs was blocked there. Please run the 5-minute [real-provider checklist](docs/TESTING.md#manual-end-to-end-checklist-real-provider) on your setup.

---

## Install

1. Download `dist/ai_texture_studio-1.0.0.zip` (or build it: `python tools/build.py`).
2. In Blender: **Edit › Preferences › Get Extensions › ▾ (top right) › Install from Disk…** and pick the zip.
   You can also drag and drop the zip onto the Blender window.
3. Enable **Allow Online Access** (Preferences › System › Network) if you use a hosted provider. The add-on respects this switch; local ComfyUI on `127.0.0.1` works without it.
4. Open **Preferences › Add-ons › AI Texture Studio**, choose a provider and set up its key (see below).
5. In the 3D Viewport press **N** and open the **AI Texture** tab.

## Configure a provider (short version, details in [docs/PROVIDERS.md](docs/PROVIDERS.md))

| Provider | What you need | Key environment variable (default) |
|---|---|---|
| OpenAI GPT Image | API key with image access; model `gpt-image-1.5` (presets: 1, 1-mini, 2) | `OPENAI_API_KEY` |
| Stability AI | API key (credits) | `STABILITY_API_KEY` |
| Google Gemini | Gemini API key; model `gemini-2.5-flash-image` | `GEMINI_API_KEY` |
| ComfyUI | A running ComfyUI server plus a checkpoint name; a depth ControlNet is recommended | none (optional token) |

**Key storage options**, set per provider:

- **Environment variable** (recommended): nothing is stored by Blender.
- **Session only**: kept in memory and gone when Blender quits.
- **System keyring**: needs the `keyring` package.
- **Blender preferences**: **plain text** in `userpref.blend`; least secure.

Keys are redacted from every message, history entry and console print.

The first request to a hosted provider shows exactly what will be sent and asks you to confirm. Hosted providers bill per generated image. A multi-view run costs **one request per enabled view**, but views with nothing left to paint are skipped for free. A correction costs one request.

## Quick start

1. Select a mesh. **Analyze Mesh** reports dimensions, UV islands, overlap, materials, existing textures, flipped normals, and how much of the surface the enabled cameras can see.
   No UVs? **Create UV Map** adds a *new* UV map (Smart UV, Lightmap or Cube); your data is untouched.
2. **Prompt & Reference:** write a prompt and, optionally, pick a reference image.
3. **Style & Resolution:** choose a style, material, detail, wear, prompt strength and an optional colour palette. Texture size is 512–4096; view size is 512–1024.
4. Click **Generate (N requests)**. Each finished view appears on the model immediately as its own layer.
5. **Local Refinement:** paint a mask on the model (**Paint Mask**) or select faces. Describe the change, then **Generate Correction**. The result is a *pending* layer: compare it, then **Accept** or **Reject**.
6. **PBR Maps:** generate roughness, metallic, normal and height approximations (each labelled as such) plus an AO map baked from the geometry.
7. **Export:** **Save Textures** (clearly named PNGs plus an optional ORM pack) and **Export Model** (`.glb` or `.fbx`).

Example prompts:

- *Weathered dark iron armor with hand-painted scratches and subtle edge wear.* Style: Hand-painted, Material: Metal, Wear 0.6.
- *Stylized low-poly orc skin, muted green tones, painted shadows, and subtle color variation.* Style: Low-poly; enable a palette.
- *Old wooden shield with chipped paint, visible wood grain, and a worn medieval appearance.* View preset: Front + Back.
- *Realistic leather straps with stitching, creases, and darkened edges.* Style: Realistic, Material: Leather.

More in [docs/USAGE.md](docs/USAGE.md).

## Feature status

### Implemented, with automated tests

- **Sidebar UI** with sections: Selected Object, Prompt & Reference, Style & Resolution, Generate, Multi-View, Local Refinement, Texture Layers (+ History), PBR Maps, Export, and AI Provider. Every panel's draw code is validated in tests.
- **Generation modes:**
  - text-only, reference-only, or text + reference generation
  - negative prompts: a native field where the provider has one, otherwise explicit "Avoid: …" text
  - fixed seed where supported
  - preview of the exact prompt and parameters sent
- **Style controls** that change the actual request: style (8 presets), material, detail, wear, prompt strength (mapped to provider parameters such as `control_strength`, `input_fidelity` and denoise), colour palette (prompt plus optional output palette lock), and pixel-art output filter.
- **Multi-view projection** onto existing UVs:
  - 10 preset cameras plus custom viewport cameras; enable, reorder and run single views
  - perspective or orthographic projection, following the object's rotation
  - depth-buffer occlusion and facing-angle selection, with deferral of grazing angles to better views
  - silhouette rejection, seam cross-fade bands, auto-alignment and colour matching
  - fill for hidden areas, clearly labelled *not AI-generated*
  - per-view layers, and inspection of each view's render / mask / raw AI output / aligned output / depth map
- **Consistent multi-object texturing:** shared art direction, seed and reference; the first object's result is fed to the rest as a reference.
- **Tileable material mode:** a seamless swatch is generated and tiled over the UVs.
- **Local refinement:**
  - masks from viewport painting (Texture Paint), faces selected in Edit Mode, or both; the mask is a normal image you can also edit in the Image Editor
  - grow and feather controls
  - viewport-based or UV-space correction
  - only masked pixels change
  - pending layer with Accept / Reject, iterative refinement and undo
- **Layers:**
  - base (import an existing texture), view, fill, correction, paint and tile layers
  - visibility, opacity, 8 blend modes, rename, duplicate, delete, reorder, merge down and flatten
  - edit layer pixels, change resolution
  - own **undo/redo**, generation history with restore, and an editable prompt history
- **Materials:** never destroys user materials. Each slot is copied (`<name>_ATS`), or one single material is used. Originals are protected and can be restored with one click.
  - Correct colour spaces, tangent-space Normal Map node bound to the UV map, and a glTF "Material Output" group for AO.
- **PBR:**
  - roughness, normal and height derived from the base colour (labelled approximations)
  - metallic from a preset or value, optionally masked (user-authored)
  - AO baked with Cycles from the geometry
  - ORM channel pack and a DirectX normal copy
- **Export and saving:** save textures, pack into the `.blend`, export glTF/FBX, and transfer (bake) the maps onto another UV map.
- **Robustness:**
  - background threads with main-thread scene updates; Blender stays responsive (≈0.1 s of main-thread work per finished view at 2048²)
  - cancellation, Retry, and rate-limit back-off honouring `Retry-After`
  - timeouts, plus validation of the image data that comes back
  - safe recovery after failed or partial runs
  - files are moved next to the `.blend` on save, with relative paths that survive "Save As"
- **Scene preservation:** mesh, topology, transforms, modifiers, rig, shape keys, vertex groups and material assignments are verified unchanged by tests.

### Partially implemented

- **Cancelling an in-flight HTTP request:** Python cannot abort the socket. The job stops at the next checkpoint and the late result is discarded; ComfyUI also receives `/interrupt`.
- **Viewport mask painting:** this uses Blender's own Texture Paint. On Blender 4.3+ (brush assets) the add-on tries to set a white brush colour, but you may need to pick white once yourself.
- **Live providers:** the provider code is complete and request formats are tested against local fakes, but it is not live-verified here (see above).

### Planned (architecture ready)

- AI-*estimated* PBR maps via dedicated material-estimation models (for example, ComfyUI workflows). Today only base colour is AI-generated.
- Reference-image (IP-Adapter) support in the built-in ComfyUI workflow. Custom workflows can already use `{{REFERENCE}}`.
- More providers through the provider interface: Black Forest Labs Flux Fill/Kontext, fal.ai, Replicate.
- Optional view upscaling, UDIM tiles, and GPU rasterization.

## Known limitations

See [docs/TROUBLESHOOTING.md#known-limitations](docs/TROUBLESHOOTING.md#known-limitations). Most important:

- **Consistency:** image models are not 3D-aware. Sequential inpainting with context, seam bands and colour matching keep views coherent, but small detail mismatches at seams can occur. Fix them with 3/4 views, a custom viewport view, or Local Refinement.
- **Hidden surfaces:** surfaces no camera can see are colour-extended, not painted. They are labelled and reported in Analyze/History.
- **Mirrored/overlapping UVs:** these islands share one texture.
- **PBR maps:** roughness, normal and height are image-derived approximations, not measured material data.

## Tests

```bash
# Blender integration + unit tests with the PyPI "bpy" module (Python 3.11)
python3.11 -m venv venv && venv/bin/pip install bpy==4.2.23 numpy==1.26.4 pytest
venv/bin/python -m pytest tests

# or inside your own Blender
blender --background --factory-startup --python tests/blender/run_in_blender.py
```

See [docs/TESTING.md](docs/TESTING.md) for what is covered and the manual checklist.

## Documentation

- [docs/USAGE.md](docs/USAGE.md): complete workflow, tips for stylized game assets.
- [docs/PROVIDERS.md](docs/PROVIDERS.md): setup, capabilities, costs, data handling, how to add a provider.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): research summary, design, multi-view algorithm, roadmap.
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md): common problems, known limitations.
- [docs/TESTING.md](docs/TESTING.md): automated tests and manual integration procedure.
