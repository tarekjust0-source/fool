# Troubleshooting

| Symptom | Cause / fix |
|---|---|
| **Generate is greyed out** | Select a mesh object, and wait for or cancel a running job. |
| "has no UV map" | Use **Create UV Map** (adds a new map) or unwrap manually. |
| "No API key configured" | Preferences › AI Texture Studio: check *Key Storage*. With *Environment Variable*, the variable must exist **before** Blender starts. The status line says whether a key was found; it never shows the key. |
| "Online access is disabled" | Preferences › System › Network › **Allow Online Access** (needed for hosted providers and remote ComfyUI). |
| HTTP 401/403 | Wrong or expired key, or the key lacks image access. |
| 402 / "quota" / "credits" | Top up your provider account. |
| 429 rate limit | Retries happen automatically with back-off; press **Retry** later if it persists. |
| Timeout | Raise *Request Timeout* (Preferences). High-quality images can take 1–2 minutes. |
| "content policy" / blocked | Rephrase the prompt; some words trigger filters. |
| "returned a blank (single-colour) image" / "could not be decoded" | Invalid response; nothing was applied. Press **Retry**. |
| ComfyUI "rejected the workflow" | The checkpoint or ControlNet file name is wrong (it must match ComfyUI's list exactly), or a custom workflow is not in **API format**. |
| Texture appears shifted or doubled on the model | Keep **Auto Align** on (OpenAI/Gemini). Inspect the view's *Render* vs *AI Output*. Lower *Prompt Strength* for closer adherence. |
| Visible seams between views | Raise *Seam Blend* (16–24), enable 3/4 views (*Character* preset), keep *Match Colours* on, then fix leftovers with Local Refinement. |
| Some areas are smeared or blurry | Those areas were hidden from every camera and colour-extended (see *Hidden-area fill* layer and History). Add a viewport view facing them, or refine them in UV space. |
| Correction says "not visible from this view" | Orbit so the masked area faces the viewport, or switch the method to *In UV Space*. |
| "mask does not touch the surface" | The mask lies in empty UV space; paint on the model or select faces. |
| Mask brush paints in colour, not white | Blender 4.3+ brush assets: pick white in the brush colour once. |
| Ctrl+Z removed my texture | Blender undo also reverts add-on data. Redo with Ctrl+Shift+Z, or restore it from **History** (files are kept on disk). Use the panel's own Undo/Redo for layer edits. |
| Texture is black or pink after moving the .blend | Keep the `ATS_textures` folder next to the .blend. Paths are relative. |
| Slow on huge meshes or 4096 textures | Projection is CPU/numpy. Use 2048 while iterating, or texture a decimated copy and use **Transfer to UV Map**. 4096 needs about 70 MB RAM per layer. |

Console output (Window › Toggle System Console on Windows) contains redacted tracebacks for unexpected errors.

## Known limitations

- **Multi-view consistency.** Hosted image models are not 3D-aware. Context inpainting, seam bands and colour matching keep views coherent, but fine details can still differ across a seam. That's why each view is a separate layer and refinement exists.
- **Texel density.** Each view is about 1024 px, so large objects at 4096 get softer detail unless you add closer viewport views.
- **Hidden surfaces.** These are colour-extended, not painted, and reported as such.
- **Overlapping or mirrored UVs** share texture. Analyze warns about them; create a fresh UV map if they must differ.
- **Deforming meshes.** The current pose and shape-key mix are projected; use the rest pose for best results.
- **PBR.** Roughness, normal and height are image-derived approximations; metallic is user-authored; AO is a real geometric bake. Only base colour is AI-generated.
- **Provider specifics:**
  - OpenAI and Gemini: no seed, soft masks; the pipeline still only applies masked pixels.
  - Stability: uses the reference image for the first view only.
  - ComfyUI: built-in workflow has no reference image (use a custom workflow).
- **Cancellation** cannot abort a request already being processed by the provider; it may still be billed.
- **UDIM / UVs outside 0–1** are not supported.
