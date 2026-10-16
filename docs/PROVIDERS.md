# AI providers

Choose a provider in **Preferences › Add-ons › AI Texture Studio**, or in the sidebar's **AI Provider** section. **Test Connection** makes a lightweight metadata request; it does not generate an image.

## Capability matrix

| | OpenAI GPT Image | Stability AI | Google Gemini | ComfyUI (local) |
|---|---|---|---|---|
| Endpoint(s) | `/v1/images/edits`, `/v1/images/generations` | `/v2beta/stable-image/control/structure`, `…/control/style-transfer`, `…/edit/inpaint`, `…/generate/core` | `/v1beta/models/{model}:generateContent` | `/upload/image`, `/prompt`, `/history`, `/view`, `/interrupt` |
| Mask | soft (alpha mask, model may redraw) | **true inpainting** | described plus a mask image (soft) | **true** (`SetLatentNoiseMask`) |
| Structure from the mesh | render as input image | **structure control** | render as input image | **depth ControlNet** (optional) |
| Reference image | yes (extra `image[]`) | first view only (style transfer) | yes (multi-image) | custom workflow (`{{REFERENCE}}`) |
| Negative prompt | as "Avoid:" text | native | as "Avoid:" text | native |
| Seed | – | yes | – | yes |
| Prompt Strength maps to | `input_fidelity` (GPT Image 1/1.5) plus wording | `control_strength`, `composition_fidelity` | wording | denoise |
| Auto-align needed | yes (enabled automatically) | no | yes | no |
| Data leaves your machine | yes (OpenAI) | yes (Stability AI) | yes (Google) | only to the server URL you set |
| Cost | per image (OpenAI pricing) | credits per call | per image (Google pricing) | free (your GPU) |

Requests per multi-view run equal the number of enabled views that still have something to paint; already finished views are skipped. A correction is one request. Prices change frequently, so check the provider's pricing page. The add-on shows the number of requests on the Generate button, and asks for confirmation before the first request to each hosted provider.

## OpenAI

1. Create an API key with image-generation access.
2. Set the environment variable `OPENAI_API_KEY` before starting Blender (recommended), or choose another key storage mode.
3. Model: `gpt-image-1.5` (default); presets include `gpt-image-1`, `gpt-image-1-mini` and `gpt-image-2`. Any model id can be typed. Quality: low, medium, high or auto.
4. *API URL* can point to a compatible gateway, for example `https://my-proxy/v1`.

Notes: output size is 1024×1024 for square views. `input_fidelity` is sent only for models known to accept it, and the request is retried automatically without it if the server rejects the parameter.

## Stability AI

1. Get an API key at platform.stability.ai; requests use credits.
2. Set `STABILITY_API_KEY`.
3. *Grow Mask* (inpaint) adds overlap at seams; the default is 5.

Endpoint selection is automatic:

- First view: **structure** control, or **style transfer** when a reference image is set.
- Following views and corrections: **inpaint**.
- Tileable swatches: **core**.

## Google Gemini

1. Create a key in Google AI Studio and set `GEMINI_API_KEY`.
2. Model: `gemini-2.5-flash-image` (default) or another image-capable Gemini model id, for example `gemini-3-pro-image-preview`.

Notes: the key is sent in the `x-goog-api-key` header, never in the URL. Free-tier data may be used by Google to improve its products; see Google's terms.

## ComfyUI (local, private)

1. Run ComfyUI (default `http://127.0.0.1:8188`).
2. *Model* is the checkpoint **file name** as listed in ComfyUI, for example `sd_xl_base_1.0.safetensors`.
3. Recommended: set *Depth ControlNet* to a depth ControlNet that matches your checkpoint. The add-on renders an exact depth map of your mesh for every view.
4. Steps, CFG, sampler and scheduler are configurable; built-in nodes only, no custom nodes required.
5. **Custom workflow:** export your graph with *Save (API Format)* and point *Custom Workflow* to it. These placeholders are replaced:
   `{{PROMPT}} {{NEGATIVE}} {{SEED}} {{DENOISE}} {{STEPS}} {{CFG}} {{IMAGE}} {{MASK}} {{DEPTH}} {{REFERENCE}} {{WIDTH}} {{HEIGHT}}`.
   Exact-match placeholders keep their numeric type.

The mask is uploaded as a greyscale PNG (white = paint) and read with `LoadImageMask` (red channel). A remote ComfyUI server needs Blender's *Allow Online Access*; `127.0.0.1` and `localhost` do not. An optional bearer token can be stored like an API key.

Model licences vary: check the licence of the checkpoint you use (for example, SDXL OpenRAIL++, or Flux-dev's non-commercial terms).

## Security and data handling

- Keys are never hardcoded, never written to generated files, and redacted from every error, history entry and console print. Both known key values and common key patterns are scrubbed.
- **Key storage modes:**

  | Mode | Behaviour |
  |---|---|
  | Environment variable | recommended |
  | Session only | memory |
  | System keyring | needs the `keyring` package |
  | Blender preferences | **plain text** in `userpref.blend` |

- What is sent: renders of the selected object from each camera (the object only, on a neutral background), masks, depth maps (ComfyUI), your prompt and your reference image. Model files and scene data are never uploaded.
- Nothing is sent until you press Generate or Generate Correction. Blender's *Allow Online Access* switch is respected.

## Adding a provider

Subclass `providers.base.Provider` and implement `_generate(req, cfg, cancel)`, which returns an `ImageResult`, and optionally `_test_connection`. Declare `capabilities()` honestly, then call `providers.register(MyProvider())`. Transport errors are classified for you: `ProviderError` kinds drive the UI messages and Retry. See `providers/stability_provider.py` for a compact example.

## Evaluated, not implemented

| Option | Reason |
|---|---|
| Meshy / Tripo "retexture" APIs | They upload the model and return new UVs/meshes, which conflicts with preserving your UVs and topology. |
| Black Forest Labs Flux Fill / Kontext | Strong inpainting and editing; a good candidate for the next provider. |
| fal.ai / Replicate | Generic model hosts; easy to add through the interface. |
