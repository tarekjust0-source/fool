# Using AI Texture Studio

All tools live in **3D Viewport › Sidebar (N) › AI Texture**.

## 1. Selected Object

- **UV**: the UV map that receives the texture. By default this is the map marked for rendering. Existing UVs are always preferred and never modified.
- **Create UV Map** adds a **new** UV map (default name `ATS_UV`):
  - **Smart UV Project**: hard-surface props.
  - **Lightmap Pack**: one island per face; no overlaps, but many seams.
  - **Cube Projection**: boxy objects.

  Automatic unwraps can produce many seams and uneven texel density; hand-made UVs give better results. Tick *Make Active for Render/Export* if your game engine should use the new map.
- **Analyze Mesh** reports:
  - size and triangle count
  - UV islands, the share of UV space used, overlaps and UVs outside 0–1
  - material slots and existing textures with their resolutions
  - modifiers, shape keys and armatures (the current pose is used)
  - shared mesh data and flipped normals
  - **how much of the surface the enabled cameras can see**

## 2. Prompt & Reference

- **Prompt:** describe surface appearance, not the object's shape (the shape comes from your mesh).
- **Negative:** things to avoid. Stability and ComfyUI use it natively; for OpenAI and Gemini it becomes an "Avoid: …" sentence.
- **Art Direction:** shared text added to every object and request, for consistent sets of assets.
- **Reference image:** guides design and colours. Load it with the folder button or pick an existing image. It is only uploaded when you generate.
- **Preview Prompt:** shows the exact text and parameters that will be sent, plus notes on what the current provider ignores.
- **Recent prompts:** click ⤓ to reuse a prompt; the text is editable in place.

## 3. Style & Resolution

| Control | Effect on the request |
|---|---|
| Style (Realistic, Stylized, Hand-painted, Low-poly, Fantasy, Sci-fi, Cartoon, Pixel art) | Style description plus style-specific negatives, and Stability `style_preset` |
| Material | Material words; also the default roughness/metallic of the PBR approximations |
| Detail | "minimal / moderate / rich / intricate detail" |
| Wear & Damage | "pristine … heavily weathered, chipped, scratched" |
| Prompt Strength | Stability `control_strength` / `composition_fidelity`, OpenAI `input_fidelity` (GPT Image 1/1.5), ComfyUI denoise, plus adherence wording |
| Even Lighting | Asks for shadow-free albedo (soft painted shading for stylized styles) |
| Colour Palette | Hex colours in the prompt; **Palette Lock** additionally pulls the final texture towards the palette |
| Pixelate (Pixel art) | Output filter: texel blocks and colour levels |
| Texture / View Size | Atlas resolution (512–4096) / resolution of each camera view (512–1024) |
| Fixed Seed | Sent to providers that support seeds (Stability, ComfyUI) |

**Stylized game assets:** use *Hand-painted*, *Low-poly* or *Cartoon* with low Detail (0.2–0.4) and a 3–4 colour palette. Keep *Even Lighting* on, so shading is painted softly rather than baked as hard shadows. Add *Palette Lock 0.3–0.6* for a tightly controlled look.

## 4. Generate

- **Multi-View Projection** (default) or **Tileable Material** (a seamless swatch tiled over the UVs at *Tiling* scale).
- **Material Setup:**
  - *Copy Each Slot* duplicates each slot's material as `<name>_ATS`, keeping per-slot settings; originals are kept and protected.
  - *Single Material* puts one `ATS_<object>` material on all slots.
- **Start From Scratch:** paint everything again. Existing layers stay in the stack, so you can compare or hide them.
- **Generate All Selected:** textures several objects in one job. The active object goes first and, with 🔗 *Share Look Across Objects*, its first result becomes a reference image for the others.

Progress, cancellation and errors appear at the top of the panel. **Retry** repeats the last request; already textured areas are skipped, so a retry after a partial failure only pays for the missing views.

## 5. Multi-View

- The view list holds preset cameras (Front, Right, Left, Back, Top, Bottom and four 3/4 views) and **custom viewport views** (➕ captures the current viewport angle; useful for armpits, undersides and other hidden areas).
- Order matters: earlier views define the design and later views continue it.
- **▶ on a row regenerates only that view.** Its previous layers are ignored while planning, so it repaints exactly the area it painted before, with every other view visible to the AI as context. The other views' layers are not touched. The old version is kept, hidden and renamed "(previous)": toggle 👁 to compare, or delete it. One request.
- **Preset menu:** 6 axis views; *Character* (overlapping 3/4 views, smoother on organic shapes); *Front + Back* (shields, signs); *All*.
- **Inspect last run:** *Render* (what the AI saw), *Mask* (what it was asked to paint), *AI Output* (raw), *Aligned*, *Depth*. The view's layer in Texture Layers is its projection result.
- **Settings:**
  - *Min Facing*: texels seen flatter than this are left for a better view.
  - *Seam Blend*: pixels of existing texture repainted and cross-faded at view seams.
  - *Auto Align*: corrects small offsets from providers that redraw the whole frame.
  - *Match Colours*: matches each new view to the existing texture.
  - *Fill Hidden Areas*: colour-extends surfaces no camera saw. The colour comes from the nearest painted surface in 3D; it is **not AI-generated**.

**What counts as already painted.** A view only paints texels that are not yet painted with confidence:

- Layer kinds:
  - **Low-angle views**: AI pixels seen only at a flat angle, kept as a fallback.
  - **Hidden-area fill**: colour extension.
- Neither kind blocks a later view. A view you add for a hidden area *will* paint it, and the obsolete fill is replaced once the run succeeds.
- Layers marked **Protect** (🔒) are never written by generation, not even by seam blending. This is the default for imported base textures, paint layers and accepted corrections.
- *Start From Scratch* ignores all unprotected layers.

**How a view is painted.** The mesh is rendered from the camera, showing the texture so far, with grey clay where nothing is painted yet. The AI paints the clay region, plus a thin band of existing texture so it continues seamlessly. The result is projected back only onto texels that camera sees well.

## 6. Local Refinement

1. Choose the mask source:
   - **Paint Mask**: paint white on the model in Texture Paint mode, then press *Done Painting*.
   - **Selected Faces**: select faces in Edit Mode, or use the face button to add them to the painted mask.
   - **Both**.
   - The mask is also an ordinary image (`ATS_<object>_Mask`), so you can edit it in the Image Editor.
2. Choose the method:
   - **From Viewport**: the AI sees the object from the current 3D view. Best for most fixes; orbit so the area faces you.
   - **In UV Space**: edits the atlas directly; works for hidden areas.
3. Type the correction (for example, *Replace this area with rusty iron*), then **Generate Correction**. The request includes the current texture around the area and your main prompt as design context.
4. The result is a **pending** layer. Toggle *Compare* to see before and after, then **Accept** or **Reject**. Only masked texels change. Repeat as often as you like; every step is undoable.

## 7. Texture Layers and History

- Top layer first. 👁 toggles visibility; you can set opacity, blend mode (8 modes), a name (double-click) and 🔒 *Protect*.
- Each layer shows its provenance:

  | Tag | Meaning |
  |---|---|
  | `AI` | view, correction or tile |
  | `AI·low` | low-angle views |
  | `extend` | colour extension, not AI |
  | `import` | imported texture |
  | `paint` | hand-painted |
  | `mixed` | merged |

  **Show Provenance Map** colours the model by where each pixel comes from:
  - green: AI
  - yellow: AI at a low angle
  - red: colour extension
  - blue: imported or painted
  - dark: unpainted

  It also reports the percentages.
- ➕ adds an empty paint layer. **Import Existing Texture** turns the object's current image into a base layer, so generation only fills what it doesn't cover.
- Layer operations: duplicate, delete, move, merge down, flatten.
- **Edit Layer Pixels** opens the layer in Texture Paint; finish with **Apply Layer Edits**.
- **Undo/Redo** buttons belong to the add-on's own layer history, which reliably tracks layer pixels. Layer files are never deleted by layer operations. Blender's Ctrl+Z also gets one step per finished generation.
- **History** lists every generation and correction with its status, provider and prompt. ⟲ restores that result as a new layer.
- **Apply Resolution** resamples all layers to the chosen texture size.
- **Clean Up Files** deletes files nothing references.

## 8. PBR Maps

Each map shows where it comes from:

| Map | Source |
|---|---|
| Base Color | AI-generated |
| Roughness | Approximation (base colour detail plus material preset; *Override Base*, *Variation*) |
| Metallic | User-authored (preset or value; *Limit to Mask* uses the refinement mask) |
| Normal | Approximation from estimated height (OpenGL convention; DirectX copy on export) |
| Height | Approximation (high-pass luminance); created but not connected, to keep export simple |
| AO | Baked from the actual geometry with Cycles; blocks Blender while baking |

Maps are wired into the Principled BSDF with correct colour spaces. AO is connected to a `glTF Material Output` group so the glTF exporter writes it; *Multiply AO into Colour* is for hand-painted looks.

## 9. Export

- **Save Textures:** writes `<Object>_BaseColor.png`, `_Roughness`, `_Metallic`, `_Normal_OpenGL`, `_Height`, `_AO`, plus optional `_ORM` (R = AO, G = Roughness, B = Metallic) and `_Normal_DirectX`.
- **Pack Into .blend:** embeds the maps.
- **Export Model:** `.glb` (modifiers applied) or `.fbx` (textures embedded), selected object only. Your selection is restored afterwards.
- **Transfer to UV Map:** re-bakes the maps onto another UV map of the mesh, for example the original game UVs when you textured on a fresh `ATS_UV`.
- **Restore Original Materials** puts the original slot materials back.

## Where files go

Layers and maps are PNG files. With a saved `.blend` they live in `//ATS_textures/<object>/`; before the first save they go to a session temp folder and are moved next to the `.blend` when you save. Paths are stored relative and remapped on *Save As*.
