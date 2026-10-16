"""Scene / object data stored in the .blend file."""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty,
                       IntProperty, PointerProperty, StringProperty)

from ..core.camera import VIEW_LABELS
from ..core.compositing import BLEND_MODES
from ..core.prompting import STYLES

RESOLUTIONS = [("512", "512", ""), ("1024", "1024", ""), ("2048", "2048", ""), ("4096", "4096",
               "Memory heavy: ~70 MB per layer in RAM")]
VIEW_SIZES = [("512", "512", "Fast, low detail"), ("768", "768", ""), ("1024", "1024", "Native size of most models")]
MATERIALS = [
    ("AUTO", "From Prompt", "Let the prompt decide the material"),
    ("METAL", "Metal", ""), ("PAINTED_METAL", "Painted Metal", ""), ("WOOD", "Wood", ""),
    ("LEATHER", "Leather", ""), ("STONE", "Stone", ""), ("FABRIC", "Fabric", ""), ("SKIN", "Skin", ""),
    ("PLASTIC", "Plastic", ""), ("CERAMIC", "Ceramic", ""),
]
# Blender 5.0+ requires FILE_PATH/DIR_PATH properties to opt in to "//" paths.
PATH_OPTS = {"PATH_SUPPORTS_BLEND_RELATIVE"} if bpy.app.version >= (5, 0, 0) else set()

PRESET_ITEMS = [(k, v, "") for k, v in VIEW_LABELS.items()]
LAYER_KINDS = [("VIEW", "View", ""), ("FILL", "Fill", ""), ("CORRECTION", "Correction", ""),
               ("BASE", "Base", ""), ("PAINT", "Paint", ""), ("TILE", "Tile", ""), ("MERGED", "Merged", "")]


def _redraw_composite(self, context):
    from . import layers
    obj = self.id_data if isinstance(self.id_data, bpy.types.Object) else None
    if obj is not None:
        layers.schedule_recomposite(obj)


class ATS_ViewItem(bpy.types.PropertyGroup):
    preset: EnumProperty(name="View", items=PRESET_ITEMS, default="FRONT")
    enabled: BoolProperty(name="Use", default=True)
    label: StringProperty(name="Label")
    custom_view: FloatVectorProperty(size=16)
    custom_proj: FloatVectorProperty(size=16)
    custom_ortho: BoolProperty(default=False)
    status: StringProperty()
    debug_dir: StringProperty()   # folder with render/mask/generated/aligned PNGs of the last run
    debug_object: StringProperty()


class ATS_LayerItem(bpy.types.PropertyGroup):
    uid: StringProperty()
    name: StringProperty(name="Name")
    kind: EnumProperty(items=LAYER_KINDS, default="VIEW")
    visible: BoolProperty(name="Visible", default=True, update=_redraw_composite)
    opacity: FloatProperty(name="Opacity", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
                           update=_redraw_composite)
    blend: EnumProperty(name="Blend", items=[(m, m.replace("_", " ").title(), "") for m in BLEND_MODES],
                        default="NORMAL", update=_redraw_composite)
    filepath: StringProperty(subtype="FILE_PATH", options=PATH_OPTS)
    pending: BoolProperty(default=False, description="Correction awaiting Accept / Reject")
    created: StringProperty()
    info: StringProperty()
    history_uid: StringProperty()


class ATS_HistoryItem(bpy.types.PropertyGroup):
    uid: StringProperty()
    timestamp: StringProperty()
    kind: StringProperty()
    status: StringProperty()
    prompt: StringProperty()
    negative: StringProperty()
    provider: StringProperty()
    model: StringProperty()
    detail: StringProperty()
    layer_uid: StringProperty()
    filepath: StringProperty(subtype="FILE_PATH", options=PATH_OPTS)
    layer_name: StringProperty()


class ATS_PromptItem(bpy.types.PropertyGroup):
    prompt: StringProperty(name="Prompt")
    negative: StringProperty(name="Negative")
    style: StringProperty()
    timestamp: StringProperty()


class ATS_ObjectState(bpy.types.PropertyGroup):
    uid: StringProperty()
    layers: CollectionProperty(type=ATS_LayerItem)
    active_layer: IntProperty(name="Active Layer", default=0)
    history: CollectionProperty(type=ATS_HistoryItem)
    active_history: IntProperty(default=0)
    undo_stack: StringProperty(default="[]")
    redo_stack: StringProperty(default="[]")
    material_state: StringProperty(default="")
    texture_size: IntProperty(default=0)
    uv_map: StringProperty(name="UV Map", description="UV map that receives the generated texture")
    analysis: StringProperty(default="")
    base_image: PointerProperty(type=bpy.types.Image)
    mask_image: PointerProperty(type=bpy.types.Image)
    maps: StringProperty(default="{}")   # json kind -> image name
    pixelate: BoolProperty(name="Pixelate Output", default=False, update=_redraw_composite,
                           description="Pixel-art output filter (block size in texels)")
    pixel_size: IntProperty(name="Pixel Size", default=8, min=2, max=64, update=_redraw_composite)
    pixel_colors: IntProperty(name="Levels", default=0, min=0, max=64, update=_redraw_composite,
                              description="Colour levels per channel (0 = keep)")
    palette_lock: FloatProperty(name="Palette Lock", default=0.0, min=0.0, max=1.0, subtype="FACTOR",
                                update=_redraw_composite,
                                description="Pull output colours towards the art-direction palette")
    gutter: IntProperty(name="Padding", default=8, min=0, max=64, update=_redraw_composite,
                        description="Edge padding (texels) around UV islands to avoid seams in mip-maps")


class ATS_SceneSettings(bpy.types.PropertyGroup):
    # Prompt & reference
    prompt: StringProperty(name="Prompt", description="Describe the desired surface appearance")
    negative: StringProperty(name="Negative", description="What to avoid (native field where supported)")
    reference: PointerProperty(type=bpy.types.Image, name="Reference",
                               description="Optional reference image guiding design and colours")
    use_reference: BoolProperty(name="Use Reference", default=True)
    art_direction: StringProperty(name="Art Direction",
                                  description="Shared description applied to every object for a consistent look")
    # Style
    style: EnumProperty(name="Style", items=[(k, v[0], v[1]) for k, v in STYLES.items()], default="STYLIZED")
    material: EnumProperty(name="Material", items=MATERIALS, default="AUTO")
    detail: FloatProperty(name="Detail", default=0.5, min=0, max=1, subtype="FACTOR")
    wear: FloatProperty(name="Wear & Damage", default=0.3, min=0, max=1, subtype="FACTOR")
    strength: FloatProperty(name="Prompt Strength", default=0.7, min=0, max=1, subtype="FACTOR",
                            description="Low: stay close to renders/reference. High: follow the prompt freely")
    delight: BoolProperty(name="Even Lighting (albedo)", default=True,
                          description="Ask for shadow-free albedo so the texture works from every angle")
    use_palette: BoolProperty(name="Colour Palette", default=False)
    palette_count: IntProperty(name="Colours", default=3, min=1, max=6)
    color_1: FloatVectorProperty(name="", subtype="COLOR_GAMMA", default=(0.30, 0.33, 0.36), min=0, max=1)
    color_2: FloatVectorProperty(name="", subtype="COLOR_GAMMA", default=(0.55, 0.42, 0.25), min=0, max=1)
    color_3: FloatVectorProperty(name="", subtype="COLOR_GAMMA", default=(0.75, 0.70, 0.58), min=0, max=1)
    color_4: FloatVectorProperty(name="", subtype="COLOR_GAMMA", default=(0.40, 0.15, 0.12), min=0, max=1)
    color_5: FloatVectorProperty(name="", subtype="COLOR_GAMMA", default=(0.18, 0.30, 0.20), min=0, max=1)
    color_6: FloatVectorProperty(name="", subtype="COLOR_GAMMA", default=(0.10, 0.10, 0.12), min=0, max=1)
    use_seed: BoolProperty(name="Fixed Seed", default=False)
    seed: IntProperty(name="Seed", default=1234, min=0)
    # Resolution / mode
    texture_size: EnumProperty(name="Texture", items=RESOLUTIONS, default="2048")
    view_size: EnumProperty(name="View Size", items=VIEW_SIZES, default="1024")
    mode: EnumProperty(name="Mode", items=[
        ("MULTIVIEW", "Multi-View Projection", "Paint the object from several cameras and project onto its UVs"),
        ("TILE", "Tileable Material", "Generate a seamless swatch and tile it over the UVs"),
    ], default="MULTIVIEW")
    tile_scale: FloatProperty(name="Tiling", default=4.0, min=0.1, max=100.0)
    material_mode: EnumProperty(name="Material Setup", items=[
        ("PER_SLOT", "Copy Each Slot", "Duplicate each slot's material and wire the textures in (keeps slot settings)"),
        ("SINGLE", "Single Material", "One new material on all slots (simplest for game engines)"),
    ], default="PER_SLOT")
    use_modifiers: BoolProperty(name="Use Modifiers", default=True,
                                description="Project onto the evaluated mesh (as displayed, with modifiers)")
    # Multi-view
    views: CollectionProperty(type=ATS_ViewItem)
    active_view: IntProperty(default=0)
    ortho: BoolProperty(name="Orthographic", default=False)
    fov: FloatProperty(name="Field of View", default=35.0, min=5.0, max=90.0)
    align_to_object: BoolProperty(name="Follow Object Rotation", default=True)
    min_facing: FloatProperty(name="Min Facing", default=0.3, min=0.0, max=0.95, subtype="FACTOR",
                              description="Texels seen at a flatter angle are left for better views")
    band_px: IntProperty(name="Seam Blend", default=12, min=0, max=64,
                         description="Pixels of existing texture repainted and cross-faded at view seams")
    auto_align: BoolProperty(name="Auto Align", default=True,
                             description="Correct small offsets in generated images (providers that may drift)")
    color_match: BoolProperty(name="Match Colours", default=True,
                              description="Match each new view's colours to the existing texture")
    fill_unseen: BoolProperty(name="Fill Hidden Areas", default=True,
                              description="Extend colour into surfaces no camera can see")
    regenerate: BoolProperty(name="Start From Scratch", default=False,
                             description="Ignore existing layers when deciding what to paint (they stay in the stack)")
    chain_reference: BoolProperty(name="Share Look Across Objects", default=True,
                                  description="When texturing several objects, feed the first result as an extra "
                                              "reference so all objects match")
    # Refinement
    refine_prompt: StringProperty(name="Correction", description="What the selected area should become")
    refine_source: EnumProperty(name="Mask", items=[
        ("PAINT", "Painted Mask", "Use the mask painted with 'Paint Mask'"),
        ("FACES", "Selected Faces", "Use faces selected in Edit Mode"),
        ("BOTH", "Both", "Union of painted mask and selected faces"),
    ], default="PAINT")
    refine_space: EnumProperty(name="Method", items=[
        ("VIEW", "From Viewport", "AI sees the object from the current 3D view (best for most fixes)"),
        ("UV", "In UV Space", "Edit the texture atlas directly (works for hidden areas)"),
    ], default="VIEW")
    refine_feather: IntProperty(name="Feather", default=6, min=0, max=64)
    refine_grow: IntProperty(name="Grow", default=2, min=0, max=64)
    # PBR
    pbr_roughness: BoolProperty(name="Roughness", default=True)
    pbr_metallic: BoolProperty(name="Metallic", default=True)
    pbr_normal: BoolProperty(name="Normal", default=True)
    pbr_height: BoolProperty(name="Height", default=False)
    pbr_ao: BoolProperty(name="Ambient Occlusion", default=False)
    rough_override: BoolProperty(name="Override Base", default=False)
    rough_base: FloatProperty(name="Base Roughness", default=0.6, min=0, max=1, subtype="FACTOR")
    rough_variation: FloatProperty(name="Variation", default=0.25, min=0, max=1, subtype="FACTOR")
    metal_value: FloatProperty(name="Metallic", default=-1.0, min=-1.0, max=1.0,
                               description="-1 = from material preset")
    metal_from_mask: BoolProperty(name="Limit to Mask", default=False,
                                  description="Metallic only where the refinement mask is painted")
    normal_strength: FloatProperty(name="Normal Strength", default=2.0, min=0.0, max=20.0)
    height_detail: FloatProperty(name="Height Detail", default=1.0, min=0.0, max=4.0)
    ao_samples: IntProperty(name="AO Samples", default=32, min=4, max=1024)
    ao_multiply: BoolProperty(name="Multiply AO into Colour", default=False,
                              description="Darken base colour with AO (common for hand-painted assets)")
    # Export
    export_dir: StringProperty(name="Folder", subtype="DIR_PATH", default="//export/", options=PATH_OPTS)
    export_format: EnumProperty(name="Format", items=[("GLB", "glTF Binary (.glb)", ""), ("FBX", "FBX (.fbx)", "")],
                                default="GLB")
    export_orm: BoolProperty(name="Write ORM Pack", default=True,
                             description="Channel-packed AO/Roughness/Metallic texture (glTF, Unreal)")
    export_dx_normal: BoolProperty(name="DirectX Normal Copy", default=False,
                                   description="Also write a green-flipped normal map (Unreal, some engines)")
    transfer_uv: StringProperty(name="Target UV")
    # History
    prompt_history: CollectionProperty(type=ATS_PromptItem)
    active_prompt: IntProperty(default=0)


CLASSES = (ATS_ViewItem, ATS_LayerItem, ATS_HistoryItem, ATS_PromptItem, ATS_ObjectState, ATS_SceneSettings)


def palette(settings: ATS_SceneSettings) -> list:
    cols = [settings.color_1, settings.color_2, settings.color_3, settings.color_4, settings.color_5,
            settings.color_6][: settings.palette_count]
    return [tuple(c) for c in cols]


def register_props():
    bpy.types.Scene.ats = PointerProperty(type=ATS_SceneSettings)
    bpy.types.Object.ats = PointerProperty(type=ATS_ObjectState)


def unregister_props():
    del bpy.types.Object.ats
    del bpy.types.Scene.ats
